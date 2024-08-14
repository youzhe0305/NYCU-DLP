import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models import MaskGit as VQGANTransformer
from utils import LoadTrainData
import yaml
from torch.utils.data import DataLoader
import random

#TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    def __init__(self, args, MaskGit_CONFIGS):
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=args.device)
        self.optim,self.scheduler = self.configure_optimizers(args)
        self.prepare_training()
        
    @staticmethod
    def prepare_training():
        os.makedirs("transformer_checkpoints", exist_ok=True)

    def train_one_epoch(self, train_loader, args, ith_epoch): # one epoch
        
        progress_bar = tqdm(train_loader) # 要能夠獲取object長度才能顯示進度條
        total_loss = torch.zeros(1).to(args.device)
        iterations = 0
        for idx, img in enumerate(progress_bar):
            
            img = img.to(args.device)
            logits, z_indices = self.model(img) # (b, n_token, n_codebook_vector), (b,h*w)
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), z_indices.view(-1)) # input: (B,C) target: (N)
            total_loss += loss
            iterations += 1
            loss.backward()
            
            if idx % args.accum_grad == 0:
                self.optim.step()
                self.optim.zero_grad()
            progress_bar.set_description(f'Training, Epoch {ith_epoch}: averge loss: {total_loss.item() / iterations}', refresh=False)
            progress_bar.set_postfix({
                'averge loss':f'{total_loss.item() / iterations}'
            }, refresh=False)
            progress_bar.refresh()     
        return total_loss.item() / iterations
        
    def eval_one_epoch(self, val_loader, args, ith_epoch):
        
        progress_bar = tqdm(val_loader)
        total_loss = torch.zeros(1).to(args.device)
        iterations = 0
        for idx, img in enumerate(progress_bar):

            img = img.to(args.device)
            logits, z_indices = self.model(img) # (b, n_token, n_codebook_vector), (b,h*w)
            loss = F.cross_entropy(logits, z_indices.view(-1, logits.shape[-1]), z_indices.view(-1)) # input: (B,C) target: (N)
            total_loss += loss
            iterations += 1
            loss.backward()
            progress_bar.set_description(f'Validation, Epoch {ith_epoch}: averge loss: {total_loss.item() / iterations}', refresh=False)
            progress_bar.set_postfix({
                'averge loss':f'{total_loss.item() / iterations}'
            }, refresh=False)   
            progress_bar.refresh()     
            return total_loss.item() / iterations

    def configure_optimizers(self, args):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        scheduler = None
        return optimizer,scheduler

    def save(self, path):
        torch.save({
            "model_parameter": self.model.state_dict(),
            "optimizer": self.optim.state_dict(),  
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self, args):
        if args.load_path != None:
            checkpoint = torch.load(args.load_path)
            self.model.load_state_dict(checkpoint['model_parameter'], strict=True) 
            self.optim.load_state_dict(checkpoint['optimizer'])

def set_seed(seed): # set the seed to ensure the result will be same
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MaskGIT")
    #TODO2:check your dataset path is correct 
    parser.add_argument('--train_d_path', type=str, default="./cat_face/train/", help='Training Dataset Path')
    parser.add_argument('--val_d_path', type=str, default="./cat_face/val/", help='Validation Dataset Path')
    parser.add_argument('--checkpoint_root', type=str, default='./transformer_checkpoints', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda:0", help='Which device the training is on.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for training.')
    parser.add_argument('--partial', type=float, default=1.0, help='Number of epochs to train (default: 50)')    
    parser.add_argument('--accum-grad', type=int, default=10, help='Number for gradient accumulation.')

    #you can modify the hyperparameters 
    parser.add_argument('--epochs', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--save_per_epoch', type=int, default=3, help='Save CKPT per ** epochs(defcault: 1)')
    parser.add_argument('--start-from-epoch', type=int, default=0, help='Number of epochs to train.')
    parser.add_argument('--ckpt-interval', type=int, default=0, help='Number of epochs to train')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate.')
    parser.add_argument('--load_path', type=str, default=None, help='the path to load ckpt')

    parser.add_argument('--MaskGitConfig', type=str, default='config/MaskGit.yml', help='Configurations for TransformerVQGAN')

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, 'r'))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root= args.train_d_path, partial=args.partial)
    train_loader = DataLoader(train_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=True)
    
    val_dataset = LoadTrainData(root= args.val_d_path, partial=args.partial)
    val_loader =  DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                drop_last=True,
                                pin_memory=True,
                                shuffle=False)
    
#TODO2 step1-5:
    set_seed(529)
    train_transformer.load_checkpoint(args)
    
    min_training_loss = 1e9
    min_validation_loss = 1e9    
    for epoch in range(args.start_from_epoch+1, args.epochs+1):
        training_avg_loss = train_transformer.train_one_epoch(train_loader, args, epoch)
        validation_avg_loss = train_transformer.eval_one_epoch(val_loader, args, epoch)
        
        if epoch % args.save_per_epoch == 0:
            train_transformer.save(f'{args.checkpoint_root}/epoch={epoch}.pt')
        if training_avg_loss < min_training_loss:
            min_training_loss = training_avg_loss
            train_transformer.save(f'{args.checkpoint_root}/train_best.pt')
        if validation_avg_loss < min_validation_loss:
            min_validation_loss = validation_avg_loss
            train_transformer.save(f'{args.checkpoint_root}/test_best.pt')