import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import hashlib

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10

def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(imgs1, imgs2) # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size): # 算跟standard gaussian distribution的KL差
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD


class kl_annealing(): # 用來平衡KL-Divergence與圖片loss的方法，避免KL過大或過小影響結果。透過改變KL的權重
    def __init__(self, args, current_epoch=0):
        # TODO
        self.type = args.kl_anneal_type
        self.cycle = args.kl_anneal_cycle # number of cycle
        self.ratio = args.kl_anneal_ratio # increasing speed of weight
        self.epoch = current_epoch
        self.total_epoch = args.num_epoch
        self.beta = 0
        if self.type == 'None':
            self.beta = 1
        
    def update(self):
        # TODO
        if self.type == 'Cyclical':
            self.frange_cycle_linear(n_iter = self.total_epoch, n_cycle = self.cycle, ratio=self.ratio)
        elif self.type == 'Monotonic':
            self.frange_cycle_linear(n_iter = self.total_epoch, n_cycle=1, ratio=self.ratio) # one cycle = Monotonic
        self.epoch += 1

    def get_beta(self):
        # TODO
        return self.beta

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1): 
        # start: beta start point, stop: the ceil of beta, ratio: the ratio to increase, otherwise stay in stop  
        # TODO
        period = np.ceil(n_iter / n_cycle)
        step = (stop-start) / (np.ceil(period*ratio)-1) # compute one step length for one cycle, 1 for 0.0, period-1 for plus to 1.0

        mod = self.epoch % period
        if mod < period * ratio:
            self.beta = start + step * mod
            if self.beta > stop:
                self.beta = stop
        else:
            pass # do nothing just stay at stop(1)

class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args
        self.device = torch.device(self.args.device)
        print('device:')
        print(self.device) # lookup device
        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)
        
        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor   = Gaussian_Predictor(args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion       = Decoder_Fusion(args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)
        
        # Generative model
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)
        
        if self.args.optim == 'AdamW':
            self.optim = optim.AdamW(self.parameters(), lr=self.args.lr)
        else:
            self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
        
        
        self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[3,7,15], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0
        
        # Teacher forcing arguments
        self.tfr = args.tfr # Teacher forcing prob
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde
        
        self.train_vi_len = args.train_vi_len # Training video length
        self.val_vi_len   = args.val_vi_len 
        self.batch_size = args.batch_size
        
        
    def forward(self, img, label):
        pass
    
    def training_stage(self):
        ###
        best_loss = 1e9
        ###
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False

            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                # img, label shape: (batch_size=2, video_frames=16, 3, 32, 64)
                img = img.to(self.device)
                label = label.to(self.device)
                loss = self.training_one_step(img, label, adapt_TeacherForcing, time_smoothing_rate=self.args.time_smoothing_rate)
                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(self.tfr, round(beta,2)), pbar, loss.detach().cpu(), lr=round(self.scheduler.get_last_lr()[0], 10))
                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(self.tfr, round(beta,2)), pbar, loss.detach().cpu(), lr=round(self.scheduler.get_last_lr()[0], 10))

            if self.current_epoch % self.args.per_save == 0:
                self.save(os.path.join(self.args.save_root, f"epoch={self.current_epoch}.ckpt"))
            ###
            eval_loss = self.eval() # 多了eval_loss變數
            if eval_loss.item() < best_loss and self.current_epoch >= 20:
                best_loss = eval_loss.item()
                self.save(os.path.join(self.args.save_root, f"best.ckpt"))
            ###
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update() # update per epoch
            
            
    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        ###
        total_loss = torch.zeros(1).to(self.device)
        ###
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.device)
            label = label.to(self.device)
            loss, psnr = self.val_one_step(img, label)
            ###
            total_loss += loss
            ###
            self.tqdm_bar('val', pbar, loss.detach().cpu(), lr=self.scheduler.get_last_lr()[0])
        ###
        return total_loss
        ###
    
    def mix_teacher_forcing(self, gt_img, pred_img, ratio):
        return gt_img * ratio + pred_img * (1-ratio)

    def training_one_step(self, img, label, adapt_TeacherForcing, time_smoothing_rate = 0.3): # return loss
        # TODO
        # img, label shape: (batch_size=2, video_frames=16, 3, frams_H=32, frams_W=64)

        pred_img = img[:,0,:,:] # predict image, for next frame's reference
        video_loss = torch.zeros(1).to(self.device)
        beta = self.kl_annealing.get_beta() # beta for this epoch
        for idx in range(1,self.train_vi_len): # start predict from img 1(2nd frame)

            if adapt_TeacherForcing == True:# previous image for reference to predict, when teacher forcing, use ground truth
                ref_img = self.mix_teacher_forcing(img[:,idx-1,:,:,:], pred_img, self.tfr) # use the mix of gt & pred 
            else:
                ref_img = pred_img # previous image for reference to predict
            
            transformed_ref_img = self.frame_transformation(ref_img) # frame encoder
            transformed_label = self.label_transformation(label[:,idx,:,:,:])
            transformed_gt_img = self.frame_transformation(img[:,idx,:,:,:]) # current image for predict gaussain districution

            z, mu, logvar = self.Gaussian_Predictor(transformed_gt_img, transformed_label)
            fused_features = self.Decoder_Fusion(transformed_ref_img, transformed_label, z)
            prediction = self.Generator(fused_features)
            pred_img = prediction # for next frame reference
            # MSE + KL(N(mean, var) | N(0,1))

            time_smoothing_loss = self.mse_criterion(prediction-ref_img, img[:,idx,:,:,:]-img[:,idx-1,:,:,:]) # 讓預測的2幀之間的差異，去跟原圖的2幀間的差異對比，看看有沒有類似，用來給予模型更多連續幀的訊息

            video_loss += self.mse_criterion(prediction, img[:,idx,:,:,:]) + beta * kl_criterion(mu, logvar, self.batch_size) + time_smoothing_rate * time_smoothing_loss
        


        video_loss.backward()
        avg_loss = video_loss / (self.train_vi_len - 1)
        self.optimizer_step()
        self.optim.zero_grad()

        return avg_loss

        raise NotImplementedError
    
    def val_one_step(self, img, label):
        # TODO
        ref_img = img[:,0,:,:,:] # predict image, for next frame's reference
        video_loss = torch.zeros(1).to(self.device)
        video_psnr = torch.zeros(1).to(self.device)
        for idx in range(1,self.train_vi_len): # start predict from img 1(2nd frame)
            
            transformed_ref_img = self.frame_transformation(ref_img) # frame encoder
            transformed_label = self.label_transformation(label[:,idx,:,:,:])

            z = torch.randn(size=(1, self.args.N_dim, self.args.frame_H, self.args.frame_W)).to(self.device) # sample from standard distribution
            fused_features = self.Decoder_Fusion(transformed_ref_img, transformed_label, z)
            prediction = self.Generator(fused_features)
            ref_img = prediction # previous image for reference to predict

            # MSE, no need to consider KL divergence, simply sample from std gaussian distirbution
            video_loss += self.mse_criterion(prediction, img[:,idx,:,:,:])
            video_psnr += Generate_PSNR(img[:,idx,:,:,:], prediction)
        
        avg_loss = video_loss / (self.train_vi_len - 1)
        avg_psnr = video_psnr / (self.train_vi_len - 1)

        return avg_loss, avg_psnr

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))
            
        new_list[0].save(img_name, format="GIF", append_images=new_list,
                    save_all=True, duration=40, loop=0)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len, \
                                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False
            
        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return train_loader
    
    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='val', video_len=self.val_vi_len, partial=1.0)  
        val_loader = DataLoader(dataset,
                                  batch_size=1,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)  
        return val_loader
    
    def teacher_forcing_ratio_update(self):
        # TODO

        if self.current_epoch >= self.tfr_sde:
            self.tfr -=  self.tfr_d_step
            self.tfr = max(self.tfr, 0)
            
    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(f"({mode}) Epoch {self.current_epoch}, lr:{lr}" , refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()
        
    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),  
            "lr"        : self.scheduler.get_last_lr()[0],
            "tfr"       :   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True) 
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']
            
            self.optim      = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler  = optim.lr_scheduler.MultiStepLR(self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()

def str_to_int(str):

    md5 = hashlib.md5()
    md5.update(str.encode())
    hash = md5.hexdigest() # 16進位
    return int(hash, 16) % 2**23

def set_seed(seed): # set the seed to ensure the result will be same
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def main(args):
    
    seed = str_to_int('Nijika')
    set_seed(seed)
    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()




if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str, choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true', help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str, required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=3,      help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,  help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int, default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int, default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,     help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,     help="Width input image to be resize")
    
    
    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,    help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,     help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int, default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,    help="Dimension of the output in Decoder_Fusion")
    
    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float, default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,   help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,  help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,    default=None,help="The path of your checkpoints")   
    
    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,    help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=5,        help="Number of epoch to use fast train mode")
    
    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type',     type=str, default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int, default=10,               help="")
    parser.add_argument('--kl_anneal_ratio',    type=float, default=1,              help="")
    
    # time smoothing
    parser.add_argument('--time_smoothing_rate',     type=float, default=0.3,       help="")

# python Trainer.py --DR dataset --save_root check_points --fast_train --device cpu
# python Trainer.py --DR dataset --save_root check_points --fast_train --device cpu --fast_partial 0.0025, for one epoch

    args = parser.parse_args()
    
    main(args)
