from model import cDDPM
import torch
import torch.nn as nn
from dataloader import Diffusion_Dataset
from torch.utils.data import DataLoader
import argparse
from diffusers import DDPMScheduler
from tqdm import tqdm
import random
import numpy as np
from evaluator import evaluation_model
# reference1: https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb
# reference2: https://huggingface.co/docs/diffusers/v0.21.0/tutorials/basic_training

class Trainer():
    def __init__(self, args):
        
        self.args = args
        self.model = cDDPM(n_object_class=24).to(f'{args.device}:{args.gpu}')
        self.dataset = Diffusion_Dataset(args.data_root, mode='train')
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[15,60,150], gamma=0.1)
        self.criterion = nn.MSELoss() # 因為DDPM的本質上，是要讓兩個gaussian distribution的KL divergence最小，這裡variance可以忽略不記。所以基本上就是算平均的差，因此用MSE當loss
        self.noise_scheduler = DDPMScheduler( # 讓noisr可以直接加到第t步的noise 或是 可以方便做去noise
            num_train_timesteps=args.time_step,
            beta_start=0.0001, # beta代表t到t+1時，noise要加的量
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2'
        )
        self.evaluator = evaluation_model()
        for para in self.evaluator.resnet18.parameters():
            para.requires_grad = False            
            

    def train_one_epoch(self, ith_epoch):
        
        device = f'{self.args.device}:{self.args.gpu}'
        progress_bar = tqdm(self.dataloader)
        total_loss = torch.zeros(1).to(device)
        iterations = 0
        for idx, (img, label) in enumerate(progress_bar):
            img = img.to(device)
            label = label.to(device)
            
            noise = torch.randn(img.shape).to(device)
            t = torch.randint(low=0, high=self.args.time_step, size=(img.shape[0],)).long().to(device) # random time_step
            noisy_img = self.noise_scheduler.add_noise(img, noise, t)
            predict_noise = self.model(noisy_img, t, label)      
            
            loss = self.criterion(predict_noise, noise) # 雖然加的不是原本的noise，而是sqrt(1-alpha_t)*noise，但反正只差常數，就讓他直接預測原來的
        
            
            total_loss += loss
            iterations += 1
            loss.backward()
            self.optimizer_step()
            
            progress_bar.set_description(f'Training, Epoch {ith_epoch}:', refresh=False)
            progress_bar.set_postfix({
                'lr': f'{self.scheduler.get_last_lr()}',
                'averge loss':f'{round(total_loss.item() / iterations, 8)}'
            }, refresh=False)
            progress_bar.refresh()     
        
        self.scheduler.step()
        avg_loss = total_loss.item() / iterations
        
        return avg_loss
    
    def train_with_eval(self):
        
        device = f'{self.args.device}:{self.args.gpu}'

        img, label = self.dataset[0] # 拿第一個label生圖來做測試，反正每次都會shuffle
        print(label)
        label = label.to(device)
        noise = torch.randn(1, 3, 64, 64).to(device)
        img = noise
        for i, t in enumerate(self.noise_scheduler.timesteps):
            print(i)
            predict_noise = self.model(img, t, label)
            img = self.noise_scheduler.step(predict_noise, t, img).prev_sample # 生成x_{t-1}
            torch.cuda.empty_cache()
        
        acc = self.evaluator.eval(img, label)
        loss = -torch.log(acc) * self.args.beta
        print(f'training with evaluator, loss:{loss.item()}')
        loss.backward()
        self.optimizer_step()

    
    def train(self):
        min_loss = 1e9
        for epoch in range(self.args.num_epoch):
            epoch_avg_loss = self.train_one_epoch(epoch)
            if epoch % self.args.save_per_epoch == 0:
                self.save_model(f'{self.args.save_root}/epoch={epoch}.ckpt')
            if epoch_avg_loss < min_loss:
                min_loss = epoch_avg_loss
                self.save_model(f'{self.args.save_root}/training_best.ckpt')
                
    def save_model(self, path):
        torch.save({
            'model_parameter': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),  
            'scheduler': self.scheduler.state_dict(),
        }, path)
        print(f"save ckpt to {path}")
    
    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   
        
def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,  default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--gpu',           type=int, default=0)
    parser.add_argument('--data_root',            type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',            type=str, required=True,  help="root to save checkpoint")
    parser.add_argument('--num_workers',   type=int, default=20)
    parser.add_argument('--num_epoch',     type=int, default=70,     help="number of total epoch")
    parser.add_argument('--time_step',     type=int, default=1000,     help="number of noise adding step")
    parser.add_argument('--save_per_epoch',     type=int, default=3,     help="per X epochs, save model")
    parser.add_argument('--beta',     type=float, default=0.1,     help="")


    args = parser.parse_args()        
    return args    
        
    
if __name__ == '__main__':
    set_seed(529)
    args = get_args()
    trainer = Trainer(args)
    trainer.train()
    
# python3 train.py --batch_size 32 --num_epoch 300 --lr 0.0001 --data_root ../iclevr --save_root ../model_weight
    