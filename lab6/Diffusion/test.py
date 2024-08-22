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
from torchvision.utils import make_grid, save_image
from evaluator import evaluation_model


# reference1: https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb
# reference2: https://huggingface.co/docs/diffusers/v0.21.0/tutorials/basic_training

class Tester():
    def __init__(self, args):
        
        self.args = args
        self.model = cDDPM(n_object_class=24).to(f'{args.device}:{args.gpu}')
        self.load_model()
        self.model.eval()
        self.dataset = Diffusion_Dataset(args.data_root, mode=args.test_type)
        self.dataloader = DataLoader(self.dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        self.noise_scheduler = DDPMScheduler( # 讓noisr可以直接加到第t步的noise 或是 可以方便做去noise
            num_train_timesteps=args.time_step,
            beta_start=0.0001, # beta代表t到t+1時，noise要加的量
            beta_end=0.02,
            beta_schedule='squaredcos_cap_v2'
        )
        self.evaluator = evaluation_model()
    
    @torch.no_grad()
    def test(self):
        
        device = f'{self.args.device}:{self.args.gpu}'
        progress_bar = tqdm(self.dataloader)  
        total_acc = 0
        test_imgs = [] 
        for idx, label in enumerate(progress_bar):
            process_img = []
            label = label.to(device)
            noise = torch.randn(1, 3, 64, 64).to(device)
            img = noise
            for i, t in enumerate(self.noise_scheduler.timesteps):
                # print(t)
                predict_noise = self.model(img, t, label)
                img = self.noise_scheduler.step(predict_noise, t, img).prev_sample # 生成x_{t-1}
                if i % 100 == 0:
                    process_img.append(img.squeeze(0))

            acc = self.evaluator.eval(img, label)
            print(acc)
            total_acc += acc
            process_img.append(img.squeeze(0))
            test_imgs.append(img.squeeze(0))
            process_img = torch.stack(process_img) # (n,c,h,w)
            grid = make_grid(process_img, normalize=True)
            save_image(grid, f'output/processing_{idx}.jpg')
        
        test_imgs = torch.stack(test_imgs)
        test_grid = make_grid(test_imgs, normalize=True, nrow=8)
        save_image(test_grid, f'output/test_result.jpg')
        
        print('averge accuracy: ',total_acc / len(self.dataset))
            
            
    def load_model(self):
        checkpoint = torch.load(args.weight_path)
        self.model.load_state_dict(checkpoint['model_parameter'])
        
        
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   
        
def get_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=1)
    parser.add_argument('--device',        type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--gpu',           type=int, default=0)
    parser.add_argument('--data_root',     type=str, required=True,  help="Your Dataset Path")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--time_step',     type=int, default=1000,     help="number of noise adding step")
    parser.add_argument('--weight_path',   type=str, required=True,  help="model weight path")
    parser.add_argument('--test_type', type=str, choices=['test', 'new_test'], default='test', help='test label json')

    args = parser.parse_args()        
    return args    
    
    
if __name__ == '__main__':
    set_seed(529)
    args = get_args()
    tester = Tester(args)
    tester.test()
    
# python3 test.py --data_root ../iclevr --weight_path ../model_weight_best/training_best.pth --test_type new_test
