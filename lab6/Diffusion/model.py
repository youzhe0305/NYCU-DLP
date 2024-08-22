import torch
import torch.nn as nn
import diffusers

# reference from https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb

class cDDPM(nn.Module):
    def __init__(self, n_object_class):
        super().__init__()
        self.label_enocder = nn.Sequential( # 因為非condition的UNet2DModel不能內部設定projection，所以另外加一個encoder
            nn.Linear(n_object_class, 64),
            nn.Linear(64,512)
        )
        self.ddpm = diffusers.UNet2DModel( # 只處理time step跟class不需要用到ConditionModel，有更高級的cnodition才要。
            sample_size=(64,64),           # the target image resolution
            in_channels=3,            # the number of input channels, 3 for RGB images
            out_channels=3,           # the number of output channels
            class_embed_type='identity',
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            block_out_channels=(128, 128, 256, 256, 512, 512), # Roughly matching our basic unet example
            down_block_types=( 
                "DownBlock2D",   
                "DownBlock2D",      
                "DownBlock2D",   
                "DownBlock2D",
                "AttnDownBlock2D",   
                "DownBlock2D",
            ), 
            up_block_types=(
                "UpBlock2D", 
                "AttnUpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D", 
                "UpBlock2D",    
            ),
        )

    def forward(self, img, t, label):
        encoded_label = self.label_enocder(label)
        return self.ddpm(sample=img, timestep=t, class_labels=encoded_label).sample
    
    
if __name__ == '__main__':
    model = cDDPM(n_object_class=24)
    print(model)
    img = torch.randn((20,3,64,64))
    print( model(img, 5, torch.randint(0,2, (20,24), dtype=torch.float)))
    
    