# Implement your ResNet34_UNet model here
import torch
import torch.nn as nn
from models.unet import DoubleConv
# ref: https://github.com/chenyuntc/pytorch-best-practice/blob/master/models/ResNet34.py

class ResBlock(nn.Module): # layers with one shortcut connection
    def __init__(self, in_channels, out_channels, downsampling=False):
        super().__init__()
        self.downsampling = downsampling

        stride = 2 if downsampling == True else 1 # when block in tail of one part, use stride 2 to scale down 2 times 
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=(1,1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=(1,1)),
            nn.BatchNorm2d(out_channels),
        )
        if downsampling == True: # shortcut will be 2 times larger than x, use conv to make shortcut smaller. channel will be changed also.
            self.shortcut_connection = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=2, padding=(1,1)),
                nn.BatchNorm2d(out_channels),
                nn.Identity()                                
            )
        elif in_channels != out_channels: # image will be same, channel will be changed
            self.shortcut_connection = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=(1,1)),
                nn.BatchNorm2d(out_channels),
                nn.Identity()                                
            )
        else:
            self.shortcut_connection = nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x):
        x2 = self.model(x)
        shortcut = self.shortcut_connection(x)
        output = self.relu(x2 + shortcut)
        return output

class UpSampling(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bilinear=True): # bilinear set True for preliminary test
        super().__init__()

        if bilinear == True:
            self.up_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    
            self.conv = DoubleConv(in_channels, mid_channels, out_channels)
        else:
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels=in_channels, kernel_size=(2,2), stride=(2,2), padding=0 ) # scale 2 times
            self.conv = DoubleConv(in_channels, mid_channels, out_channels)
        # self.norm = nn.BatchNorm2d(in_channels)

    def forward(self, x, concat=None):
        # (batch, channel, height, width)
        if concat != None:
            concated_x = torch.cat((concat, x), dim=1) # concat img from constracting path, get high resolution features
        else:
            concated_x = x
        up = self.up_conv(concated_x) # half
        # up = self.norm(up)
        return self.conv(up)

class ResNet34_Unet(nn.Module):
    def __init__(self, in_channels, n_class, bilinear = True, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

        self.Res_part1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=(7,7), stride=2, padding_mode='reflect', padding=(3,3)), # scale down 2
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1) ) # scale down 2
        )
        self.Res_part2 = self.make_part(3, in_channels=64, out_channels=64, downspmpling=False)
        self.Res_part3 = self.make_part(4, in_channels=64, out_channels=128) # scale down 2
        self.Res_part4 = self.make_part(6, in_channels=128, out_channels=256) # scale down 2
        self.Res_part5 = self.make_part(3, in_channels=256, out_channels=512) # scale down 2
        self.Res_part6 = self.make_part(1, in_channels=512, out_channels=256, downspmpling=False)
        self.expansive_path_part1 = UpSampling(in_channels=256+512, mid_channels=32, out_channels=32, bilinear=bilinear) # scale up 2
        self.expansive_path_part2 = UpSampling(in_channels=32+256, mid_channels=32, out_channels=32, bilinear=bilinear) # scale up 2
        self.expansive_path_part3 = UpSampling(in_channels=32+128, mid_channels=32, out_channels=32, bilinear=bilinear) # scale up 2
        self.expansive_path_part4 = UpSampling(in_channels=32+64, mid_channels=32, out_channels=32, bilinear=bilinear) # scale up 2
        self.expansive_path_part5 = UpSampling(in_channels=32, mid_channels=32, out_channels=32, bilinear=bilinear) # scale up 2
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=n_class, kernel_size=(3,3), padding_mode='zeros', padding=1),
        )

    def make_part(self, n_block, in_channels, out_channels, downspmpling = True):

        part = nn.Sequential()
        part.append(ResBlock(in_channels, out_channels, downsampling=downspmpling))
        for i in range(n_block-1):
            part.append(ResBlock(out_channels, out_channels))
        return part

    def forward(self, x):
        '''
            c1                                      d5        
                c2              ->              d4
                    c3          ->          d3
                        c4      ->      d2
                            c5  ->  d1
                                c6
        '''
        c1 = self.Res_part1(x)
        c2 = self.Res_part2(c1)
        c3 = self.Res_part3(c2)
        c4 = self.Res_part4(c3)
        c5 = self.Res_part5(c4)
        c6 = self.Res_part6(c5)

        d1 = self.expansive_path_part1(c6, c5)
        d2 = self.expansive_path_part2(d1, c4)
        d3 = self.expansive_path_part3(d2, c3)
        d4 = self.expansive_path_part4(d3, c2)
        d5 = self.expansive_path_part5(d4)
        output = self.output(d5)
        return output
    
    def get_loss(self, Y_pred, Y):
        
        # pred_Y: (batch, classes, H, W)
        # Y: (batch, 1, H, W) -> (batch, H, W) when class is indice(store as value)
        Y = Y.view(Y.shape[0], Y.shape[2], Y.shape[3]).type(torch.LongTensor).to(self.device) # indice should be store as LongTensor
        loss = self.criterion(Y_pred, Y)
        return loss