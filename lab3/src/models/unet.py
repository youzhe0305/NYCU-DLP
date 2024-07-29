# Implement your UNet model here
import torch
import torch.nn as nn

# reference from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py

class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.first_conv = nn.Sequential(
            # padding = 1 -> maintain same size
            nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=(3,3), padding_mode='reflect', padding=1),
            nn.BatchNorm2d(mid_channels), # feature channel
            nn.ReLU()
        )
        self.second_conv = nn.Sequential(
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=(3,3), padding_mode='reflect', padding=1),
            nn.BatchNorm2d(out_channels), # feature channel
            nn.ReLU()
        )

    def forward(self, x):
        x_mid = self.first_conv(x)
        x_out = self.second_conv(x_mid)
        return x_out

class DownSampling(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2,2)),
            DoubleConv(in_channels, mid_channels, out_channels)
        )
    def forward(self, x):
        return self.model(x)

class UpSampling(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, bilinear=True): # bilinear set True for preliminary test
        super().__init__()
        '''
            with two mode: bilinear or not
            bilinear: Just scale up by bilinear interplot
                      pros: faster & lower cost
                      cons: cannot learning 
            TransposedConv: Trough Padding to scale up image, than use convlution to get larger image including features
                            pros: kernel can be learn
                            cons: lower & higher cost
        '''

        if bilinear == True:
            # about align_corner: True: make source points on corner, False: make source point on crosshatch of target points 
            # ref: https://blog.csdn.net/wangweiwells/article/details/101820932
            self.up_conv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)    
            self.conv = DoubleConv(in_channels, mid_channels, out_channels)
        else:
            # channel / 2 for add concat channel 
            # strid-1 = the pixel between 2 source pixel
            # kernel_size - padding - 1 = padding number
            # Height = (s*(height-1)+1)+(k-p-1)*2 - (k-1) = s*height + k - 2*p - s, Width same method
            # ref: https://blog.csdn.net/qq_37541097/article/details/120709865 
            self.up_conv = nn.ConvTranspose2d(in_channels, out_channels=in_channels//2, kernel_size=(2,2), stride=(2,2), padding=0 ) # scale 2 times
            self.conv = DoubleConv(in_channels, mid_channels, out_channels)

        
    def forward(self, x, concat):
        # (batch, channel, height, width)
        x = self.up_conv(x) # half
        concated_x = torch.cat((concat, x), dim=1) # concat img from constracting path, get high resolution features
        return self.conv(concated_x)


        
class UNet(nn.Module):

    def __init__(self, in_channels, n_class, bilinear=True, device=torch.device('cpu')):
        super().__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss(reduction='mean') # get mean loss of all pixel

        
        factor = 2 if bilinear==True else 1 # if bilinear, the channel will not be half in up-conv -> devided by factor to get half channel
        self.constracting_path_part1 = DoubleConv(in_channels=in_channels, mid_channels=64, out_channels=64)
        self.constracting_path_part2 = DownSampling(in_channels=64, mid_channels=128, out_channels=128)
        self.constracting_path_part3 = DownSampling(in_channels=128, mid_channels=256, out_channels=256)
        self.constracting_path_part4 = DownSampling(in_channels=256, mid_channels=512, out_channels=512)
        self.constracting_path_part5 = DownSampling(in_channels=512, mid_channels=1024//factor, out_channels=1024//factor)

        self.expansive_path_part1 = UpSampling(in_channels=1024, mid_channels=512//factor, out_channels=512//factor, bilinear=bilinear)
        self.expansive_path_part2 = UpSampling(in_channels=512, mid_channels=256//factor, out_channels=256//factor, bilinear=bilinear)
        self.expansive_path_part3 = UpSampling(in_channels=256, mid_channels=128//factor, out_channels=128//factor, bilinear=bilinear)
        self.expansive_path_part4 = UpSampling(in_channels=128, mid_channels=64, out_channels=64, bilinear=bilinear)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=n_class, kernel_size=(1,1), padding_mode='reflect', padding=0),
        )


    def forward(self, x):
        '''
            c1              ->              d4
                c2          ->          d3
                    c3      ->      d2
                        c4  ->  d1
                            c5
        '''
        c1 = self.constracting_path_part1(x)
        c2 = self.constracting_path_part2(c1)
        c3 = self.constracting_path_part3(c2)
        c4 = self.constracting_path_part4(c3)
        c5 = self.constracting_path_part5(c4)
        
        d1 = self.expansive_path_part1(c5, c4)
        d2 = self.expansive_path_part2(d1, c3)
        d3 = self.expansive_path_part3(d2, c2)
        d4 = self.expansive_path_part4(d3, c1)
        output = self.output(d4)

        return output

    def get_loss(self, Y_pred, Y):

        # pred_Y: (batch, classes, H, W)
        # Y: (batch, 1, H, W) -> (batch, H, W) when class is indice(store as value)
        Y = Y.view(Y.shape[0], Y.shape[2], Y.shape[3]).type(torch.LongTensor).to(self.device) # indice should be store as LongTensor
        loss = self.criterion(Y_pred, Y)
        return loss

