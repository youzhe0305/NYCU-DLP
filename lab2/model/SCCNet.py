# implement SCCNet model

import torch
import torch.nn as nn
import math

# reference paper: https://ieeexplore.ieee.org/document/8716937
class SquareLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        pass

class SCCNet(nn.Module):
    def __init__(self, numClasses=0, timeSample=0, Nu=0, C=0, Nc=0, Nt=1, dropoutRate=0.5): 
        super(SCCNet, self).__init__()
        
        self.criterion = nn.CrossEntropyLoss()
        # Zero-padding and batch normalization were applied to both the first and second convolutions
        # Spatial component analysis 空間
        self.first_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=Nu, kernel_size=(C, Nt), padding_mode='zeros', padding=Nt-1), # input: (batch, channel, Height, Width)
            nn.LeakyReLU(),
            nn.Dropout(dropoutRate)
        )
        
        # Spatiotemporal(時空) filtering
        self.secood_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(Nu, 12), padding_mode='zeros', padding=(0,5) ),
            nn.LeakyReLU(),
            nn.Dropout(dropoutRate)
        )
        
        # Temporal(時間的) smoothing
        self.third_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=(1,62), stride=(1,12)), # input: (batch, channel, Height, Width)
            nn.LeakyReLU(),
            nn.Dropout(dropoutRate)    
        )
        
        # output shape: (20,T/12)
        self.fc = nn.Linear(in_features= 20*int((438-62)/12 + 1), out_features=4) 

        pass

    def forward(self, x):

        # input (batch, 22, 438)
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        fist_block_output = self.first_block(x) # output shape: (batch,22,1,438)
        fist_block_output = fist_block_output.view(fist_block_output.shape[0], 1, fist_block_output.shape[1], fist_block_output.shape[3])
        second_block_output = self.secood_block(fist_block_output)
        second_block_output = second_block_output.view(second_block_output.shape[0], 1, second_block_output.shape[1], second_block_output.shape[3])
        third_block_output = self.third_block(second_block_output)
        third_block_output = third_block_output.view(third_block_output.shape[0],-1) # (batch, flatten)
        output = self.fc(third_block_output) # need to be add softmax when prediction
        return output
        pass
    
    def get_loss(self, Y_pred, Y):

        Y = Y.type(torch.LongTensor)
        Y = nn.functional.one_hot(Y, num_classes=4).type(torch.float)
        loss = self.criterion(Y_pred, Y)
        return loss

    # if needed, implement the get_size method for the in channel of fc layer
    def get_size(self, C, N):
        pass