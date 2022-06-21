import torch
from torch import nn
import math

import numpy as np
import os
import random

import torch.nn.functional as F

BN_MOMENTUM = 0.1

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True
    
class MM_PA(nn.Module):
    def __init__(self, n_length=5, mm_mode='mm1'):
        super(MM_PA, self).__init__()
        
        self.mm_mode = mm_mode
        
        # default mode (shallow layers to extract general pattern)
        if self.mm_mode == 'mm1':
            self.shallow_conv = nn.Conv2d(in_channels=3,out_channels=8,kernel_size=5,stride=1,padding=2)
        
        # deeper layers to extract/abstract general pattern
        elif self.mm_mode == 'mm2':
            self.shallow_conv = nn.Sequential(
                            nn.Conv2d(in_channels=3, out_channels=8, 
                                      kernel_size=5, padding=2),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=8, out_channels=8, 
                                  kernel_size=5, padding=2)
                )
        
        #self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0) # max pool before / kernel
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # max pool before / kernel
        
        self.n_length = n_length
        
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.001)
                nn.init.constant_(m.bias.data, 0)
        '''
        
    def forward(self, x): # x.size(): b, 3*n_length, h, w
        x_rgb = x[:,-3:, :,:] # the final frame of the input sequence
    
        h, w = x.size(-2), x.size(-1) 
        x = x.view((-1, 3) + x.size()[-2:])
        
        # downsample by 2 for efficiency purpose (e.g., FLOPs)
        x_small = self.maxpool(x)
        hs, ws = x_small.size(-2), x_small.size(-1)
        
        if self.mm_mode == 'mm0':
            x = x_small # taking rgb diff directly
        
        elif self.mm_mode == 'mm1' or self.mm_mode == 'mm2':
            x = self.shallow_conv(x_small) # if taking 1 or 2 layer general pattern
          
        ind = 0
        x = x.view(-1, self.n_length, x.size(-3), x.size(-2)*x.size(-1)) # torch.Size([16, 5, 8, 82944])
        
        for i in range(3): #self.n_length-2
            inc = self.n_length // 3 
            
            # pairwise (e.g., 1-2, 2-3, 3-4)
            #d_i = nn.PairwiseDistance(p=2)(x[:,i,:,:], x[:,i+1,:,:]).unsqueeze(1) # torch.Size([16, 1, 82944])
            
            # TARGET - OTHERS (e.g., 1-4, 2-4, 3-4)
            d_i = nn.PairwiseDistance(p=2)(x[:,-5+i,:,:], x[:,-1,:,:]).unsqueeze(1) # torch.Size([16, 1, 82944])
            
            d = d_i if i == 0 else torch.cat((d, d_i), 1)
            ind += inc
        
        mm = d.view(-1, 1*(3), hs, ws)
        mm = F.interpolate(mm, [h,w]) # upsample by 2 to recover spatial dimension
        
        return mm, x_rgb