from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import torch

from torch import nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import time

import numpy as np
import random
import os

import math

from .DCNv2.dcn_v2 import DCN

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.benchmark = False    
    torch.backends.cudnn.deterministic = True
    
def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
  
class MOC_Branch(nn.Module):
    def __init__(self, input_channel, arch, head_conv, branch_info, K):
        super(MOC_Branch, self).__init__()
        assert head_conv > 0
        wh_head_conv = 64 if (arch == 'resnet' or arch == 'mobile_v2' or arch == 'shuffle_v2') else head_conv
        
     
        self.shrink = nn.Sequential(
            nn.Conv2d(input_channel, input_channel//4, 
                      kernel_size=1, padding=0, bias=False, groups=1),
            
            nn.BatchNorm2d(num_features=input_channel//4),
            )
        
        
        ## Center branch
        self.hm = nn.Sequential(
            nn.Conv2d(K*input_channel//4, head_conv, 
                      kernel_size=3, padding=1, bias=True, groups=1),
            
            ## using deformable convolution as below also works (perhaps even better to dynamically capture features)
            #DCN((3)*input_channel//4, head_conv,
            #         kernel_size=(3, 3), stride=1,
            #         padding=1, dilation=1, deformable_groups=1),
            
            
            nn.ReLU(inplace=True))
        
        self.hm_cls = nn.Sequential(nn.Conv2d(head_conv, branch_info['hm'],
                                              kernel_size=1, stride=1,
                                              padding=0, bias=True, groups=1))
        
        
        self.hm_cls[-1].bias.data.fill_(-2.19) # -2.19
        
        
        ## Trajectory branch
        # increasing kernel size to 5 appears to improve localization (can reduce the head_conv accordingly)
        self.mov = nn.Sequential(
            nn.Conv2d(K*input_channel//4, head_conv, 
                      kernel_size=3, padding=1, bias=True, groups=1),
            
            nn.ReLU(inplace=True))
        
        self.mov_cls = nn.Sequential(nn.Conv2d(head_conv, (branch_info['mov']), 
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        
        fill_fc_weights(self.mov)
        fill_fc_weights(self.mov_cls)
 
        ## Box branch
        self.wh = nn.Sequential(
            nn.Conv2d(input_channel//4, wh_head_conv,
                      kernel_size=3, padding=1, bias=True),
            
            nn.ReLU(inplace=True),
            nn.Conv2d(wh_head_conv, branch_info['wh'] // K,
                      kernel_size=1, stride=1,
                      padding=0, bias=True))
        fill_fc_weights(self.wh)
        
        
        self.init_weights()
        
    def init_weights(self):
       
        for name, m in self.shrink.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        
        ## Explore different initialization; matters little to the final accuracy        
        '''
        for name, m in self.hm.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        for name, m in self.hm_cls.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                # normal distribution here gives a very strange spike / scale for the loss (due to different range of std?)
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', a=math.sqrt(5), nonlinearity='leaky_relu')
     
        for name, m in self.mov.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        
        for name, m in self.mov_cls.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
       
        for name, m in self.wh.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        '''
    def vis_feat(self, x, t, c):
        
        x_np = x.cpu().detach().numpy()
        
        # x_np.shape: nt, c, h, w
        
        tar_feat = np.abs(x_np[t, c, :,:]) # shape: h, w
        plt.imshow(tar_feat)
        plt.title('Channel ' + str(c+1) + '| Time ' + str(t+1))
        plt.colorbar()
        plt.show()
        
    def forward(self, input_chunk, K):
        
        # ===================================
        bbK, cc, hh, ww = input_chunk.size()
        
        input_chunk_small = self.shrink(input_chunk) #bbK, cr, hh, ww
        
        output = {}
        
        output_hm = self.hm(input_chunk_small.view(-1, cc*(K)//4, hh, ww))
        output['hm'] = self.hm_cls(output_hm).sigmoid_() 
        
        output_wh = (self.wh(input_chunk_small))
        output_wh = output_wh.view(bbK // K, -1, hh, ww)
        output['wh'] =  output_wh
     
        output_mov = self.mov(input_chunk_small.view(-1, cc*K//4, hh, ww))
        output['mov'] = self.mov_cls(output_mov)
        
        return output