from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch 
from torch import nn

from .branch_mod import MOC_Branch

from .resnet import MOC_ResNet, AMMA_ResNet
from .deconv import deconv_layers
from .mm import MM_PA
from.MobileNetV2 import AMMA_MobileNetV2
from.ShuffleNetV2 import AMMA_ShuffleNetV2

import numpy as np
import cv2
import matplotlib.pyplot as plt 

import torch.nn.functional as F

import math

from einops import rearrange, repeat

backbone = {
    'resnet': MOC_ResNet
}

class AMMA_Net(nn.Module):
    def __init__(self, opt, arch, num_layers, branch_info, head_conv, K, flip_test=False):
        super(AMMA_Net, self).__init__()
        self.flip_test = flip_test
        self.K = K
        
        mm_mode = 'mm1' # hardcoded here; mm0 == taking rgb diff directly
        self.mm = MM_PA(n_length=opt.ninput, mm_mode=mm_mode)
        
        if opt.arch == 'resnet_18':
            self.backbone = AMMA_ResNet(num_layers, rgb_ws='TTTFF')
        elif opt.arch == 'mobile_v2':
            self.backbone = AMMA_MobileNetV2()
        elif opt.arch == 'shuffle_v2':
            self.backbone = AMMA_ShuffleNetV2()
            
        self.deconv_layer = deconv_layers(inplanes=512, BN_MOMENTUM=0.1)
        self.branch = MOC_Branch(256, arch, head_conv, branch_info, K) # self.backbone.output_channel == 64
        
        self.init_weights()
        
        
    def forward(self, input):
        if self.flip_test:
            assert(self.K == len(input) // 2)
            chunk1 = [self.backbone(input[i]) for i in range(self.K)]
            chunk2 = [self.backbone(input[i + self.K]) for i in range(self.K)]

            return [self.branch(chunk1), self.branch(chunk2)]
        else:
            # parallel processing (squeeze into batch dim)   
            bb, cc, hh, ww = input[0].size()
            input_all = torch.cat(input, dim=1)
            input_all = input_all.view(-1, cc, hh, ww)
            
            '''
            # debug: original image
            ninput = input_all.size()[1] // 3
            for ii in range(self.K-1, self.K):
                for i in range(ninput):
                    self.vis_feat(input_all[ii:ii+1,i*3:i*3+3,:,:].cpu())
            '''
            
            input_all, input_rgb = self.mm(input_all) 
            chunk = self.backbone(input_rgb, input_all)
            
            '''
            # debug: pa image
            for ii in range(self.K-1, self.K):
                for i in range(input_all.size()[1]):
                    self.vis_feat(input_all[ii:ii+1,i,:,:].cpu())
            '''
            
            chunk = self.deconv_layer(chunk)
            chunk = F.interpolate(chunk, [36, 36])
            
            return [self.branch(chunk, self.K)]
            
    def init_weights(self):
        # print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def vis_feat(self, image):
        # ADDED: vis for debug
        # data[i] = ((data[i] / 255.) - mean) / std
        if image.size()[1] == 3:
            image_temp = image.numpy().squeeze().transpose(1,2,0)
            image_temp = ((image_temp * [0.28863828, 0.27408164, 0.27809835] + [0.40789654, 0.44719302, 0.47026115]) * 255).astype(np.uint8)
            image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB)
        else: 
            image_temp = image.numpy().squeeze().astype(np.float32)
        plt.imshow(image_temp)
        plt.axis('off')
        plt.show()   

class MOC_Net(nn.Module):
    def __init__(self, opt, arch, num_layers, branch_info, head_conv, K, flip_test=False):
        super(MOC_Net, self).__init__()
        self.flip_test = flip_test
        self.K = K
        
        self.backbone = backbone[arch](num_layers) if 'resnet' in opt.arch else backbone[arch]()
        self.deconv_layer = deconv_layers(inplanes=512, BN_MOMENTUM=0.1)
        self.branch = MOC_Branch(256, arch, head_conv, branch_info, K) # self.backbone.output_channel == 64
        
        self.init_weights()
        
    def forward(self, input):
        if self.flip_test:
            assert(self.K == len(input) // 2)
            chunk1 = [self.backbone(input[i]) for i in range(self.K)]
            chunk2 = [self.backbone(input[i + self.K]) for i in range(self.K)]

            return [self.branch(chunk1), self.branch(chunk2)]
        else:
            
            # TODO: alternative: parallel processing (squeeze into batch dim)   
            bb, cc, hh, ww = input[0].size()
            input_all = torch.cat(input, dim=1)
            input_all = input_all.view(-1, cc, hh, ww)
            
            '''
            # debug: original image
            ninput = input_all.size()[1] // 3
            for ii in range(self.K):
                for i in range(ninput):
                    self.vis_feat(input_all[ii:ii+1,i*3:i*3+3,:,:].cpu())
            '''
            
            
            chunk = self.backbone(input_all)
            
            '''
            # debug: pa image
            for ii in range(self.K):
                for i in range(input_all.size()[1]):
                    self.vis_feat(input_all[ii:ii+1,i,:,:].cpu())
            '''
            
            chunk = self.deconv_layer(chunk)
            chunk = F.interpolate(chunk, [36, 36])
            
            return [self.branch(chunk, self.K)]
            
    # ADDED: to separate deconv layer (??)
    def init_weights(self):
        # print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def vis_feat(self, image):
        # ADDED: vis for debug
        # data[i] = ((data[i] / 255.) - mean) / std
        if image.size()[1] == 3:
            image_temp = image.numpy().squeeze().transpose(1,2,0)
            image_temp = ((image_temp * [0.28863828, 0.27408164, 0.27809835] + [0.40789654, 0.44719302, 0.47026115]) * 255).astype(np.uint8)
            image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB)
        else: 
            image_temp = image.numpy().squeeze().astype(np.float32)
        plt.imshow(image_temp)
        plt.show()