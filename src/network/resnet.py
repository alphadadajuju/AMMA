# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Dequan Wang and Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

from .deconv import deconv_layers

import matplotlib.pyplot as plt
import numpy as np

import torch.nn.functional as F

BN_MOMENTUM = 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        
    # ADDED: debug feat map
    def vis_feat(self, x, t, c):
        
        x_np = x.cpu().detach().numpy()
        
        # x_np.shape: nt, c, h, w
        
        tar_feat = x_np[t, c, :,:] # shape: h, w
        plt.imshow(tar_feat)
        plt.title('Channel ' + str(c+1) + '| Time ' + str(t+1))
        plt.colorbar()
        plt.show()
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


class MOC_ResNet(nn.Module):
    def __init__(self, num_layers):
        super(MOC_ResNet, self).__init__()
        self.output_channel = 64
        block, layers = resnet_spec[num_layers]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)

        # used for deconv layers
        #self.deconv_layer = deconv_layers(self.inplanes, BN_MOMENTUM)
        
        self.init_weights()
        
    def forward(self, input):
        x = self.conv1(input) # 144x144x64
    
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        #x_copy = x.clone() 
        
        x = self.layer1(x) # 72x72x64
        x = self.layer2(x) # 36x36x128
        x = self.layer3(x) # 18x18x256
        x = self.layer4(x) # 9x9x512
        
        #x = self.deconv_layer(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_weights(self):
        pass
        # print('=> init deconv weights from normal distribution')
        #for name, m in self.deconv_layer.named_modules():
        #    if isinstance(m, nn.BatchNorm2d):
        #        nn.init.constant_(m.weight, 1)
        #        nn.init.constant_(m.bias, 0)

class AMMA_ResNet(nn.Module):
    def __init__(self, num_layers, rgb_ws):
        super(AMMA_ResNet, self).__init__()
        self.output_channel = 64
        block, layers = resnet_spec[num_layers]
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.rgb_ws = rgb_ws
        
        if self.rgb_ws[0] == 'T':
            self.conv1_5 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1_5 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        
            self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
            self.rgb_weight1 = nn.Parameter(torch.zeros(1) + 0.6) #0.6
            
        self.layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1)
        if self.rgb_ws[1] == 'T':
            self.resnext_layer1 = self._make_layer(BasicBlock, 64, layers[0], stride=1)
            
            self.rgb_weight2 = nn.Parameter(torch.zeros(1) + 0.6) #0.6
            
        
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2,)
        if self.rgb_ws[2] == 'T':
            self.inplanes = 64 # added to fake channel shape for make_layer
            self.resnext_layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2)
            
            self.rgb_weight3 = nn.Parameter(torch.zeros(1) + 0.6) #0.6
            
            
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
        
        if self.rgb_ws[3] == 'T':
            self.inplanes = 128 # added to fake channel shape for make_layer
            self.resnext_layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2)
            
            self.rgb_weight4 = nn.Parameter(torch.zeros(1) + 0.6) #0.6
        
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
        
        if self.rgb_ws[4] == 'T':
            self.inplanes = 256 # added to fake channel shape for make_layer
            self.resnext_layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2)
            
            self.rgb_weight5 = nn.Parameter(torch.zeros(1) + 0.6) #0.6
        
    def forward(self, input, motion):
        x = self.conv1(input)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        if self.rgb_ws[0] == 'T':
            x_c5 = self.relu(self.bn1_5(self.conv1_5(motion)))
            x_diff_orig = self.maxpool_diff(1.0/1.0*x_c5)
            temp_out_diff1 = x_diff_orig 
        
            # sum fusion
            x = self.rgb_weight1 * x + (1.0 - self.rgb_weight1) * temp_out_diff1
        
        x = self.layer1(x)
        
        if self.rgb_ws[1] == 'T':
            x_diff1 = self.resnext_layer1(x_diff_orig)
            
            # sum
            x = self.rgb_weight2 * x + (1.0 - self.rgb_weight2) * x_diff1
        
        x = self.layer2(x) 
        
        if self.rgb_ws[2] == 'T':
            x_diff2 = self.resnext_layer2(x_diff1)
            
            # sum
            x = self.rgb_weight3 * x + (1.0 - self.rgb_weight3) * x_diff2
        
        x = self.layer3(x) 
        
        if self.rgb_ws[3] == 'T':
            x_diff3 = self.resnext_layer3(x_diff2)
            x = self.rgb_weight4 * x + (1.0 - self.rgb_weight4) * x_diff3
        
        x = self.layer4(x) 
        
        if self.rgb_ws[4] == 'T':
            x_diff4 = self.resnext_layer4(x_diff3)
            x = self.rgb_weight5 * x + (1.0 - self.rgb_weight5) * x_diff4
        
        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def init_weights(self):
        # print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
