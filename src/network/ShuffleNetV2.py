import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups
    
    # reshape
    x = x.view(batchsize, groups, 
        channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x
    
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup//2
        
        if self.benchmodel == 1:
            #assert inp == oup_inc
        	self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )                
        else:                  
            self.banch1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )        
    
            self.banch2 = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
          
    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)        

    def forward(self, x):
        if 1==self.benchmodel:
            x1 = x[:, :(x.shape[1]//2), :, :]
            x2 = x[:, (x.shape[1]//2):, :, :]
            out = self._concat(x1, self.banch2(x2))
        elif 2==self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class AMMA_ShuffleNetV2(nn.Module):
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(AMMA_ShuffleNetV2, self).__init__()
        
        assert input_size % 32 == 0
        
        self.stage_repeats = [4, 8, 4]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 24,  48,  96, 192, 1024]
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 224, 488, 976, 2048]
     
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = conv_bn(3, input_channel, 2)    
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # added: for mm stream
        self.conv1_mm = conv_bn(3, input_channel, 2)    
        self.maxpool_mm = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.features = []
        self.features_mm = []
        # building inverted residual blocks
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage+2]
            for i in range(numrepeat):
                if i == 0:
	            #inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                    
                    # added: for mm stream
                    self.features_mm.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                    
                    # added: for mm stream
                    self.features_mm.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
                
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)
        
        self.features_mm = nn.Sequential(*self.features_mm[:12])

        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])
        #self.globalpool = nn.Sequential(nn.AvgPool2d(int(input_size/32)))              
    
    	# building classifier; no need for our task
        #self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))
        
        # add: shrink (1024 -> 512) to match the spatial dimension of AMMA's backbone feature
        self.shrink = nn.Sequential(
            nn.Conv2d(1024, 512, 
                      kernel_size=1, padding=0, bias=False, groups=1),
            
            nn.BatchNorm2d(num_features=512)
            )
        
        self.rgb_weight1 = nn.Parameter(torch.zeros(1) + 0.6) #0.6
        self.rgb_weight2 = nn.Parameter(torch.zeros(1) + 0.6) #0.6
        self.rgb_weight3 = nn.Parameter(torch.zeros(1) + 0.6) #0.6
        
        
    def forward(self, rgb, motion):
        x = rgb
        
        x = self.conv1(x)
        x = self.maxpool(x)
        
        # goal: fuse at 3 blocks: index 3, 6, and 10 
        # block 1 fusion
        motion = self.conv1_mm(motion)
        motion = self.maxpool_mm(motion)
        
        x = self.rgb_weight1 * x + (1.0 - self.rgb_weight1) * motion
        
        fuse_indices = [3,11] 
        
        # block 2 fusion
        for i in range(0, fuse_indices[0] + 1):
            x = self.features[i](x)
            motion = self.features_mm[i](motion)
        
        x = self.rgb_weight2 * x + (1.0 - self.rgb_weight2) * motion
    
        # block 3 fusion
        for i in range(fuse_indices[0] + 1, fuse_indices[1] + 1):
            x = self.features[i](x)
            motion = self.features_mm[i](motion)
        
        x = self.rgb_weight3 * x + (1.0 - self.rgb_weight3) * motion
        
        # follows through the rest of the feedforward network
        for i in range(fuse_indices[1] + 1, len(self.features)):
            x = self.features[i](x)
            
        x = self.conv_last(x)
        x = self.shrink(x)
        
        # not needeed for this task
        #x = self.globalpool(x)
        #x = x.view(-1, self.stage_out_channels[-1])
        #x = self.classifier(x)
        
        return x
