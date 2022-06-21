import torch
from torch import nn

import torch.nn.functional as F


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class AMMA_MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None):
        super(AMMA_MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        
        # mm stream: duplicate orig feature layer design up to point X 
        # goal: fuse 3 times after ... index 1, 3, and 6 
        features_pa = [ConvBNReLU(3, input_channel, stride=2)]
        
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                
                features_pa.append(block(input_channel, output_channel, stride, expand_ratio=t))
                    
                input_channel = output_channel
        
        features_pa = features_pa[:11] # hardcoded: end of the third block: 7 / 4 block: 11
        
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.features_pa = nn.Sequential(*features_pa)
        
        # add: shrink (1280 -> 512)
        self.shrink = nn.Sequential(
            nn.Conv2d(1280, 512, 
                      kernel_size=1, padding=0, bias=False, groups=1),
            
            nn.BatchNorm2d(num_features=512)
            )
        
        self.rgb_weight1 = nn.Parameter(torch.zeros(1) + 0.6) #0.6
        self.rgb_weight2 = nn.Parameter(torch.zeros(1) + 0.6) #0.6
        self.rgb_weight3 = nn.Parameter(torch.zeros(1) + 0.6) #0.6
        
        self.reset_parameters()

    def reset_parameters(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, rgb, motion):
        x = rgb 
        
        # goal: fuse at 3 blocks: index 1, 3, and 6 (the selected indices loosely match the fusion design done in resnet18)
        fuse_indices = [1,3,6] #[3,6,10] # [1,3,6]
        
        # block 1 fusion
        for i in range(fuse_indices[0] + 1):
            x = self.features[i](x)
            motion = self.features_pa[i](motion)
        
        x = self.rgb_weight1 * x + (1.0 - self.rgb_weight1) * motion
        
        # block 2 fusion
        for i in range(fuse_indices[0] + 1, fuse_indices[1] + 1):
            x = self.features[i](x)
            motion = self.features_pa[i](motion)
        
        x = self.rgb_weight2 * x + (1.0 - self.rgb_weight2) * motion
        
        # block 3 fusion
        for i in range(fuse_indices[1] + 1, fuse_indices[2] + 1):
            x = self.features[i](x)
            motion = self.features_pa[i](motion)
        
        x = self.rgb_weight3 * x + (1.0 - self.rgb_weight3) * motion
        
        # follows through the rest of the feedforward network
        for i in range(fuse_indices[2] + 1, len(self.features)):
            x = self.features[i](x)
        
        # channel reudction 1280->580
        final_features = self.shrink(x)
        
        return final_features

###
# original MobileNetV2 architecture for SSDlite for reference
### 
class MobileNetV2(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        
        # detector head
        '''
        self.extras = nn.ModuleList([
            InvertedResidual(1280, 512, 2, 0.2),
            InvertedResidual(512, 256, 2, 0.5),
            InvertedResidual(256, 256, 2, 0.5),
            InvertedResidual(256, 64, 2, 0.5)
        ])
        '''
        
        # add: shrink (1280 -> 512) to match the spatial dimension of AMMA's backbone feature
        self.shrink = nn.Sequential(
            nn.Conv2d(1280, 512, 
                      kernel_size=1, padding=0, bias=False, groups=1),
            
            nn.BatchNorm2d(num_features=512)
            )
        
        self.reset_parameters()

    def reset_parameters(self):
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        
        features = []
        
        for i in range(len(self.features)):
            x = self.features[i](x)
        #features.append(x)
        
        # for SSDlite detector head
        '''
        for i in range(14):
            x = self.features[i](x)
        features.append(x)

        for i in range(14, len(self.features)):
            x = self.features[i](x)
        features.append(x)
        
        for i in range(len(self.extras)):
            x = self.extras[i](x)
            features.append(x)
        '''
        
        final_features = self.shrink(x)
        
        return final_features
