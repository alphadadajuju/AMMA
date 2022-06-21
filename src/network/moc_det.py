from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from torch import nn
#from .branch import MOC_Branch
from .branch_mod import MOC_Branch
#from .dla import MOC_DLA
from .resnet import MOC_ResNet, AMMA_ResNet
from .MobileNetV2 import AMMA_MobileNetV2
from .ShuffleNetV2 import AMMA_ShuffleNetV2

from .deconv import deconv_layers

import torch.nn.functional as F
import matplotlib.pyplot as plt 
import numpy as np
import cv2

backbone = {
    #'dla': MOC_DLA,
    'resnet': MOC_ResNet
}

class AMMA_Backbone(nn.Module):
    def __init__(self, arch, num_layers, rgb_ws='TTTFF'):
        super(AMMA_Backbone, self).__init__()
        
        # TODO
        if arch == 'mobile_v2':
            self.backbone = AMMA_MobileNetV2()
        
        elif arch == 'shuffle_v2':
            self.backbone = AMMA_ShuffleNetV2()
            
        else:
            self.backbone = AMMA_ResNet(num_layers, rgb_ws=rgb_ws)
         
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
        
        
    def forward(self, input_rgb, input_mo): # input_mo.size(): torch.Size([5, 3, 288, 288]) (same as input_rgb)
        '''
        # debug: visualize rgb and motion
        for ii in range(max(input_mo.size()[0] - 3, 0), input_mo.size()[0]):
            for i in range(input_mo.size()[1]):
                self.vis_feat(input_mo[ii:ii+1,i,:,:].cpu())
        '''
        
        return self.backbone(input_rgb, input_mo)
    
class MOC_Backbone(nn.Module):
    def __init__(self, arch, num_layers,):
        super(MOC_Backbone, self).__init__()
        self.backbone = backbone[arch](num_layers)
        
    def forward(self, input):
        return self.backbone(input)
    
class MOC_Deconv(nn.Module):
    def __init__(self, inplanes, BN_MOMENTUM):
        super(MOC_Deconv, self).__init__()
        
        self.deconv_layer = deconv_layers(inplanes=512, BN_MOMENTUM=0.1)
        #self.init_weights() # should not use for inference (??) -> shouldn't matter as long as you load the correct weight later
    
    def forward(self, input):
        return self.deconv_layer(input)
    
    # separate deconv layer
    def init_weights(self):
        # print('=> init deconv weights from normal distribution')
        for name, m in self.deconv_layer.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
class MOC_Det(nn.Module):
    def __init__(self, backbone, branch_info, arch, head_conv, K, flip_test=False):
        super(MOC_Det, self).__init__()
        self.flip_test = flip_test
        self.K = K
        self.branch = MOC_Branch(256, arch, head_conv, branch_info, K) # backbone.backbone.output_channel == 64

    def forward(self, chunk1):
        assert(self.K == len(chunk1))
        
        chunk1 = F.interpolate(chunk1, [36, 36])
        return [self.branch(chunk1, self.K)]
                