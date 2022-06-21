from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import torchvision.models as models
import torch
import torch.nn as nn

import torch.utils.model_zoo as model_zoo
from network.moc_net import MOC_Net, AMMA_Net
from network.moc_det import MOC_Det, AMMA_Backbone, MOC_Deconv, MOC_Backbone
from network.mm import MM_PA

def create_model_rgb(opt, arch, branch_info, head_conv, K, flip_test=False):
    
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    model = MOC_Net(opt, arch, num_layers, branch_info, head_conv, K, flip_test=flip_test)
    
    return model

def create_model_mm(opt, arch, branch_info, head_conv, K, flip_test=False):
    
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    model = AMMA_Net(opt, arch, num_layers, branch_info, head_conv, K, flip_test=flip_test)
    
    return model

def create_modellite_mm(opt, arch, branch_info, head_conv, K, flip_test=False):
    
    model = AMMA_Net(opt, arch, -1, branch_info, head_conv, K, flip_test=flip_test)
    
    return model

def create_inference_model_mm(opt, arch, branch_info, head_conv, K, flip_test=False):
    
    # hard-coded: mobilenet does not care about this variable
    num_layers = 0
    if arch == 'resnet_18':
        num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
        arch = arch[:arch.find('_')] if '_' in arch else arch
    
    mm = MM_PA(n_length=opt.ninput)
    backbone = AMMA_Backbone(arch, num_layers, opt.rgb_ws)
    deconv = MOC_Deconv(inplanes=256, BN_MOMENTUM=0.1) # 
    branch = MOC_Det(deconv, branch_info, arch, head_conv, K, flip_test=flip_test)
    
    return mm, backbone, deconv, branch

def load_inference_model_mm(mm, backbone, deconv, branch, model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
        
        ###
        # for some older models use different naming convention (comment them if no longer needing them)
        # e.g,. pa vs mm
        ###
        if k.startswith('pa'):
            del state_dict[k]
            k_new = k.replace('pa', 'mm')
            state_dict[k_new] = state_dict_[k]
        
        # e.g., branch.mov.2.weight", "branch.mov.2.bias" vs "branch.mov_cls.0.weight", "branch.mov_cls.0.bias".
        if k == 'branch.mov.2.weight':
            del state_dict[k]
            k_new = "branch.mov_cls.0.weight"
            state_dict[k_new] = state_dict_[k]
            
        if k == 'branch.mov.2.bias':
            del state_dict[k]
            k_new = "branch.mov_cls.0.bias"
            state_dict[k_new] = state_dict_[k]
        
    # strict state_dict matching 
    state_dict_mm = {}
    state_dict_backbone = {}
    state_dict_deconv = {}
    state_dict_branch = {}
    
    for k in state_dict:
        if k.startswith('mm'): 
            state_dict_mm[k[3:]] = state_dict[k] 
            
        elif k.startswith('backbone'): 
            state_dict_backbone[k] = state_dict[k]
        elif k.startswith('deconv_layer'): 
            state_dict_deconv[k] = state_dict[k]
        else:
            state_dict_branch[k] = state_dict[k]
            
    mm.load_state_dict(state_dict_mm, strict=True)      
    backbone.load_state_dict(state_dict_backbone, strict=True) 
    deconv.load_state_dict(state_dict_deconv, strict=True) 
    branch.load_state_dict(state_dict_branch, strict=True)

    return mm, backbone, deconv, branch

def save_model(path, model, optimizer=None, epoch=0, best=100):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'best': best,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)

def load_model(model, model_path, optimizer=None, lr=None, ucf_pretrain=False):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location='cpu')
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}
    
    # convert data_parallal to model
    for k in state_dict_:
        
        if ucf_pretrain:
            if k.startswith('branch.hm') or k.startswith('branch.mov'):
                continue
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
           state_dict[k] = state_dict_[k]
            
    check_state_dict(model.state_dict(), state_dict)
    model.load_state_dict(state_dict, strict=True) # strict=False

    # resume optimizer parameters
    if optimizer is not None:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print('Resumed optimizer with start lr', lr)
        else:
            print('No optimizer parameters in checkpoint.')

    if 'best' in checkpoint:
        best = checkpoint['best']
    else:
        best = 100
    if optimizer is not None:
        return model, optimizer, start_epoch, best
    else:
        return model

def load_coco_pretrained_modellite(opt, model):
    
    if opt.arch == 'mobile_v2':
        model_path = '../experiment/modelzoo/mbv2_coco_ssdlite.pth' 
        
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict_ = checkpoint['model']
        state_dict = {}
        
        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
                
        new_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('backbone.features'):
             
                new_state_dict[key] = value
           
                if opt.mm_model != '':
             
                    new_key = key.replace('backbone.features', 'backbone.features_pa')
                    new_state_dict[new_key] = value
                    
    elif opt.arch == 'shuffle_v2': 
        model_path = '../experiment/modelzoo/shufflenetv2_x1_69.402_88.374.pth.tar'
        
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict_ = checkpoint
        state_dict = {}
        
        # convert data_parallal to model
        for k in state_dict_:
            if k.startswith('module') and not k.startswith('module_list'):
                state_dict[k[7:]] = state_dict_[k]
            else:
                state_dict[k] = state_dict_[k]
                
        new_state_dict = {}
        
        for key, value in state_dict.items():
            
            if key.startswith('features'): #
                
                new_key = 'backbone.' + key
                new_state_dict[new_key] = value
                
                if opt.mm_model != '':
             
                    new_key = new_key.replace('backbone.features', 'backbone.features_mm')
                    new_state_dict[new_key] = value
            
            elif key.startswith('conv1'):
                new_key = 'backbone.' + key
                new_state_dict[new_key] = value
                
                if opt.mm_model != '':
             
                    new_key = new_key.replace('backbone.conv1', 'backbone.conv1_mm')
                    new_state_dict[new_key] = value
                    
            elif key.startswith('conv_last'):
                new_key = 'backbone.' + key
                new_state_dict[new_key] = value
            
    coco_model_path = '../experiment/modelzoo/coco_resdcn18.pth'
    checkpoint = torch.load(coco_model_path, map_location='cpu')
    print('loaded {}, epoch {}'.format(coco_model_path, checkpoint['epoch']))
    coco_state_dict_ = checkpoint['state_dict']
    coco_state_dict = {}

    # convert data_parallal to model
    for k in coco_state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            coco_state_dict[k[7:]] = coco_state_dict_[k]
        else:
            coco_state_dict[k] = coco_state_dict_[k]
    
    for key, value in coco_state_dict.items():
        
        if key.startswith('wh'):
            
            new_key = 'branch.' + key
            new_state_dict[new_key] = value
        
        '''   
        # can reuse deconv layer weights from COCO pretrained?
        elif key.startswith('deconv_layers'):
            
            new_key = 'deconv_layer.' + key
            new_state_dict[new_key] = value
        '''
        
    model.load_state_dict(new_state_dict, strict=False)
    
    return model

def load_coco_pretrained_model(opt, model):
    if opt.arch == 'dla_34':
        print('load coco pretrained dla_34')
        model_path = '../experiment/modelzoo/coco_dla.pth'
    elif opt.arch == 'resnet_18':
        print('load coco pretrained resnet_18')
        model_path = '../experiment/modelzoo/coco_resdcn18.pth'
    
    elif opt.arch == 'resnet_101':
        print('load coco pretrained resnet_101')
        model_path = '../experiment/modelzoo/coco_resdcn101.pth'
    else:
        raise NotImplementedError
        
    checkpoint = torch.load(model_path, map_location='cpu')
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('wh'):
            
            new_key = 'branch.' + key
            new_state_dict[new_key] = value
        
        elif key.startswith('hm') or key.startswith('reg'):
            pass
        
        # ADDED: separating the deconv layers
        elif key.startswith('deconv_layers'):
            
            new_key = 'deconv_layer.' + key
            new_state_dict[new_key] = value
        
        else:
            new_key = 'backbone.' + key
            new_state_dict[new_key] = value
        
        ###
        # duplicate COCO pretrained weights for the mm stream
        ###
        if opt.mm_model != '':
            # fusion stage 1
            if key == 'conv1.weight':
                new_key = 'backbone.' + 'conv1_5.weight'
                new_state_dict[new_key] = value
            
            elif key == 'bn1.weight':
                new_key = 'backbone.' + 'bn1_5.weight'
                new_state_dict[new_key] = value
   
            # fusion stage 2 
            elif key[:6] == 'layer1':
                new_key = 'backbone.resnext_' + key
                new_state_dict[new_key] = value
            
            # fusion stage 3 
            elif key[:6] == 'layer2':
                new_key = 'backbone.resnext_' + key
                new_state_dict[new_key] = value
            
            # Uncomment below for higher extent of appearance-motion fusion
            '''
            # fusion stage 3
            elif key[:6] == 'layer3':
                new_key = 'backbone.resnext_' + key
                new_state_dict[new_key] = value
            
            # fusion stage 3 (beyond TDN design)
            elif key[:6] == 'layer4':
                new_key = 'backbone.resnext_' + key
                new_state_dict[new_key] = value
            '''
                
    if 'resnet' in opt.arch:
        pass
        #new_state_dict = convert_resnet_dcn(new_state_dict)
        
    print('load coco pretrained successfully')
    if opt.print_log:
        check_state_dict(model.state_dict(), new_state_dict)
        print('check done!')
        
    model.load_state_dict(new_state_dict, strict=False)
    
    return model

def check_state_dict(load_dict, new_dict):
    # check loaded parameters and created model parameters
    for k in new_dict:
        if k in load_dict:
            if new_dict[k].shape != load_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}.'.format(
                          k, load_dict[k].shape, new_dict[k].shape))
                new_dict[k] = load_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in load_dict:
        if not (k in new_dict):
            print('No param {}.'.format(k))
            new_dict[k] = load_dict[k]
            
