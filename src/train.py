#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 08:51:55 2021

@author: alphadadajuju
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import numpy as np
import random
import tensorboardX
import time

from opts import opts
from datasets.init_dataset import get_dataset
from trainer.logger import Logger
from MOC_utils.model import create_model_rgb, create_model_mm, create_modellite_mm, load_coco_pretrained_model, load_coco_pretrained_modellite, save_model, load_model
from trainer.moc_trainer import MOCTrainer
from inference.normal_inference import normal_inference
from ACT import frameAP

from adamW import AdamW
from ptflops.flops_counter import get_model_complexity_info


GLOBAL_SEED =  317

# for flop computation 
def prepare_input(resolution):
    
    x1 = torch.FloatTensor(1, *resolution)
    x2 = torch.FloatTensor(1, *resolution)
    x3 = torch.FloatTensor(1, *resolution)
    x4 = torch.FloatTensor(1, *resolution)
    x5 = torch.FloatTensor(1, *resolution)
    
    return [x1, x2, x3, x4, x5]
  
    
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    #torch.backends.cudnn.benchmark = False    
    #torch.backends.cudnn.deterministic = True
    
def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)
    
def main(opt):
    # added to specify gpu id; the gpus arg in the provided code does not work 
    torch.cuda.set_device(opt.gpus[0])
    set_seed(opt.seed)

    print('dataset: ' + opt.dataset + '   task:  ' + opt.task)
    Dataset = get_dataset(opt.dataset)
    opt = opts().update_dataset(opt, Dataset)
    
    train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train'))
    epoch_train_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'train_epoch'))
    val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val'))
    epoch_val_writer = tensorboardX.SummaryWriter(log_dir=os.path.join(opt.log_dir, 'val_epoch'))
    
    logger = Logger(opt, epoch_train_writer, epoch_val_writer)
    opt.device = torch.device('cuda')
    
    if opt.rgb_model != '':
        model = create_model_rgb(opt, opt.arch, opt.branch_info, opt.head_conv, opt.K)
    
    elif opt.mm_model != '':
        if opt.arch == 'resnet_18':
            model = create_model_mm(opt, opt.arch, opt.branch_info, opt.head_conv, opt.K)
        
        elif opt.arch == 'mobile_v2' or opt.arch == 'shuffle_v2':
            model = create_modellite_mm(opt, opt.arch, opt.branch_info, opt.head_conv, opt.K)
        
    '''
    # Complexity analysis
    with torch.cuda.device(1):

      macs, params = get_model_complexity_info(model, (opt.ninput * 3, 288, 288), input_constructor=prepare_input, as_strings=True,
                                               print_per_layer_stat=True, verbose=True)
      print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
      print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    '''
    
    '''
    optimizer = torch.optim.Adam([{"params": model.backbone.parameters(), "lr": opt.lr},
                                      {"params": model.deconv_layer.parameters(), "lr": opt.lr},
                                      {"params": model.branch.parameters(), "lr": opt.lr}], opt.lr)
    '''
    
    optimizer = AdamW(model.parameters(), lr=opt.lr, weight_decay=3e-2)
    
    
    '''
    # debug: freezing early layers to reduce overfitting; slight drop of accuracy on jh_s1
    print('Frozen layers:')
    for name, param in model.named_parameters():
        
        continue
        
        base_mod = name.split('.')[0]
        layer = name.split('.')[1]
        
        # freeze the first conv
        if base_mod == 'backbone':
            if layer == 'conv1' or layer == 'bn1':
                param.requires_grad = False
                print (name)
            
            if layer == 'conv1_5' or layer == 'bn1_5':
                param.requires_grad = False
                print (name)
            
            
        if base_mod == 'backbone' or base_mod == 'deconv_layer':
            param.requires_grad = False
            print (name)
    
    '''
    
    start_epoch = opt.start_epoch
    
    # ADDED: allowing automatic lr dropping upon resuming a training
    step_count = 0
    for step in range(len(opt.lr_step)):
        if start_epoch >= opt.lr_step[step]:
            step_count += 1
    opt.lr = opt.lr * (opt.lr_drop**step_count)
    
    if 'resnet' in opt.arch or 'dla' in opt.arch:
        if opt.pretrain_model == 'coco':
            
            model = load_coco_pretrained_model(opt, model)

        elif opt.pretrain_model == 'imagenet':
            raise NotImplementedError("Pretrained models should be either COCO.")
        else:
            raise NotImplementedError("Pretrained models should be either COCO.")
    
    elif 'mobile' in opt.arch or 'shuffle' in opt.arch:
        if opt.pretrain_model == 'coco':
           
            model = load_coco_pretrained_modellite(opt, model)
        else:
            raise NotImplementedError("Pretrained models should be either COCO.")
            
    if opt.load_model != '':
        model, optimizer, _, _ = load_model(model, opt.load_model, optimizer, opt.lr, opt.ucf_pretrain)
    
    trainer = MOCTrainer(opt, model, optimizer)
    trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
    
    train_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'train'),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=opt.pin_memory,
        drop_last=True,
        worker_init_fn=worker_init_fn
    )

    print('training...')
    print('GPU allocate:', opt.chunk_sizes)
    best_ap = 0
    best_epoch = 0
    stop_step = 0
    
    # added: to ensure no decrease of lr too early (for jh s1?)
    if stop_step == 0:
        drop_early_flag = False # False for more reproducible results; do not load models trained after a few number of epochs (despite good validation accuracy)
    else: 
        drop_early_flag = True
        
    set_seed(opt.seed)
    
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        print('eopch is ', epoch)
        log_dict_train = trainer.train(epoch, train_loader, train_writer)
        logger.write('epoch: {} |'.format(epoch))
        for k, v in log_dict_train.items():
            logger.scalar_summary('epcho/{}'.format(k), v, epoch, 'train')
            logger.write('train: {} {:8f} | '.format(k, v))
        logger.write('\n')
        
        if opt.save_all and not opt.auto_stop:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            model_name = 'model_[{}]_{}.pth'.format(epoch, time_str)
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])
        else:
            model_name = 'model_last.pth'
            save_model(os.path.join(opt.save_dir, model_name),
                       model, optimizer, epoch, log_dict_train['loss'])

        # this step evaluates the model
        if opt.val_epoch:
            with torch.no_grad():
                log_dict_val = trainer.val(epoch, val_loader, val_writer)
            for k, v in log_dict_val.items():
                logger.scalar_summary('epcho/{}'.format(k), v, epoch, 'val')
                logger.write('val: {} {:8f} | '.format(k, v))
        logger.write('\n')
        
        if opt.auto_stop:
            tmp_rgb_model = opt.rgb_model
            tmp_mm_model = opt.mm_model
            
            if opt.rgb_model != '':
                opt.rgb_model = os.path.join(opt.rgb_model, model_name)
            elif opt.mm_model != '':
                opt.mm_model = os.path.join(opt.mm_model, model_name)
            
            normal_inference(opt)
            ap = frameAP(opt, print_info=opt.print_log)
            print ('frame mAP: {}'.format(ap) )
            
            os.system("rm -rf tmp")
            if ap > best_ap:
                best_ap = ap
                best_epoch = epoch
                saved1 = os.path.join(opt.save_dir, model_name)
                saved2 = os.path.join(opt.save_dir, 'model_best.pth')
                os.system("cp " + str(saved1) + " " + str(saved2))
            
            if stop_step < len(opt.lr_step) and epoch >= opt.lr_step[stop_step]:
                
                # Don't want to decrease lr too early when validation mAP was high by accident (non-reproducible models)
             
                # hard-coded: if best happens before target step - 1 or 2 (further away would be misleading?)
                if best_epoch ==  opt.lr_step[0] - 1:# or best_epoch ==  opt.lr_step[0] - 2:
                    drop_early_flag = True
                
                if drop_early_flag is False: 
                    model, optimizer, _, _ = load_model(
                    model, os.path.join(opt.save_dir, 'model_last.pth'), optimizer, opt.lr) # model_best -> model_last?
                    drop_early_flag = True
                    print('load epoch is ', epoch)
                    
                else: # after the first drop, the rest could drop based on mAP
                    model, optimizer, _, _ = load_model(
                        model, os.path.join(opt.save_dir, 'model_best.pth'), optimizer, opt.lr) # model_best -> model_last?
                    print('load epoch is ', best_epoch)
                
                opt.lr = opt.lr * opt.lr_drop
                logger.write('Drop LR to ' + str(opt.lr) + '\n')
                # added for debug
                print('Drop LR to ' + str(opt.lr) + '\n')
                
                for ii, param_group in enumerate(optimizer.param_groups):
                    param_group['lr'] = opt.lr
                    
                
                torch.cuda.empty_cache()
                trainer = MOCTrainer(opt, model, optimizer)
                trainer.set_device(opt.gpus, opt.chunk_sizes, opt.device)
                stop_step = stop_step + 1
            
            opt.rgb_model = tmp_rgb_model
            opt.mm_model = tmp_mm_model
            
        else:
            # this step drop lr
            if epoch in opt.lr_step:
                lr = opt.lr * (opt.lr_drop ** (opt.lr_step.index(epoch) + 1))
                logger.write('Drop LR to ' + str(lr) + '\n')
                
                # added for debug
                print('Drop LR to ' + str(lr) + '\n')
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
    if opt.auto_stop:
        print('best epoch is ', best_epoch)
        
    logger.close()
if __name__ == '__main__':
    os.system("rm -rf tmp")
    opt = opts().parse()
    main(opt)
    