from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        # basical experiment settings
        self.parser.add_argument('--task', default='MOC_train',
                                 help='current task')
        self.parser.add_argument('--exp_id', default='default')
        self.parser.add_argument('--model_name', default='None',
                                 help='corrent model name')
        self.parser.add_argument('--load_model', default='',
                                 help='path to load model')
        self.parser.add_argument('--rgb_model', default='',
                                 help='path to rgb model')
        self.parser.add_argument('--flow_model', default='',
                                 help='path to flow model')
        self.parser.add_argument('--mm_model', default='',
                                 help='path to mm model')
        self.parser.add_argument('--mm_fuse_mode', default='',
                                 help='mode to fuse mm: ')
        self.parser.add_argument('--rgb_ws', type=str, default='TTTFF',
                                 help='Lateral connection weight from motion to rgb: ')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')
        self.parser.add_argument('--ninput', type=int, default=1,
                                 help='length of input, 1 for rgb, 5 for flow by default')
        self.parser.add_argument('--ninputrgb', type=int, default=5,
                                 help='length of input GAP for rgb sparse stream, 5 by default')
        self.parser.add_argument('--mm_mode', type=str, default='mm1',
                                 help='mm modes: mm0 (rgb difference), mm1 (1 lay) or mm2')
        
        # makes it flexible (for different gpus / colab training environment)
        self.parser.add_argument('--save_root', type=str, default='../experiment/result_model/',
                                 help='Directory to save trained model + log')
        # model seeting
        self.parser.add_argument('--arch', default='resnet_18',
                                 help='model architecture. Currently tested'
                                      'resnet_18 | resnet_101 | dla_34 / mobile_v2 / shuffle_v2')
        self.parser.add_argument('--set_head_conv', type=int, default=-1,
                                 help='conv layer channels for output head'
                                      'default setting is 256 for dla and 256 for resnet(except for wh branch) ')
        self.parser.add_argument('--down_ratio', type=int, default=4,#4
                                 help='output stride. Currently only supports 4.')
        self.parser.add_argument('--K', type=int, default=7,
                                 help='length of action tube')

        # system settings
        self.parser.add_argument('--gpus', default='0,1,2,3,4,5,6,7',
                                 help='visible gpu list, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=16,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--batch_size', type=int, default=128,
                                 help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                 help='batch size on the master gpu. -1 by default')

        # learning rate settings
        self.parser.add_argument('--lr', type=float, default=5e-4,
                                 help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='6,8',
                                 help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=30,
                                 help='total training epochs.')
        self.parser.add_argument('--lr_drop', type=float, default=0.5,
                                 help='learning rate drop after N steps.')

        # dataset seetings
        self.parser.add_argument('--dataset', default='hmdb',
                                 help='ucf101 | hmdb')
        self.parser.add_argument('--split', type=int, default=1,
                                 help='1 split for UCF101, 3 splits for JHMDB')
        self.parser.add_argument('--resize_height', type=int, default=288,
                                 help='input image height')
        self.parser.add_argument('--resize_width', type=int, default=288,
                                 help='input image width')

        # training settings
        self.parser.add_argument('--pretrain_model', default='coco',
                                 help='training pretrain_model, coco | imagenet')
        self.parser.add_argument('--ucf_pretrain', action='store_true',
                                 help='use ucf pretrain for jhmdb')

        self.parser.add_argument('--auto_stop', action='store_true',
                                 help='auto_stop when training, used for jhmdb')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save each epoch training model')
        self.parser.add_argument('--val_epoch', action='store_true',
                                 help='val after each epoch')
        self.parser.add_argument('--visual_per_inter', type=int, default=25, # 100
                                 help='iter for draw loss by tensorboardX')

        self.parser.add_argument('--start_epoch', type=int, default=0,
                                 help='strat epoch, used for recover experiment')
        self.parser.add_argument('--pin_memory', action='store_true',
                                 help='set pin_memory True')

        # loss ratio settings
        self.parser.add_argument('--hm_weight', type=float, default=1,
                                 help='loss weight for center heatmaps.')
        self.parser.add_argument('--mov_weight', type=float, default=1,
                                 help='loss weight for moving offsets.')
        self.parser.add_argument('--wh_weight', type=float, default=0.1, #0.1
                                 help='loss weight for bbox regression.')

        # inference settings
        self.parser.add_argument('--redo', action='store_true',
                                 help='redo for count APs')
        self.parser.add_argument('--flip_test', action='store_true',
                                 help='flip data augmentation.')
        self.parser.add_argument('--N', type=int, default=100,
                                 help='max number of output objects.')
        self.parser.add_argument('--inference_dir', default='tmp',
                                 help='directory for inferencing')
        self.parser.add_argument('--th', type=float, default=0.5,
                                 help='threshod for ACT.py')
        
        self.parser.add_argument('--evaluation_mode', type=str, default='trimmed',
                                 help='trimmed or untrimmed evaluation (untrimmed is for ucf24)')
        
        self.parser.add_argument('--inference_mode', type=str, default='normal',
                                 help='stream or normal inference')
        
        # fusion settings for rgb and flow
        self.parser.add_argument('--hm_fusion_rgb', type=float, default=0.5,
                                 help='rgb : th, flow: 1 - th')
        self.parser.add_argument('--mov_fusion_rgb', type=float, default=0.8, # 0.8
                                 help='rgb : th, flow: 1 - th')
        self.parser.add_argument('--wh_fusion_rgb', type=float, default=0.8, # 0.8
                                 help='rgb : th, flow: 1 - th')

        # log
        self.parser.add_argument('--print_log', action='store_true',
                                 help='print log info')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        # This line created problem for setting gpu id?
        #opt.gpus = [i for i in range(len(opt.gpus))]
        
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]

        if opt.set_head_conv != -1:
            opt.head_conv = opt.set_head_conv
        elif 'dla' in opt.arch:
            opt.head_conv = 256
        elif 'resnet' in opt.arch:
            opt.head_conv = 256
        elif 'mobile' in opt.arch:
            opt.head_conv = 256
        elif 'shuffle' in opt.arch:
            opt.head_conv = 256
            
        if opt.ninput == 1 and opt.flow_model != '':
            opt.ninput = 5

        opt.mean = [0.40789654, 0.44719302, 0.47026115]
        opt.std = [0.28863828, 0.27408164, 0.27809835]
        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)
        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..')
        # ORIG: when there exist only two modalities
        #opt.save_dir = opt.rgb_model if opt.rgb_model != '' else opt.flow_model
        if opt.rgb_model != '':
            #opt.save_dir = opt.rgb_model
            opt.save_dir = opt.save_root + opt.rgb_model
        elif opt.flow_model != '':
            opt.save_dir = opt.flow_model
        else:
            opt.save_dir = opt.save_root + opt.mm_model
        
        opt.log_dir = opt.save_dir + '/logs_tensorboardX'

        return opt

    def update_dataset(self, opt, dataset):
        opt.num_classes = dataset.num_classes
        opt.branch_info = {'hm': opt.num_classes,
                           'mov': 2 * (opt.K), # ORIG:opt.K; -2 is hardcoded for now (==n_mem)
                           'wh': 2 * (opt.K)} # ORIG:opt.K
        return opt
