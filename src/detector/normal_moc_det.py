from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np

import torch

from MOC_utils.model import create_model_rgb, load_model, create_model_mm, create_modellite_mm
from MOC_utils.data_parallel import DataParallel
from .decode import moc_decode
from MOC_utils.utils import flip_tensor

#from torchvision import utils
import matplotlib.pyplot as plt 

import time 
def vistensor(tensor, ch=0, allkernels=False, nrow=8, padding=1):
    n, c, w, h = tensor.shape
    
    for i in range(n):
        for ch in range(c):
            plt.imshow(tensor[i, ch, :, :].detach().numpy())
            plt.colorbar()
            plt.show()
    
class MOCDetector(object):
    def __init__(self, opt):
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        self.rgb_model = None
        
        self.mm_model = None
        if opt.rgb_model != '':
            print('create rgb model')
            self.rgb_model = create_model_rgb(opt, opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
            self.rgb_model = load_model(self.rgb_model, opt.save_root + opt.rgb_model)
            
            self.rgb_model = DataParallel(
                self.rgb_model, device_ids=opt.gpus,
                chunk_sizes=opt.chunk_sizes).to(opt.device)
            
            self.rgb_model.eval()
        
        if opt.mm_model != '':
            print('create MM model')
            
            if opt.arch == 'resnet_18':
                self.mm_model = create_model_mm(opt, opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
        
            elif opt.arch == 'mobile_v2' or opt.arch == 'shuffle_v2':
                self.mm_model = create_modellite_mm(opt, opt.arch, opt.branch_info, opt.head_conv, opt.K, flip_test=opt.flip_test)
                
            self.mm_model = load_model(self.mm_model, opt.save_root + opt.mm_model)

            self.mm_model = DataParallel(
                self.mm_model, device_ids=opt.gpus, #[0]
                chunk_sizes=opt.chunk_sizes).to(opt.device)
            self.mm_model.eval()
         
        self.num_classes = opt.num_classes
        self.opt = opt
        
        # added: for speed measurement
        self.total_time = 0
        
    def pre_process(self, images, is_flow=False, ninput=1):

        K = self.opt.K
        images = [cv2.resize(im, (self.opt.resize_height, self.opt.resize_width), interpolation=cv2.INTER_LINEAR) for im in images]

        if self.opt.flip_test:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(K * 2)]
        else:
            data = [np.empty((3 * ninput, self.opt.resize_height, self.opt.resize_width), dtype=np.float32) for i in range(K)]

        mean = np.tile(np.array(self.opt.mean, dtype=np.float32)[:, None, None], (ninput, 1, 1))
        std = np.tile(np.array(self.opt.std, dtype=np.float32)[:, None, None], (ninput, 1, 1))

        for i in range(K):
            for ii in range(ninput):
                data[i][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i*ninput + ii], (2, 0, 1)) # added: *ninput
                if self.opt.flip_test:
                    # TODO
                    if is_flow:
                        temp = images[i + ii].copy()
                        temp = temp[:, ::-1, :]
                        temp[:, :, 2] = 255 - temp[:, :, 2]
                        data[i + K][3 * ii:3 * ii + 3, :, :] = np.transpose(temp, (2, 0, 1))
                    else:
                        data[i + K][3 * ii:3 * ii + 3, :, :] = np.transpose(images[i + ii], (2, 0, 1))[:, :, ::-1]
            # normalize
            data[i] = ((data[i] / 255.) - mean) / std
            if self.opt.flip_test:
                data[i + K] = ((data[i + K] / 255.) - mean) / std
        return data

    def process(self, images, flows):
        with torch.no_grad():
            if self.rgb_model is not None:
                rgb_output = self.rgb_model(images)
                #rgb_hm = rgb_output[0]['hm'].sigmoid_()
                rgb_hm = rgb_output[0]['hm']
                rgb_wh = rgb_output[0]['wh']
                rgb_mov = rgb_output[0]['mov']
                
                if self.opt.flip_test:
                    rgb_hm_f = rgb_output[1]['hm'].sigmoid_()
                    rgb_wh_f = rgb_output[1]['wh']

                    rgb_hm = (rgb_hm + flip_tensor(rgb_hm_f)) / 2
                    rgb_wh = (rgb_wh + flip_tensor(rgb_wh_f)) / 2
            
            if self.mm_model is not None:
                mm_output = self.mm_model(flows)
                hm = mm_output[0]['hm'] #.sigmoid_()
                wh = mm_output[0]['wh']
                mov = mm_output[0]['mov']
            
            elif self.rgb_model is not None:
                hm = rgb_hm
                wh = rgb_wh
                mov = rgb_mov
            
            else:
                print('No model exists.')
                assert 0
        
            detections = moc_decode(hm, wh, mov, N=self.opt.N, K=self.opt.K)
            return detections

    def post_process(self, detections, height, width, output_height, output_width, num_classes, K):
        detections = detections.detach().cpu().numpy()
        
        results = []
        for i in range(detections.shape[0]): # batch
            top_preds = {}
            for j in range((detections.shape[2] - 2) // 2):
                # tailor bbox to prevent out of bounds
                detections[i, :, 2 * j] = np.maximum(0, np.minimum(width - 1, detections[i, :, 2 * j] / output_width * width))
                detections[i, :, 2 * j + 1] = np.maximum(0, np.minimum(height - 1, detections[i, :, 2 * j + 1] / output_height * height))
            classes = detections[i, :, -1]
            # gather bbox for each class
            for c in range(self.opt.num_classes):
                inds = (classes == c)
                top_preds[c + 1] = detections[i, inds, :4 * (K-0) + 1].astype(np.float32) # ORIG: just K
            results.append(top_preds)
        return results

    def run(self, data):

        flows = None
        images = None

        if self.rgb_model is not None:
            images = data['images']
            
            for i in range(len(images)):
                
                '''
                # ADDED: vis for debug
                # data[i] = ((data[i] / 255.) - mean) / std
                image_temp = images[i].numpy().squeeze().transpose(1,2,0)
                image_temp = ((image_temp * self.opt.std + self.opt.mean) * 255).astype(np.uint8)
                image_temp = cv2.cvtColor(image_temp, cv2.COLOR_BGR2RGB)
                plt.imshow(image_temp)
                plt.show()
                '''  
                images[i] = images[i].to(self.opt.device)
       
        if self.mm_model is not None:
            flows = data['flows']
            for i in range(len(flows)):
                flows[i] = flows[i].to(self.opt.device)
        
        meta = data['meta']
        meta = {k: v.numpy()[0] for k, v in meta.items()}
        
        detection_start = time.time()
        detections = self.process(images, flows)
        detection_end = time.time()
        self.total_time += detection_end - detection_start
        
        detections = self.post_process(detections, meta['height'], meta['width'],
                                       meta['output_height'], meta['output_width'],
                                       self.opt.num_classes, self.opt.K)

        return detections, self.total_time
