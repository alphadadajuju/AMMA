from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from progress.bar import Bar
import torch
import pickle

from opts import opts
from datasets.init_dataset import switch_dataset
#from detector.stream_moc_det import MOCDetector
from detector.MMstream_moc_det import MOCDetector
import random

import time
import matplotlib.pyplot as plt 

GLOBAL_SEED = 317


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def worker_init_fn(dump):
    set_seed(GLOBAL_SEED)


class PrefetchDataset(torch.utils.data.Dataset):
    def __init__(self, opt, dataset, pre_process, pre_process_single_clip):
        self.pre_process = pre_process
        
        # treat a "clip" as a basic unit
        self.pre_process_single_clip = pre_process_single_clip
        
        self.opt = opt
        self.vlist = dataset._test_videos[dataset.split - 1]
        self.gttubes = dataset._gttubes
        self.nframes = dataset._nframes
        self.imagefile = dataset.imagefile
        self.flowfile = dataset.flowfile
        self.resolution = dataset._resolution
        self.input_h = dataset._resize_height
        self.input_w = dataset._resize_width
        self.output_h = self.input_h // self.opt.down_ratio
        self.output_w = self.input_w // self.opt.down_ratio
        self.indices = []
    
        self.n_mem = self.opt.K - 1
    
        total_num_frames = 0
     
        for v in self.vlist:
            
            total_num_frames += self.nframes[v]
            use_ind_flag = True # sample frames when True
            ind_cd = self.opt.ninput 
            if self.opt.mm_model != '':
                for i in range(min(self.opt.K * self.opt.ninput , self.nframes[v]) - self.opt.ninput + 1, 1 + self.nframes[v]): # start: self.opt.K ||+ (self.opt.ninput - 1)
                    
                    # also sample a frame reaching the final frame of a video
                    if (use_ind_flag is True) or i == self.nframes[v]:
                            self.indices += [(v, i)]
                            use_ind_flag = False # correct: False (to skip frames)
                            ind_cd = self.opt.ninput # sample every other ninput (e.g., 5) frame
                            
                    ind_cd -= 1
                    if ind_cd == 0:
                        use_ind_flag = True
                    
            elif self.opt.rgb_model != '':
                for i in reversed(range(min(self.opt.K * self.opt.ninputrgb - self.opt.ninputrgb + 1 , self.nframes[v]), 1 + self.nframes[v])): # start: self.opt.K ||+ (self.opt.ninput - 1)
                    if not os.path.exists(self.outfile(v, i)):
                        self.indices += [(v, i)]
        
        print ('Finished loading det indices.')
        print ('There is a total of {} frames.'.format(total_num_frames))
        
        self.img_buffer = []
        self.flow_buffer = []
        self.img_buffer_flip = []
        self.flow_buffer_flip = []
        self.last_video = -1
        self.last_frame = -1
        
        # debug: to keep track of what frames actually being detected
        self.im_list_history = []
        
    def __getitem__(self, index):
        v, frame = self.indices[index]
        h, w = self.resolution[v]
        images = []
        flows = []
        video_tag = 0
        
        # if there is a new video
        if (v == self.last_video and frame == self.last_frame + self.opt.ninput) or (v == self.last_video and frame == self.nframes[v]):  #and frame == self.last_frame + 1:
            video_tag = 1 # correct: 1
        else:
            video_tag = 0 # 0

        self.last_video = v
        self.last_frame = frame
        
        if video_tag == 0:
            
            # clear out history for a fresh start
            self.im_list_history = []
            
            # Current version does not support stream mode for rgb models; adapting it from micro-motion models should be fairly straightforward
            if self.opt.rgb_model != '':
                images = [cv2.imread(self.imagefile(v, frame + i)).astype(np.float32) for i in range(self.opt.K)]
                images = self.pre_process(images)
                if self.opt.flip_test:
                    self.img_buffer = images[:self.opt.K]
                    self.img_buffer_flip = images[self.opt.K:]
                else:
                    self.img_buffer = images
            
            # Current version does not support stream mode for flow models; adapting it from micro-motion models should be fairly straightforward
            if self.opt.flow_model != '':
                flows = [cv2.imread(self.flowfile(v, min(frame + i, self.nframes[v]))).astype(np.float32) for i in range(self.opt.K + self.opt.ninput - 1)]
                flows = self.pre_process(flows, is_flow=True, ninput=self.opt.ninput)

                if self.opt.flip_test:
                    self.flow_buffer = flows[:self.opt.K]
                    self.flow_buffer_flip = flows[self.opt.K:]
                else:
                    self.flow_buffer = flows
                    
            if self.opt.mm_model != '':
                
                n_mem =  self.n_mem
                im_inds = []
                
                # key frame of the input sequence
                for _ in range(1): 
                    im_inds.append(frame - 1)
                    self.im_list_history.append(frame)
                    
                cur_f = frame
                
                # rest of frames of the input sequence
                for _ in range(1, n_mem+1):
                    cur_f = np.maximum(cur_f - self.opt.ninput, 1)
                    im_inds.append(cur_f - 1)
                    self.im_list_history.append(cur_f)
                
                # debug
                im_inds_mm = []
                for idx, i in enumerate(im_inds): 
                    for ii in range(self.opt.ninput):
                        flows.append(cv2.imread(self.imagefile(v, max(i + 1 - ii, 1))).astype(np.float32))
                        im_inds_mm.append(max(i - ii, 0))
                
                im_inds.reverse()
                flows.reverse()
                im_inds_mm.reverse()
                self.im_list_history.reverse()
                #print ('num of frames of this vid: {}'.format(self.nframes[v]))
                #print(self.im_list_history)
                
                flows = self.pre_process(flows, is_flow=False, ninput=self.opt.ninput)
                self.img_buffer = flows
        
        else:
            # Current version does not support stream mode for rgb models
            if self.opt.rgb_model != '':
                
                image = cv2.imread(self.imagefile(v, frame + self.opt.K - 1)).astype(np.float32)
                image, image_flip = self.pre_process_single_frame(image)
                del self.img_buffer[0]
                self.img_buffer.append(image)
                if self.opt.flip_test:
                    raise NotImplementedError('Flip test not supported in current version.')
                    del self.img_buffer_flip[0]
                    self.img_buffer_flip.append(image_flip)
                    images = self.img_buffer + self.img_buffer_flip
                else:
                    images = self.img_buffer
                    
            # Current version does not support stream mode for flow models;
            if self.opt.flow_model != '':
                flow = cv2.imread(self.flowfile(v, min(frame + self.opt.K + self.opt.ninput - 2, self.nframes[v]))).astype(np.float32)
                data_last_flip = self.flow_buffer_flip[-1] if self.opt.flip_test else None
                data_last = self.flow_buffer[-1]
                flow, flow_flip = self.pre_process_single_frame(flow, is_flow=True, ninput=self.opt.ninput, data_last=data_last, data_last_flip=data_last_flip)
                del self.flow_buffer[0]
                self.flow_buffer.append(flow)
                if self.opt.flip_test:
                    raise NotImplementedError('Flip test not supported in current version.')
                    del self.flow_buffer_flip[0]
                    self.flow_buffer_flip.append(flow_flip)
                    flows = self.flow_buffer + self.flow_buffer_flip
                else:
                    flows = self.flow_buffer
            
            if self.opt.mm_model != '': 
                
                im_inds_mm = []
                flow_clip = []
                
                for ii in range(self.opt.ninput):
                    flow_clip.append(cv2.imread(self.imagefile(v, max(frame - ii, 1))).astype(np.float32))
                    im_inds_mm.append(max(frame - ii - 1, 0))
                
                self.im_list_history.append(frame)
                #print(self.im_list_history)
                
                flow_clip.reverse()
                im_inds_mm.reverse()
                
                flow_clip = self.pre_process_single_clip(flow_clip, is_flow=False, ninput=self.opt.ninput)
                flow_clip = flow_clip[0] # simply b/c it is a list of size 1
                
                del self.img_buffer[0] # len(self.img_buffer = K)
                self.img_buffer.append(flow_clip)
                
                if self.opt.flip_test:
                    raise NotImplementedError('Flip test not supported in current version.')
                    #del self.img_buffer_flip[0]
                    #self.img_buffer_flip.append(image_flip)
                    #flows = self.img_buffer + self.img_buffer_flip
                else:
                    flows = self.img_buffer
                    
        outfile = self.outfile(v, frame)
        if not os.path.isdir(os.path.dirname(outfile)):
            os.system("mkdir -p '" + os.path.dirname(outfile) + "'")

        return {'outfile': outfile, 'images': images, 'flows': flows, 'meta': {'height': h, 'width': w, 'output_height': self.output_h, 'output_width': self.output_w}, 'video_tag': video_tag}

    def outfile(self, v, i):
        return os.path.join(self.opt.inference_dir, v, "{:0>5}.pkl".format(i))

    def __len__(self):
        return len(self.indices)

def interpolate_detection(dets, list_update):
    for label in dets[0]:
        tubelets = dets[0][label]
        
        interval = len(list_update) - 1
        for ii in range(interval):
            lo_bound = tubelets[:,ii*4:ii*4+4]
            hi_bound = tubelets[:,ii*4+4:ii*4+8]
            scores = tubelets[:,-1]
            assert len(lo_bound) == len(hi_bound)
            
            for d in range(len(lo_bound)): 
                lo_box = lo_bound[d]
                hi_box = hi_bound[d]
                sc = scores[d]
                
                #assert lo_box[-1] == hi_box[-1] and lo_box[-2] == hi_box[-2] # score and label should be equavelent
                
                diff_box = (hi_box - lo_box) / ((list_update[ii+1] - list_update[ii]) + 0.001)

def vis_feat(image):
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
        
def stream_inference(opt):
    torch.cuda.set_device(opt.gpus[0])
    # torch.backends.cudnn.benchmark = True

    Dataset = switch_dataset[opt.dataset]
    opt = opts().update_dataset(opt, Dataset)

    dataset = Dataset(opt, 'test')
    detector = MOCDetector(opt)
    prefetch_dataset = PrefetchDataset(opt, dataset, detector.pre_process, detector.pre_process_single_clip)
    data_loader = torch.utils.data.DataLoader(
        prefetch_dataset,
        batch_size=1, # orig: 1 (?)
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        worker_init_fn=worker_init_fn)

    num_iters = len(data_loader)

    bar = Bar(opt.exp_id, max=num_iters)
    
    print('inference chunk_sizes:', opt.chunk_sizes)
    print('Length of process data: {}'.format(len(data_loader)))
    
    data_time_start = time.time()
    data_time = 0.0
    
    save_display_time = 0.0
    for iter, data in enumerate(data_loader):
        '''
        # debug / qualitative analysis: visualize
        input_all = data['flows']
        for ii in range(opt.K-3, opt.K):
            for i in range(opt.ninput):
                vis_feat(input_all[ii][:, i*3:i*3+3,:,:].cpu())
        '''
        data_time_end = time.time()
        data_time += data_time_end - data_time_start
        
        outfile = data['outfile']
        detections, det_time = detector.run(data)
        
        if iter % 1000 == 0:
            print('Data time {} seconds.'.format(data_time))
            print('Processed {}/{} frames; {} seconds.'.format(iter+1, num_iters, det_time))
            
        # TODO: interpolation between frames
        # In fact, interp fits better in ACT_build
        # If done here, then ACT build frame index need to be largely modified 
        # interpolation inference can be estimated in ACT_build?
        
        #interpolate_detection(detections, data['K_frames'])
        
        save_display_start = time.time()
        
        for i in range(len(outfile)):
            with open(outfile[i], 'wb') as file:
                pickle.dump(detections[i], file)
        
        Bar.suffix = 'inference: [{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            iter, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

        bar.next()
        
        save_display_end = time.time()
        save_display_time += save_display_end - save_display_start 
        
        data_time_start = time.time()
    bar.finish()
    
    print('Processed all frames; data {} seconds.'.format(data_time))
    print('Processed all frames; det {} seconds.'.format(det_time))
    print('Processed all frames; save_display {} seconds.'.format(save_display_time))
    print('Processed all frames; total {} seconds.'.format(det_time + data_time + save_display_time))
