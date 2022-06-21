from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle

import torch.utils.data as data

from ACT_utils.ACT_utils import tubelet_in_out_tubes, tubelet_has_gt

class BaseDataset(data.Dataset):

    def __init__(self, opt, mode, ROOT_DATASET_PATH, pkl_filename):

        super(BaseDataset, self).__init__()
        pkl_file = os.path.join(ROOT_DATASET_PATH, pkl_filename)

        with open(pkl_file, 'rb') as fid:
            pkl = pickle.load(fid, encoding='iso-8859-1')
        
        
        
        # uncomment the block to reduce video samples (to quickly check if the training/testing pipeline function normally)
        '''
        for sp in range(len(pkl['train_videos'])):
            # Only target split 1 for now
            
            train_videos_reduced = []
            for id, video in enumerate(pkl['train_videos'][sp]):
                if id % 1000 == 0:
                    train_videos_reduced.append(video)
            
            test_videos_reduced = []
            for id, video in enumerate(pkl['test_videos'][sp]):
                if id % 1 == 0:
                    test_videos_reduced.append(video)
            
            pkl['train_videos'][sp] = train_videos_reduced
            pkl['test_videos'][sp] = test_videos_reduced
        '''
        for k in pkl:
            setattr(self, ('_' if k != 'labels' else '') + k, pkl[k])

        self.split = opt.split
        self.mode = mode
        self.K = opt.K
        self.opt = opt

        self._mean_values = [104.0136177, 114.0342201, 119.91659325]
        self._ninput = opt.ninput
        self._resize_height = opt.resize_height
        self._resize_width = opt.resize_width
        
        # added for only the rgb stream (separate from the original ninput now used for mm_model)
        self._ninputrgb = opt.ninputrgb
        
        # comment below when NOT training the full dataset
        #assert len(self._train_videos[self.split - 1]) + len(self._test_videos[self.split - 1]) == len(self._nframes)
        
        self._indices = []
        if self.mode == 'train':
            # get train video list
            video_list = self._train_videos[self.split - 1]
        else:
            # get test video list
            video_list = self._test_videos[self.split - 1]
        if self._ninput < 1:
            raise NotImplementedError('Not implemented: ninput < 1')

        # ie. [v]: Basketball/v_Basketball_g01_c01
        #     [vtubes] : a dict(key, value)
        #                key is the class;  value is a list of <frame number> <x1> <y1> <x2> <y2>. for exactly [v]
        # ps: each v refer to one vtubes in _gttubes (vtubes = _gttubes[v])
        #                                          or (each video has many frames with one classes)
        
        v_count = 0
   
        for v in video_list:
            
            vtubes = sum(self._gttubes[v].values(), [])
            
            new_indices = []
            
            for i in reversed(range(min(self.K * self.opt.ninputrgb, self._nframes[v]) - self.opt.ninputrgb + 1, self._nframes[v] + 1)): # orig: self.K
             
                if tubelet_in_out_tubes(vtubes, i, -1*(min(self.K*self.opt.ninputrgb, self._nframes[v]) - self.opt.ninputrgb + 1)) and tubelet_has_gt(vtubes, i, -1*(min(self.K*self.opt.ninputrgb, self._nframes[v]) - self.opt.ninputrgb + 1)):

                    new_indices += [(v, i, self._nframes[v])]
                
            self._indices += new_indices
            
            v_count += 1
            if v_count % 200 == 0: 
                print ('Finished sampling {} videos.'.format(v_count))
         
        
        print ('Finished pre-sampling!')
        
        '''
        ######
        ## Check if there exists any serious class inbalance problem; add more data samples to those classes
        ## Did not help much in accuracy
        ######
        frames_count = {}
        for seq in range(len(self._indices)):
            tar_cls = self._indices[seq][0].split('/')[0]
            if not (tar_cls in frames_count):
                frames_count[tar_cls] = 0
            else: 
                frames_count[tar_cls] += 1
        
        print ('Finished checking if there exists class inbalance.')
        
        ## filling in self._indices for classes that have fewer examples?
        max_frames = max(frames_count.values())
        min_frames = min(frames_count.values())
        
        update_indices = self._indices.copy()
        
        while True:
            new_ind = random.randint(0, len(self._indices)-1)
            tar_cls = self._indices[new_ind][0].split('/')[0]
            
            
            if frames_count[tar_cls] < max_frames:
                frames_count[tar_cls] += 1
                update_indices.append(self._indices[new_ind])
            
            if min(frames_count.values()) == max_frames:
                break
        
        self._indices = update_indices
        print ('Finished filling in data for classes with fewer examples.')
        '''
        
        
        self.distort_param = {
            'brightness_prob': 0.5,
            'brightness_delta': 32,
            'contrast_prob': 0.5,
            'contrast_lower': 0.5,
            'contrast_upper': 1.5,
            'hue_prob': 0.5,
            'hue_delta': 18,
            'saturation_prob': 0.5,
            'saturation_lower': 0.5,
            'saturation_upper': 1.5,
            'random_order_prob': 0.0,
        }
        self.expand_param = {
            'expand_prob': 0.5, #0.5
            'max_expand_ratio': 2.0, #ORIG: 4.0
        }
        
        self.batch_samplers = [{
            'sampler': {},
            'max_trials': 1,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.6, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.1, }, # 0.1
            'max_trials': 50, #50
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.6, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.3, }, # 0.3
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.5, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.5, }, #0.5
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.7, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'min_jaccard_overlap': 0.9, },
            'max_trials': 50,
            'max_sample': 1,
        }, {
            'sampler': {'min_scale': 0.3, 'max_scale': 1.0, 'min_aspect_ratio': 0.5, 'max_aspect_ratio': 2.0, },
            'sample_constraint': {'max_jaccard_overlap': 1.0, },
            'max_trials': 50,
            'max_sample': 1,
        }, ]
        
        self.max_objs = 128

    def __len__(self):
        return len(self._indices)

    def imagefile(self, v, i):
        raise NotImplementedError

    def flowfile(self, v, i):
        raise NotImplementedError


"""
Abstract class for handling dataset of tubes.

Here we assume that a pkl file exists as a cache. The cache is a dictionary with the following keys:
    labels: list of labels
    train_videos: a list with nsplits elements, each one containing the list of training videos
    test_videos: idem for the test videos
    nframes: dictionary that gives the number of frames for each video
    resolution: dictionary that output a tuple (h,w) of the resolution for each video
    gttubes: dictionary that contains the gt tubes for each video.
                Gttubes are dictionary that associates from each index of label, a list of tubes.
                A tube is a numpy array with nframes rows and 5 columns, <frame number> <x1> <y1> <x2> <y2>.
"""
