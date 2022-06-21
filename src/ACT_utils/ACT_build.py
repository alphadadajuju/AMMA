from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import pickle

import numpy as np

from progress.bar import Bar

from datasets.init_dataset import get_dataset

from .ACT_utils import nms2d, nms_tubelets, iou2d

import matplotlib.pyplot as plt
import cv2

import time

def load_frame_detections_stream(opt, dataset, K, vlist, inference_dir):
    
    total_interp_time = 0
    
    alldets = []  # list of numpy array with <video_index> <frame_index> <ilabel> <score> <x1> <y1> <x2> <y2>
    bar = Bar('{}'.format('FrameAP'), max=len(vlist))
    
    for iv, v in enumerate(vlist):
        
        use_ind_flag = True
        ind_cd = opt.ninputrgb 
        opt_ninput = opt.ninputrgb 
         
        h, w = dataset._resolution[v]

        # aggregate the results for each frame from detection pkl file
        vdets = {i: np.empty((0, 6), dtype=np.float32) for i in range(1, 1 + dataset._nframes[v])} # 1 # x1, y1, x2, y2, score, ilabel

        # sparsely sampled frame indices for evaluation (remember that detection did not take place at all frames)
        last_k_ind = []
        last_k_ind_init = min(K  * opt_ninput, dataset._nframes[v]) - opt_ninput + 1
        last_k_ind.append(last_k_ind_init)
         
        for _ in range(opt.K - 1):
            last_k_ind_init = max(1, last_k_ind_init - opt_ninput)
            last_k_ind.append(last_k_ind_init)
        last_k_ind.reverse()
        
        for i in range(min(K  * opt_ninput, dataset._nframes[v]) - opt_ninput + 1, 1 + dataset._nframes[v]): 
            
            # use_ind_flag init as True: always process the first frame
            if use_ind_flag is False and i != dataset._nframes[v]: 
                
                # count down to process a frame
                ind_cd -= 1
                
                if ind_cd == 0: # process this i frame 
                    use_ind_flag = True 
                    ind_cd = opt.ninputrgb
                    
                # skip frame(s)
                else: 
                    continue
            
            use_ind_flag = False # correct: False (in order to skip some frames)
            
            if i  != min(K  * opt_ninput, dataset._nframes[v]) - opt_ninput + 1:
                last_k_ind.append(i)
                
                if opt.K != 1:
                    if len(last_k_ind) > opt.K: # always only keep the latest K frames
                        del last_k_ind[0]
                
            pkl = os.path.join(inference_dir, v, "{:0>5}.pkl".format(i))
            if not os.path.isfile(pkl):
                print("ERROR: Missing extracted tubelets " + pkl)
                sys.exit()

            with open(pkl, 'rb') as fid:
                dets = pickle.load(fid)

            for label in dets:
                
                # dets  : {label:  N, 4K+1}
                # 4*K+1 : (x1, y1, x2, y2) * K, score
                tubelets = dets[label]
                
                # filter out low-condidence boxes (only for visualization purpose)
                #tubelets = tubelets[tubelets[:,-1] > 0.012]
                
                labels = np.empty((tubelets.shape[0], 1), dtype=np.int32)
                labels[:, 0] = label - 1
               
                n_mem = K - 1
            
                list_update = []
                
                for j in reversed(range(0, K)):
                    if vdets[last_k_ind[j]].shape[0] < 10000:
                        vdets[last_k_ind[j]] = np.concatenate((vdets[last_k_ind[j]], np.concatenate((tubelets[:, np.array([4*(n_mem), 1+4*(n_mem), 2+4*(n_mem), 3+4*(n_mem), -1])], labels), axis=1)), axis=0)
                        n_mem = n_mem - 1
                        list_update.append(last_k_ind[j])
                
                ###
                # debug: visualize image and detected boxes?
                ###
                '''
                k_ind_history = last_k_ind
                for kk in range(opt.K):
                    v_file = '/home/alphadadajuju/projects/MMT/data/JHMDB/Frames/' + v + '/' + str(k_ind_history[k_ind_history.index(i) - kk]).zfill(5) + '.png'  
                    print(v_file)
                    
                    imk = cv2.imread(v_file)
                    fig = plt.figure()
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.axis('off')
                    ax.imshow(cv2.cvtColor(imk, cv2.COLOR_BGR2RGB), interpolation='nearest')
                    #plt.imshow(cv2.cvtColor(imk, cv2.COLOR_BGR2RGB))
                    
                    tind = opt.K - kk - 1
                    x1y1x2y2 = tubelets[:,tind*4:tind*4+4]
                    scores = tubelets[:,-1]
                    
                    for n_t in range(len(x1y1x2y2)):
                        x1, y1, x2, y2 = x1y1x2y2[n_t, :]
                        
                        if n_t == 0:
                            box_color = 'r'
                        elif n_t == 1:
                            box_color = 'g'
                        elif n_t == 2:
                            box_color = 'b'
                        else:
                            box_color = 'w'
                        
                        ax.add_patch(plt.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            fill=False,
                            edgecolor=box_color,
                            linewidth=3))
                        
                        text = "%.2f" % scores[n_t]

                        ax.text(x1 - 2, y1 - 10,
                                text,
                                bbox=dict(facecolor='navy', alpha=0.7),
                                fontsize=12, color='yellow')
                    
                    fig.suptitle(v_file[-9:], fontsize=20, color='w')
                    plt.show()
                    '''
                interp_start = time.time()
                
                # extrapolate intra-frame detection results
                list_update.reverse()
                #print(list_update)
                interval = len(list_update) - 1
                for ii in range(interval):
                    lo_bound = tubelets[:,ii*4:ii*4+4]
                    hi_bound = tubelets[:,ii*4+4:ii*4+8]
                    scores = tubelets[:,-1]
                    assert len(lo_bound) == len(hi_bound)
                    
                    if list_update[ii + 1] - list_update[ii] > 1: # if there are gaps between
                        for d in range(len(lo_bound)): 
                            lo_box = lo_bound[d]
                            hi_box = hi_bound[d]
                            sc = scores[d]
                            
                            #assert lo_box[-1] == hi_box[-1] and lo_box[-2] == hi_box[-2] # score and label should be equavelent
                            
                            diff_box = (hi_box - lo_box) / ((list_update[ii+1] - list_update[ii]))
                            
                            for iii in range(list_update[ii]+1, list_update[ii+1]):
                                if vdets[iii].shape[0] < 10000:
                                    vdets[iii] = np.concatenate((vdets[iii], np.array([np.concatenate([lo_box + (iii-list_update[ii])*diff_box, [sc], [label-1]])])))
                
                interp_end = time.time()
                total_interp_time += interp_end - interp_start
        
        # Perform NMS in each frame
        # vdets : {frame_num:  K*N, 6} ---- x1, x2, y1, y2, score, label

        for i in vdets:
            num_objs = vdets[i].shape[0]
            for ilabel in range(len(dataset.labels)):
                vdets[i] = vdets[i].astype(np.float32)
                a = np.where(vdets[i][:, 5] == ilabel)[0]
                if a.size == 0:
                    continue
                vdets[i][vdets[i][:, 5] == ilabel, :5] = nms2d(vdets[i][vdets[i][:, 5] == ilabel, :5], 0.6)
            # alldets: N,8 --------> ith_video, ith_frame, label, score, x1, x2, y1, y2
            alldets.append(np.concatenate((iv * np.ones((num_objs, 1), dtype=np.float32), i * np.ones((num_objs, 1),
                                                                                                      dtype=np.float32), vdets[i][:, np.array([5, 4, 0, 1, 2, 3], dtype=np.int32)]), axis=1))
        
        # debug: visualize (incomplete code?)
        #vis_frame_detection_alllabels(v, vdets, opt, target_class_index)
        
        Bar.suffix = '[{0}/{1}]:{2}|Tot: {total:} |ETA: {eta:} '.format(iv + 1, len(vlist), v, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    bar.finish()
    
    print ('Total time to interpolate frame detection: {} seconds'.format(total_interp_time))
    
    return np.concatenate(alldets, axis=0)

def load_frame_detections(opt, dataset, K, vlist, inference_dir):
    alldets = []  # list of numpy array with <video_index> <frame_index> <ilabel> <score> <x1> <y1> <x2> <y2>
    bar = Bar('{}'.format('FrameAP'), max=len(vlist))
    
    for iv, v in enumerate(vlist):
        
        h, w = dataset._resolution[v]

        # aggregate the results for each frame from detection pkl file
        vdets = {i: np.empty((0, 6), dtype=np.float32) for i in range(1, 1 + dataset._nframes[v])} # 1 # x1, y1, x2, y2, score, ilabel

        opt_ninput = opt.ninputrgb
        
        for i in range(min(K  * opt_ninput , dataset._nframes[v]) - opt_ninput + 1  , 1 + dataset._nframes[v]): # short-range mem: K, 1 + dataset._nframes[v] # + opt.ninput - 1
            
            pkl = os.path.join(inference_dir, v, "{:0>5}.pkl".format(i))
            if not os.path.isfile(pkl):
                print("ERROR: Missing extracted tubelets " + pkl)
                sys.exit()

            with open(pkl, 'rb') as fid:
                dets = pickle.load(fid)

            for label in dets:
                # dets  : {label:  N, 4K+1}
                # 4*K+1 : (x1, y1, x2, y2) * K, score
                
                tubelets = dets[label]
                labels = np.empty((tubelets.shape[0], 1), dtype=np.int32)
                labels[:, 0] = label - 1
                
                n_mem = K - 1
                
                # linked clip
                for j in range(0, K):
                    if vdets[max(i-opt_ninput*j, 1)].shape[0] <= 10000: #10000
                        vdets[max(i-opt_ninput*j, 1)] = np.concatenate((vdets[max(i-opt_ninput*j, 1)], np.concatenate((tubelets[:, np.array([4*(n_mem), 1+4*(n_mem), 2+4*(n_mem), 3+4*(n_mem), -1])], labels), axis=1)), axis=0)
                        n_mem = n_mem - 1
        
        # Perform NMS in each frame
        # vdets : {frame_num:  K*N, 6} ---- x1, x2, y1, y2, score, label
        for i in vdets:
            
            num_objs = vdets[i].shape[0]
            for ilabel in range(len(dataset.labels)):
                vdets[i] = vdets[i].astype(np.float32)
                a = np.where(vdets[i][:, 5] == ilabel)[0]
                if a.size == 0:
                    continue
                vdets[i][vdets[i][:, 5] == ilabel, :5] = nms2d(vdets[i][vdets[i][:, 5] == ilabel, :5], 0.6)
            # alldets: N,8 --------> ith_video, ith_frame, label, score, x1, x2, y1, y2
            alldets.append(np.concatenate((iv * np.ones((num_objs, 1), dtype=np.float32), i * np.ones((num_objs, 1),
                                                                                                      dtype=np.float32), vdets[i][:, np.array([5, 4, 0, 1, 2, 3], dtype=np.int32)]), axis=1))
        Bar.suffix = '[{0}/{1}]:{2}|Tot: {total:} |ETA: {eta:} '.format(iv + 1, len(vlist), v, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
    bar.finish()
    return np.concatenate(alldets, axis=0)

def BuildTubes(opt):
    redo = opt.redo
    if not redo:
        print('load previous linking results...')
        print('if you want to reproduce it, please add --redo')
    Dataset = get_dataset(opt.dataset)
    inference_dirname = opt.inference_dir
    K = opt.K
    split = 'val'
    dataset = Dataset(opt, split)

    print('inference_dirname is ', inference_dirname)
    vlist = dataset._test_videos[opt.split - 1]
    bar = Bar('{}'.format('BuildTubes'), max=len(vlist))
    
    total_build_tube_time = 0.0
    total_load_detection_time = 0.0
    total_save_detection_time = 0.0
    total_interp_time = 0.0
    processing_time_start = time.time()
    
    for iv, v in enumerate(vlist):
        
        load_detection_time_start = time.time()
    
        outfile = os.path.join(inference_dirname, v + "_tubes.pkl")
        
        
        if os.path.isfile(outfile) and not redo:
            continue

        RES = {}
        nframes = dataset._nframes[v]
        
        if opt.inference_mode == 'stream': # this condition may not be necessary?
            # record the latest K frame index (for the final frame allocate detection)
            last_k_ind = []
            last_k_ind_init = min(K  * opt.ninput, dataset._nframes[v]) - opt.ninput + 1
            last_k_ind.append(last_k_ind_init)
            for _ in range(opt.K - 1):
                last_k_ind_init = max(1, last_k_ind_init - opt.ninput)
                last_k_ind.append(last_k_ind_init)
            last_k_ind.reverse()
            
            k_ind_history = last_k_ind.copy()
    
        # load detected tubelets
        VDets = {}
       
        for startframe in range(min(K  * opt.ninput , dataset._nframes[v]) - opt.ninput + 1  , 1 + dataset._nframes[v]):
            
            if opt.inference_mode == 'stream': # otherwise ignore
                
                if startframe not in k_ind_history:
                    continue    
            
                # incrementally append more frame index processed by stream inference mode    
                if startframe + opt.ninput >= dataset._nframes[v]:
                    k_ind_history.append(dataset._nframes[v])
                else:
                    k_ind_history.append(startframe + opt.ninput)
                 
            if startframe != min(K  * opt.ninput, dataset._nframes[v]) - opt.ninput + 1: # not initial frame (ex: 16)
                
                if opt.inference_mode == 'stream': # otherwise ignore
                    last_k_ind.append(startframe)
    
                    if len(last_k_ind) > opt.K: # only keep the last K index
                        del last_k_ind[0]
                        
            resname = os.path.join(inference_dirname, v, "{:0>5}.pkl".format(startframe))
            if not os.path.isfile(resname):
                print("ERROR: Missing extracted tubelets " + resname)
                sys.exit()

            with open(resname, 'rb') as fid:
                VDets[startframe] = pickle.load(fid)
        
        
        # Start recording build tube time here; beforehand it was data loading
        load_detection_time_end = time.time()
        total_load_detection_time += load_detection_time_end - load_detection_time_start
        build_tube_time_start = time.time()
        
        k_ind_history = k_ind_history[:-1] 
        # added: may not be correct but proceed with learning tube building
        first_endframe = list(VDets.keys())[0]
               
        for ilabel in range(len(dataset.labels)):
             
            FINISHED_TUBES = []
            CURRENT_TUBES = []  # tubes is a list of tuple (end frame, lstubelets)
            # calculate average scores of tubelets in tubes

            def tubescore(tt):
                return np.mean(np.array([tt[i][1][-1] for i in range(len(tt))])) # a tube could contain multiple linked mini-tubes (linked over time); hence the for loop
  
            for frame in range(min(K  * opt.ninput , dataset._nframes[v]) - opt.ninput + 1  , 1 + dataset._nframes[v]):
                # load boxes of the new frame and do nms while keeping Nkeep highest scored
                
                if opt.inference_mode == 'stream':
                    if frame not in k_ind_history:
                        continue
                
                ltubelets = VDets[frame][ilabel + 1]  # [:,range(4*K) + [4*K + 1 + ilabel]]  Nx(4K+1) with (x1 y1 x2 y2)*K ilabel-score
                # orig
                ltubelets = nms_tubelets(ltubelets, 0.6, top_k=10)
                
                ###
                # debug: visualize image and detected boxes?
                ###
                '''
                for kk in range(opt.K):
                    if opt.inference_mode == 'stream':
                        if opt.dataset == 'hmdb':
                            v_file = '/home/alphadadajuju/projects/MMT/data/JHMDB/Frames/' + v + '/' + str(k_ind_history[k_ind_history.index(frame) - kk]).zfill(5) + '.png'  
                        else: 
                            v_file = '/home/alphadadajuju/projects/MMT/data/ucf24/rgb-images/' + v + '/' + str(k_ind_history[k_ind_history.index(frame) - kk]).zfill(5) + '.jpg'  
                    else:
                        if opt.dataset == 'hmdb':
                            v_file = '/home/alphadadajuju/projects/MMT/data/JHMDB/Frames/' + v + '/' + str(max(1, frame - kk*opt.ninput)).zfill(5) + '.png'  
                        else:
                            v_file = '/home/alphadadajuju/projects/MMT/data/ucf24/rgb-images/' + v + '/' + str(max(1, frame - kk*opt.ninput)).zfill(5) + '.jpg'  
                    print(v_file)
                    
                    imk = cv2.imread(v_file)
                    fig = plt.figure()
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.axis('off')
                    ax.imshow(cv2.cvtColor(imk, cv2.COLOR_BGR2RGB), interpolation='nearest')
                    #plt.imshow(cv2.cvtColor(imk, cv2.COLOR_BGR2RGB))
                    
                    tind = opt.K - kk - 1
                    x1y1x2y2 = ltubelets[:,tind*4:tind*4+4]
                    scores = ltubelets[:,-1]
                    
                    for n_t in range(len(x1y1x2y2)):
                        x1, y1, x2, y2 = x1y1x2y2[n_t, :]
                        
                        ax.add_patch(plt.Rectangle(
                            (x1, y1),
                            x2 - x1,
                            y2 - y1,
                            fill=False,
                            edgecolor='r',
                            linewidth=3))
                        
                        text = "%.2f" % scores[n_t]

                        ax.text(x1 - 2, y1 - 10,
                                text,
                                bbox=dict(facecolor='navy', alpha=0.7),
                                fontsize=12, color='yellow')
                    
                    fig.suptitle(v_file[-9:], fontsize=20, color='w')
                    plt.show()
                    '''
                
                # just start new tubes
                if frame == first_endframe: # orig: 1 
                    for i in range(ltubelets.shape[0]):
                        CURRENT_TUBES.append([(first_endframe, ltubelets[i, :])]) # orig: 1
                    continue

                # sort current tubes according to average score
                avgscore = [tubescore(t) for t in CURRENT_TUBES]
                argsort = np.argsort(-np.array(avgscore))
                CURRENT_TUBES = [CURRENT_TUBES[i] for i in argsort]
                # loop over tubes
                finished = []
                for it, t in enumerate(CURRENT_TUBES):
                    # compute ious between the last box of t and ltubelets # mine interpretation: for each tube in the memory, associates with any possible current tubelet 
                    last_endframe, last_tubelet = t[-1] # confusing -1? -> # a tube could contain multiple linked mini-tubes (linked over time)
                    ious = []
                    offset =  round((frame - last_endframe) / opt.ninput) # orig: frame - last_endframe
                    if offset < K:
                        nov = K - offset # number of overlaps
                        ious = sum([iou2d(ltubelets[:, 4 * iov:4 * iov + 4], last_tubelet[4 * (iov + offset):4 * (iov + offset + 1)]) for iov in range(nov)]) / float(nov)
                    else: # never once entered
                        ious = iou2d(ltubelets[:, :4], last_tubelet[4 * K - 4:4 * K])# head and tail matching

                    valid = np.where(ious >= 0.5)[0]

                    if valid.size > 0: # ONLY match the best QUERY tube to the database, then delete this query
                        # take the one with maximum score
                        idx = valid[np.argmax(ltubelets[valid, -1])]
                        CURRENT_TUBES[it].append((frame, ltubelets[idx, :]))
                        ltubelets = np.delete(ltubelets, idx, axis=0)
                    else:
                        if offset >= opt.K:
                            finished.append(it)

                # finished tubes that are done
                for it in finished[::-1]:  # process in reverse order to delete them with the right index why --++--
                    FINISHED_TUBES.append(CURRENT_TUBES[it][:])
                    del CURRENT_TUBES[it]

                # start new tubes
                for i in range(ltubelets.shape[0]):
                    CURRENT_TUBES.append([(frame, ltubelets[i, :])])

            # all tubes are not finished
            FINISHED_TUBES += CURRENT_TUBES

            # build real tubes
            output = []
            
            for t_i, t in enumerate(FINISHED_TUBES):
                # added: another filter step to make sure at least linked once?
                if len(t) < 2: 
                    continue
                score_tube = tubescore(t)

                # just start new tubes
                if score_tube < 0.005: 
                    continue

                beginframe = max(t[0][0] - opt.ninput*(K-1), 1)
                endframe = t[-1][0] 
                
                length = endframe + 1 - beginframe

                # delete tubes with short duraton (contibuting to many fp?)
                if length < min(dataset._nframes[v], opt.ninput*(opt.K)): # 15
                    continue
                
                # build final tubes by average the tubelets
                out = np.zeros((length, 6), dtype=np.float32)
                out[:, 0] = np.arange(beginframe, endframe + 1)
                n_per_frame = np.zeros((length, 1), dtype=np.int32) # orig: zeros
                
                for i in range(len(t)):
                    frame, box = t[i] # frame: end frame of a tube
                    n_mem = K - 1
                    
                    if opt.inference_mode == 'stream':
                        # for stream detection
                        if frame != nframes:
                            for k in range(K):
                                out[max(frame - k*opt.ninput, 1) - beginframe, 1:5] += box[4 * n_mem:4 * n_mem + 4]
                                out[max(frame - k*opt.ninput, 1) - beginframe, -1] += box[-1] 
                                n_per_frame[max(frame - k*opt.ninput, 1) - beginframe, 0] += 1
                                n_mem -= 1
                            
                        # for the last frame
                        else: 
                            for k in reversed(range(0, K)):
                                out[last_k_ind[k] - beginframe, 1:5] += box[4 * n_mem:4 * n_mem + 4]
                                out[last_k_ind[k] - beginframe, -1] += box[-1] 
                                n_per_frame[last_k_ind[k] - beginframe, 0] += 1
                                n_mem -= 1
                    
                    else:
                        for k in range(K):
                            out[max(frame - k*opt.ninput, 1) - beginframe, 1:5] += box[4 * n_mem:4 * n_mem + 4]
                            out[max(frame - k*opt.ninput, 1) - beginframe, -1] += box[-1] 
                            n_per_frame[max(frame - k*opt.ninput, 1) - beginframe, 0] += 1
                            n_mem -= 1
                        
                nonzero_ind =n_per_frame!=0 # sparse! would be dividing a lot of zeros
                out[nonzero_ind[:,0], 1:] /= n_per_frame[nonzero_ind[:,0]]
                # orig
                #out[:, 1:] /= n_per_frame
                
                
                interp_time_start = time.time()
                
                if 0 in out[:,-1]: # if any frame index contains 0 (meaning not filled! This line was creating issues!)
                    #print ('Frame detection interpolation takes place!')
                    
                    # Detection extrapolation for intra-frames
                    nonzero_ind = np.where(nonzero_ind==True)[0]
                    nz_v_prev = -5 # why -5? hardcoded?
                    nonzero_nonlink = []
                    for nz_i, nz_v in enumerate(nonzero_ind):
                        nz_offset = nz_v - nz_v_prev
                        
                        if nz_offset == 1: 
                            nz_v_prev = nz_v
                            continue
                        
                        if nz_i > 0:
                            nonzero_nonlink.append((nz_v_prev, nz_v))
                        nz_v_prev = nz_v
                    
                    for idx, lo_hi in enumerate(nonzero_nonlink):
                        lo_hi_dist = lo_hi[1] - lo_hi[0]
                        lo_box = out[lo_hi[0], 1:]
                        hi_box = out[lo_hi[1], 1:]
                        score = (out[lo_hi[0], -1] + out[lo_hi[1], -1]) / 2.
                        diff_box = (hi_box - lo_box) / lo_hi_dist
                        
                        for offset in range(1, lo_hi_dist):
                            if out[lo_hi[0] + offset, -1] == 0: # if the cell was not filled before
                                out[lo_hi[0] + offset, 1:] = lo_box + offset*diff_box
                                out[lo_hi[0] + offset, -1] = score
                            
                
                interp_time_end = time.time()
                total_interp_time += interp_time_end - interp_time_start
                
                output.append([out, score_tube])
                # out: [num_frames, (frame idx, x1, y1, x2, y2, score)]

            RES[ilabel] = output
            
            # debug: visualize linked results along with video frames
            #vis_linked_detection(v, output, opt)
        
        build_tube_time_end = time.time()
        total_build_tube_time += build_tube_time_end - build_tube_time_start
        
        # TODO: incomplete
        # debug: visualize linked results for all labels (with high confidence)
        #vis_linked_detection_alllabels(v, RES, opt, )
        
        save_detection_time_start = time.time()
        
        # RES{ilabel:[(out[length,6],score)]}ilabel[0,...]
        with open(outfile, 'wb') as fid:
            pickle.dump(RES, fid)
        Bar.suffix = '[{0}/{1}]:{2}|Tot: {total:} |ETA: {eta:} '.format(
            iv + 1, len(vlist), v, total=bar.elapsed_td, eta=bar.eta_td)
        bar.next()
        
        save_detection_time_end = time.time()
        total_save_detection_time += save_detection_time_end - save_detection_time_start
        
    bar.finish()
    
    print('Total time to process this script: ', time.time() - processing_time_start)
    print('Total time to load detection: ', total_load_detection_time)
    print('Total time to save tube: ', total_save_detection_time)
    print('Total time to interpolate: ', total_interp_time)
    print('Total time to build tube (excluding loadig detection): ', total_build_tube_time)
    
# Debug code; incomplete! 
def vis_linked_detection_alllabels(v, res, opt):
    
    # rgb image directory depends upon the dataset
    if opt.dataset == 'hmdb':
        v_file_root = '/home/alphadadajuju/projects/MMT/data/JHMDB/Frames/' + v + '/'
        v_file_tail = '.png'
    else:
        v_file_root = '/home/alphadadajuju/projects/MMT/data/ucf24/rgb-images/' + v + '/'
        v_file_tail = '.jpg'
    

    for ilabel in range(len(res)):
        for n_tube in res[ilabel]:
            
            # ignore super low-confidence tube
            if n_tube[1] < 0.1:
                continue
            else: 
                for n_f in n_tube[0]:
                    
                    v_file = v_file_root   + str(np.int(n_f[0])).zfill(5) + v_file_tail
                    
                    imk = cv2.imread(v_file)
                    fig = plt.figure()
                    ax = fig.add_axes([0, 0, 1, 1])
                    ax.axis('off')
                    ax.imshow(cv2.cvtColor(imk, cv2.COLOR_BGR2RGB), interpolation='nearest')
                        
                    #plt.imshow(cv2.cvtColor(imk, cv2.COLOR_BGR2RGB))
                    
                    x1y1x2y2 = n_f[1:]
                    x1, y1, x2, y2, score = x1y1x2y2[:]
                            
                    ax.add_patch(plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor='r',
                        linewidth=3))
                            
                    text = "%.2f" % score
    
                    ax.text(x1 - 2, y1 - 10,
                            text,
                            bbox=dict(facecolor='navy', alpha=0.7),
                            fontsize=12, color='yellow')
                            
                    
                    fig.suptitle(v_file[-9:], fontsize=20, color='b')
                
    plt.show()
                

# Debug code; incomplete! 
def vis_linked_detection(v, output, opt):
  
    # rgb image directory depends upon the dataset
    if opt.dataset == 'hmdb':
        v_file_root = '/home/alphadadajuju/projects/MMT/data/JHMDB/Frames/' + v + '/'
        v_file_tail = '.png'
    else:
        v_file_root = '/home/alphadadajuju/projects/MMT/data/ucf24/rgb-images/' + v + '/'
        v_file_tail = '.jpg'
    
    for n_tube in output:
        
        # ignore super low-confidence tube
        if n_tube[1] < 0.0:
            continue
        else: 
            for n_f in n_tube[0]:
                v_file = v_file_root   + str(np.int(n_f[0])).zfill(5) + v_file_tail
                
                imk = cv2.imread(v_file)
                fig = plt.figure()
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis('off')
                ax.imshow(cv2.cvtColor(imk, cv2.COLOR_BGR2RGB), interpolation='nearest')
                #plt.imshow(cv2.cvtColor(imk, cv2.COLOR_BGR2RGB))
                
                x1y1x2y2 = n_f[1:]
                x1, y1, x2, y2, score = x1y1x2y2[:]
                        
                ax.add_patch(plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    edgecolor='r',
                    linewidth=3))
                        
                text = "%.2f" % score

                ax.text(x1 - 2, y1 - 10,
                        text,
                        bbox=dict(facecolor='navy', alpha=0.7),
                        fontsize=12, color='yellow')
                        
                
                fig.suptitle(v_file[-9:], fontsize=20, color='b')
                plt.show()
                
   
