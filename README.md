# AMMA
Pytorch implementation of our "Accumulating Micro-Motion Representations for Lightweight Online Action Detection in Real-time" has been released.

## AMMA Overview
AMMA is a real-time tubelet detector operating on lightweight 2D CNN backbones and raw video clips. It adopts a coarse-level tubelet detection scheme, acquiring actions' spatiotemporal context by combining sparsely sampled visual cues and complementary dynamic cues across a wider temporal window. Due to the smoothness nature of continuous actions, our detector can efficiently infer temporally coarse action tubelets across a sparse set of frames while interpolating intra-frame detection. To explicitly encode short-term action dynamics in an efficient manner, we devise a simple yet effective motion representation by accumulating learnable motion boundaries over each video clip (referred to as "micro-motion"). In AMMA, micro-motion is computed on-the-fly from RGB frames, whose abstract features can then be adaptively fused with the appearance ones at multiple scales to produce temporal-aware features via 2D CNN. On top of its spatiotemporal backbone, AMMA aggregates multiple temporal-aware features from successive clips at its detector head, permitting long-range action modeling. AMMA is one of the few works primarily focusing on highly efficient action detection solutions for realistic deployment on low-end devices.

![alt text](https://github.com/alphadadajuju/TEDdet/blob/master/images/pipeline.jpg)

* We propose a compact micro-motion representation to encode short-term action dynamics. Compensating for the low efficiency of traditional optical flow methods, our motion representation can be generated on-the-fly from video streams in real-time.

* We devise a lightweight action tubelet detector integrating 2D CNN backbones, micro-motion generation \& fusion, and cooperative detection branches. It adopts a coarse-to-fine detection paradigm to efficiently infer actions in online settings.

* We tailor the proposed detection pipeline with three ultra-lightweight CNN backbones and validate their overall-superior performances in high precision, high speed, and low complexity on JHMDB-21 and UCF-24. 

## AMMA Usage
### 1. Installation and Dataset
Please refer to https://github.com/MCG-NJU/MOC-Detector for detailed instructions.

### 2. Train
The current version of AMMA supports ResNet18, MobileNetV2, and ShuffleNetV2 as the feature extraction backbones. To proceed with training, first download the pretrained weights in [Google Drive](https://drive.google.com/drive/folders/1r2uYo-4hL6oOzRARFsYIn5Pu2Lv7VS6m) for ResNet18. COCO pretrained models come from [CenterNet](https://github.com/xingyizhou/CenterNet). Move pretrained models to ```${AMMA_ROOT}/experiment/modelzoo</mark>.```

To train your own model, run the ```train.py``` script along with relevant input arguments. For instance:

```
$python train.py --K 5 --exp_id K5_model --rgb_model TED_K5 --batch_size 16 --master_batch 16 --lr 2.5e-4 --gpus 0 --num_worker 16 --num_epochs 10 --lr_step 5 --dataset hmdb --split 1 --down_ratio 8 --lr_drop 0.1 --ninput 1 --ninputrgb 5 --auto_stop --pretrain coco 
```

Note that by setting ```--auto_save```, a validation step will be carried out at the end of each training epoch in order to save the best-performing model (i.e., model_best.pth). Instead, using ```--save_all``` will save every epoch's training model. For more details on possible input arguments, refer to ```${TEDdet_ROOT}/src/opts.py.```

### 3. Evaluation
To evaluate a trained model, one performs two sequential steps from the ```det.py``` and ```ACT.py``` scripts. The former performs model inference and saves detection results (i.e., .pkl) accordingly. The latter loads the detection results and evaluates them based on designated metrics. For instance:

#### Model inference 
```
$python det.py --task stream --K 5 --gpus 0 --batch_size 1 --master_batch 1 --num_workers 1 --rgb_model TED_K5/model_best.pth --inference_dir ../data0/TED_K5_stream --ninput 1 --dataset hmdb --split 1 --down_ratio 8 --ninputrgb 5 
```

Note that TEDdet supports two inference modes: normal and stream mode. Stream mode takes into account a FIFO-based feature-caching mechanism to more efficiently process incoming video frames (the reported speed in the article corresponds to this mode). The mode needs to be specified explicitly during both inference and evaluation.

#### Evaluating mAP

```
$python ACT.py --task frameAP --K 5 --th 0.5 --inference_dir ../data0/TED_K5_stream --dataset hmdb --split 1 --evaluation_model trimmed --inference_mode stream --ninputrgb 5
```
To compute videoAP, one needs to additionally generate action tubes using ACT.py by first configuring ```--task``` to ```BuildTubes``` and then evaluate ```videoAP```.

## References
Our codes refer to the work of [CenterNet](https://github.com/xingyizhou/CenterNet), [MOC](https://github.com/MCG-NJU/MOC-Detector) and [TEA](https://github.com/Phoenix1327/tea-action-recognition). We thank the authors for their structured implementations and comments!
