> [GIFT: Learning Transformation-Invariant Dense Visual Descriptors via Group CNNs](https://arxiv.org/abs/1911.05932)  
> Yuan Liu, Zehong Shen, Zhixuan Lin, Sida Peng, Hujun Bao, Xiaowei Zhou   
> NeurIPS 2019
> [Project Page](https://zju3dv.github.io/GIFT/)

Any questions or discussions are welcomed!

## Requirements & Compilation
1. Requirements

Required packages are listed in [requirements.txt](requirements.txt). 

Note that an old version of OpenCV (3.4.2) is needed since the code uses SIFT module of OpenCV.

The code is tested using Python-3.7.3 with pytorch 1.3.0.

2. Compile hard example mining functions
```shell script
cd hard_mining
python setup.py build_ext --inplace
```

3. Compile extend utilities
```shell script
cd utils/extend_utils
python build_extend_utils_cffi.py
```
According to your installation path of CUDA, 
you may need to revise the variables cuda_version, cuda_include and cuda_library in 
[build_extend_utils_cffi.py](utils/extend_utils/build_extend_utils_cffi.py).

## Testing

### Download pretrained models

1. Pretrained GIFT model can be found at [here](https://1drv.ms/u/s!AoUMOA44sUHpgQNh3RCTOEvq3XSn?e=cLXHXU). 

2. Pretrained SuperPoint model can be found at [here](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork/blob/master/superpoint_v1.pth). 

3. Make a directory called data and arrange these files like the following. 
```
data/
├── superpoint/
|   └── superpoint_v1.pth
└── model/
    └── GIFT-stage2-pretrain/
        └── 20001.pth
```

### Demo

We provide some examples of relative pose estimation in [demo.ipynb](demo.ipynb).

### Test on HPatches dataset

1. Download Resized HPatches, ER-HPatches and ES-HPatches datasets at [here](https://1drv.ms/u/s!AoUMOA44sUHpgQUCbAlJK8RviqgJ?e=NSZWea). 
Optionally, you can also generate these datasets from 
[original Hpatches sequences](http://icvl.ee.ic.ac.uk/vbalnt/hpatches/hpatches-sequences-release.tar.gz)
using [correspondence_database.py](dataset/correspondence_database.py). 

2. Extract these datasets like the following.
```
data/
├── hpatches_resize/
├── hpatches_erotate/
├── hpatches_erotate_illm/
├── hpatches_escale/
└── hpatches_escale_illm/
```

3. Evaluation
```shell script
# use keypoints detected by superpoint and descriptors computed by GIFT
python run.py --task=eval_original \
              --det_cfg=configs/eval/superpoint_det.yaml \
              --desc_cfg=configs/eval/gift_pretrain_desc.yaml \
              --match_cfg=configs/eval/match_v2.yaml

# use keypoints detected by superpoint and descriptors computed by superpoint
python run.py --task=eval_original \
              --det_cfg=configs/eval/superpoint_det.yaml \
              --desc_cfg=configs/eval/superpoint_desc.yaml \
              --match_cfg=configs/eval/match_v2.yaml
```

The output is ``es superpoint_det gift_pretrain_desc match_v2 pck-5 0.290 -2 0.132 -1 0.057 cost 267.826 s``,
which are ``datasetname detector_name descriptor_name matching_strategy PCK-5 PCK-2 PCK-1``. 
``PCK-5`` means that a correspondence is correct if the distance between the matched keypoint and its ground truth location is less than 5 pixels.  

### Test on relative pose estimation dataset

1. Download the st_peters_squares dataset from [here](https://1drv.ms/u/s!AoUMOA44sUHpgQRQ0Ok4a6wppNDJ?e=qJlRzg).
(This dataset is a part of [st_peters](http://webhome.cs.uvic.ca/~kyi/files/2018/learned-correspondence/st_peters_square.tar.gz).) 

2. Extract the dataset and arrange directories like the following.
```
data
└── st_peters_square_dataset/
    └── test/
```

3. Evaluation
```shell script
# use keypoints detected by superpoint and descriptors computed by GIFT
python run.py --task=rel_pose \
              --det_cfg=configs/eval/superpoint_det.yaml \
              --desc_cfg=configs/eval/gift_pretrain_desc.yaml \
              --match_cfg=configs/eval/match_v0.yaml

# use keypoints detected by superpoint and descriptors computed by superpoint
python run.py --task=rel_pose \
              --det_cfg=configs/eval/superpoint_det.yaml \
              --desc_cfg=configs/eval/superpoint_desc.yaml \
              --match_cfg=configs/eval/match_v0.yaml
```

The output is ``sps_100_200_first_100 superpoint_det gift_pretrain_desc match_v0 ang diff 24.100 inlier 62.240 correct-5 0.170 -10 0.350 -20 0.650``
which is ``datasetname detector_name descriptor_name matching_strategy average_angle_difference average_inlier_number correct_rate_5_degree  correct_rate_10_degree correct_rate_20_degree``.
In relative pose estimation, we can compute the angle difference between the estimated rotation and the ground truth rotation.
``average_angle_difference`` is the average angle difference among all image pairs.
``average_inlier_number`` is the number of inlier keypoints after RANSAC.
``correct_rate_5_degree`` indicate the percentage of image pairs whose angle difference is less than 5 degree.

## Training
1. Download the [train-2014](http://images.cocodataset.org/zips/train2014.zip) and [val-2014](http://images.cocodataset.org/zips/val2014.zip) set of COCO dataset and the [SUN397 dataset](http://groups.csail.mit.edu/vision/SUN/releases/SUN2012pascalformat.tar.gz).

2. Organize files like the following
```
data
├── SUN2012Images/
|   └── JPEGImages/
└── coco/
    ├── train2014/ 
    └── val2014/
```

3. Training
```shell script
mkdir data/record
python run.py --task=train --cfg=configs/GIFT-stage1.yaml # train group extractor (Vanilla CNN)
python run.py --task=train --cfg=configs/GIFT-stage2.yaml # train group embedder (Group CNNs)
```

## Acknowledgements

We have used codes or datasets from following projects:
- [HPatches](https://github.com/hpatches/hpatches-dataset): Homography-patches dataset
- [GeoDesc](https://github.com/lzx551402/geodesc): Learning Local Descriptors by Integrating Geometry Constraints (code in [opencvhelper.py](utils/opencvhelper.py))
- [SuperPoint](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork): Self-Supervised Interest Point Detection (code in [superpoint_utils.py](utils/superpoint_utils.py)) and Description and [its tensorflow version](https://github.com/rpautrat/SuperPoint)
- [GL3D](https://github.com/lzx551402/GL3D): a large-scale database created for 3D reconstruction and geometry-related learning problems
- [Learning to Find Good Correspondences](https://github.com/vcg-uvic/learned-correspondence-release)

## Copyright

This work is affliated with ZJU-SenseTime Joint Lab of 3D Vision, and its intellectual property belongs to SenseTime Group Ltd.

```
Copyright (c) ZJU-SenseTime Joint Lab of 3D Vision. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
