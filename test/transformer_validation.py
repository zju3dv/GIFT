import sys
sys.path.append('.')

import yaml
from dataset.transformer import TransformerCV
from network.extractor import NetworkShallow
from skimage.io import imread, imsave
import torch
import numpy as np
import time
import cv2

img=imread('test/data/DJI_0246.JPG')
# img=cv2.resize(img,(512,512),interpolation=cv2.INTER_LINEAR)
img=img[:480,:640,:]
test_num=20
pts_num=108
pts=np.random.uniform(0,1,[pts_num,2])
pts[:,0]*=img.shape[1]
pts[:,1]*=img.shape[0]

def test_warp(img):
    with open('configs/default.yaml','r') as f:
        config=yaml.load(f)
    transformer=TransformerCV(config)

    begin=time.time()
    for k in range(test_num):
        results=transformer.transform(img,pts)
        for warp_id,img_warp in enumerate(results['img']):
            for pt_warp in results['pts'][warp_id]:
                img_warp=cv2.circle(np.ascontiguousarray(img_warp),tuple(pt_warp.astype(np.int32)),5,(255,0,0))
            imsave('test/results/{}.jpg'.format(warp_id),img_warp)
        break

    print('warp average cost {} s'.format((time.time()-begin)/test_num))

test_warp(img)



