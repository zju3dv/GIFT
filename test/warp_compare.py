import sys
sys.path.append('.')

import yaml
from dataset.transformer import TransformerCV
from network.wrapper import GroupTransformer
from network.extractor import NetworkShallow
from skimage.io import imread, imsave
import torch
import numpy as np
import time
import cv2

config={}
config['sample_rotate_num']=5
config['sample_scale_num']=5
config['sample_scale_begin']=0
config['sample_rotate_begin']=-90
config['sample_rotate_end']=90
config['sample_type']= 'rotate_scale'
config['sample_gconv_ksize']=5
config['sample_base_factor']=0.5 # 1.4142135623730951

img=imread('test/data/DJI_0246.JPG')
# img=cv2.resize(img,(512,512),interpolation=cv2.INTER_LINEAR)
img=img[:480,:640,:]
test_num=20
pts_num=108
pts=np.random.uniform(0,1,[pts_num,2])
pts[:,0]*=img.shape[1]
pts[:,1]*=img.shape[0]

def test_transformer(img):
    img=torch.tensor(img.astype(np.float32),dtype=torch.float32).cuda()
    img=img.permute(2,0,1)[None,:,:,:]

    transformer=GroupTransformer(config).cuda()
    imgs,As=transformer(img)
    begin=time.time()
    for k in range(test_num):
        with torch.no_grad():
            imgs,As=transformer(img)
        torch.cuda.synchronize()
    print('transformer average cost {} s'.format((time.time()-begin)/test_num))

    net=NetworkShallow(config).cuda()
    begin=time.time()
    for k in range(test_num):
        with torch.no_grad():
            imgs,As=transformer(img)
            for img in imgs:
                feats=net(img)
        torch.cuda.synchronize()
    print('transformer+forward average cost {} s'.format((time.time()-begin)/test_num))

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
            imsave('{}.jpg'.format(warp_id),img_warp)
        break

    print('warp average cost {} s'.format((time.time()-begin)/test_num))

test_warp(img)



