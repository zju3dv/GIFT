import sys
sys.path.append('.')

import cv2
import numpy as np
from utils.base_utils import tensor_to_image, get_img_patch
from torch.utils.data import DataLoader
from skimage.io import imsave
from dataset.correspondence_dataset import  CorrespondenceDataset
from dataset.correspondence_database import CorrespondenceDatabase, worker_init_fn
import time
import yaml

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='time')
parser.add_argument('--cfg', type=str, default='configs/default.yaml')
parser.add_argument('--max_num', type=int, default=5)
flags=parser.parse_args()

with open(flags.cfg,'r') as f:
    cfg=yaml.load(f, Loader=yaml.FullLoader)

batch_size=cfg['batch_size']
worker_num=cfg['worker_num']

database=CorrespondenceDatabase()
dataset=CorrespondenceDataset(cfg,database.coco_set*20)
loader=DataLoader(dataset,batch_size,shuffle=True,num_workers=worker_num,worker_init_fn=worker_init_fn)

def test_time():
    begin=time.time()
    for data_i,data in enumerate(loader):
        if data_i>flags.max_num: break
    print('batch size {} worker num {} cost {} s'.format(batch_size,worker_num,(time.time()-begin)/(data_i+1)))

def test_outputs():
    batch_index=0
    for data_i,data in enumerate(loader):
        img_list0,pts_list0,pix_pos0,grid_list0,img_list1,pts_list1,pix_pos1,grid_list1,scale_offset,rotate_offset,H = data
        for k in range(cfg['sample_scale_num']*cfg['sample_rotate_num']):
            if data_i<2:
                if k==0: print('scale,rotate,H', scale_offset.shape, rotate_offset.shape, H.shape)
                print('left',k,img_list0[k][0].shape,pts_list0[k][batch_index].shape,grid_list0[k][batch_index].shape)
                print('right',k,img_list1[k][0].shape,pts_list0[k][batch_index].shape,grid_list1[k][batch_index].shape)

            # img_warp0=draw_pts(tensor_to_image(img_list0[k].numpy()[batch_index]), pts_list0[k].numpy()[batch_index].reshape([-1, 2]))
            # imsave('test/results/{}_{}_left.jpg'.format(data_i, k), img_warp0)
            #
            # img_warp1=draw_pts(tensor_to_image(img_list1[k].numpy()[batch_index]), pts_list1[k].numpy()[batch_index].reshape([-1, 2]))
            # imsave('test/results/{}_{}_right.jpg'.format(data_i, k), img_warp1)
            #
            # img_grid0=draw_pts(tensor_to_image(img_list0[k].numpy()[batch_index]), grid_list0[k].numpy()[batch_index,5].reshape([-1, 2]))
            # imsave('test/results/{}_{}_grid_left.jpg'.format(data_i, k), img_grid0)
            #
            # img_grid1=draw_pts(tensor_to_image(img_list1[k].numpy()[batch_index]), grid_list1[k].numpy()[batch_index,5].reshape([-1, 2]))
            # imsave('test/results/{}_{}_grid_right.jpg'.format(data_i, k), img_grid1)

        # test scale and rotate offset
        pn=pts_list0[0].shape[0]
        pi=np.random.randint(0,pn)
        so=scale_offset[batch_index,pi].numpy()
        ro=rotate_offset[batch_index,pi].numpy()
        patch0=get_regions(img_list0,pts_list0,batch_index,pi)
        patch1=get_regions(img_list1,pts_list1,batch_index,pi)
        imsave('test/results/patch_{}_left_{}_{}.jpg'.format(data_i,so,ro),patch0)
        imsave('test/results/patch_{}_right_{}_{}.jpg'.format(data_i,so,ro),patch1)
        if data_i>flags.max_num: break

def get_regions(img_list0,pts_list0,batch_index,pi):
    imgs = []
    for si in range(cfg['sample_scale_num']):
        scale_imgs=[]
        for ri in range(cfg['sample_rotate_num']):
            k=si*cfg['sample_rotate_num']+ri
            img0 = tensor_to_image(img_list0[k][batch_index].numpy())
            pt0 = pts_list0[k][batch_index,pi].numpy()
            pat0 = get_img_patch(img0, pt0, 20)
            scale_imgs.append(pat0)
        imgs.append(np.concatenate(scale_imgs,1))
    return np.concatenate(imgs,0)

def draw_pts(img, pts):
    img = np.ascontiguousarray(img.copy())
    for pt in pts: img = cv2.circle(img, tuple(pt.astype(np.int32)), 3, (255, 0, 0), 2)
    return img

if __name__=="__main__":
    name2func={
        'outputs':test_outputs,
        'time':test_time
    }
    name2func[flags.task]()