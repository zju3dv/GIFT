import time

import os
from struct import unpack
import numpy as np

from utils.base_utils import read_pickle, save_pickle

def read_corr(file_path):
    """Read the match correspondence file.
    Args:
        file_path: file path.
    Returns:
        matches: list of match data, each consists of two image indices and Nx15 match matrix, of
        which each line consists of two 2x3 transformations, geometric distance and two feature
        indices.
    """
    matches = []
    with open(file_path, 'rb') as fin:
        while True:
            rin = fin.read(24)
            if rin == '' or len(rin)!=24:
                # EOF
                break
            idx0, idx1, num = unpack('Q' * 3, rin)
            bytes_theta = num * 60
            corr = np.fromstring(fin.read(bytes_theta), dtype=np.float32).reshape(-1, 15)
            matches.append([idx0, idx1, corr])
    return matches

def read_image_list(file_path):
    with open(file_path,'r') as f:
        return [line.strip() for line in f.readlines() if len(line.strip())>0]

def get_gl3d_dataset(root='data/GL3D'):
    pkl_path='data/GL3D_dataset.pkl'
    if os.path.exists(pkl_path):
        return read_pickle(pkl_path)

    print('begin preparing gl3d dataset...')
    dataset=[]
    print('total pid num {}'.format(len(os.listdir(root))))
    begin=time.time()
    for k,pid in enumerate(os.listdir(root)):
        if k%20==0:
            print('{} processed cost {} s!'.format(k,time.time()-begin))
        list_fn=os.path.join(root, pid, 'image_list.txt')
        corr_fn=os.path.join(root, pid, 'geolabel/corr.bin')
        if os.path.exists(list_fn):
            image_list=read_image_list(list_fn)
        else:
            print('{} missing'.format(list_fn))
            continue

        if os.path.exists(corr_fn):
            matches=read_corr(corr_fn)
        else:
            print('{} missing'.format(corr_fn))
            continue

        for match_index, match in enumerate(matches):
            img_id0=match[0]
            img_id1=match[1]
            img_pth0=image_list[img_id0].split('/')[-1]
            img_pth1=image_list[img_id1].split('/')[-1]
            img_pth0='{}/{}/images_desc_train/{}'.format(root,pid,img_pth0)
            img_pth1='{}/{}/images_desc_train/{}'.format(root,pid,img_pth1)

            if not os.path.exists(img_pth0) or not os.path.exists(img_pth1):
                print(img_pth0,img_pth1)
                continue

            if match[2].shape[0]==0:
                print('{} {} no match points'.format(img_pth0,img_pth1))
                continue

            data={}
            data['img0_pth']=img_pth0
            data['img1_pth']=img_pth1
            pts0,pts1=[],[]
            for corr_index, corr in enumerate(match[2]):
                pt0=(corr[:6].reshape([2,3])[:,2]+1.0)/2.0
                pt1=(corr[6:12].reshape([2,3])[:,2]+1.0)/2.0
                pts0.append(pt0)
                pts1.append(pt1)

            data['pix_pos0']=np.asarray(pts0)
            data['pix_pos1']=np.asarray(pts1)
            data['type']='gl3d'
            dataset.append(data)

    print('end preparing gl3d dataset...')
    save_pickle(dataset,pkl_path)
    return dataset


if __name__=="__main__":
    pass