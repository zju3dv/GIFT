import easydict
import os
import numpy as np
import h5py
from utils.base_utils import read_pickle, save_pickle

def read_calibration_file(fn):
    with h5py.File(fn, 'r') as f:
        return f['R'].value, f['T'].value, f['K'].value, f['dc'].value, f['imsize'].value

def get_rel_img_dataset(name, set_name='test'):
    pickle_name='img_info.pkl'
    if os.path.exists(f'data/{name}/{set_name}/{pickle_name}'):
        return read_pickle(f'data/{name}/{set_name}/{pickle_name}')
    img_pths=np.loadtxt(f'data/{name}/{set_name}/images.txt',dtype=str)
    cal_pths=np.loadtxt(f'data/{name}/{set_name}/calibration.txt',dtype=str)

    img_info_list=[]
    for id, (img_pth, cal_pth) in enumerate(zip(img_pths, cal_pths)):
        R,T,K,dc,imsize=read_calibration_file(os.path.join('data',name,set_name,cal_pth))

        fn_id=os.path.basename(cal_pth).split('.')[0].split('_')[-1]
        visibility=np.loadtxt(os.path.join('data',name,set_name,'visibility','vis_'+fn_id+'.txt'))

        cx = (imsize[0,0] - 1.0) * 0.5
        cy = (imsize[0,1] - 1.0) * 0.5
        K[0,2]+=cx
        K[1,2]+=cy

        img_info=easydict.EasyDict(
            path=os.path.join('data',name,set_name,img_pth),
            visibility=visibility,R=R,T=T,K=K,dc=dc,imgsize=imsize
        )
        img_info_list.append(img_info)

    save_pickle(img_info_list,f'data/{name}/{set_name}/{pickle_name}')
    return img_info_list

def get_rel_pair_dataset(vis_min: float, vis_max: float, name, set_name='test'):
    pickle_name=f'{vis_min:.3f}_{vis_max:.3f}_pair_info.pkl'
    if os.path.exists(f'data/{name}/{set_name}/{pickle_name}'):
        return read_pickle(f'data/{name}/{set_name}/{pickle_name}')

    img_info_list=get_rel_img_dataset(name, set_name)

    visibilities=[info.visibility for info in img_info_list]
    visibilities=np.asarray(visibilities)

    mask=np.logical_and(visibilities>=vis_min,visibilities<=vis_max)
    print('pair num {}'.format(np.sum(mask)/2))

    pair_info_list=[]
    img_num=len(img_info_list)
    for id0 in range(img_num-1):
        for id1 in range(id0+1,img_num):
            if not mask[id0,id1]: continue
            img_info0=img_info_list[id0]
            img_info1=img_info_list[id1]

            R0,T0,K0=img_info0.R,img_info0.T,img_info0.K
            R1,T1,K1=img_info1.R,img_info1.T,img_info1.K
            R = R1 @ R0.T
            T = T1.reshape([3,1]) - R1 @ R0.T @ T0.reshape([3,1])

            pair_info=easydict.EasyDict(path0=img_info0.path, path1=img_info1.path,
                                        R01=R, T01=T, K0=K0, K1=K1, vis=visibilities[id0,id1])
            pair_info_list.append(pair_info)

    save_pickle(pair_info_list,f'data/{name}/{set_name}/{pickle_name}')
    return pair_info_list

