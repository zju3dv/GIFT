import sys
import time
import torch
from torch.nn import DataParallel


sys.path.append('.')

from torch.utils.data import DataLoader
from network.wrapper import TrainWrapper
from dataset.correspondence_dataset import  CorrespondenceDataset
from dataset.correspondence_database import CorrespondenceDatabase, worker_init_fn
from train.train_tools import to_cuda, overwrite_configs
import yaml

import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='time')
parser.add_argument('--cfg', type=str, default='configs/default.yaml')
parser.add_argument('--max_num', type=int, default=5)
parser.add_argument('--single', dest='single', action='store_true')
parser.add_argument('--multiple', dest='single', action='store_false')
parser.set_defaults(single=True)
flags=parser.parse_args()

with open('configs/default.yaml','r') as f:
    default_cfg=yaml.load(f, Loader=yaml.FullLoader)
with open(flags.cfg,'r') as f:
    cfg=yaml.load(f, Loader=yaml.FullLoader)
cfg=overwrite_configs(default_cfg,cfg)

batch_size=cfg['batch_size']
worker_num=cfg['worker_num']

if flags.single:
    batch_size=2
    wrapper=TrainWrapper(cfg).cuda()
else:
    wrapper=DataParallel(TrainWrapper(cfg)).cuda()
database=CorrespondenceDatabase()
dataset=CorrespondenceDataset(cfg,database.coco_set)
loader=DataLoader(dataset,batch_size,shuffle=True,num_workers=worker_num,worker_init_fn=worker_init_fn)

def test_time():
    begin=time.time()
    for data_i, data in enumerate(loader):
        if data_i==1: begin=time.time()
        if flags.single:
            img_list0, pts_list0, pts0, grid_list0, img_list1, pts_list1, pts1, grid_list1, scale_offset, rotate_offset, H = to_cuda(data)
        else:
            img_list0,pts_list0,pts0,grid_list0,img_list1,pts_list1,pts1,grid_list1,scale_offset,rotate_offset,H = data
        results=wrapper(img_list0, pts_list0, pts0, grid_list0, img_list1, pts_list1, pts1, grid_list1, scale_offset, rotate_offset, 5, cfg['loss_type'])
        torch.mean(results['triplet_loss']).backward()
        if data_i>flags.max_num: break
    print('batch size {} cost {} s/b'.format(batch_size,(time.time()-begin)/(data_i)))

if __name__=="__main__":
    name2func={
        'time':test_time
    }
    name2func[flags.task]()
