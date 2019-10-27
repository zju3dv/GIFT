import sys
import time

import torch

sys.path.append('.')

from torch.utils.data import DataLoader
from network.wrapper import TrainWrapper
from dataset.correspondence_dataset import  CorrespondenceDataset
from dataset.correspondence_database import CorrespondenceDatabase, worker_init_fn
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
wrapper=TrainWrapper(cfg).cuda()

def to_cuda(data):
    results=[]
    for i,item in enumerate(data):
        if type(item).__name__=="Tensor":
            results.append(item.cuda())
        elif type(item).__name__=='list':
            tensor_list=[]
            for tensor in item:
                tensor_list.append(tensor.cuda())
            results.append(tensor_list)
        else:
            raise NotImplementedError
    return results

def test_time():
    begin=time.time()
    for data_i, data in enumerate(loader):
        img_list0,pts_list0,pts0,grid_list0,img_list1,pts_list1,pts1,grid_list1,scale_offset,rotate_offset,H = to_cuda(data)
        results=wrapper(img_list0, pts_list0, pts0, grid_list0, img_list1, pts_list1, pts1, grid_list1, scale_offset, rotate_offset, 5, cfg['loss_type'])
        results['triplet_loss'].backward()
        if data_i>flags.max_num: break
    print('batch size {} cost {} s/b'.format(batch_size,(time.time()-begin)/(data_i+1)))

if __name__=="__main__":
    name2func={
        'time':test_time
    }
    name2func[flags.task]()
