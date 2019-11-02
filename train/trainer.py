import time
from collections import OrderedDict

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import os
import yaml
import numpy as np

from dataset.correspondence_dataset import CorrespondenceDataset
from dataset.correspondence_database import CorrespondenceDatabase, worker_init_fn
from dataset.transformer import TransformerCV
from network.wrapper import TrainWrapper
from train.train_tools import overwrite_configs, Recorder, to_cuda, adjust_learning_rate, reset_learning_rate, \
    dim_extend
from torch.optim import Adam
from skimage.io import imread

from utils.base_utils import perspective_transform
from utils.detector import detect_and_compute_sift_np, detect_and_compute_harris_np
from utils.match_utils import compute_match_pairs, keep_valid_kps_feats


class Trainer:
    def __init__(self,cfg_file):
        self._train_set_ready=False
        self._network_ready=False
        self._init_config(cfg_file)

    def _init_config(self, cfg_file):
        with open(os.path.join('configs/default.yaml'), 'r') as f:
            default_train_config = yaml.load(f)

        with open(cfg_file, 'r') as f:
            overwrite_cfg = yaml.load(f)
            self.config = overwrite_configs(default_train_config, overwrite_cfg)

        self.name = self.config['name']
        self.recorder = Recorder(os.path.join('data', 'record', self.name), os.path.join('data', 'record', self.name + '.log'))
        self.model_dir = os.path.join('data', 'model', self.name)

    def _init_train_set(self):
        if self._train_set_ready:
            print('train set is ready')
            return

        print('begin preparing train set...')
        database = CorrespondenceDatabase()
        self.database = database

        train_set = []
        for name in self.config['train_set']: train_set += database.__getattr__(name + "_set")
        self.train_set = CorrespondenceDataset(self.config, train_set)
        self.train_loader = DataLoader(self.train_set,self.config['batch_size'],shuffle=True,
                                       num_workers=self.config['worker_num'],worker_init_fn=worker_init_fn)
        self._train_set_ready=True
        print('train set is ready')

    def _init_network(self):
        if self._network_ready:
            print('network is ready')
            return

        print('begin preparing network...')
        self.network=TrainWrapper(self.config)
        self.extractor=self.network.extractor_wrapper
        self.embedder=self.network.embedder_wrapper
        self.network=DataParallel(self.network).cuda()

        paras=[]
        if self.config['train_extractor']: paras+=self.extractor.parameters()
        if self.config['train_embedder']: paras+=self.embedder.parameters()
        self.optim=Adam(paras,lr=1e-3)

        if self.config['pretrain']:
            self._load_model(self.config['pretrain_model_path'],self.config['pretrain_step'],
                             self.config['pretrain_extractor'],self.config['pretrain_embedder'],False)

        self.step = 0
        self._load_model(self.model_dir, -1, True, True, True)

        self._network_ready=True
        self.transformer = TransformerCV(self.config)
        print('network is ready')

    def _get_hem_thresh(self,):
        hem_thresh_begin = self.config['hem_thresh_begin']
        hem_thresh_end = self.config['hem_thresh_end']
        warm_up_step = self.config['warm_up_step']
        median_step = self.config['median_step']
        if self.step<warm_up_step:
            return hem_thresh_begin
        if warm_up_step<=self.step<=median_step:
            return hem_thresh_begin-(hem_thresh_begin-hem_thresh_end)/(median_step-warm_up_step)*(self.step - warm_up_step)
        if self.step>median_step:
            return hem_thresh_end

    def _get_lr(self):
        lr_warm_up=self.config['lr_warm_up']
        lr_begin=self.config['lr_begin']
        lr_median=self.config['lr_median']
        lr_end=self.config['lr_end']
        warm_up_step = self.config['warm_up_step']
        median_step = self.config['median_step']
        if self.step<warm_up_step:
            return lr_warm_up
        if warm_up_step<=self.step<=median_step:
            return lr_begin-(lr_begin-lr_median)/(median_step-warm_up_step)*(self.step-warm_up_step)
        if self.step>median_step:
            return lr_median-(lr_median-lr_end)/(self.config['train_step']-median_step)*(self.step-median_step)

    def train(self):
        self._init_network()
        self._init_train_set()

        batch_begin=time.time()
        for data in self.train_loader:
            lr=reset_learning_rate(self.optim,self._get_lr())
            hem_thresh=self._get_hem_thresh()

            loss_info = OrderedDict()
            # img_list0,pts_list0,pts0,grid_list0,img_list1,pts_list1,pts1,grid_list1,scale_offset,rotate_offset,H=to_cuda(data)
            img_list0,pts_list0,pts0,grid_list0,img_list1,pts_list1,pts1,grid_list1,scale_offset,rotate_offset,H=data
            data_time = time.time() - batch_begin

            results = self.network(img_list0, pts_list0, pts0, grid_list0, img_list1, pts_list1, pts1, grid_list1,
                                   scale_offset, rotate_offset, hem_thresh, self.config['loss_type'])

            loss = 0.0
            for k, v in results.items():
                v = torch.mean(v)
                if k.endswith('loss'): loss = loss + v
                loss_info[k] = v.cpu().detach().numpy()

            self.network.zero_grad()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            batch_time = time.time() - batch_begin
            # record
            loss_info['data_time'] = data_time
            loss_info['batch_time'] = batch_time
            loss_info['lr'] = lr
            loss_info['hem_thresh'] = hem_thresh
            total_step=self.step
            self.recorder.rec_loss(loss_info, total_step, total_step, 'train', ((total_step + 1) % self.config['info_step']) == 0)

            # save model
            if (total_step+1)%self.config['save_step']==0:
                self._save_model()
                print('model saved!')

            batch_begin = time.time()
            self.step+=1
            if self.step>self.config['train_step']: break

        self._save_model()
        print('model saved!')

    def evaluate(self,dataset, dataset_name, use_raw_feats=False, kpts_type='superpoint',thresh=5):
        evaluate_name=self.name if not use_raw_feats else kpts_type

        corrects5, corrects2, corrects1 = [], [], []
        begin=time.time()
        for data in dataset:
            pth0, pth1, H = data['img0_pth'], data['img1_pth'], data['H'].copy()
            img0, img1 = imread(pth0), imread(pth1)
            if kpts_type!='sift':
                kps0, feats0 = self._get_feats_kps(pth0, self.model_name_to_dir_name[kpts_type])
                kps1, feats1 = self._get_feats_kps(pth1, self.model_name_to_dir_name[kpts_type])
            else:
                kps0, feats0 = detect_and_compute_sift_np(img0)
                kps1, feats1 = detect_and_compute_sift_np(img1)

            if kps0.shape[0]==0 or kps1.shape[0]==0 or len(kps0)==0 or len(kps1)==0:
                continue

            kps0, feats0 = keep_valid_kps_feats(kps0, feats0, H, img1.shape[0], img1.shape[1])
            kps1, feats1 = keep_valid_kps_feats(kps1, feats1, np.linalg.inv(H), img0.shape[0], img0.shape[1])
            if kps0.shape[0]==0 or kps1.shape[0]==0 or len(kps0)==0 or len(kps1)==0:
                continue

            if not use_raw_feats:
                feats0 = self._extract_feats(img0,kps0)
                feats1 = self._extract_feats(img1,kps1)

            pr01, gt01, pr10, gt10 = compute_match_pairs(feats0, kps0, feats1, kps1, H)
            dist0 = np.linalg.norm(pr01 - gt01, 2, 1)
            dist1 = np.linalg.norm(pr10 - gt10, 2, 1)
            corrects5.append((np.mean(dist0 < 5) + np.mean(dist1 < 5)) / 2)
            corrects2.append((np.mean(dist0 < 2) + np.mean(dist1 < 2)) / 2)
            corrects1.append((np.mean(dist0 < 1) + np.mean(dist1 < 1)) / 2)

        print('{} {} pck-5 {} -2 {} -1 {} cost {} s'.format(evaluate_name,dataset_name,np.mean(corrects5),np.mean(corrects2),np.mean(corrects1),time.time()-begin))

    model_name_to_dir_name = {
        'superpoint': 'sp_hpatches',
        'geodesc': 'gd_hpatches',
        'geodesc_ds': 'gd_hpatches_ds',
        'sift':'sift',
        'lf_net':'lf_hpatches',
        'harris':'harris'
    }

    @staticmethod
    def _get_feats_kps(pth, model_name):
        if model_name=='sift':
            kps, feats = detect_and_compute_sift_np(imread(pth))
        elif model_name=='harris':
            kps, feats = detect_and_compute_harris_np(imread(pth),2048)
        else:
            subpths = pth.split('/')
            npzfn = '_'.join([subpths[-2], subpths[-1].split('.')[0]]) + '.npz'
            data_dir = subpths[-3]
            fn = os.path.join('data', model_name, data_dir)
            fn = os.path.join(fn, npzfn)
            if os.path.exists(fn):
                npzfile = np.load(fn)
                kps, feats = npzfile['kpts'], npzfile['descs']
            else:
                kps, feats= np.zeros([0,2]), np.zeros([0,128])
                print('{} not found !'.format(fn))
        return kps, feats

    def _extract_feats(self, img, pts):
        if not self._network_ready: self._init_network()
        transformed_imgs=self.transformer.transform(img,pts)
        with torch.no_grad():
            img_list,pts_list=to_cuda(self.transformer.postprocess_transformed_imgs(transformed_imgs))
            gfeats=self.extractor(dim_extend(img_list),dim_extend(pts_list))
            efeats=self.embedder(gfeats)[0].detach().cpu().numpy()
        return efeats

    def _save_model(self):
        os.system('mkdir -p {}'.format(self.model_dir))
        state_dict = {
            'extractor': self.extractor.state_dict(),
            'optim': self.optim.state_dict(),
            'step': self.step
        }
        if self.embedder is not None: state_dict['embedder'] = self.embedder.state_dict()
        torch.save(state_dict, os.path.join(self.model_dir, '{}.pth'.format(self.step)))

    def _load_model(self, model_dir, step=-1, load_extractor=True, load_embedder=False, load_optimizer=True):
        if not os.path.exists(model_dir):
            return 0

        pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
        if len(pths) == 0:
            return 0
        if step == -1:
            pth = max(pths)
        else:
            pth = step

        pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
        if load_extractor and self.extractor is not None:
            state_dict = pretrained_model['extractor']
            self.extractor.load_state_dict(state_dict)
        if load_embedder and self.embedder is not None and 'embedder' in pretrained_model:
            self.embedder.load_state_dict(pretrained_model['embedder'])
        if load_optimizer: self.optim.load_state_dict(pretrained_model['optim'])
        print('load {} step {}'.format(model_dir, pretrained_model['step']))
        self.step = pretrained_model['step'] + 1


