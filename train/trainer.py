import time
from collections import OrderedDict

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import os
import yaml

from dataset.correspondence_dataset import CorrespondenceDataset
from dataset.correspondence_database import CorrespondenceDatabase, worker_init_fn
from dataset.transformer import TransformerCV
from network.wrapper import TrainWrapper
from train.train_tools import overwrite_configs, Recorder, reset_learning_rate
from torch.optim import Adam

torch.multiprocessing.set_sharing_strategy('file_system')

class Trainer:
    def __init__(self,cfg_file):
        self._train_set_ready=False
        self._network_ready=False
        self._init_config(cfg_file)

    def _init_config(self, cfg_file):

        with open(cfg_file, 'r') as f:
            overwrite_cfg = yaml.load(f,Loader=yaml.FullLoader)

        if 'default_config' in overwrite_cfg:
            with open(os.path.join(overwrite_cfg['default_config']), 'r') as f:
                default_train_config = yaml.load(f,Loader=yaml.FullLoader)
                self.config = overwrite_configs(default_train_config, overwrite_cfg)
        else:
            self.config = overwrite_cfg

        self.name = self.config['name']
        self.recorder = Recorder(os.path.join('data', 'record', self.name), os.path.join('data', 'record', self.name + '.log'))
        self.model_dir = os.path.join('data', 'model', self.name)
        self.hem_thresh=self.config['hem_thresh_begin']
        self.hem_hit_count=0

    def _init_train_set(self):
        if self._train_set_ready:
            print('training set is ready')
            return

        print('begin preparing training set...')
        database = CorrespondenceDatabase()
        self.database = database

        train_set = []
        for name in self.config['train_set']: train_set += database.__getattr__(name + "_set")
        self.train_set = CorrespondenceDataset(self.config, train_set)
        self.train_loader = DataLoader(self.train_set,self.config['batch_size'],shuffle=True,
                                       num_workers=self.config['worker_num'],worker_init_fn=worker_init_fn)
        self._train_set_ready=True
        print('training set is ready')

    def _init_network(self):
        if self._network_ready:
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

    def _adjust_hem_thresh(self):
        decay_num = (self.step + 1) // self.config['hem_thresh_decay_step']
        old_hem_thresh=self.hem_thresh
        self.hem_thresh = self.config['hem_thresh_begin'] - decay_num * self.config['hem_thresh_decay_rate']
        self.hem_thresh = max(self.hem_thresh, self.config['hem_thresh_end'])
        if self.hem_thresh!=old_hem_thresh:
            print('hem_thresh adjust from {} to {}'.format(old_hem_thresh,self.hem_thresh))

    def _get_warm_up_lr(self):
        if self.step <= 2500:
            lr = 1e-4 * (self.step//250 + 1)
        elif self.step <= 5000:
            lr = 1e-3
        else:
            # 1e-3 to 5e-5
            lr = 1e-3 - (1e-3 - 1e-4) / (15000//250) * ((self.step-5000)//250)
            lr = max(lr, 1e-4)
        return lr

    def _get_finetune_lr(self):
        # 5e-4 to 1e-5
        lr = 5e-4 - (5e-4 - 1e-5) / (10000//250) * (self.step//250)
        lr = max(lr, 1e-5)
        return lr

    def _get_finetune_gl3d_lr(self):
        # 5e-4 to 1e-5
        lr = 5e-4 - (5e-4 - 1e-5) / (20000//500) * (self.step//500)
        lr = max(lr, 1e-5)
        return lr

    def train(self):
        self._init_network()
        self._init_train_set()

        batch_begin=time.time()
        for data in self.train_loader:
            lr=self.__getattribute__('_get_{}_lr'.format(self.config['lr_type']))()
            reset_learning_rate(self.optim,lr)
            self._adjust_hem_thresh()

            loss_info = OrderedDict()
            img_list0,pts_list0,pts0,grid_list0,img_list1,pts_list1,pts1,grid_list1,scale_offset,rotate_offset,H=data
            data_time = time.time() - batch_begin

            results = self.network(img_list0, pts_list0, pts0, grid_list0, img_list1, pts_list1, pts1, grid_list1,
                                   scale_offset, rotate_offset, self.hem_thresh, self.config['loss_type'])

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
            loss_info['hem_thresh'] = self.hem_thresh
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


