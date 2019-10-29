import time
from collections import OrderedDict

import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
import os
import yaml

from dataset.correspondence_dataset import CorrespondenceDataset
from dataset.correspondence_database import CorrespondenceDatabase, worker_init_fn
from network.wrapper import TrainWrapper
from train.train_tools import overwrite_configs, Recorder, to_cuda, adjust_learning_rate, reset_learning_rate
from torch.optim import Adam


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
        self.optim=Adam(self.network.parameters(),lr=1e-3)
        self.step=0

        self._load_model(self.model_dir, -1, True, True, True)

        self._network_ready=True
        print('network is ready')

    def _get_hem_thresh(self):
        if self.step<2000:
            return 32
        if 2000<=self.step<=10000:
            return 32-31/8000*(self.step-2000)
        if self.step>10000:
            return 1

    def _get_lr(self):
        if self.step<1000:
            return 1e-4
        if 1000<=self.step<=10000:
            return 1e-3-(1e-3-5e-5)/8000*(self.step-2000)
        if self.step>10000:
            return 5e-5

    def train(self):
        self._init_network()
        self._init_train_set()

        batch_begin=time.time()
        for data in self.train_loader:
            lr=reset_learning_rate(self.optim,self._get_lr())
            hem_thresh=self._get_hem_thresh()

            loss_info = OrderedDict()
            img_list0,pts_list0,pts0,grid_list0,img_list1,pts_list1,pts1,grid_list1,scale_offset,rotate_offset,H=to_cuda(data)
            data_time = time.time() - batch_begin

            results = self.network(img_list0, pts_list0, pts0, grid_list0, img_list1, pts_list1, pts1, grid_list1,
                                   scale_offset, rotate_offset, hem_thresh, self.config['loss_type'])

            loss = 0.0
            for k, v in results.items():
                v = torch.mean(v)
                if k.endswith('loss'): loss = loss + v
                loss_info[k] = v.cpu().detach().numpy()

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            batch_time = time.time() - batch_begin
            # record
            loss_info['data_time'] = data_time
            loss_info['batch_time'] = batch_time
            loss_info['lr'] = lr
            total_step=self.step
            self.recorder.rec_loss(loss_info, total_step, total_step, 'train', ((total_step + 1) % self.config['info_step']) == 0)

            # save model
            if (total_step+1)%self.config['save_step']==0:
                self._save_model()
                print('model saved!')

            batch_begin = time.time()
            self.step+=1

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


