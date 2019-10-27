from network.embedder import *
from network.extractor import *

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network.operator import GaussianConv, normalize_coordinates
from utils.base_utils import get_rot_m

name2embedder={}
name2extractor={
    "NetworkShallow": NetworkShallow,
}

class GroupTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.srn=cfg['sample_rotate_num']
        self.ssn=cfg['sample_scale_num']
        self.ssb=cfg['sample_scale_begin']
        self.srb=cfg['sample_rotate_begin']
        self.sre=cfg['sample_rotate_end']
        self.st=cfg['sample_type']
        self.sbf=cfg['sample_base_factor']

        self.sri=(self.sre-self.srb)/180*np.pi/(self.srn-1) if self.srn>1 else 0
        self.srb=0 if self.srn==1 else self.srb
        self.Ms=[]
        self.gconv=GaussianConv(cfg['sample_gconv_ksize'])

        if self.st=='rotate_scale':
            for si in range(self.ssn):
                MRs=[]
                for ri in range(self.srn):
                    sm=self.sbf**(si+self.ssb)
                    sm=np.diag([sm,sm])
                    rm=get_rot_m(self.sri*ri+self.srb/180*np.pi)
                    M=sm @ rm
                    MRs.append(M)
                self.Ms.append(MRs)
        else:
            raise NotImplementedError

    def forward(self, img):
        b,f,h,w=img.shape
        pts0=np.asarray([[0,0],[0,h],[w,h],[w,0]],np.float32)
        imgs,As=[],[]
        cimg=img
        for si, MRs in enumerate(self.Ms):
            if si+self.ssb>0:
                cimg=self.gconv(cimg)
            for M in MRs:
                # compute target size
                center=np.mean(pts0,0)
                pts1= (pts0 - center[None,:]) @ M.transpose()
                min_pts1=np.min(pts1,0)
                tw,th=np.round(np.max(pts1-min_pts1[None,:],0)).astype(np.int32)

                # compute A
                offset= - M @ center - min_pts1
                A=np.concatenate([M,offset[:,None]],1)

                # sample grid
                xs,ys=np.meshgrid(np.arange(tw),np.arange(th))
                pts=np.concatenate([xs[:,:,None],ys[:,:,None]],2).reshape([th*tw,2])
                pts=(pts-A[:,2:].transpose()) @ np.linalg.inv(A[:,:2]).transpose()
                pts=pts.reshape([th,tw,2])
                pts=torch.tensor(pts,dtype=torch.float32,device=img.device)
                pts=normalize_coordinates(pts,h,w)
                img_out=F.grid_sample(cimg,pts.unsqueeze(0).repeat(b,1,1,1),'bilinear')

                A = torch.tensor(A, dtype=torch.float32, device=cimg.device)
                As.append(A)
                imgs.append(img_out)

        return imgs, As

def interpolate_feats(img,pts,feats):
    # compute location on the feature map (due to pooling)
    _, _, h, w = feats.shape
    pool_num = img.shape[-1] // feats.shape[-1]
    pts_warp=(pts+0.5)/pool_num-0.5
    pts_norm=normalize_coordinates(pts_warp,h,w)
    pts_norm=torch.unsqueeze(pts_norm, 1)  # b,1,n,2

    # interpolation
    pfeats=F.grid_sample(feats, pts_norm, 'bilinear')[:, :, 0, :]  # b,f,n
    pfeats=pfeats.permute(0,2,1) # b,n,f
    return pfeats

class ExtractorWrapper(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.extractor=name2extractor[cfg['extractor']](cfg)
        self.embedder=name2embedder[cfg['embedder']](cfg)
        self.sn, self.rn = cfg['sample_scale_num'], cfg['sample_rotate_num']

    def forward(self,img_list,pts_list,grid_list=None):
        '''

        :param img_list:  list of [b,3,h,w]
        :param pts_list:  list of [b,n,2]
        :param grid_list:  list of [b,hn,wn,2]
        :return:gefeats [b,n,f,sn,rn]
        '''
        assert(len(img_list)==self.rn*self.sn)
        pfeats_list,neg_feats_list=[],[]
        # feature extraction
        for img_index,img in enumerate(img_list):
            # extract feature
            feats=self.extractor(img)
            pfeats_list.append(interpolate_feats(img,pts_list[img_index],feats)[:,:,:,None])
            if grid_list is not None:
                _,hn,wn,_=grid_list[img_index].shape
                grid_pts=grid_list[img_index].reshape(-1,hn*wn,2)
                neg_feats_list.append(interpolate_feats(img,grid_pts,feats)[:,:,:,None])


        pfeats_list=torch.cat(pfeats_list,3) # b,n,f,sn*rn
        if grid_list is not None:
            neg_feats_list = torch.cat(neg_feats_list, 3) # b,hn*wn,f,sn*rn
            b,hn,wn,_=grid_list[0].shape
            b,_,f,srn=neg_feats_list.shape
            neg_feats_list=neg_feats_list.reshape(b,hn,wn,f,srn) # b,hn,wn,f,sn*rn
            return pfeats_list, neg_feats_list
        else:
            return pfeats_list

class EmbedderWrapper(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.embedder=name2embedder[cfg['embedder']](cfg)
        self.sn, self.rn = cfg['sample_scale_num'], cfg['sample_rotate_num']

    def forward(self,pfeats):
        # group cnns
        b,n,f,srn=pfeats.shape
        assert(srn==self.sn*self.rn)
        gfeats=pfeats.reshape(b,n,f,self.sn,self.rn)
        gefeats=self.embedder(gfeats) # b,n,f
        return gefeats

class TrainingWrapper(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.extractor_wrapper=ExtractorWrapper(cfg)
        self.embedder_wrapper=EmbedderWrapper(cfg)

    def forward(self, img_list0, pts_list0, img_list1, pts_list1):
        pass
