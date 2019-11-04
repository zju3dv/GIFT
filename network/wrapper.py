from network.embedder import *
from network.extractor import *

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from network.loss import scale_rotate_offset_dist, sample_semi_hard_feature, clamp_loss_all
from network.operator import normalize_coordinates

name2embedder={
    "GREBilinearPool":GREBilinearPool,
    "GREBilinearPoolPaddingDeepMore":GREBilinearPoolPaddingDeepMore,
    "None": lambda cfg: None,
}
name2extractor={
    "NetworkShallow": NetworkShallow,
    "NetworkShallowPool2": NetworkShallowPool2,
    "None": lambda cfg: None,
}

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
        self.sn, self.rn = cfg['sample_scale_num'], cfg['sample_rotate_num']

    def forward(self,img_list,pts_list,grid_list=None):
        '''

        :param img_list:  list of [b,3,h,w]
        :param pts_list:  list of [b,n,2]
        :param grid_list:  list of [b,hn,wn,2]
        :return:gefeats [b,n,f,sn,rn]
        '''
        assert(len(img_list)==self.rn*self.sn)
        gfeats_list,neg_gfeats_list=[],[]
        # feature extraction
        for img_index,img in enumerate(img_list):
            # extract feature
            feats=self.extractor(img)
            gfeats_list.append(interpolate_feats(img,pts_list[img_index],feats)[:,:,:,None])
            if grid_list is not None:
                _,hn,wn,_=grid_list[img_index].shape
                grid_pts=grid_list[img_index].reshape(-1,hn*wn,2)
                neg_gfeats_list.append(interpolate_feats(img,grid_pts,feats)[:,:,:,None])

        gfeats_list=torch.cat(gfeats_list,3)  # b,n,f,sn*rn
        b,n,f,_=gfeats_list.shape
        gfeats_list=gfeats_list.reshape(b,n,f,self.sn,self.rn)
        if grid_list is not None:
            neg_gfeats_list = torch.cat(neg_gfeats_list, 3) # b,hn*wn,f,sn*rn
            b,hn,wn,_=grid_list[0].shape
            b,_,f,srn=neg_gfeats_list.shape
            neg_gfeats_list=neg_gfeats_list.reshape(b,hn,wn,f,self.sn,self.rn) # b,hn,wn,f,sn*rn
            return gfeats_list, neg_gfeats_list
        else:
            return gfeats_list

class EmbedderWrapper(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.embedder=name2embedder[cfg['embedder']](cfg)
        self.sn, self.rn = cfg['sample_scale_num'], cfg['sample_rotate_num']

    def forward(self, gfeats):
        # group cnns
        b,n,f,sn,rn=gfeats.shape
        assert(sn==self.sn and rn==self.rn)
        gefeats=self.embedder(gfeats) # b,n,f
        return gefeats

class TrainWrapper(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.extractor_wrapper=ExtractorWrapper(cfg)
        self.embedder_wrapper=EmbedderWrapper(cfg)
        self.sn, self.rn = cfg['sample_scale_num'], cfg['sample_rotate_num']
        self.loss_margin = cfg['loss_margin']
        self.hem_interval = cfg['hem_interval']
        self.train_embedder = cfg['train_embedder']
        self.train_extractor = cfg['train_extractor']

    # def validate_scale_rotate_offset_dist(self, img_list0, pts_list0, pts0, grid_list0, img_list1, pts_list1, pts1,
    #                                       grid_list1, scale_offset, rotate_offset, hem_thresh, loss_type='gfeats'):
    #     gfeats0 = self.extractor_wrapper(img_list0,pts_list0)  # [b,n,fg,sn,rn]
    #     gfeats1, gfeats_neg = self.extractor_wrapper(img_list1,pts_list1,grid_list1) # [b,n,fg,sn,rn] [b,hn,wn,fg,sn,rn]
    #     b, n, fg, sn, rn = gfeats0.shape
    #     assert(sn==self.sn and rn==self.rn)
    #
    #     # pos distance [b,n]
    #     dis_pos = scale_rotate_offset_dist(gfeats0.permute(0, 1, 3, 4, 2), gfeats1.permute(0, 1, 3, 4, 2), scale_offset, rotate_offset, self.sn, self.rn)
    #
    #     sp=scale_offset+1; sp[sp>self.sn-1]=self.sn-1
    #     sn=scale_offset-1; sn[sn<-self.sn+1]=-self.sn+1
    #     dis_pos_sp1 = scale_rotate_offset_dist(gfeats0.permute(0, 1, 3, 4, 2), gfeats1.permute(0, 1, 3, 4, 2), sp, rotate_offset, self.sn, self.rn)
    #     dis_pos_sn1 = scale_rotate_offset_dist(gfeats0.permute(0, 1, 3, 4, 2), gfeats1.permute(0, 1, 3, 4, 2), sn, rotate_offset, self.sn, self.rn)
    #
    #     rp=rotate_offset+1; rp[rp>self.rn-1]=self.rn-1
    #     rn=rotate_offset-1; rn[rn<-self.rn+1]=-self.rn+1
    #     dis_pos_rp1 = scale_rotate_offset_dist(gfeats0.permute(0, 1, 3, 4, 2), gfeats1.permute(0, 1, 3, 4, 2), scale_offset, rp, self.sn, self.rn)
    #     dis_pos_rn1 = scale_rotate_offset_dist(gfeats0.permute(0, 1, 3, 4, 2), gfeats1.permute(0, 1, 3, 4, 2), scale_offset, rn, self.sn, self.rn)
    #
    #     mask_sp1=(dis_pos-dis_pos_sp1)>0
    #     mask_sn1=(dis_pos-dis_pos_sn1)>0
    #     mask_rp1=(dis_pos-dis_pos_rp1)>0
    #     mask_rn1=(dis_pos-dis_pos_rn1)>0
    #
    #     print(scale_offset[:,:10])
    #     print(rotate_offset[:,:10])
    #     print(torch.sum(mask_sp1),torch.sum(mask_sn1),torch.sum(mask_rp1),torch.sum(mask_rn1),mask_rn1.shape)

    def forward(self, img_list0, pts_list0, pts0, grid_list0, img_list1, pts_list1, pts1, grid_list1, scale_offset, rotate_offset, hem_thresh, loss_type='gfeats'):
        '''
        :param img_list0:   [sn,rn,b,3,h,w]
        :param pts_list0:   [sn,rn,b,n,2]
        :param pts0:        [b,n,2]
        :param grid_list0:  [sn,rn,b,hn,wn,2]
        :param img_list1:   [sn,rn,b,3,h,w]
        :param pts_list1:   [sn,rn,b,n,2]
        :param pts1:        [b,n,2]
        :param grid_list1:  [sn,rn,b,hn,wn,2]
        :param scale_offset:  [b,n]
        :param rotate_offset: [b,n]
        :param hem_thresh:
        :param loss_type: 'gfeats' or 'gefeats'
        :return:
        '''
        gfeats0 = self.extractor_wrapper(img_list0,pts_list0)  # [b,n,fg,sn,rn]
        if self.train_extractor:
            gfeats1, gfeats_neg = self.extractor_wrapper(img_list1,pts_list1,grid_list1) # [b,n,fg,sn,rn] [b,hn,wn,fg,sn,rn]
        else:
            with torch.no_grad():
                gfeats1, gfeats_neg = self.extractor_wrapper(img_list1, pts_list1, grid_list1)  # [b,n,fg,sn,rn] [b,hn,wn,fg,sn,rn]

        b, hn, wn, fg, sn, rn = gfeats_neg.shape
        b, n, fg, sn, rn = gfeats0.shape
        assert(sn==self.sn and rn==self.rn)
        pts_shem_gt=pts1/self.hem_interval
        hem_thresh=hem_thresh/self.hem_interval

        if loss_type=='gfeats':
            # pos distance [b,n]
            dis_pos = scale_rotate_offset_dist(gfeats0.permute(0, 1, 3, 4, 2), gfeats1.permute(0, 1, 3, 4, 2), scale_offset, rotate_offset, self.sn, self.rn)
            dis_pos=dis_pos[:,None,None,:].repeat(1, sn, rn,1).reshape(b*sn*rn,n) # b*sn*rn,n

            # neg search
            gfeats_neg=gfeats_neg.permute(0,4,5,3,1,2)
            gfeats_neg=gfeats_neg.reshape(b*sn*rn,fg,hn,wn)

            pts_shem_gt = pts_shem_gt[:, None, None, :, :].repeat(1, sn, rn, 1, 1).reshape(b * sn * rn, n, 2)  # b*sn*rn,n

            gfeats0=gfeats0.permute(0, 3, 4, 1, 2)
            gfeats0=gfeats0.reshape(b * sn * rn, n, fg)
            gfeats_shem_neg=sample_semi_hard_feature(gfeats_neg, dis_pos, gfeats0, pts_shem_gt, 1, hem_thresh, self.loss_margin) # b*sn*rn,n,fg
            # neg distance [b,n]
            dis_neg = torch.norm(gfeats0-gfeats_shem_neg, 2, 2) # b*sn*rn,n
        else:
            assert(loss_type=='gefeats')
            if self.train_embedder:
                efeats0=self.embedder_wrapper(gfeats0)     # b,n,fe
                efeats1=self.embedder_wrapper(gfeats1)     # b,n,fe
                efeats_neg=self.embedder_wrapper(gfeats_neg.reshape(b,hn*wn,fg,sn,rn)) # b,hn*wn,fe
            else:
                with torch.no_grad():
                    efeats0=self.embedder_wrapper(gfeats0)     # b,n,fe
                    efeats1=self.embedder_wrapper(gfeats1)     # b,n,fe
                    efeats_neg=self.embedder_wrapper(gfeats_neg.reshape(b,hn*wn,fg,sn,rn)) # b,hn*wn,fe

            # pos distance
            dis_pos=torch.norm(efeats0-efeats1, 2, 2)

            # neg search
            fe=efeats_neg.shape[-1]
            efeats_neg=efeats_neg.reshape(b,hn,wn,fe).permute(0,3,1,2)
            efeats_shem_neg=sample_semi_hard_feature(efeats_neg,dis_pos,efeats0,pts_shem_gt,1,hem_thresh,self.loss_margin)

            # neg distance [b,n]
            dis_neg = torch.norm(efeats0-efeats_shem_neg, 2, 2)

        triplet_loss, triplet_neg_rate = clamp_loss_all(dis_pos - dis_neg + self.loss_margin)

        results = {
            'triplet_loss': triplet_loss,
            'triplet_neg_rate': triplet_neg_rate,
            'dis_pos': dis_pos,
            'dis_neg': dis_neg,
        }
        return results

