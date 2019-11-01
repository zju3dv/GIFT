import numpy as np
import torch

from utils.base_utils import get_rot_m, normalize_image
import cv2

class TransformerCV:
    def __init__(self, cfg):
        ssb = cfg['sample_scale_begin']
        ssi = cfg['sample_scale_inter']
        ssn = cfg['sample_scale_num']

        srb = cfg['sample_rotate_begin'] / 180 * np.pi
        sri = cfg['sample_rotate_inter'] / 180 * np.pi
        srn = cfg['sample_rotate_num']

        self.scales = [ssi ** (si + ssb) for si in range(ssn)]
        self.rotations = [sri * ri + srb for ri in range(srn)]

        self.grid_interval=cfg['hem_interval']
        self.ssi=ssi

        self.SRs=[]
        for scale in self.scales:
            Rs=[]
            for rotation in self.rotations:
                Rs.append(scale*get_rot_m(rotation))
            self.SRs.append(Rs)

    def transform(self, img, pts=None, output_grid=False):
        '''

        :param img:
        :param pts:
        :param output_grid:  output a list of transformed grid coordinate for training
        :return:
        '''
        h,w,_=img.shape
        pts0=np.asarray([[0,0],[0,h],[w,h],[w,0]],np.float32)
        center = np.mean(pts0, 0)

        if output_grid:
            xs, ys=np.meshgrid(np.arange(0,w,self.grid_interval),np.arange(0,h,self.grid_interval))
            gh,gw=xs.shape[0],xs.shape[1]
            grid=np.concatenate([xs[:,:,None],ys[:,:,None]],2).reshape([gh*gw, 2]) # hem_h*hem_w,2

        pts_warps, img_warps, grid_warps = [], [], []
        img_cur=img.copy()
        for si,Rs in enumerate(self.SRs):
            if si>0:
                if self.ssi<0.6:
                    img_cur=cv2.GaussianBlur(img_cur,(5,5),1.5)
                else:
                    img_cur=cv2.GaussianBlur(img_cur,(3,3),0.75)
            for M in Rs:
                pts1 = (pts0 - center[None, :]) @ M.transpose()
                min_pts1 = np.min(pts1, 0)
                tw, th = np.round(np.max(pts1 - min_pts1[None, :], 0)).astype(np.int32)

                # compute A
                offset = - M @ center - min_pts1
                A = np.concatenate([M, offset[:, None]], 1)
                # note!!!! the border type is constant 127!!!! because in the subsequent processing, we will subtract 127
                img_warp=cv2.warpAffine(img_cur,A,(tw,th),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(127,127,127))
                img_warps.append(img_warp[:,:,:3])
                if pts is not None:
                    pts_warp = pts @ M.transpose() + offset[None, :]
                    pts_warps.append(pts_warp)
                if output_grid:
                    grid_warp = grid @ M.transpose() + offset[None, :]
                    grid_warps.append(grid_warp.reshape([gh,gw,2]))

        outputs={'img':img_warps}
        if pts is not None: outputs['pts']=pts_warps
        if output_grid: outputs['grid']=grid_warps

        return outputs


    @staticmethod
    def postprocess_transformed_imgs(results, output_grid=False):
        img_list,pts_list,grid_list=[],[],[]
        for img_id, img in enumerate(results['img']):
            img_list.append(normalize_image(img))
            pts_list.append(torch.tensor(results['pts'][img_id],dtype=torch.float32))
            if output_grid:
                grid_list.append(torch.tensor(results['grid'][img_id],dtype=torch.float32))

        if output_grid:
            return img_list, pts_list, grid_list
        else:
            return img_list, pts_list

