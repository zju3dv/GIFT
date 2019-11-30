import cv2
import os
import time

import numpy as np
import yaml
from easydict import EasyDict
from pyflann import FLANN
from skimage.io import imread

from network.wrapper import GIFTDescriptor
from utils.base_utils import perspective_transform, gray_repeats, save_pickle
from utils.detector import SIFTDetector, HarrisDetector, GridDetector
from utils.extend_utils.extend_utils_fn import find_nearest_point_idx, find_first_and_second_nearest_point
from utils.match_utils import keep_valid_kps_feats, compute_angle
from utils.superpoint_utils import SuperPointWrapper, SuperPointDescriptor


class Matcher:
    def __init__(self, cfg):
        self.mutual_best=cfg['mutual_best']
        self.ratio_test=cfg['ratio_test']
        self.ratio=cfg['ratio']
        self.use_cuda=cfg['cuda']
        self.flann=FLANN()
        if self.use_cuda:
            self.match_fn_1=lambda desc0,desc1: find_nearest_point_idx(desc1, desc0)
            self.match_fn_2=lambda desc0,desc1: find_first_and_second_nearest_point(desc1, desc0)
        else:
            self.match_fn_1=lambda desc0,desc1: self.flann.nn(desc1, desc0, 1, algorithm='linear')
            self.match_fn_2=lambda desc0,desc1: self.flann.nn(desc1, desc0, 2, algorithm='linear')

    def match(self,desc0,desc1):
        mask=np.ones(desc0.shape[0],dtype=np.bool)
        if self.ratio_test:
            idxs,dists = self.match_fn_2(desc0,desc1)

            dists=np.sqrt(dists) # note the distance is squared
            ratio_mask=dists[:,0]/dists[:,1]<self.ratio
            mask&=ratio_mask
            idxs=idxs[:,0]
        else:
            idxs,_=self.match_fn_1(desc0,desc1)

        if self.mutual_best:
            idxs_mutual,_=self.match_fn_1(desc1,desc0)
            mutual_mask = np.arange(desc0.shape[0]) == idxs_mutual[idxs]
            mask&=mutual_mask

        matches=np.concatenate([np.arange(desc0.shape[0])[:,None],idxs[:,None]],axis=1)
        matches=matches[mask]

        return matches

    def match_original(self, desc0, desc1):
        idxs01, _ = self.match_fn_1(desc0, desc1)
        idxs10, _ = self.match_fn_1(desc1, desc0)
        return idxs01, idxs10

class DummyDescriptor:
    def __init__(self,cfg):
        pass

    def __call__(self, img, kps):
        return np.zeros([kps.shape[0],0])

class CorrespondenceEstimator:
    name2det={
        'superpoint': SuperPointWrapper,
        'sift': SIFTDetector,
        'harris': HarrisDetector,
        'grid':GridDetector,
    }
    name2desc={
        'none': DummyDescriptor,
        'gift': GIFTDescriptor,
        'superpoint': SuperPointDescriptor
    }
    def __init__(self,det_cfg,desc_cfg,match_cfg):
        self.det_cfg=det_cfg
        self.desc_cfg=desc_cfg
        self.detector=self.name2det[det_cfg['type']](det_cfg)
        self.descriptor=self.name2desc[desc_cfg['type']](desc_cfg)
        self.matcher=Matcher(match_cfg)

    def __call__(self, img0, img1, kps_filter=None):
        kps0, desc0 = self.detector(img0)
        kps1, desc1 = self.detector(img1)

        if kps_filter is not None:
            kps0, desc0 = kps_filter(kps0, desc0)

        if kps0.shape[0] == 0 or kps1.shape[0] == 0:
            return np.zeros([0,2]), np.zeros([0,2]), np.zeros([0,128]), np.zeros([0,128]), np.zeros([0,2])

        if self.desc_cfg['type']=='none':
            assert(desc0 is not None and desc1 is not None)
        else:
            desc0 = self.descriptor(img0, kps0)
            desc1 = self.descriptor(img1, kps1)

        matches = self.matcher.match(desc0,desc1)

        return kps0, kps1, desc0, desc1, matches

    def estimate_original(self, img0, img1, H):
        kps0, desc0 = self.detector(img0)
        kps1, desc1 = self.detector(img1)

        if kps0.shape[0] != 0 and kps1.shape[0] != 0:
            kps0, desc0 = keep_valid_kps_feats(kps0, desc0, H, img1.shape[0], img1.shape[1])
            kps1, desc1 = keep_valid_kps_feats(kps1, desc1, np.linalg.inv(H), img0.shape[0], img0.shape[1])

        if kps0.shape[0] == 0 or kps1.shape[0] == 0:
            return np.zeros([0]), np.zeros([0]), np.zeros([0]), np.zeros([0])

        if self.desc_cfg['type']=='none':
            assert(desc0 is not None and desc1 is not None)
        else:
            desc0 = self.descriptor(img0, kps0)
            desc1 = self.descriptor(img1, kps1)

        idxs01, idxs10 = self.matcher.match_original(desc0, desc1)

        pr01 = kps1[idxs01]
        gt01 = perspective_transform(kps0, H)

        pr10 = kps0[idxs10]
        gt10 = perspective_transform(kps1, np.linalg.inv(H))

        return gt01, gt10, pr01, pr10

class ThreshCorrectMetric:
    def __init__(self,thresh):
        self.corrects=[]
        self.corrects_num=[]
        self.max_corrects=[]
        self.max_corrects_num=[]
        self.thresh=thresh

    def update(self,dists,min_dists):
        self.corrects.append(np.mean(dists < self.thresh))
        self.corrects_num.append(np.sum(dists < self.thresh))
        self.max_corrects.append(np.mean(min_dists < self.thresh))
        self.max_corrects_num.append(np.sum(min_dists < self.thresh))

    def update_zero(self):
        self.corrects.append(0)
        self.corrects_num.append(0)
        self.max_corrects.append(0)
        self.max_corrects_num.append(0)

    def __str__(self):
        return '-{} {:.3f}/{:.3f} num {:.3f}/{:.3f}'.format(
            self.thresh,np.mean(self.corrects),np.mean(self.max_corrects),
            np.mean(self.corrects_num), np.mean(self.max_corrects_num))

class EvaluationWrapper:
    @staticmethod
    def get_stem(path):
        return os.path.basename(path).split('.')[0]

    @staticmethod
    def load_cfg(path):
        with open(path, 'r') as f:
            return yaml.load(f,Loader=yaml.FullLoader)

    def log(self,msg):
        print(msg)

    def __init__(self,det_cfg_file,desc_cfg_file,match_cfg_file):
        self.det_name=self.get_stem(det_cfg_file)
        self.desc_name=self.get_stem(desc_cfg_file)
        self.match_name=self.get_stem(match_cfg_file)
        self.eval_name=f'{self.det_name}_{self.desc_name}_{self.match_name}'
        self.correspondence_estimator=CorrespondenceEstimator(
            self.load_cfg(det_cfg_file),self.load_cfg(desc_cfg_file),self.load_cfg(match_cfg_file))

    def evaluate_homography(self,dataset, dataset_name):
        thresh=[7,5,3,1]
        thresh_metrics=[ThreshCorrectMetric(t) for t in thresh]
        begin=time.time()
        for data in dataset:
            pth0, pth1, H = data['img0_pth'], data['img1_pth'], data['H'].copy()
            img0, img1 = imread(pth0), imread(pth1)
            kps_filter = lambda kps, desc: keep_valid_kps_feats(kps, desc, H, img1.shape[0], img1.shape[1])
            kps0, kps1, desc0, desc1, matches = self.correspondence_estimator(img0, img1, kps_filter)

            if (kps0.shape[0]==0) or matches.shape[0]==0:
                for tm in thresh_metrics: tm.update_zero()
                continue

            gt = perspective_transform(kps0, H)
            dists = np.linalg.norm(gt[matches[:,0]]-kps1[matches[:,1]],2,1)
            idxs, _ = find_nearest_point_idx(kps1, gt)
            min_dists = np.linalg.norm(kps1[idxs[matches[:,0]]]-gt[matches[:,0]],2,1)

            for tm in thresh_metrics: tm.update(dists,min_dists)

        metric_str=' '.join([str(tm) for tm in thresh_metrics])
        msg = '{} {} {} {} pck {} cost {:.3f} s'.format(dataset_name, self.det_name, self.desc_name,
                                                        self.match_name, metric_str,time.time()-begin)

        self.log(msg)

    def evaluate_homography_original(self, dataset, dataset_name):
        corrects5, corrects2, corrects1 = [], [], []
        begin=time.time()
        for data in dataset:
            pth0, pth1, H = data['img0_pth'], data['img1_pth'], data['H'].copy()
            img0, img1 = imread(pth0), imread(pth1)
            gt01, gt10, pr01, pr10 = self.correspondence_estimator.estimate_original(img0, img1, H)

            if len(gt01)==0:
                continue

            dists01 = np.linalg.norm(gt01-pr01,2,1)
            dists10 = np.linalg.norm(gt10-pr10,2,1)

            corrects5.append((np.mean(dists01<5)+np.mean(dists10<5))/2)
            corrects2.append((np.mean(dists01<2)+np.mean(dists10<2))/2)
            corrects1.append((np.mean(dists01<1)+np.mean(dists10<1))/2)

        msg = '{} {} {} {} pck-5 {:.3f} -2 {:.3f} -1 {:.3f} cost {:.3f} s'.format(
            dataset_name, self.det_name, self.desc_name, self.match_name, np.mean(corrects5),
            np.mean(corrects2), np.mean(corrects1), time.time()-begin)

        self.log(msg)

    @staticmethod
    def estimate_relative_pose_from_correspondence(pts1, pts2, K1, K2):
        f_avg = (K1[0, 0] + K2[0, 0]) / 2
        pts1, pts2 = np.ascontiguousarray(pts1, np.float32), np.ascontiguousarray(pts2, np.float32)

        pts_l_norm = cv2.undistortPoints(np.expand_dims(pts1, axis=1), cameraMatrix=K1, distCoeffs=None)
        pts_r_norm = cv2.undistortPoints(np.expand_dims(pts2, axis=1), cameraMatrix=K2, distCoeffs=None)

        E, mask = cv2.findEssentialMat(pts_l_norm, pts_r_norm, focal=1.0, pp=(0., 0.),
                                       method=cv2.RANSAC, prob=0.999, threshold=3.0 / f_avg)
        points, R_est, t_est, mask_pose = cv2.recoverPose(E, pts_l_norm, pts_r_norm)
        return mask[:,0].astype(np.bool), R_est, t_est

    def relative_pose_estimation(self,dataset,dataset_name):
        print('dataset {} len {}'.format(dataset_name,len(dataset)))
        begin=time.time()
        ang_diffs,inliers=[],[]
        for data in dataset:
            if type(data)==dict:
                img0, img1, K = data['img0'].copy(), data['img1'].copy(), data['K'].copy()
                R=data['T'][:3,:3]
                K0, K1 = K.copy(), K.copy()
            else:
                assert(type(data)==EasyDict)
                img0, img1 = gray_repeats(imread(data.path0)), gray_repeats(imread(data.path1))
                K0, K1, R = data.K0, data.K1, data.R01

            kps0, kps1, desc0, desc1, matches = self.correspondence_estimator(img0,img1)

            if kps0.shape[0]<5 or kps1.shape[0]<5 or matches.shape[0]<5:
                ang_diffs.append(180), inliers.append(0)
                continue

            mask, R_est, t_est = self.estimate_relative_pose_from_correspondence(
                kps0[matches[:,0]],kps1[matches[:,1]],K0,K1)

            ang_diff=compute_angle(R.T @ R_est)
            inlier=np.sum(mask)

            ang_diffs.append(ang_diff)
            inliers.append(inlier)

        save_pickle(ang_diffs,f'data/results/{self.eval_name}_{dataset_name}.pkl')
        ang_diffs = np.asarray(ang_diffs)
        ang_diffs[np.isnan(ang_diffs)] = 0

        msg='{} {} {} {} ang diff {:.3f} inlier {:.3f} correct-5 {:.3f} -10 {:.3f} -20 {:.3f} cost {:.3f} s'.format(
            dataset_name, self.det_name, self.desc_name, self.match_name, np.mean(ang_diffs), np.mean(inliers),
            np.mean(ang_diffs < 5), np.mean(ang_diffs < 10),np.mean(ang_diffs < 20),time.time()-begin)
        self.log(msg)
