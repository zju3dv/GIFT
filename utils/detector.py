import cv2
import numpy as np

from utils.opencvhelper import SiftWrapper

def detect_dog_keypoints(img):
    sift = cv2.xfeatures2d.SIFT_create()
    if len(img.shape)==3: img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kps=sift.detect(img,None)
    kps_np=np.asarray([kp.pt for kp in kps])
    return np.round(kps_np).astype(np.int32)


def detect_and_compute_sift_np(img):
    sift = cv2.xfeatures2d.SIFT_create()
    if len(img.shape)==3: img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(img,None)
    kp=np.asarray([p.pt for p in kp])
    return kp, des

def detect_and_compute_sift_np_num(img,num=1024):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=num)
    if len(img.shape)==3: img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(img,None)
    kp=np.asarray([p.pt for p in kp])
    return kp, des

def detect_and_compute_sift_np_v2(img):
    sift = cv2.xfeatures2d.SIFT_create(4096)
    if len(img.shape)==3: img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(img,None)
    kp=np.asarray([p.pt for p in kp])
    return kp, des

def detect_and_compute_harris_np(img,num):
    smooth_img = cv2.GaussianBlur(img, (5, 5), 1.5)
    if len(smooth_img.shape)==3:
        smooth_img=cv2.cvtColor(smooth_img, cv2.COLOR_RGB2GRAY)
    harris_img = cv2.cornerHarris(smooth_img.astype(np.float32), 2, 3, 0.04)
    element=np.sort(harris_img.flatten())[-num]
    mask=harris_img>=element
    hs,ws=np.nonzero(mask)
    kps=np.concatenate([ws[:,None],hs[:,None]],1)
    des=np.zeros([kps.shape[0],128],np.float32)
    return kps, des

class SIFTDetector:
    def __init__(self,cfg):
        self.wrapper=SiftWrapper(n_sample=cfg['sample_num'])
        self.wrapper.half_sigma = True
        self.wrapper.pyr_off = False
        self.wrapper.ori_off = False
        self.wrapper.create()

    def __call__(self,img):
        if len(img.shape)>2:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray_img=img.copy()

        _, cv_kpts = self.wrapper.detect(gray_img)
        if len(cv_kpts)==0:
            return np.zeros([0,2]), np.zeros([0,128])

        sift_desc = np.asarray(self.wrapper.compute(img,cv_kpts),np.float32)
        sift_desc/=np.linalg.norm(sift_desc,2,1,True)

        np_kpts=np.asarray([kp.pt for kp in cv_kpts])
        return np_kpts, sift_desc

class HarrisDetector:
    def __init__(self,cfg):
        self.feature_num=cfg['feature_num']

    def __call__(self, img):
        return detect_and_compute_harris_np(img,self.feature_num)

class GridDetector:
    def __init__(self,cfg):
        self.interval=cfg['interval']

    def __call__(self, img):
        h, w = img.shape[:2]
        th, tw = h//self.interval, w//self.interval
        xs, ys = np.meshgrid(np.arange(tw), np.arange(th))
        pts = np.concatenate([xs[:, :, None], ys[:, :, None]], 2).reshape([-1, 2])
        return pts*self.interval, np.zeros([pts.shape[0],128],np.float32)