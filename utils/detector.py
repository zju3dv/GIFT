import cv2
import numpy as np

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

def detect_and_compute_harris_np(img,num):
    smooth_img = cv2.GaussianBlur(img, (5, 5), 1.5)
    harris_img = cv2.cornerHarris(cv2.cvtColor(smooth_img, cv2.COLOR_RGB2GRAY).astype(np.float32), 2, 3, 0.04)
    element=np.sort(harris_img.flatten())[-num]
    mask=harris_img>=element
    hs,ws=np.nonzero(mask)
    kps=np.concatenate([ws[:,None],hs[:,None]],1)
    des=np.zeros([kps.shape[0],128],np.float32)
    return kps, des
