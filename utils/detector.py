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