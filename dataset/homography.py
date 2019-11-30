import cv2
import numpy as np
from scipy.stats import truncnorm

from utils.base_utils import perspective_transform

def rotate_pts(pts, max_angle,sample_type='rvs'):
    if sample_type=='rvs':
        angle=truncnorm.rvs(-2,2,loc=0,scale=max_angle/2)
    elif sample_type=='uniform':
        angle=np.random.uniform(-max_angle,max_angle)
    else: raise NotImplementedError
    rot_m = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],np.float32)
    center=np.mean(pts,0,keepdims=True)
    return np.matmul(pts-center,rot_m.transpose())+center

def perspective_pts(pts, h, w, perspective_amplitude=0.2, direction='lr', perspective_short_amplitude=0.2):
    displacement = np.random.uniform(-perspective_amplitude,perspective_amplitude) # truncnorm.rvs(-1, 1, loc=0, scale=perspective_amplitude)
    ds = np.random.uniform(-perspective_short_amplitude, 0)
    if direction=='lr':
        displacement*=h
        ds*=w
        pts += np.asarray([[ds, displacement],[ds, -displacement],[-ds, displacement],[-ds, -displacement]], np.float32)
    elif direction=='ud':
        displacement*=w
        ds*=h
        pts += np.asarray([[displacement, ds],[-displacement, -ds],[displacement, -ds], [-displacement, ds]], np.float32)
    else: raise NotImplementedError
    return pts

def scale_pts(pts,max_scale_ratio,base_ratio=2):
    scale=base_ratio**np.random.uniform(-max_scale_ratio,max_scale_ratio)
    center=np.mean(pts,0,keepdims=True)
    return (pts-center)*scale+center

def translate_pts(pts, h, w, overflow_val=0.2):
    n_pts=pts.copy()

    n_pts[:,0]/=w
    n_pts[:,1]/=h
    min_x,min_y=np.min(n_pts, 0)
    max_x,max_y=np.max(n_pts, 0)

    beg_x=min(-overflow_val-min_x,0)
    end_x=max(overflow_val+1.0-max_x,0)
    if beg_x<end_x: offset_x=np.random.uniform(beg_x,end_x)
    else: offset_x=0

    beg_y=min(-overflow_val-min_y,0)
    end_y=max(overflow_val+1.0-max_y,0)
    if beg_x<end_x: offset_y=np.random.uniform(beg_y,end_y)
    else: offset_y=0

    pts_off=n_pts.copy()
    pts_off[:,0]+=offset_x
    pts_off[:,1]+=offset_y
    pts_off[:,0]*=w
    pts_off[:,1]*=h

    return pts_off

def nearest_identity(pts, h, w):
    pts=scale_pts(pts, 0.15)
    pts=rotate_pts(pts, 5 / 180 * np.pi)
    pts=translate_pts(pts, h, w, 0.05)
    return pts

def nearest_identity_strictly(pts, h, w):
    pts=scale_pts(pts, 0.05)
    pts=rotate_pts(pts, 5 / 180 * np.pi)
    # pts=translate_pts(pts, h, w, 0.05)
    return pts

def left_right_move(pts, h, w):
    pts=perspective_pts(pts, h, w, 0.3, 'lr', 0.3)
    pts=scale_pts(pts, 0.15)
    pts=rotate_pts(pts, 5 / 180 * np.pi)
    return pts

def up_down_move(pts, h, w):
    pts=perspective_pts(pts, h, w, 0.2, 'ud', 0.2)
    pts=scale_pts(pts, 0.15)
    pts=rotate_pts(pts, 5 / 180 * np.pi)
    return pts

def forward_backward_move(pts, h, w):
    pts=scale_pts(pts,1.0)
    pts=rotate_pts(pts,30/180*np.pi)
    pts=translate_pts(pts,h,w,0.05)
    return pts

def rotate_move(pts, h, w):
    pts=scale_pts(pts,0.15)
    pts=rotate_pts(pts,60/180*np.pi,'uniform')
    return pts

def scale_move(pts, h, w):
    pts=scale_pts(pts,1.2)
    pts=rotate_pts(pts,5/180*np.pi)
    return pts

def sample_homography(h, w):
    pts1 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    pts2 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    shape=np.asarray([w,h],np.float32).reshape([1,2])
    pts1*=shape
    pts2*=shape

    fns=[nearest_identity, left_right_move, up_down_move, forward_backward_move, scale_move, rotate_move]
    pts2=np.random.choice(fns,p=[0.15,0.25,0.2,0.1,0.25,0.05])(pts2,h,w)
    H=cv2.getPerspectiveTransform(pts2.astype(np.float32),pts1.astype(np.float32))
    return H

def compute_approximated_affine(H, x, y):
    x_h=np.asarray([x,y,1])
    x_n=np.asarray([[x,y]])
    x_p=perspective_transform(x_n,H)[0] # [2,]
    z_h=np.dot(H[2,:],x_h)
    return (H[:2,:2]-x_p[:,None] @ H[2,:2][None,:])/z_h

def compute_approximated_affine_batch(H, pts):
    x_h=np.concatenate([pts,np.ones([pts.shape[0],1])],1)  # n,3
    x_p=perspective_transform(pts,H)                       # [n,2]
    z_h=np.dot(H[2,:],x_h.transpose()).transpose()         # [n,2]

    return (H[:2,:2][None,:,:]-x_p[:,:, None] @ H[2,:2][None,:])/z_h[:,None,None]
