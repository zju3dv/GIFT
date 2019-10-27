import math
import time

import cv2
import numpy as np
from transforms3d.euler import euler2mat
from scipy.stats import truncnorm

from utils.base_utils import perspective_transform


def get_homography(angx,angy,angz,scale,focal=200,x0=128,y0=128):
    K=np.asarray([[focal,0,x0 ],
                  [0,focal,y0 ],
                  [0,    0,1.0]],np.float32)
    K_inv=np.linalg.inv(K)
    R=euler2mat(angx,angy,angz,axes='sxyz')
    K_scale=K.copy()
    K_scale[2,2]=scale
    H=np.matmul(np.matmul(K_scale,R),K_inv)
    return H

def compute_homography(h,w,perspective=True,scaling=True,rotation=True,translation=True,
                       min_patch_ratio=1.0,max_patch_ratio=1.0,perspective_amplitude_x=0.2,perspective_amplitude_y=0.2,
                       scaling_amplitude=0.1,max_angle=np.pi/6, translation_overflow=0.,
                       allow_artifacts=False):
    patch_ratio = truncnorm.rvs(-2,2,loc=max_patch_ratio,scale=(max_patch_ratio-min_patch_ratio)/2.0)
    patch_ratio = 2.0-patch_ratio if patch_ratio>=1.0 else patch_ratio

    pts1 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    margin = (1 - patch_ratio) / 2
    pts2 = np.asarray([[0, 0], [0, patch_ratio], [patch_ratio, patch_ratio], [patch_ratio, 0]], np.float32) + margin

    if perspective and (allow_artifacts or margin>1e-2):
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        # perspective_displacement = np.random.uniform(-perspective_amplitude_y, perspective_amplitude_y)
        # h_displacement_left ,h_displacement_right = np.random.uniform(-perspective_amplitude_x , perspective_amplitude_x, 2)
        perspective_displacement = truncnorm.rvs(-2, 2, loc=0, scale=perspective_amplitude_y/2)
        h_displacement_left ,h_displacement_right = truncnorm.rvs(-2,2, loc=0 , scale=perspective_amplitude_x/2, size=2)
        pts2 += np.asarray([[h_displacement_left, perspective_displacement],
                            [h_displacement_left, -perspective_displacement],
                            [h_displacement_right, perspective_displacement],
                            [h_displacement_right, -perspective_displacement]],np.float32)

    if rotation:
        rotation_num=50
        angles = np.linspace(-max_angle, max_angle, rotation_num)
        angles = np.append(angles, 0.0)
        rot_ms = np.asarray([[np.cos(angles), -np.sin(angles)], [np.sin(angles), np.cos(angles)]],np.float32) # rn+1,3,3
        rot_ms = rot_ms.transpose(2,0,1)

        center = np.mean(pts2, axis=0, keepdims=True)
        rotated = np.tile(np.expand_dims(pts2 - center, 0), (rotation_num+1, 1, 1))
        rotated = np.matmul(rotated, rot_ms) + center[None,:,:]
        if allow_artifacts:
            valid=np.arange(rotation_num)
        else:
            valid=np.nonzero(np.all(np.all(np.logical_and(rotated >= 0., rotated < 1.0), 2), 1))[0]
        pts2=rotated[np.random.choice(valid,1)[0]]

    if scaling and allow_artifacts:
        scale_num=25
        scales=np.random.uniform(0.4/patch_ratio,1.0/patch_ratio,scale_num)
        scales=np.append(scales,1.0)
        center=np.mean(pts2,0,keepdims=True)
        scaled=(np.expand_dims(pts2-center,0)*scales[:,None,None])+center[None,:,:]
        if allow_artifacts:
            valid=np.arange(scale_num)
        else:
            valid=np.nonzero(np.all(np.all(np.logical_and(scaled >= 0., scaled < 1.0), 2), 1))[0]
        pts2=scaled[np.random.choice(valid,1)[0]]



    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min, t_max=t_min+translation_overflow, t_max+translation_overflow
        pts2+=np.asarray([np.random.uniform(-t_min[0],t_max[0]),np.random.uniform(-t_min[1],t_max[1])])[None,:]

    shape=np.asarray([w,h],np.float32).reshape([1,2])
    pts1*=shape
    pts2*=shape
    return cv2.getPerspectiveTransform(pts2.astype(np.float32),pts1.astype(np.float32))

def compute_homography_discrete(h, w, scale=True, rotation=True, translation=True, perspective=True,
                                base_scale_ratio=2.0, scale_factor_range=1, max_scale_disturb=0.0,
                                max_angle=np.pi/6, translation_overflow=0.,
                                perspective_amplitude_x=0.2,perspective_amplitude_y=0.2):
    pts1 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    pts2 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    scale_offset = 0

    if perspective:
        # pts2[:,0]+=np.random.uniform(-perspective_amplitude_x,perspective_amplitude_x,pts2[:,0].shape)
        # pts2[:,1]+=np.random.uniform(-perspective_amplitude_y,perspective_amplitude_y,pts2[:,1].shape)
        perspective_displacement = np.random.uniform(-perspective_amplitude_y, perspective_amplitude_y)
        h_displacement_left ,h_displacement_right = np.random.uniform(-perspective_amplitude_x , perspective_amplitude_x, 2)
        # print(perspective_amplitude_y,h_displacement_left ,h_displacement_right,perspective_displacement)
        pts2 += np.asarray([[h_displacement_left, perspective_displacement],
                            [h_displacement_left, -perspective_displacement],
                            [h_displacement_right, perspective_displacement],
                            [h_displacement_right, -perspective_displacement]],np.float32)

    if scale:
        # scale_ratio=np.random.choice(scale_range,1)*(1+np.random.uniform(-max_scale_disturb,max_scale_disturb))
        scale_offset=np.random.randint(-scale_factor_range, scale_factor_range + 1)
        scale_ratio=base_scale_ratio**scale_offset*(1+np.random.uniform(-max_scale_disturb,max_scale_disturb))
        pts2=(pts2-0.5)*scale_ratio+0.5

    if translation:
        t_min, t_max = np.min(pts2, axis=0) + translation_overflow, \
                       np.min(1 - pts2, axis=0) + translation_overflow
        if -t_min[0]>=t_max[0]: tx=0
        else: tx=np.random.uniform(-t_min[0],t_max[0])
        if -t_min[1]>=t_max[1]: ty=0
        else: ty=np.random.uniform(-t_min[1],t_max[1])
        pts2[:,0]+=tx
        pts2[:,1]+=ty

    if rotation:
        angle = np.random.uniform(-max_angle, max_angle)
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_m = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],np.float32)
        pts2 = np.matmul((pts2 - center), rot_m.transpose()) + center

    pts2+=np.random.uniform(0,0.05,pts2.shape)

    shape=np.asarray([w,h],np.float32).reshape([1,2])
    pts1*=shape
    pts2*=shape
    return cv2.getPerspectiveTransform(pts2.astype(np.float32),pts1.astype(np.float32)), scale_offset

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
    pts=scale_pts(pts,1.5)
    pts=rotate_pts(pts,5/180*np.pi)
    return pts

def sample_homography(h, w):
    pts1 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    pts2 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    shape=np.asarray([w,h],np.float32).reshape([1,2])
    pts1*=shape
    pts2*=shape

    fns=[nearest_identity, left_right_move, up_down_move, forward_backward_move, scale_move]
    pts2=np.random.choice(fns,p=[0.2,0.2,0.2,0.25,0.15])(pts2,h,w)
    H=cv2.getPerspectiveTransform(pts2.astype(np.float32),pts1.astype(np.float32))
    return H

def sample_homography_v2(h, w):
    pts1 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    pts2 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    shape=np.asarray([w,h],np.float32).reshape([1,2])
    pts1*=shape
    pts2*=shape

    fns=[nearest_identity, left_right_move, up_down_move, forward_backward_move, scale_move, rotate_move]
    pts2=np.random.choice(fns,p=[0.2,0.2,0.2,0.15,0.15,0.1])(pts2,h,w)
    H=cv2.getPerspectiveTransform(pts2.astype(np.float32),pts1.astype(np.float32))
    return H

def sample_homography_test(h, w):
    pts1 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    pts2 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    shape=np.asarray([w,h],np.float32).reshape([1,2])
    pts1*=shape
    pts2*=shape

    pts2=rotate_pts(pts1,90/180*np.pi)
    H=cv2.getPerspectiveTransform(pts2.astype(np.float32),pts1.astype(np.float32))
    return H

def sample_identity(h,w):
    pts1 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    pts2 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    shape=np.asarray([w,h],np.float32).reshape([1,2])
    pts1*=shape
    pts2*=shape

    pts2=nearest_identity_strictly(pts2,h,w)
    H=cv2.getPerspectiveTransform(pts2.astype(np.float32),pts1.astype(np.float32))
    return H

def compute_similar_affine(H,x,y):
    x_h=np.asarray([x,y,1])
    x_n=np.asarray([[x,y]])
    x_p=perspective_transform(x_n,H)[0] # [2,]
    z_h=np.dot(H[2,:],x_h)
    return (H[:2,:2]-x_p[:,None] @ H[2,:2][None,:])/z_h

def compute_similar_affine_batch(H,pts):
    x_h=np.concatenate([pts,np.ones([pts.shape[0],1])],1)  # n,3
    x_p=perspective_transform(pts,H)                       # [n,2]
    z_h=np.dot(H[2,:],x_h.transpose()).transpose()         # [n,2]

    return (H[:2,:2][None,:,:]-x_p[:,:, None] @ H[2,:2][None,:])/z_h[:,None,None]

def rotate_45(pts,h,w):
    pts=rotate_pts(pts,45/180*np.pi,'uniform')
    # pts=scale_pts(pts,0.5)
    pts=translate_pts(pts,h,w,0.05)
    return pts

def scale_half(pts,h,w):
    pts=scale_pts(pts,0.5)
    pts=translate_pts(pts,h,w,0.05)
    return pts

def sample_4_rotate_3_scale(h, w):
    pts1 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    pts2 = np.asarray([[0., 0.], [0., 1.], [1., 1.], [1., 0.]])
    shape=np.asarray([w,h],np.float32).reshape([1,2])
    pts1*=shape
    pts2*=shape

    fns=[nearest_identity, rotate_45, scale_half]
    pts2=np.random.choice(fns,p=[0.3,0.35,0.35])(pts2,h,w)
    H=cv2.getPerspectiveTransform(pts2.astype(np.float32),pts1.astype(np.float32))
    return H
