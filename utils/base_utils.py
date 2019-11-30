import os

import cv2
import torch

import numpy as np
import pickle

from skimage.io import imread
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.decomposition import PCA



def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, pkl_path):
    os.system('mkdir -p {}'.format(os.path.dirname(pkl_path)))
    with open(pkl_path, 'wb') as f:
        pickle.dump(data, f)

#####################depth and image###############################

def mask_zbuffer_to_pts(mask, zbuffer, K):
    ys,xs=np.nonzero(mask)
    zbuffer=zbuffer[ys, xs]
    u,v,f=K[0,2],K[1,2],K[0,0]
    depth = zbuffer / np.sqrt((xs - u + 0.5) ** 2 + (ys - v + 0.5) ** 2 + f ** 2) * f

    pts=np.asarray([xs, ys, depth], np.float32).transpose()
    pts[:,:2]*=pts[:,2:]
    return np.dot(pts,np.linalg.inv(K).transpose())

def mask_depth_to_pts(mask,depth,K,output_2d=False):
    hs,ws=np.nonzero(mask)
    pts_2d=np.asarray([ws,hs],np.float32).transpose()
    depth=depth[hs,ws]
    pts=np.asarray([ws,hs,depth],np.float32).transpose()
    pts[:,:2]*=pts[:,2:]
    if output_2d:
        return np.dot(pts,np.linalg.inv(K).transpose()), pts_2d
    else:
        return np.dot(pts,np.linalg.inv(K).transpose())

def read_render_zbuffer(dpt_pth,max_depth,min_depth):
    zbuffer=imread(dpt_pth)
    mask=zbuffer>0
    zbuffer=zbuffer.astype(np.float64)/2**16*(max_depth-min_depth)+min_depth
    return mask, zbuffer

def zbuffer_to_depth(zbuffer,K):
    u,v,f=K[0,2],K[1,2],K[0,0]
    x=np.arange(zbuffer.shape[1])
    y=np.arange(zbuffer.shape[0])
    x,y=np.meshgrid(x,y)
    x=np.reshape(x,[-1,1])
    y=np.reshape(y,[-1,1])
    depth = np.reshape(zbuffer,[-1,1])

    depth = depth / np.sqrt((x - u + 0.5) ** 2 + (y - v + 0.5) ** 2 + f ** 2) * f
    return np.reshape(depth,zbuffer.shape)

def project_points(pts,RT,K):
    pts = np.matmul(pts,RT[:,:3].transpose())+RT[:,3:].transpose()
    pts = np.matmul(pts,K.transpose())
    dpt = pts[:,2]
    pts2d = pts[:,:2]/dpt[:,None]
    return pts2d, dpt

#######################image processing#############################

def gray_repeats(img_raw):
    if len(img_raw.shape) == 2: img_raw = np.repeat(img_raw[:, :, None], 3, axis=2)
    if img_raw.shape[2] > 3: img_raw = img_raw[:, :, :3]
    return img_raw

def normalize_image(img,mask=None):
    if mask is not None: img[np.logical_not(mask.astype(np.bool))]=127
    img=(img.transpose([2,0,1]).astype(np.float32)-127.0)/128.0
    return torch.tensor(img,dtype=torch.float32)

def tensor_to_image(tensor):
    return (tensor * 128 + 127).astype(np.uint8).transpose(1,2,0)

def round_coordinates(coord,h,w):
    coord=np.round(coord).astype(np.int32)
    coord[coord[:,0]<0,0]=0
    coord[coord[:,0]>=w,0]=w-1
    coord[coord[:,1]<0,1]=0
    coord[coord[:,1]>=h,1]=h-1
    return coord

def get_img_patch(img,pt,size):
    h,w=img.shape[:2]
    x,y=pt.astype(np.int32)
    xmin=max(0,x-size)
    xmax=min(w-1,x+size)
    ymin=max(0,y-size)
    ymax=min(h-1,y+size)
    patch=np.full([size*2,size*2,3],127,np.uint8)
    patch[ymin-y+size:ymax-y+size,xmin-x+size:xmax-x+size]=img[ymin:ymax,xmin:xmax]
    return patch

def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])

    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def perspective_transform(pts, H):
    tpts = np.concatenate([pts, np.ones([pts.shape[0], 1])], 1) @ H.transpose()
    tpts = tpts[:, :2] / np.abs(tpts[:, 2:]) # todo: why only abs? this one is correct
    return tpts

def get_rot_m(angle):
    return np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], np.float32) # rn+1,3,3

def equal_hist(img):
    if len(img.shape)==3:
        img0=cv2.equalizeHist(img[:,:,0])
        img1=cv2.equalizeHist(img[:,:,1])
        img2=cv2.equalizeHist(img[:,:,2])
        img=np.concatenate([img0[...,None],img1[...,None],img2[...,None]],2)
    else:
        img=cv2.equalizeHist(img)
    return img


def draw_correspondence(img0, img1, kps0, kps1, matches, colors=None):
    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    h = max(h0, h1)
    w = w0 + w1
    out_img = np.zeros([h, w, 3], np.uint8)
    out_img[:h0, :w0] = img0
    out_img[:h1, w0:] = img1

    for pt in kps0:
        pt = np.round(pt).astype(np.int32)
        cv2.circle(out_img, tuple(pt), 1, (255, 0, 0), -1)

    for pt in kps1:
        pt = np.round(pt).astype(np.int32)
        pt = pt.copy()
        pt[0] += w0
        cv2.circle(out_img, tuple(pt), 1, (255, 0, 0), -1)

    for mi,m in enumerate(matches):
        pt = np.round(kps0[m[0]]).astype(np.int32)
        pr_pt = np.round(kps1[m[1]]).astype(np.int32)
        pr_pt[0] += w0
        if colors is None:
            cv2.line(out_img, tuple(pt), tuple(pr_pt), (0, 255, 0), 1)
        elif type(colors)==list:
            color=(int(c) for c in colors[mi])
            cv2.line(out_img, tuple(pt), tuple(pr_pt), color, 1)
        else:
            color=(int(c) for c in colors)
            cv2.line(out_img, tuple(pt), tuple(pr_pt), color, 1)

    return out_img

def draw_keypoints(img,kps,colors=None,radius=2):
    out_img=img.copy()
    for pi, pt in enumerate(kps):
        pt = np.round(pt).astype(np.int32)
        if colors is not None:
            color=[int(c) for c in colors[pi]]
            cv2.circle(out_img, tuple(pt), radius, color, -1)
        else:
            cv2.circle(out_img, tuple(pt), radius, (255,0,0), -1)
    return out_img