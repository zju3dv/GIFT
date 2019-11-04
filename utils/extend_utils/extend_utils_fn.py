import sys
import time

sys.path.append('.')
from utils.extend_utils._extend_utils import lib, ffi
import numpy as np


def find_nearest_point_idx(ref_pts,que_pts):
    '''
    for every point in que_pts, find the nearest point in ref_pts
    :param ref_pts:  pn1, f
    :param que_pts:  pn2, f
    :return:  idxs pn2
    '''
    assert(ref_pts.shape[1]==que_pts.shape[1])
    pn1=ref_pts.shape[0]
    pn2=que_pts.shape[0]
    dim=ref_pts.shape[1]

    ref_pts=np.ascontiguousarray(ref_pts[None,:,:],np.float32)
    que_pts=np.ascontiguousarray(que_pts[None,:,:],np.float32)
    idxs=np.zeros([1,pn2],np.int32)

    ref_pts_ptr=ffi.cast('float *',ref_pts.ctypes.data)
    que_pts_ptr=ffi.cast('float *',que_pts.ctypes.data)
    idxs_ptr=ffi.cast('int *',idxs.ctypes.data)
    lib.findNearestPointIdxLauncher(ref_pts_ptr,que_pts_ptr,idxs_ptr,1,pn1,pn2,dim,0)

    return idxs[0]

def find_nearest_point_idx_v2(ref_pts,que_pts):
    '''
    for every point in que_pts, find the nearest point in ref_pts
    :param ref_pts:  pn1, f
    :param que_pts:  pn2, f
    :return:  idxs pn2
    '''
    dist=find_feats_distance(ref_pts,que_pts)
    return np.argmin(dist,0)

def find_nearest_set_point_idx(ref_pts,que_pts):
    '''
    for every point in que_pts, find the nearest point in ref_pts
    :param ref_pts:  pn1, k, f
    :param que_pts:  pn2, k, f
    :return:  idxs pn2
    '''
    assert(ref_pts.shape[1]==que_pts.shape[1] and ref_pts.shape[2]==que_pts.shape[2])
    pn1=ref_pts.shape[0]
    pn2=que_pts.shape[0]
    dim=ref_pts.shape[2]
    k=ref_pts.shape[1]

    ref_pts=np.ascontiguousarray(ref_pts[None,:,:,:],np.float32)
    que_pts=np.ascontiguousarray(que_pts[None,:,:,:],np.float32)
    idxs=np.zeros([1,pn2],np.int32)

    ref_pts_ptr=ffi.cast('float *',ref_pts.ctypes.data)
    que_pts_ptr=ffi.cast('float *',que_pts.ctypes.data)
    idxs_ptr=ffi.cast('int *',idxs.ctypes.data)
    lib.findNearestSetPointIdxLauncher(ref_pts_ptr,que_pts_ptr,idxs_ptr,1,pn1,pn2,k,dim,0)

    return idxs[0]

def find_nearest_set_point_idx_v2(ref_pts,que_pts):
    '''
    for every point in que_pts, find the nearest point in ref_pts
    :param ref_pts:  pn1, k, f
    :param que_pts:  pn2, k, f
    :return:  idxs pn2
    '''
    assert(ref_pts.shape[1]==que_pts.shape[1] and ref_pts.shape[2]==que_pts.shape[2])
    pn1=ref_pts.shape[0]
    pn2=que_pts.shape[0]
    dim=ref_pts.shape[2]
    k=ref_pts.shape[1]

    ref_pts=np.ascontiguousarray(ref_pts[None,:,:,:],np.float32)
    que_pts=np.ascontiguousarray(que_pts[None,:,:,:],np.float32)
    dist=np.zeros([1,pn1,pn2],np.float32)
    dist=np.ascontiguousarray(dist,np.float32)

    ref_pts_ptr=ffi.cast('float *',ref_pts.ctypes.data)
    que_pts_ptr=ffi.cast('float *',que_pts.ctypes.data)
    idxs_ptr=ffi.cast('float *',dist.ctypes.data)
    lib.findSetDistanceLauncher(ref_pts_ptr,que_pts_ptr,idxs_ptr,1,pn1,pn2,k,dim)

    idxs=np.argmin(dist[0],0)
    return idxs

def find_nearest_set_point_idx_v3(ref_pts,que_pts):
    '''
    for every point in que_pts, find the nearest point in ref_pts
    :param ref_pts:  pn1, k, f
    :param que_pts:  pn2, k, f
    :return:  idxs pn2
    '''
    assert(ref_pts.shape[1]==que_pts.shape[1] and ref_pts.shape[2]==que_pts.shape[2])
    pn1=ref_pts.shape[0]
    pn2=que_pts.shape[0]
    dim=ref_pts.shape[2]
    k=ref_pts.shape[1]

    ref_pts=np.ascontiguousarray(ref_pts[None,:,:,:],np.float32)
    que_pts=np.ascontiguousarray(que_pts[None,:,:,:],np.float32)
    dist=np.zeros([1,pn1,pn2],np.float32)
    dist=np.ascontiguousarray(dist,np.float32)

    ref_pts_ptr=ffi.cast('float *',ref_pts.ctypes.data)
    que_pts_ptr=ffi.cast('float *',que_pts.ctypes.data)
    idxs_ptr=ffi.cast('float *',dist.ctypes.data)
    lib.findSetDistanceLauncher(ref_pts_ptr,que_pts_ptr,idxs_ptr,1,pn1,pn2,k,dim)

    idxs0=np.argmin(dist[0],0)
    idxs1=np.argmin(dist[0],1)
    return idxs0, idxs1

def find_feats_distance(ref_pts,que_pts):
    assert(ref_pts.shape[1]==que_pts.shape[1])
    pn1=ref_pts.shape[0]
    pn2=que_pts.shape[0]
    dim=ref_pts.shape[1]

    ref_pts=np.ascontiguousarray(ref_pts[None,:,:],np.float32)
    que_pts=np.ascontiguousarray(que_pts[None,:,:],np.float32)
    dist=np.zeros([1,pn1,pn2],np.float32)

    ref_pts_ptr=ffi.cast('float *',ref_pts.ctypes.data)
    que_pts_ptr=ffi.cast('float *',que_pts.ctypes.data)
    dist_ptr=ffi.cast('float *',dist.ctypes.data)
    lib.findFeatsDistanceLauncher(ref_pts_ptr,que_pts_ptr,dist_ptr,1,pn1,pn2,dim)

    return dist[0]

def find_group_distance(ref_pts,que_pts,max_sn,max_rn,inter_distance=False):
    '''

    :param ref_pts:  n1,sn,rn,f
    :param que_pts:  n2,sn,rn,f
    :param max_sn:
    :param max_rn:
    :return:
    '''
    n1,sn,rn,f=ref_pts.shape
    n2,sn,rn,f=que_pts.shape
    dists=[]
    for si in range(-max_sn,max_sn+1):
        for ri in range(-max_rn,max_rn+1):
            if si<0:
                feats_ref=ref_pts[:,-si:]
                feats_que=que_pts[:,:sn+si]
            elif si>0:
                feats_ref=ref_pts[:,:sn-si]
                feats_que=que_pts[:,si:]
            else:
                feats_ref=ref_pts
                feats_que=que_pts

            if ri<0:
                feats_ref=feats_ref[:,:,-ri:]
                feats_que=feats_que[:,:,:rn+ri]
            elif ri>0:
                feats_ref=feats_ref[:,:,:rn-ri]
                feats_que=feats_que[:,:,ri:]
            else:
                feats_ref=feats_ref
                feats_que=feats_que

            search_size=(rn-abs(ri))*(sn-abs(si))
            feats_que=feats_que.reshape([n2,-1])
            feats_ref=feats_ref.reshape([n1,-1])
            dist_cur=find_feats_distance(feats_ref,feats_que)/search_size
            dists.append(dist_cur[:,:,None])
    if inter_distance:
        return np.concatenate(dists,2)
    else:
        return np.min(np.concatenate(dists,2),2)

def find_group_min_idxs(ref_pts,que_pts,max_sn,max_rn):
    dist=find_group_distance(ref_pts,que_pts,max_sn,max_rn)
    return np.argmin(dist,0), np.argmin(dist,1)

def count_neighborhood(pts,radius):
    pn,dim=pts.shape

    pts=np.ascontiguousarray(pts,np.float32)
    count=np.zeros([pn],np.int32)

    pts_ptr=ffi.cast('float *',pts.ctypes.data)
    count_ptr=ffi.cast('int *',count.ctypes.data)
    lib.countNeighborhoodLauncher(pts_ptr,count_ptr,radius,pn,dim)

    return count
