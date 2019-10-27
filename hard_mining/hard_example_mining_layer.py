import hard_mining.hard_example_mining as hem
import torch

def hard_example_mining_layer(feats,feats_ref,gt_loc,interval=32,thresh=8,cross_batch=False):
    '''

    :param feats:       b,f,h,w
    :param feats_ref:   b,n,f
    :param gt_loc:      b,n,2
    :param interval:
    :param thresh:
    :param cross_batch:
    :return: b,n,3
    '''
    b,n,f=feats_ref.shape
    feats=feats.permute(0,2,3,1).float().contiguous()
    feats_ref=feats_ref.float().contiguous()
    gt_loc=gt_loc.float().contiguous()
    hard_idxs=torch.zeros([b,n,3],dtype=torch.int32,device=feats.device).contiguous()
    hem.hard_example_mining(feats,feats_ref,gt_loc,hard_idxs,interval,thresh*thresh,cross_batch)
    return hard_idxs

def knn_hard_example_mining_layer(feats,feats_ref,gt_loc,k,interval=32,thresh=8):
    '''

    :param feats:       b,f,h,w
    :param feats_ref:   b,n,f
    :param gt_loc:      b,n,2
    :param interval:
    :param thresh:
    :return: b,n,3
    '''
    b,n,f=feats_ref.shape
    feats=feats.permute(0,2,3,1).float().contiguous()
    feats_ref=feats_ref.float().contiguous()
    gt_loc=gt_loc.float().contiguous()
    hard_idxs=torch.zeros([b,n,k,3],dtype=torch.int32,device=feats.device).contiguous()
    hem.knn_hard_example_mining(feats,feats_ref,gt_loc,hard_idxs,interval,thresh*thresh)
    return hard_idxs

def semi_hard_example_mining_layer(feats,feats_dist,feats_ref,gt_loc,interval=32,thresh=8,margin=0.0,cross_batch=False):
    '''

    :param feats:       b,f,h,w
    :param feats_dist:  b,n
    :param feats_ref:   b,n,f
    :param gt_loc:      b,n,2
    :param interval:
    :param thresh:
    :param margin:
    :param cross_batch:
    :return: b,n,3
    '''
    b,n,f=feats_ref.shape
    feats=feats.permute(0,2,3,1).float().contiguous()
    feats_dist=feats_dist.float().contiguous()
    feats_ref=feats_ref.float().contiguous()
    gt_loc=gt_loc.float().contiguous()
    hard_idxs=torch.zeros([b,n,3],dtype=torch.int32,device=feats.device).contiguous()
    hem.semi_hard_example_mining(feats,feats_dist,feats_ref,gt_loc,hard_idxs,interval,thresh*thresh,margin,cross_batch)
    return hard_idxs

def knn_semi_hard_example_mining_layer(feats,feats_dist,feats_ref,gt_loc,k,interval=32,thresh=8,margin=1.0):
    '''

    :param feats:       b,f,h,w
    :param feats_dist:  b,n
    :param feats_ref:   b,n,f
    :param gt_loc:      b,n,2
    :param interval:
    :param thresh:
    :param margin:
    :return: b,n,3
    '''
    b,n,f=feats_ref.shape
    feats=feats.permute(0,2,3,1).float().contiguous()
    feats_dist=feats_dist.float().contiguous()
    feats_ref=feats_ref.float().contiguous()
    gt_loc=gt_loc.float().contiguous()
    hard_idxs=torch.zeros([b,n,k,3],dtype=torch.int32,device=feats.device).contiguous()
    hem.knn_semi_hard_example_mining(feats,feats_dist,feats_ref,gt_loc,hard_idxs,interval,thresh*thresh,margin)
    return hard_idxs

def knn_cuda(ref_feats,que_feats,k):
    que_nb=que_feats.shape[0]
    knn_idxs=torch.zeros([k,que_nb],device=que_feats.device,dtype=torch.int32)
    knn_dist=torch.zeros([k,que_nb],device=que_feats.device,dtype=torch.float32)
    ref_feats=ref_feats.transpose(1,0).contiguous()
    que_feats=que_feats.transpose(1,0).contiguous()

    hem.knn_search(ref_feats,que_feats,knn_dist,knn_idxs)

    return knn_idxs.transpose(1,0), knn_dist.transpose(1,0)

def knn_cuda_permute(ref_feats,que_feats,k):
    que_nb=que_feats.shape[1]
    knn_idxs=torch.zeros([k,que_nb],device=que_feats.device,dtype=torch.int32)
    knn_dist=torch.zeros([k,que_nb],device=que_feats.device,dtype=torch.float32)
    ref_feats=ref_feats.contiguous()
    que_feats=que_feats.contiguous()

    hem.knn_search(ref_feats,que_feats,knn_dist,knn_idxs)

    return knn_idxs, knn_dist