import torch

from hard_mining.hard_example_mining_layer import semi_hard_example_mining_layer, sq_semi_hard_example_mining_layer

def scale_rotate_offset_dist(feats0,feats1,scale_offset,rotate_offset,max_sn,max_rn):
    '''

    :param feats0:          b,n,ssn,srn,f
    :param feats1:          b,n,ssn,srn,f
    :param scale_offset:    b,n
    :param rotate_offset:   b,n
    :param max_sn:
    :param max_rn:
    :return: dist: b,n
        scale_offset<0: img0 0:ssn+scale_offset corresponds to img1 -scale_offset:ssn
        scale_offset>0: img0 scale_offset:ssn corresponds to img1 0:ssn-scale_offset
        rotate_offset<0: img0 0:srn+rotate_offset corresponds to img1 -rotate_offset:srn
        rotate_offset>0: img0 rotate_offset:srn corresponds to img1 0:srn-rotate_offset
    '''
    b,n,ssn,srn,f=feats0.shape
    dist=torch.zeros([b,n],dtype=torch.float32,device=feats0.device)
    for si in range(-max_sn,max_sn+1):
        for ri in range(-max_rn,max_rn+1):
            mask=(scale_offset==si) & (rotate_offset==ri)
            if torch.sum(mask)==0: continue
            cfeats0=feats0[mask] # n, ssn, srn, f
            cfeats1=feats1[mask] # n, ssn, srn, f
            if si<0:
                cfeats0=cfeats0[:,:ssn+si]
                cfeats1=cfeats1[:,-si:]
            elif si>0:
                cfeats0=cfeats0[:,si:]
                cfeats1=cfeats1[:,:ssn-si]
            else:
                cfeats0=cfeats0
                cfeats1=cfeats1

            if ri<0:
                cfeats0=cfeats0[:,:,:srn+ri]
                cfeats1=cfeats1[:,:,-ri:]
            elif ri>0:
                cfeats0=cfeats0[:,:,ri:]
                cfeats1=cfeats1[:,:,:srn-ri]
            else:
                cfeats0=cfeats0
                cfeats1=cfeats1

            cdist=torch.norm(cfeats0-cfeats1,2,3) # n,
            n,csn,crn=cdist.shape
            cdist=torch.mean(cdist.reshape(n,csn*crn),1)
            dist[mask]=dist[mask]+cdist

    return dist

def sample_semi_hard_feature(feats, dis_pos, feats_pos, pix_pos, interval, thresh, margin, loss_square=False):
    '''
    :param feats:      b,f,h,w
    :param dis_pos:    b,n
    :param feats_pos:  b,n,f
    :param pix_pos:    b,n,2
    :param interval:
    :param thresh:
    :param margin:
    :param loss_square:
    :return:
    '''
    with torch.no_grad():
        if loss_square:
            pix_neg = sq_semi_hard_example_mining_layer(feats, dis_pos, feats_pos, pix_pos, interval, thresh, margin).long()  # b,n,3
        else:
            pix_neg=semi_hard_example_mining_layer(feats, dis_pos, feats_pos, pix_pos, interval, thresh, margin).long() # b,n,3
    feats=feats.permute(0,2,3,1)
    feats_neg=feats[pix_neg[:,:,0],pix_neg[:,:,2],pix_neg[:,:,1]]   # b,n,f
    return feats_neg

def clamp_loss_all(loss):
    """
    max(loss, 0) with hard-negative mining
    """
    loss = torch.clamp(loss, min=0.0) # b,n
    num = torch.sum(loss>1e-6).float()
    total_num=1
    for k in loss.shape: total_num*=k
    return torch.sum(loss)/(num+1), num/total_num