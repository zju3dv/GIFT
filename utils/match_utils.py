from pyflann.index import FLANN
import cv2

from utils.base_utils import perspective_transform
from utils.extend_utils.extend_utils_fn import find_nearest_point_idx
import numpy as np


def nn_match(feats_que, feats_ref, use_cuda=True):
    if use_cuda:
        idxs = find_nearest_point_idx(feats_ref, feats_que)
    else:
        flann = FLANN()
        idxs, dists = flann.nn(feats_ref, feats_que, 1, algorithm='linear', trees=4)
    return idxs


def compute_match_pairs(feats0, kps0, feats1, kps1, H):
    # 0 to 1
    idxs = nn_match(feats0, feats1, True)
    pr01 = kps1[idxs]
    gt01 = perspective_transform(kps0, H)

    idxs = nn_match(feats1, feats0, True)
    pr10 = kps0[idxs]
    gt10 = perspective_transform(kps1, np.linalg.inv(H))

    return pr01, gt01, pr10, gt10


def keep_valid_kps_feats(kps, feats, H, h, w):
    n, _ = kps.shape
    warp_kps = perspective_transform(kps, H)
    mask = (warp_kps[:, 0] >= 0) & (warp_kps[:, 0] < w) & \
           (warp_kps[:, 1] >= 0) & (warp_kps[:, 1] < h)
    return kps[mask], feats[mask]


def compute_angle(rotation_diff):
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.) / 2.))
    return angular_distance
