import random
import time
import numpy as np
import os
import cv2
from skimage.io import imread, imsave

from utils.base_utils import perspective_transform, read_pickle, save_pickle
from utils.config import cfg


def worker_init_fn(worker_id):
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2 ** 16))))


def scale_transform_img(img, min_ratio=0.15, max_ratio=0.25, base_ratio=2, flip=True):
    h, w = img.shape[0], img.shape[1]
    pts0 = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

    scale_ratio = base_ratio ** (-np.random.uniform(min_ratio, max_ratio))
    if np.random.random() < 0.5 and flip:
        scale_ratio = 1.0 / scale_ratio
    center = np.mean(pts0, 0, keepdims=True)
    pts1 = (pts0 - center) * scale_ratio + center
    if scale_ratio > 1:
        min_pt = np.min(pts1, 0)  # <0
        max_pt = np.max(pts1, 0)  # >w,h
        min_w, min_h = -(max_pt - np.asarray([w, h]))
        max_w, max_h = -min_pt
    else:
        min_pt = np.min(pts1, 0)  # >0
        max_pt = np.max(pts1, 0)  # <w,h
        min_w, min_h = -min_pt
        max_w, max_h = np.asarray([w, h]) - max_pt

    offset_h = np.random.uniform(min_h, max_h)
    offset_w = np.random.uniform(min_w, max_w)
    pts1 += np.asarray([[offset_w, offset_h]], np.float32)

    th, tw = h, w  # int(h * scale_ratio), int(w * scale_ratio)
    H = cv2.getPerspectiveTransform(pts0.astype(np.float32), pts1.astype(np.float32))

    img1 = cv2.warpPerspective(img, H, (tw, th), flags=cv2.INTER_LINEAR)
    return img1, H


def perspective_transform_img(img, perspective_type='lr', min_ratio=0.05, max_ratio=0.1):
    h, w = img.shape[0], img.shape[1]
    pts0 = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    pts1 = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], np.float32)

    # left right
    if perspective_type == 'lr':
        val = h * np.random.uniform(min_ratio, max_ratio)
        if np.random.random() < 0.5: val *= -1
        pts1[0, 1] -= val
        pts1[1, 1] += val
        pts1[2, 1] -= val
        pts1[3, 1] += val

        val = h * np.random.uniform(min_ratio, max_ratio)
        pts1[0, 0] += val
        pts1[1, 0] -= val
        pts1[2, 0] -= val
        pts1[3, 0] += val
    else:  # 'ud'
        val = w * np.random.uniform(min_ratio, max_ratio)
        if np.random.random() < 0.5: val *= -1
        pts1[0, 0] += val
        pts1[1, 0] -= val
        pts1[2, 0] += val
        pts1[3, 0] -= val

        val = h * np.random.uniform(min_ratio, max_ratio)
        pts1[0, 1] += val
        pts1[1, 1] += val
        pts1[2, 1] -= val
        pts1[3, 1] -= val

    pts1 = pts1 - np.min(pts1, 0, keepdims=True)
    tw, th = np.max(pts1, 0)
    H = cv2.getPerspectiveTransform(pts0.astype(np.float32), pts1.astype(np.float32))
    img1 = cv2.warpPerspective(img, H, (tw, th), flags=cv2.INTER_LINEAR)
    return img1, H


def rotate_transform_img(img, min_angle=0, max_angle=360, random_flip=False):
    h, w = img.shape[0], img.shape[1]

    pts0 = np.asarray([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    center = np.mean(pts0, 0, keepdims=True)
    theta = np.random.uniform(min_angle / 180 * np.pi, max_angle / 180 * np.pi)
    if random_flip and np.random.random() < 0.5: theta = -theta
    R = np.asarray([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]], np.float32)
    pts1 = (pts0 - center) @ R + center
    H = cv2.getPerspectiveTransform(pts0.astype(np.float32), pts1.astype(np.float32))

    img1 = cv2.warpPerspective(img, H, (w, h))
    return img1, H


class CorrespondenceDatabase:
    @staticmethod
    def get_SUN2012_image_paths():
        img_dir = os.path.join(cfg.data_dir, 'SUN2012Images', 'JPEGImages')
        img_pths = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)]
        return img_pths

    @staticmethod
    def get_COCO_image_paths():
        img_dir = os.path.join(cfg.data_dir, 'coco', 'train2014')
        img_pths = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)]
        img_dir = os.path.join(cfg.data_dir, 'coco', 'val2014')
        img_pths += [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)]
        return img_pths

    @staticmethod
    def generate_homography_database(img_list):
        return [{'type': 'homography', 'img_pth': img_pth} for img_pth in img_list]

    @staticmethod
    def get_hpatch_sequence_database(name='resize', max_size=480):
        """
        Get hpatches_resize if it exists,
            else generate one
        """

        def resize_and_save(pth_in, max_size, pth_out):
            img = imread(pth_in)
            h, w = img.shape[:2]
            ratio = max_size / max(h, w)
            h, w = int(h * ratio), int(w * ratio)
            img = cv2.resize(img, (w, h))
            imsave(pth_out, img)
            return ratio

        root_dir = os.path.join('data', 'hpatches_sequence')
        output_dir = os.path.join('data', 'hpatches_{}'.format(name))
        pkl_file = os.path.join(output_dir, 'info.pkl')
        if os.path.exists(pkl_file):
            return read_pickle(pkl_file)

        if not os.path.exists(output_dir): os.mkdir(output_dir)
        illumination_dataset = []
        viewpoint_dataset = []
        for dir in os.listdir(root_dir):
            if not os.path.exists(os.path.join(output_dir, dir)):
                os.mkdir(os.path.join(output_dir, dir))

            img_pattern = os.path.join(root_dir, dir, '{}.ppm')
            hmg_pattern = os.path.join(root_dir, dir, 'H_1_{}')
            omg_pattern = os.path.join(output_dir, dir, '{}.png')

            ratio0 = resize_and_save(img_pattern.format(1), max_size, omg_pattern.format(1))
            # resize image
            for k in range(2, 7):
                ratio1 = resize_and_save(img_pattern.format(k), max_size, omg_pattern.format(k))
                H = np.loadtxt(hmg_pattern.format(k))
                H = np.matmul(np.diag([ratio1, ratio1, 1.0]), np.matmul(H, np.diag([1 / ratio0, 1 / ratio0, 1.0])))
                data = {'type': 'hpatch',
                        'img0_pth': omg_pattern.format(1),
                        'img1_pth': omg_pattern.format(k),
                        'H': H}
                if dir.startswith('v'):
                    viewpoint_dataset.append(data)
                if dir.startswith('i'):
                    illumination_dataset.append(data)

        save_pickle([illumination_dataset, viewpoint_dataset], pkl_file)
        return illumination_dataset, viewpoint_dataset

    @staticmethod
    def add_homography_background(img, H):
        img_dir = os.path.join(cfg.data_dir, 'SUN2012Images', 'JPEGImages')
        background_pths = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)]
        bpth = background_pths[np.random.randint(0, len(background_pths))]

        h, w, _ = img.shape
        bimg = cv2.resize(imread(bpth), (w, h))
        if len(bimg.shape) == 2: bimg = np.repeat(bimg[:, :, None], 3, axis=2)
        if bimg.shape[2] > 3: bimg = bimg[:, :, :3]
        msk_tgt = cv2.warpPerspective(np.ones([h, w], np.uint8), H, (w, h), flags=cv2.INTER_NEAREST).astype(np.bool)
        img[np.logical_not(msk_tgt)] = bimg[np.logical_not(msk_tgt)]
        return img

    @staticmethod
    def make_hpatch_transform_database_combine(in_dataset, output_name, transform, add_background):
        hpatch_transform_root_dir = os.path.join(cfg.data_dir, 'hpatches_{}'.format(output_name))
        if not os.path.exists(hpatch_transform_root_dir): os.mkdir(hpatch_transform_root_dir)

        hpatch_transform_pkl = os.path.join(hpatch_transform_root_dir, 'info.pkl')
        if os.path.exists(hpatch_transform_pkl):
            return read_pickle(hpatch_transform_pkl)

        print('begin making {} dataset'.format(output_name))
        dataset = []
        random.shuffle(in_dataset)
        for in_data in in_dataset:
            pth0 = in_data['img0_pth']
            pth1 = in_data['img1_pth']
            in_dir = pth1.split('/')[-2]
            in_id = pth1.split('/')[-1].split('.')[-2]
            output_dir = os.path.join(hpatch_transform_root_dir, in_dir)
            if not os.path.exists(output_dir): os.mkdir(output_dir)

            img1 = imread(pth1)
            img1, H = transform(img1)

            if add_background: img1 = CorrespondenceDatabase.add_homography_background(img1, H)
            img1_pth = os.path.join(output_dir, '{}.png'.format(in_id))
            imsave(img1_pth, img1)
            data = {'type': 'hpatch',
                    'img0_pth': pth0,
                    'img1_pth': img1_pth,
                    'H': H @ in_data['H']}
            dataset.append(data)

        save_pickle(dataset, hpatch_transform_pkl)
        return dataset

    @staticmethod
    def warp_flow(H, norm_flow):
        h, w, _ = norm_flow.shape
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        pts = np.concatenate([xs.reshape([-1, 1]), ys.reshape([-1, 1])], 1)

        norm_flow = norm_flow.reshape([h * w, 2])
        nan_mask = np.sum(np.isnan(norm_flow), 1) > 0
        norm_flow[nan_mask] = (0, 0)
        norm_flow *= np.asarray((w, h))

        norm_flow += pts
        outside_mask = np.logical_or(np.logical_or(norm_flow[:, 0] < 0, norm_flow[:, 0] >= w),
                                     np.logical_or(norm_flow[:, 1] < 0, norm_flow[:, 1] >= h))
        norm_flow = perspective_transform(norm_flow, H)
        norm_flow -= pts
        norm_flow /= np.asarray((w, h))
        norm_flow[nan_mask] = np.nan
        norm_flow[outside_mask] = np.nan

        norm_flow = norm_flow.reshape([h, w, 2])
        return norm_flow

    @staticmethod
    def make_sun3d_all_dataset(img_num=499):
        dataset = []
        for k in range(img_num):
            data = {
                'img0_pth': 'data/sun3d_all/img0/{}.png'.format(k),
                'img1_pth': 'data/sun3d_all/img1/{}.png'.format(k),
                'flow_pth': 'data/sun3d_all/flow_01/{}.npy'.format(k),
                'type': 'sun3d_all'
            }
            dataset.append(data)
        return dataset

    @staticmethod
    def make_transformed_sun3d_dataset(dataset_in, dataset_out_name, warp_fn, add_background=True):
        dir_out = os.path.join('data', dataset_out_name)
        if not os.path.exists(dir_out): os.mkdir(dir_out)
        info_pkl = os.path.join(dir_out, 'info.pkl')
        if os.path.exists(info_pkl):
            return read_pickle(info_pkl)

        if not os.path.exists(os.path.join(dir_out, 'img1')):
            os.mkdir(os.path.join(dir_out, 'img1'))
        if not os.path.exists(os.path.join(dir_out, 'flow_01')):
            os.mkdir(os.path.join(dir_out, 'flow_01'))

        print('begin making {} dataset'.format(dataset_out_name))

        dataset_out = []
        for data_in in dataset_in:
            image_id = data_in['img0_pth'].split('/')[-1].split('.')[0]
            img1 = imread(data_in['img1_pth']).astype(np.uint8)
            flow = np.load(data_in['flow_pth']).astype(np.float32).transpose([1, 2, 0])

            img1, H = warp_fn(img1)
            if add_background:
                img1 = CorrespondenceDatabase.add_homography_background(img1, H)

            flow_warp = CorrespondenceDatabase.warp_flow(H, flow)

            data_out = {
                'img0_pth': data_in['img0_pth'],
                'img1_pth': os.path.join(dir_out, 'img1', '{}.png'.format(image_id)),
                'flow_pth': os.path.join(dir_out, 'flow_01', '{}.npy'.format(image_id)),
                'type': 'sun3d_all'
            }
            imsave(data_out['img1_pth'], img1)
            np.save(data_out['flow_pth'], flow_warp.astype(np.float32))
            dataset_out.append(data_out)

        save_pickle(dataset_out, info_pkl)
        return dataset_out

    def __getattr__(self, item):
        if item == 'coco_set':
            self.coco_set = self.generate_homography_database(self.get_COCO_image_paths())
            print('coco_len {}'.format(len(self.coco_set)))
            return self.coco_set
        if item== 'hi_set':
            self.hi_set, self.hv_set = self.get_hpatch_sequence_database()
            return self.hi_set
        if item== 'hv_set':
            self.hi_set, self.hv_set = self.get_hpatch_sequence_database()
            return self.hv_set
        if item == 'single_set':
            self.single_set = self.generate_homography_database(self.get_COCO_image_paths()[:1]*10000)
            print('single_len {}'.format(len(self.coco_set)))
            return self.single_set
        else:
            super(CorrespondenceDatabase, self).__getattribute__(item)

    def __init__(self):
        pass
