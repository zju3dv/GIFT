import os

import torch
from scipy import stats

from skimage.io import imread
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter

from dataset.homography import compute_approximated_affine_batch, sample_homography
from dataset.photometric_augmentation import *
from dataset.transformer import TransformerCV
from utils.augmentation_utils import add_noise, gaussian_blur, jpeg_compress
from utils.base_utils import gray_repeats, get_rot_m


class CorrespondenceDataset(Dataset):
    def __init__(self, train_cfg, database):
        super(CorrespondenceDataset,self).__init__()
        self.base_scale=train_cfg['sample_scale_inter']
        self.base_rotate=train_cfg['sample_rotate_inter']/180*np.pi
        self.database=database
        self.transformer=TransformerCV(train_cfg)

        self.args=train_cfg['augmentation_args']
        self.jitter = ColorJitter(self.args['brightness'], self.args['contrast'], self.args['saturation'], self.args['hue'])

        img_dir = os.path.join('data', 'SUN2012Images', 'JPEGImages')
        self.background_pths = [os.path.join(img_dir, fn) for fn in os.listdir(img_dir)]

        self.name2func = {
            'jpeg': lambda img_in: jpeg_compress(img_in, self.args['jpeg_low'], self.args['jpeg_high']),
            'blur': lambda img_in: gaussian_blur(img_in, self.args['blur_range']),
            'jitter': lambda img_in: np.asarray(self.jitter(Image.fromarray(img_in))),
            'noise': lambda img_in: add_noise(img_in),
            'none': lambda img_in: img_in,
            'sp_gaussian_noise': lambda img_in: additive_gaussian_noise(img_in,self.args['sp_gaussian_range']),
            'sp_speckle_noise': lambda img_in: additive_speckle_noise(img_in,self.args['sp_speckle_prob_range']),
            'sp_additive_shade': lambda img_in: additive_shade(img_in,self.args['sp_nb_ellipse'],self.
                                                               args['sp_transparency_range'],self.args['sp_kernel_size_range']),
            'sp_motion_blur': lambda img_in: motion_blur(img_in,self.args['sp_max_kernel_size']),
            'resize_blur': lambda img_in: resize_blur(img_in, self.args['resize_blur_min_ratio'])
        }

    def __len__(self):
        return len(self.database)

    def __getitem__(self, index):
        return self.decode(self.database[index])

    def decode_homography(self, data):
        img_raw = imread(data['img_pth'])
        th, tw = self.args['h'],self.args['w']
        img_raw = gray_repeats(img_raw)

        if np.random.random()<0.8:
            img_raw,_,_=self.get_image_region(img_raw,th*np.random.uniform(0.8,1.2),tw*np.random.uniform(0.8,1.2))

        H = self.generate_homography()

        # warp image
        img0=cv2.resize(img_raw,(tw,th),interpolation=cv2.INTER_LINEAR)          # [h,w,3]
        img1=cv2.warpPerspective(img0,H,(tw,th),flags=cv2.INTER_LINEAR)          # [h,w,3]

        # add random background
        if self.args["add_background"]:
            img1=self.add_homography_background(img1, data['img_pth'], H)
        else:
            # black is 127 here because it should be consistent with image_normalization
            img1=self.add_black_background(img1,H)

        pix_pos0, pix_pos1 = self.sample_ground_truth(img0,H)
        scale_offset, rotate_offset = self.compute_scale_rotate_offset(H,pix_pos0)
        return img0, img1, pix_pos0, pix_pos1, H, scale_offset, rotate_offset

    def decode_gl3d(self, data):
        img0=imread(data['img0_pth'])
        img1=imread(data['img1_pth'])
        img0=cv2.resize(img0,(self.args['w'],self.args['h']),interpolation=cv2.INTER_LINEAR)
        img1=cv2.resize(img1,(self.args['w'],self.args['h']),interpolation=cv2.INTER_LINEAR)

        pix_pos0=data['pix_pos0']
        pix_pos1=data['pix_pos1']

        pix_pos0[:,0]*=img0.shape[1]
        pix_pos0[:,1]*=img0.shape[0]
        pix_pos1[:,0]*=img1.shape[1]
        pix_pos1[:,1]*=img1.shape[0]

        idxs=self.sample_indices(self.args['sample_num'], pix_pos0.shape[0])

        pix_pos0=pix_pos0[idxs]
        pix_pos1=pix_pos1[idxs]

        H=np.zeros([3,3],dtype=np.float32)
        scale_offset=np.zeros(pix_pos0.shape[0],dtype=np.float32)
        rotate_offset=np.zeros(pix_pos0.shape[0],dtype=np.float32)
        return img0, img1, pix_pos0, pix_pos1, H, scale_offset, rotate_offset

    def decode(self, data):
        if data['type']=='homography':
            img0, img1, pix_pos0, pix_pos1, H, scale_offset, rotate_offset = self.decode_homography(data)
        elif data['type']=='gl3d':
            img0, img1, pix_pos0, pix_pos1, H, scale_offset, rotate_offset = self.decode_gl3d(data)
        else:
            raise NotImplementedError

        if self.args['augment']:
            img0 = self.augment(img0)
            img1 = self.augment(img1)

        results0=self.transformer.transform(img0, pix_pos0, output_grid=True)
        results1=self.transformer.transform(img1, pix_pos1, output_grid=True)

        img_list0,pts_list0,grid_list0=self.transformer.postprocess_transformed_imgs(results0, True)
        img_list1,pts_list1,grid_list1=self.transformer.postprocess_transformed_imgs(results1, True)

        # note!!! since the grid is offset by a random grid_begins, so here we substract it
        pix_pos0=torch.tensor(pix_pos0-results0['grid_begins'][None,:],dtype=torch.float32)
        pix_pos1=torch.tensor(pix_pos1-results1['grid_begins'][None,:],dtype=torch.float32)
        scale_offset=torch.tensor(scale_offset,dtype=torch.int32)
        rotate_offset=torch.tensor(rotate_offset,dtype=torch.int32)
        H=torch.tensor(H,dtype=torch.float32)

        return img_list0,pts_list0,pix_pos0,grid_list0,img_list1,pts_list1,pix_pos1,grid_list1,scale_offset,rotate_offset,H

    def add_homography_background(self, img, cur_path, H):
        bpth = self.generate_background_pth(cur_path)

        h, w, _ = img.shape
        bimg = cv2.resize(imread(bpth), (w, h))
        if len(bimg.shape) == 2: bimg = np.repeat(bimg[:, :, None], 3, axis=2)
        if bimg.shape[2] > 3: bimg = bimg[:, :, :3]
        msk_tgt = cv2.warpPerspective(np.ones([h, w], np.uint8), H, (w, h), flags=cv2.INTER_NEAREST).astype(np.bool)
        img[np.logical_not(msk_tgt)] = bimg[np.logical_not(msk_tgt)]
        return img

    @staticmethod
    def add_black_background(img, H):
        h, w, _ = img.shape
        msk_tgt = cv2.warpPerspective(np.ones([h, w], np.uint8), H, (w, h), flags=cv2.INTER_NEAREST).astype(np.bool)
        img[np.logical_not(msk_tgt)] = 127
        return img

    def get_homography_correspondence(self, h, w, th, tw, H):
        coords = [np.expand_dims(item, 2) for item in np.meshgrid(np.arange(w), np.arange(h))]
        coords = np.concatenate(coords, 2).astype(np.float32)
        coords_target = cv2.perspectiveTransform(np.reshape(coords, [1, -1, 2]), H.astype(np.float32))
        coords_target = np.reshape(coords_target, [h, w, 2])

        source_mask = np.logical_and(np.logical_and(0 <= coords_target[:, :, 0], coords_target[:, :, 0] < tw ),
                                     np.logical_and(0 <= coords_target[:, :, 1], coords_target[:, :, 1] < th ))
        coords_target[np.logical_not(source_mask)] = 0

        return coords_target, source_mask

    def sample_correspondence(self, img, pix_pos, msk):
        h, w = img.shape[0], img.shape[1]
        val_msk = []

        if self.args['test_harris']:
            harris_img = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), 2, 3, 0.04)
            harris_msk = harris_img > np.percentile(harris_img.flatten(), self.args['harris_percentile'])
            val_msk.append(harris_msk)

        if self.args['test_edge']:
            edge_thresh = self.args['edge_thresh']
            edge_msk = np.ones_like(msk)
            edge_msk[:edge_thresh, :] = False
            edge_msk[:, :edge_thresh] = False
            edge_msk[h - edge_thresh:h, :] = False
            edge_msk[:, w - edge_thresh:w] = False
            edge_msk=np.logical_and(edge_msk,np.logical_and(pix_pos[:,:,0]<w-edge_thresh, pix_pos[:,:,0]>edge_thresh))
            edge_msk=np.logical_and(edge_msk,np.logical_and(pix_pos[:,:,1]<h-edge_thresh, pix_pos[:,:,1]>edge_thresh))
            msk = np.logical_and(msk, edge_msk)

        if len(val_msk) == 0:
            val_msk = np.ones_like(msk)
        else:
            val_msk_out = np.zeros_like(msk)
            for item in val_msk:
                val_msk_out = np.logical_or(val_msk_out, item)
            val_msk = val_msk_out
        val_msk = np.logical_and(msk, val_msk)

        hs, ws = np.nonzero(val_msk)
        pos_num = len(hs)
        if pos_num == 0:
            hs, ws = np.random.randint(0, h, 3000), np.random.randint(0, w, 3000)
            pos_num = len(hs)

        idxs = self.sample_indices(self.args['sample_num'], pos_num)
        pix_pos0 = np.concatenate([ws[idxs][:, None], hs[idxs][:, None]], 1)  # sn,2
        pix_pos1 = pix_pos[pix_pos0[:, 1], pix_pos0[:, 0]]
        return pix_pos0, pix_pos1

    def sample_ground_truth(self, img, H, img_tgt=None):
        h,w,_=img.shape
        if img_tgt is not None: th,tw,_=img_tgt.shape
        else: th,tw=h,w
        pix_pos, msk = self.get_homography_correspondence(h, w, th, tw, H)
        pix_pos0, pix_pos1 = self.sample_correspondence(img, pix_pos, msk)

        if self.args['perturb']:
            lower, upper = -self.args['perturb_max'], self.args['perturb_max']
            mu, sigma = 0, self.args['perturb_max']/2
            X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
            pix_pos1 += X.rvs(pix_pos1.shape)
        return pix_pos0, pix_pos1

    @staticmethod
    def get_image_region(img, th, tw):
        th, tw=int(th), int(tw)
        h, w, _ = img.shape
        if h > th:
            hbeg = np.random.randint(0, h - th)
            hend = hbeg + th
        else:
            hbeg, hend = 0, h

        if w > tw:
            wbeg = np.random.randint(0, w - tw)
            wend = wbeg + tw
        else:
            wbeg, wend = 0, w

        return img[hbeg:hend, wbeg:wend], wbeg, hbeg

    @staticmethod
    def resize_image_to_min(img, th, tw):
        oh,ow,_=img.shape
        if th<=oh and tw<=ow:
            return img, 1.0

        ratio=max((th+1)/oh, (tw+1)/ow)
        img=cv2.resize(img,(int(ratio*ow),int(ratio*oh)))
        return img, ratio

    @staticmethod
    def sample_indices(sample_num,pos_num):
        if pos_num >= sample_num:
            idxs = np.arange(pos_num)
            np.random.shuffle(idxs)
            idxs = idxs[:sample_num]
        else:
            idxs = np.arange(pos_num)
            idxs = np.append(idxs, np.random.choice(idxs, sample_num - pos_num))
        return idxs

    def generate_background_pth(self, cur_pth):
        bpth=self.background_pths[np.random.randint(0,len(self.background_pths))]
        while bpth==cur_pth: bpth=self.background_pths[np.random.randint(0,len(self.background_pths))]
        return bpth

    def generate_homography(self):
        return sample_homography(self.args['h'], self.args['w'])

    def augment(self, img):
        # ['jpeg','blur','noise','jitter','none']
        if len(self.args['augment_classes']) > self.args['augment_num']:
            augment_classes = np.random.choice(self.args['augment_classes'], self.args['augment_num'], False, p=self.args['augment_classes_weight'])
        elif 0 < len(self.args['augment_classes']) <= self.args['augment_num']:
            augment_classes = self.args["augment_classes"]
        else:
            return img

        for ac in augment_classes:
            img = self.name2func[ac](img)
        return img

    def compute_scale_rotate_offset(self, H, pix_pos0):
        As=compute_approximated_affine_batch(H, pix_pos0)
        scale=np.sqrt(np.linalg.det(As))
        Us,_,Vs=np.linalg.svd(As)
        R=Us@Vs
        rotate=np.arctan2(R[:,1,0],R[:,0,0])
        scale_offset=np.round(np.log(scale)/np.log(self.base_scale)).astype(np.int32)
        rotate_offset=np.round(rotate/self.base_rotate).astype(np.int32)
        return scale_offset, rotate_offset
