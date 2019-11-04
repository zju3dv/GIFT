import numpy as np
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_hue
from PIL import Image
import cv2

def additive_gaussian_noise(image, stddev_range=(5, 95)):
    stddev = np.random.uniform(*stddev_range)
    noise = np.random.normal(0.0 , stddev, image.shape)
    noisy_image = image.astype(np.float32)+noise
    noisy_image=np.clip(noisy_image,0,255)
    return noisy_image.astype(np.uint8)

def additive_speckle_noise(image, prob_range=(0.0, 0.005)):
    prob = np.random.uniform(*prob_range)
    sample = np.random.rand(*image.shape)
    noisy_image = image.astype(np.float32).copy()
    noisy_image[sample<prob] = 0.0
    noisy_image[sample>=(1-prob)] = 255
    return noisy_image.astype(np.uint8)

def random_brightness(image, max_change=0.3):
    return np.asarray(adjust_brightness(Image.fromarray(image),max_change))

def random_contrast(image, max_change=0.5):
    return np.asarray(adjust_contrast(Image.fromarray(image),max_change))


def additive_shade(image, nb_ellipses=20, transparency_range=(-0.5, 0.8),
                   kernel_size_range=(250, 350)):

    def _py_additive_shade(img):
        img=img.astype(np.float32)
        min_dim = min(img.shape[:2]) / 4
        mask = np.zeros(img.shape[:2], np.uint8)
        for i in range(nb_ellipses):
            ax = int(max(np.random.rand() * min_dim, min_dim / 5))
            ay = int(max(np.random.rand() * min_dim, min_dim / 5))
            max_rad = max(ax, ay)
            x = np.random.randint(max_rad, img.shape[1] - max_rad)  # center
            y = np.random.randint(max_rad, img.shape[0] - max_rad)
            angle = np.random.rand() * 90
            cv2.ellipse(mask, (x, y), (ax, ay), angle, 0, 360, 255, -1)

        transparency = np.random.uniform(*transparency_range)
        kernel_size = np.random.randint(*kernel_size_range)
        if (kernel_size % 2) == 0:  # kernel_size has to be odd
            kernel_size += 1
        mask = cv2.GaussianBlur(mask.astype(np.float32), (kernel_size, kernel_size), 0)
        shaded = img * (1 - transparency * mask[..., np.newaxis]/255.)
        return np.clip(shaded, 0, 255).astype(np.uint8)

    return _py_additive_shade(image)


def motion_blur(image, max_kernel_size=10):

    def _py_motion_blur(img):
        # Either vertial, hozirontal or diagonal blur
        img=img.astype(np.float32)
        mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
        ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
        center = int((ksize-1)/2)
        kernel = np.zeros((ksize, ksize))
        if mode == 'h':
            kernel[center, :] = 1.
        elif mode == 'v':
            kernel[:, center] = 1.
        elif mode == 'diag_down':
            kernel = np.eye(ksize)
        elif mode == 'diag_up':
            kernel = np.flip(np.eye(ksize), 0)
        var = ksize * ksize / 16.
        grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
        gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        img = cv2.filter2D(img, -1, kernel)
        img=np.clip(img,0,255)
        return img.astype(np.uint8)

    return _py_motion_blur(image)

def resize_blur(img, max_ratio=0.15):
    h,w,_=img.shape
    ratio_w,ratio_h=np.random.uniform(max_ratio,1.0,2)
    return cv2.resize(cv2.resize(img,(int(w*ratio_w),int(h*ratio_h)),interpolation=cv2.INTER_LINEAR),
                      (w,h),interpolation=cv2.INTER_LINEAR)
