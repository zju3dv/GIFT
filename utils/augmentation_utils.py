import numpy as np
import cv2
import math

def random_rotate(img, pts, rot_ang_min, rot_ang_max):
    h,w=img.shape[0],img.shape[1]
    degree=np.random.uniform(rot_ang_min,rot_ang_max)
    R=cv2.getRotationMatrix2D((w/2,h/2), degree, 1)
    img=cv2.warpAffine(img,R,(w, h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)

    pts=np.concatenate([pts,np.ones([pts.shape[0],1])],1) # n,3
    last_row=np.asarray([[0,0,1]],np.float32)
    pts=np.matmul(pts, np.concatenate([R, last_row], 0).transpose())

    return img, pts[:,:2]

def random_rotate_img(img, rot_ang_min, rot_ang_max):
    h,w=img.shape[0],img.shape[1]
    degree=np.random.uniform(rot_ang_min,rot_ang_max)
    R=cv2.getRotationMatrix2D((w/2,h/2), degree, 1)
    img=cv2.warpAffine(img,R,(w, h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return img

def crop_or_pad_to_fixed_size(img, pts=None, ratio=1.0):
    h,w=img.shape[0],img.shape[1]
    th,tw=int(math.ceil(h*ratio)),int(math.ceil(w*ratio))
    img=cv2.resize(img,(tw,th),interpolation=cv2.INTER_LINEAR)
    if pts is not None:
        pts*=ratio

    if ratio>1.0:
        # crop
        hbeg,wbeg=np.random.randint(0,th-h),np.random.randint(0,tw-w)
        result_img=img[hbeg:hbeg+h,wbeg:wbeg+w]
        if pts is not None:
            pts[:, 0] -= wbeg
            pts[:, 1] -= hbeg
    else:
        # padding
        if len(img.shape)==2:
            result_img=np.zeros([h,w],img.dtype)
        else:
            result_img = np.zeros([h, w, img.shape[2]], img.dtype)
        hbeg,wbeg=(h-th)//2,(w-tw)//2
        result_img[hbeg:hbeg+th,wbeg:wbeg+tw]=img
        if pts is not None:
            pts[:, 0] += wbeg
            pts[:, 1] += hbeg

    if pts is not None:
        return result_img, pts
    else:
        return result_img


def add_noise(image):
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 0.75:
        row,col,ch= image.shape
        mean = 0
        var = np.random.rand(1) * 0.3 * 256
        sigma = var**0.5
        gauss = sigma * np.random.randn(row,col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype(np.uint8)
    else:
        # motion blur
        sizes = [3, 5, 7, 9]
        size = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((size, size))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        else:
            kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)
    return noisy


def gaussian_blur(img,blur_range=[3,5,7,9,11]):
    sigma=np.random.choice(blur_range,1)
    return cv2.GaussianBlur(img,(sigma,sigma),0)

def jpeg_compress(img,quality_low=15,quality_high=75):
    quality=np.random.randint(quality_low,quality_high)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    img=cv2.imdecode(encimg,1)
    return img


