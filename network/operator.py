import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

class GaussianConv(nn.Module):
    def __init__(self, fsize, stride=1):
        super(GaussianConv,self).__init__()
        assert(stride==2 or stride==1)
        assert(fsize%2==1)
        kernel=np.asarray(cv2.getGaussianKernel(fsize, 0),np.float32).reshape([1,1,fsize,1])
        kernel=torch.tensor(kernel,dtype=torch.float32)
        self.register_buffer('kernel',kernel)
        self.stride=stride
        self.fsize=fsize
        if stride==1:
            self.padder=nn.ReplicationPad2d(fsize//2)
        else:
            self.padder=nn.ReplicationPad2d(fsize//2-1)
            self.padder_vert=nn.ReplicationPad2d((0,0,0,1))
            self.padder_hori=nn.ReplicationPad2d((0,1,0,0))


    def forward(self, x):
        '''

        :param x: b,...,h,w
        :return:
        '''
        input_shape=list(x.shape)
        x=x.reshape(x.shape[0],-1,x.shape[-2],x.shape[-1])
        b,f,h,w=x.shape
        with torch.no_grad():
            conv_kernel=torch.zeros([f,f,self.fsize,1],dtype=torch.float32,device=self.kernel.device)
            for k in range(f): conv_kernel[k,k]=self.kernel

        x=self.padder(x)
        # if self.stride==2 and h%2==0:
        #     x=self.padder_vert(x)
        # if self.stride==2 and w%2==0:
        #     x=self.padder_hori(x)

        x = F.conv2d(x, conv_kernel, stride=(self.stride,1), padding=0)
        x = F.conv2d(x, conv_kernel.permute(0,1,3,2), stride=(1,self.stride), padding=0)
        input_shape[-2]=x.shape[-2]
        input_shape[-1]=x.shape[-1]
        return x.reshape(input_shape)

def l2_normalize(x,ratio=1.0,axis=1):
    norm=torch.unsqueeze(torch.clamp(torch.norm(x,2,axis),min=1e-6),axis)
    x=x/norm*ratio
    return x

def normalize_coordinates(coords, h, w):
    h=h-1
    w=w-1
    coords=coords.clone().detach()
    coords[:, :, 0]-= w / 2
    coords[:, :, 1]-= h / 2
    coords[:, :, 0]/= w / 2
    coords[:, :, 1]/= h / 2
    return coords