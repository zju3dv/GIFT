import torch
import torch.nn as nn
import torch.nn.functional as F

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

class RotationalConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', rot_axis='w'):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        assert(padding_mode=='zeros')
        if type(padding) is tuple:
            if len(padding)==2:
                padding=list(padding)
            else:
                 raise NotImplementedError
        elif type(padding) is int:
            padding=[padding,padding]
        else:
            raise NotImplementedError

        # padding: left right top bottom
        self.rotate_padding=[0,0,0,0]
        self.padding=padding # h_padding, w_padding
        if rot_axis=='h':
            self.rotate_padding[2]=self.padding[0] # top
            self.rotate_padding[3]=self.padding[0] # bottom
            self.padding[0]=0
        elif rot_axis=='w':
            self.rotate_padding[0]=self.padding[1]
            self.rotate_padding[1]=self.padding[1]
            self.padding[1]=0
        else:
            raise NotImplementedError

    def forward(self, input):
        input=F.pad(input,self.rotate_padding,'circular')
        return self.conv2d_forward(input, self.weight)
