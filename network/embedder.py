import torch
import torch.nn as nn

from network.operator import l2_normalize, RotationalConv2D

class BilinearGCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.network1_embed1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network1_embed1_short = nn.Conv2d(32, 64, 1, 1)
        self.network1_embed1_relu = nn.ReLU(True)

        self.network1_embed2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network1_embed2_short = nn.Conv2d(64, 64, 1, 1)
        self.network1_embed2_relu = nn.ReLU(True)

        self.network1_embed3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 8, 3, 1, 1),
        )

        ###########################
        self.network2_embed1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network2_embed1_short = nn.Conv2d(32, 64, 1, 1)
        self.network2_embed1_relu = nn.ReLU(True)

        self.network2_embed2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network2_embed2_short = nn.Conv2d(64, 64, 1, 1)
        self.network2_embed2_relu = nn.ReLU(True)

        self.network2_embed3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 3, 1, 1),
        )

    def forward(self, x):
        '''

        :param x:  b,n,f,ssn,srn
        :return:
        '''
        b, n, f, ssn, srn = x.shape
        assert (ssn == 5 and srn == 5)
        x = x.reshape(b * n, f, ssn, srn)

        x1 = self.network1_embed1_relu(self.network1_embed1(x) + self.network1_embed1_short(x))
        x1 = self.network1_embed2_relu(self.network1_embed2(x1) + self.network1_embed2_short(x1))
        x1 = self.network1_embed3(x1)

        x2 = self.network2_embed1_relu(self.network2_embed1(x) + self.network2_embed1_short(x))
        x2 = self.network2_embed2_relu(self.network2_embed2(x2) + self.network2_embed2_short(x2))
        x2 = self.network2_embed3(x2)

        x1 = x1.reshape(b * n, 8, 25)
        x2 = x2.reshape(b * n, 16, 25).permute(0, 2, 1)  # b*n,25,16
        x = torch.bmm(x1, x2).reshape(b * n, 128)  # b*n,8,25
        assert (x.shape[1] == 128)
        x=x.reshape(b,n,128)
        x=l2_normalize(x,axis=2)
        return x

class BilinearRotationGCNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.network1_embed1 = nn.Sequential(
            RotationalConv2D(32, 64, 3, 1, 1),
            nn.ReLU(True),
            RotationalConv2D(64, 64, 3, 1, 1),
        )
        self.network1_embed1_short = RotationalConv2D(32, 64, 1, 1)
        self.network1_embed1_relu = nn.ReLU(True)

        self.network1_embed2 = nn.Sequential(
            RotationalConv2D(64, 64, 3, 1, 1),
            nn.ReLU(True),
            RotationalConv2D(64, 64, 3, 1, 1),
        )
        self.network1_embed2_short = RotationalConv2D(64, 64, 1, 1)
        self.network1_embed2_relu = nn.ReLU(True)

        self.network1_embed3 = nn.Sequential(
            RotationalConv2D(64, 64, 3, 1, 1),
            nn.ReLU(True),
            RotationalConv2D(64, 8, 3, 1, 1),
        )

        ###########################
        self.network2_embed1 = nn.Sequential(
            RotationalConv2D(32, 64, 3, 1, 1),
            nn.ReLU(True),
            RotationalConv2D(64, 64, 3, 1, 1),
        )
        self.network2_embed1_short = RotationalConv2D(32, 64, 1, 1)
        self.network2_embed1_relu = nn.ReLU(True)

        self.network2_embed2 = nn.Sequential(
            RotationalConv2D(64, 64, 3, 1, 1),
            nn.ReLU(True),
            RotationalConv2D(64, 64, 3, 1, 1),
        )
        self.network2_embed2_short = RotationalConv2D(64, 64, 1, 1)
        self.network2_embed2_relu = nn.ReLU(True)

        self.network2_embed3 = nn.Sequential(
            RotationalConv2D(64, 64, 3, 1, 1),
            nn.ReLU(True),
            RotationalConv2D(64, 16, 3, 1, 1),
        )

    def forward(self, x):
        '''

        :param x:  b,n,f,ssn,srn
        :return:
        '''
        b, n, f, ssn, srn = x.shape
        assert (ssn == 5 and srn == 8)
        x = x.reshape(b * n, f, ssn, srn)

        x1 = self.network1_embed1_relu(self.network1_embed1(x) + self.network1_embed1_short(x))
        x1 = self.network1_embed2_relu(self.network1_embed2(x1) + self.network1_embed2_short(x1))
        x1 = self.network1_embed3(x1)

        x2 = self.network2_embed1_relu(self.network2_embed1(x) + self.network2_embed1_short(x))
        x2 = self.network2_embed2_relu(self.network2_embed2(x2) + self.network2_embed2_short(x2))
        x2 = self.network2_embed3(x2)

        x1 = x1.reshape(b * n, 8, 40)
        x2 = x2.reshape(b * n, 16, 40).permute(0, 2, 1)  # b*n,40,16
        x = torch.bmm(x1, x2).reshape(b * n, 128)        # b*n,8,40
        assert (x.shape[1] == 128)
        x=x.reshape(b,n,128)
        x=l2_normalize(x,axis=2)
        return x


class GRENone(nn.Module):
    def __init__(self,cfg):
        super().__init__()

    def forward(self, x):
        '''

        :param x:  b,n,f,ssn,srn
        :return:
        '''
        x=l2_normalize(x,axis=2)
        return x[:,:,:,0,0]


class GREBilinearPoolPaddingDeepMoreScale(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.network1_embed1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network1_embed1_short = nn.Conv2d(32, 64, 1, 1)
        self.network1_embed1_relu = nn.ReLU(True)

        self.network1_embed2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network1_embed2_short = nn.Conv2d(64, 64, 1, 1)
        self.network1_embed2_relu = nn.ReLU(True)

        self.network1_embed3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 8, 3, 1, 1),
        )

        ###########################
        self.network2_embed1 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network2_embed1_short = nn.Conv2d(32, 64, 1, 1)
        self.network2_embed1_relu = nn.ReLU(True)

        self.network2_embed2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.network2_embed2_short = nn.Conv2d(64, 64, 1, 1)
        self.network2_embed2_relu = nn.ReLU(True)

        self.network2_embed3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 3, 1, 1),
        )

    def forward(self, x):
        '''

        :param x:  b,n,f,ssn,srn
        :return:
        '''
        b, n, f, ssn, srn = x.shape
        assert (ssn == 8 and srn == 5)
        x = x.reshape(b * n, f, ssn, srn)

        x1 = self.network1_embed1_relu(self.network1_embed1(x) + self.network1_embed1_short(x))
        x1 = self.network1_embed2_relu(self.network1_embed2(x1) + self.network1_embed2_short(x1))
        x1 = self.network1_embed3(x1)

        x2 = self.network2_embed1_relu(self.network2_embed1(x) + self.network2_embed1_short(x))
        x2 = self.network2_embed2_relu(self.network2_embed2(x2) + self.network2_embed2_short(x2))
        x2 = self.network2_embed3(x2)

        x1 = x1.reshape(b * n, 8, 40)
        x2 = x2.reshape(b * n, 16, 40).permute(0, 2, 1)  # b*n,40,16
        x = torch.bmm(x1, x2).reshape(b * n, 128)  # b*n,8,40
        assert (x.shape[1] == 128)
        x=x.reshape(b,n,128)
        x=l2_normalize(x,axis=2)
        return x