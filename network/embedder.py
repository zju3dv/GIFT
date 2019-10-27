import torch
import torch.nn as nn

class GREBilinearPoolPaddingDeepMoreNoRes(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.network1=nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 8, 3, 1, 1)
        )
        self.network2=nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 16, 3, 1, 1)
        )

    def forward(self, x):
        '''

        :param x:  b,n,f,ssn,srn
        :return:
        '''
        b,n,f,ssn,srn=x.shape
        assert(ssn==5 and srn==5)
        x=x.reshape(b*n,f,ssn,srn)
        x1=self.network1(x) # b*n,8,5,5
        x2=self.network2(x) # b*n,16,5,5
        x1=x1.reshape(b*n,8,25)
        x2=x2.reshape(b*n,16,25).permute(0,2,1) # b*n,25,16
        x=torch.bmm(x1,x2).reshape(b*n,128) # b*n,8,25
        assert(x.shape[1]==128)
        return x