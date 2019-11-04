import torch.nn.functional as F
import torch.nn as nn
import torch

import numpy as np

feats=np.arange(9).reshape([1,1,3,3])
feats=torch.tensor(feats,dtype=torch.float32)

feats_pad=F.pad(feats,(1,1,0,0),'circular')
print(feats_pad.shape)
print(feats_pad[0,0])