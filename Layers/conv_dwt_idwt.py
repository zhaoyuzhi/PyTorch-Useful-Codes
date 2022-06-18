import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class DWTForward(nn.Module):
    def __init__(self):
        super(DWTForward, self).__init__()

        #ll lh hl hh shape(2,2)
        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])

        #filts shape(4,1,2,2)
        filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                          hl[None,::-1,::-1], hh[None,::-1,::-1]],
                         axis = 0)

        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad = False)

    def forward(self, x):
        C = x.shape[1]
        filters = torch.cat([self.weight, ] * C, dim = 0)
        y = F.conv2d(x, filters, groups = C, stride = 2)
        return y

class DWTInverse(nn.Module):
    def __init__(self):
        super(DWTInverse, self).__init__()

        ll = np.array([[0.5, 0.5], [0.5, 0.5]])
        lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
        hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
        hh = np.array([[0.5, -0.5], [-0.5, 0.5]])

        filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                          hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                         axis = 0)

        self.weight = nn.Parameter(
            torch.tensor(filts).to(torch.get_default_dtype()),
            requires_grad = False)

    def forward(self, x):
        C = int(x.shape[1] / 4)
        filters = torch.cat([self.weight, ] * C, dim = 0)
        y = F.conv_transpose2d(x, filters, groups = C, stride = 2)
        return y

DWT = DWTForward()
IWT = DWTInverse()

x = torch.randn(1, 3, 256, 256)
dwt1 = DWT(x)
out = IWT(dwt1)
residual = x - out
#print(residual)
print(residual.shape)

'''
# DWT
输入灰度图的形状： torch.Size([1, 3, 256, 256])
生成下采样参数的形状： torch.Size([4, 1, 2, 2])
生成下采样卷积核的形状： torch.Size([12, 1, 2, 2])
经过下采样输出的形状： torch.Size([1, 12, 128, 128])
# IWT
输入进行上采样的形状： torch.Size([1, 12, 128, 128])
上采样生成参数的形状： torch.Size([4, 1, 2, 2])
生成上采样卷积核的形状： torch.Size([12, 1, 2, 2])
经过上采样输出的形状： torch.Size([1, 3, 256, 256])
# residual.shape
torch.Size([1, 3, 256, 256])
'''
