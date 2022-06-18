import os
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

#离散小波变换  下采样操作
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

        print("输入灰度图的形状：", x.shape)  #torch.Size([1, 1, 512, 512])
        C = x.shape[1]  ##通道数

        print("生成下采样参数的形状：", self.weight.shape)  # torch.Size([4, 1, 2, 2])

        #如果通道数是C，那们filters:torch.Size([4C, 1, 2, 2])
        filters = torch.cat([self.weight,] * C, dim = 0)
        print("生成下采样卷积核的形状：", filters.shape)  # torch.Size([12, 1, 2, 2])

        ##其实这里进行分组卷积操作  由定义的卷积进行下采样操作
        #输入X:(1,1,512,512)  filters:(4,1,2,2) group=1,stride=2  padding = 1
        #输出Y:(1,4,256,256)
        y = F.conv2d(x, filters, groups = C, stride = 2) #默认padding=0

        print("经过下采样输出的形状：", y.shape)  #torch.Size([1, 1, 512, 512])
        return y

#离散小波变换  上采样操作
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
            requires_grad=False)

    def forward(self, x):
        
        print("输入进行上采样的形状：", x.shape)  #torch.Size([1, 4, 256, 256])

        #进行上采样，通道数缩小为原来的四倍，H,W各变为原来的1/2
        C = int(x.shape[1] / 4)   # 1
        print("上采样生成参数的形状：", self.weight.shape)  # torch.Size([4, 1, 2, 2])

        filters = torch.cat([self.weight,] * C, dim = 0)
        print("生成上采样卷积核的形状：", filters.shape)  # torch.Size([12, 1, 2, 2])

        #逆卷积的操作过程  输入大小H*W ——》H1 = H +（S-1）*（H-1） 插值 中间插0
        #然后按照S=1 不变，padding= 核大小-padding-1,进行分组卷积
        #总结一下：先进行插值，256*256-》511*511，然后进行正常分组卷积 512*512

        y = F.conv_transpose2d(x, filters, groups = C, stride = 2) #默认pading = 0

        print("经过上采样输出的形状：", y.shape)  #torch.Size([1, 1, 512, 512])
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
