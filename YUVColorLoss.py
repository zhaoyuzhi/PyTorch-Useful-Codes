# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:31:22 2019
@author: ZHAO Yuzhi
"""

import torch
import torch.nn as nn


class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.L1loss = nn.L1Loss()
    
    def RGB2YUV(self, RGB):
        YUV = RGB.clone()
        YUV[:,0,:,:] = 0.299 * RGB[:,0,:,:] + 0.587 * RGB[:,1,:,:] + 0.114 * RGB[:,2,:,:]
        YUV[:,1,:,:] = -0.14713 * RGB[:,0,:,:] - 0.28886 * RGB[:,1,:,:] + 0.436 * RGB[:,2,:,:]
        YUV[:,2,:,:] = 0.615 * RGB[:,0,:,:] - 0.51499 * RGB[:,1,:,:] - 0.10001 * RGB[:,2,:,:]
        return YUV

    def forward(self, x, y):
        yuv_x = self.RGB2YUV(x)
        yuv_y = self.RGB2YUV(y)
        return self.L1loss(yuv_x, yuv_y)
