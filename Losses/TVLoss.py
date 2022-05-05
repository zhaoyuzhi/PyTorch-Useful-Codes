# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:31:22 2019

@author: ZHAO Yuzhi
"""

import torch
import torch.nn as nn

class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()
        
    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    
    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]
