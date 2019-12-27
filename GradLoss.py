# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:31:22 2019

@author: ZHAO Yuzhi
"""

import torch
import torch.nn as nn

class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()
        self.MSEloss = nn.MSELoss()
        
    def forward(self, x, y):
        h_x = x.size()[2]
        w_x = x.size()[3]
        x_h_grad = x[:, :, 1:, :] - x[:, :, :h_x - 1, :]
        x_w_grad = x[:, :, :, 1:] - x[:, :, :, :w_x - 1]
        y_h_grad = y[:, :, 1:, :] - y[:, :, :h_x - 1, :]
        y_w_grad = y[:, :, :, 1:] - y[:, :, :, :w_x - 1]
        
        h_loss = self.MSEloss(x_h_grad, y_h_grad)
        w_loss = self.MSEloss(x_w_grad, y_w_grad)
        
        return self.GradLoss_weight * (h_loss + w_loss)
