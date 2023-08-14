import torch
import torch.nn as nn
import torch.nn.functional as F

def build_conv_layer(kernel):
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = torch.cat((kernel, kernel, kernel), 1)
    return kernel

bias = torch.FloatTensor([0])
kernel1 = [[0.03797616, 0.044863533, 0.03797616],
            [0.044863533, 0.053, 0.044863533],
            [0.03797616, 0.044863533, 0.03797616]]
kernel2 = [[1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9],
            [1/9, 1/9, 1/9]]
kernel3 = [[-1/9, -1/9, -1/9],
            [-1/9, 17/9, -1/9],
            [-1/9, -1/9, -1/9]]

kernel = build_conv_layer(kernel1)
print(kernel.shape, kernel.dtype)

x = torch.randn(1, 3, 28, 28)
out = F.conv2d(x, kernel, bias, stride = 1, padding = 1)
print(out.shape)