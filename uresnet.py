import torch
import torch.nn as nn
from network_module import *

class UResNet444(nn.Module):
    def __init__(self, opt):
        super(UResNet444, self).__init__()
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T2 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T3 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T4 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D4 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'sigmoid')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 64 * 256 * 256
        E2 = self.E2(E1)                                        # out: batch * 128 * 128 * 128
        E3 = self.E3(E2)                                        # out: batch * 256 * 64 * 64
        E4 = self.E4(E3)                                        # out: batch * 512 * 32 * 32
        # Bottleneck
        x = self.T1(E4)                                         # out: batch * 512 * 32 * 32
        x = self.T2(x)                                          # out: batch * 512 * 32 * 32
        x = self.T3(x)                                          # out: batch * 512 * 32 * 32
        x = self.T4(x)                                          # out: batch * 512 * 32 * 32
        # Decode the center code
        D1 = self.D1(x)                                         # out: batch * 256 * 64 * 64
        D1 = torch.cat((D1, E3), 1)                             # out: batch * 512 * 64 * 64
        D2 = self.D2(D1)                                        # out: batch * 128 * 128 * 128
        D2 = torch.cat((D2, E2), 1)                             # out: batch * 256 * 128 * 128
        D3 = self.D3(D2)                                        # out: batch * 64 * 256 * 256
        D3 = torch.cat((D3, E1), 1)                             # out: batch * 128 * 256 * 256
        x = self.D4(D3)                                         # out: batch * out_channel * 256 * 256

        return x

class UResNet464(nn.Module):
    def __init__(self, opt):
        super(UResNet464, self).__init__()
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T2 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T3 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T4 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T5 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T6 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D4 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'sigmoid')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 64 * 256 * 256
        E2 = self.E2(E1)                                        # out: batch * 128 * 128 * 128
        E3 = self.E3(E2)                                        # out: batch * 256 * 64 * 64
        E4 = self.E4(E3)                                        # out: batch * 512 * 32 * 32
        # Bottleneck
        x = self.T1(E4)                                         # out: batch * 512 * 32 * 32
        x = self.T2(x)                                          # out: batch * 512 * 32 * 32
        x = self.T3(x)                                          # out: batch * 512 * 32 * 32
        x = self.T4(x)                                          # out: batch * 512 * 32 * 32
        x = self.T5(x)                                          # out: batch * 512 * 32 * 32
        x = self.T6(x)                                          # out: batch * 512 * 32 * 32
        # Decode the center code
        D1 = self.D1(x)                                         # out: batch * 256 * 64 * 64
        D1 = torch.cat((D1, E3), 1)                             # out: batch * 512 * 64 * 64
        D2 = self.D2(D1)                                        # out: batch * 128 * 128 * 128
        D2 = torch.cat((D2, E2), 1)                             # out: batch * 256 * 128 * 128
        D3 = self.D3(D2)                                        # out: batch * 64 * 256 * 256
        D3 = torch.cat((D3, E1), 1)                             # out: batch * 128 * 256 * 256
        x = self.D4(D3)                                         # out: batch * out_channel * 256 * 256

        return x

class UResNet484(nn.Module):
    def __init__(self, opt):
        super(UResNet484, self).__init__()
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T2 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T3 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T4 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T5 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T6 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T7 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T8 = ResConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D4 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'sigmoid')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 64 * 256 * 256
        E2 = self.E2(E1)                                        # out: batch * 128 * 128 * 128
        E3 = self.E3(E2)                                        # out: batch * 256 * 64 * 64
        E4 = self.E4(E3)                                        # out: batch * 512 * 32 * 32
        # Bottleneck
        x = self.T1(E4)                                         # out: batch * 512 * 32 * 32
        x = self.T2(x)                                          # out: batch * 512 * 32 * 32
        x = self.T3(x)                                          # out: batch * 512 * 32 * 32
        x = self.T4(x)                                          # out: batch * 512 * 32 * 32
        x = self.T5(x)                                          # out: batch * 512 * 32 * 32
        x = self.T6(x)                                          # out: batch * 512 * 32 * 32
        x = self.T7(x)                                          # out: batch * 512 * 32 * 32
        x = self.T8(x)                                          # out: batch * 512 * 32 * 32
        # Decode the center code
        D1 = self.D1(x)                                         # out: batch * 256 * 64 * 64
        D1 = torch.cat((D1, E3), 1)                             # out: batch * 512 * 64 * 64
        D2 = self.D2(D1)                                        # out: batch * 128 * 128 * 128
        D2 = torch.cat((D2, E2), 1)                             # out: batch * 256 * 128 * 128
        D3 = self.D3(D2)                                        # out: batch * 64 * 256 * 256
        D3 = torch.cat((D3, E1), 1)                             # out: batch * 128 * 256 * 256
        x = self.D4(D3)                                         # out: batch * out_channel * 256 * 256

        return x

class UResNet565(nn.Module):
    def __init__(self, opt):
        super(UResNet565, self).__init__()
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none')
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        self.E5 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 4, 2, 1, pad_type = opt.pad, norm = opt.norm)
        # Bottleneck
        self.T1 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T2 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T3 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T4 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T5 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        self.T6 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels, 3, 1, 1, pad_type = opt.pad, norm = opt.norm, scale_factor = 2)
        self.D5 = Conv2dLayer(opt.start_channels * 2, opt.out_channels, 7, 1, 3, pad_type = opt.pad, norm = 'none', activation = 'sigmoid')

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        E1 = self.E1(x)                                         # out: batch * 64 * 256 * 256
        E2 = self.E2(E1)                                        # out: batch * 128 * 128 * 128
        E3 = self.E3(E2)                                        # out: batch * 256 * 64 * 64
        E4 = self.E4(E3)                                        # out: batch * 256 * 32 * 32
        E5 = self.E5(E4)                                        # out: batch * 256 * 16 * 16
        # Bottleneck
        x = self.T1(E5)                                         # out: batch * 256 * 16 * 16
        x = self.T2(x)                                          # out: batch * 256 * 16 * 16
        x = self.T3(x)                                          # out: batch * 256 * 16 * 16
        x = self.T4(x)                                          # out: batch * 256 * 16 * 16
        x = self.T5(x)                                          # out: batch * 256 * 16 * 16
        x = self.T6(x)                                          # out: batch * 256 * 16 * 16
        # Decode the center code
        D1 = self.D1(x)                                         # out: batch * 256 * 32 * 32
        D1 = torch.cat((D1, E4), 1)                             # out: batch * 512 * 32 * 32
        D2 = self.D2(D1)                                        # out: batch * 256 * 64 * 64
        D2 = torch.cat((D2, E3), 1)                             # out: batch * 512 * 64 * 64
        D3 = self.D3(D2)                                        # out: batch * 128 * 128 * 128
        D3 = torch.cat((D3, E2), 1)                             # out: batch * 256 * 128 * 128
        D4 = self.D4(D3)                                        # out: batch * 64 * 256 * 256
        D4 = torch.cat((D4, E1), 1)                             # out: batch * 128 * 256 * 256
        x = self.D5(D4)                                         # out: batch * out_channel * 256 * 256

        return x
