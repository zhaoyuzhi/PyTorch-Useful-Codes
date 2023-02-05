import torch
import torch.nn as nn

# An implemention of non-local blocks
class Self_Attn_FM(nn.Module):
    """ Self attention Layer for Feature Map dimension"""
    def __init__(self, in_dim, latent_dim = 8):
        super(Self_Attn_FM, self).__init__()
        self.channel_in = in_dim
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X N X N (N is Height * Width)
        """
        batchsize, C, height, width = x.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query  = self.query_conv(x).view(batchsize, -1, height * width).permute(0, 2, 1)
        # proj_query: reshape to B x c x N, N = H x W
        proj_key =  self.key_conv(x).view(batchsize, -1, height * width)
        # transpose check, energy: B x N x N, N = H x W
        energy =  torch.bmm(proj_query, proj_key)
        # attention: B x N x N, N = H x W
        attention = self.softmax(energy)
        # proj_value is normal convolution, B x C x N
        proj_value = self.value_conv(x).view(batchsize, -1, height * width)
        # out: B x C x N
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, height, width)
        
        out = self.gamma * out + x
        return out, attention

class Self_Attn_C(nn.Module):
    """ Self attention Layer for Channel dimension"""
    def __init__(self, in_dim, latent_dim = 8):
        super(Self_Attn_C, self).__init__()
        self.chanel_in = in_dim
        self.channel_latent = in_dim // latent_dim
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim // latent_dim, kernel_size = 1)
        self.out_conv = nn.Conv2d(in_channels = in_dim // latent_dim, out_channels = in_dim, kernel_size = 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim = -1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps(B X C X H X W)
            returns :
                out : self attention value + input feature 
                attention: B X c X c
        """
        batchsize, C, height, width = x.size()
        # proj_query: reshape to B x N x c, N = H x W
        proj_query  = self.query_conv(x).view(batchsize, -1, height * width).permute(0, 2, 1)
        # proj_query: reshape to B x c x N, N = H x W
        proj_key =  self.key_conv(x).view(batchsize, -1, height * width)
        # transpose check, energy: B x c x c
        energy =  torch.bmm(proj_key, proj_query)
        # attention: B x c x c
        attention = self.softmax(energy)
        # proj_value is a convolution, B x c x N
        proj_value = self.value_conv(x).view(batchsize, -1, height * width)
        # out: B x C x N
        out = torch.bmm(attention.permute(0, 2, 1), proj_value)
        out = out.view(batchsize, self.channel_latent, height, width)
        out = self.out_conv(out)
        
        out = self.gamma * out + x
        return out, attention
      