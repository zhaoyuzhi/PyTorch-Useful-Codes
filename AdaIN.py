import torch
import torch.nn as nn
import torch.nn.functional as F

# FeatureAdaIN receives two feature maps
class FeatureAdaIN(nn.Module):
    
    def __init__(self, out_size, eps = 1e-5):
        super(FeatureAdaIN, self).__init__()
        self.eps = eps
        self.out_size = out_size

    def forward(self, content, style):
        
        # content, style are feature maps from a CNN
        assert(content.size(0) == style.size(0))
        assert(content.size(2) == style.size(2))
        assert(content.size(3) == style.size(3))
        assert(content.size(1) == self.out_size)
        assert(style.size(1) == self.out_size)
        N, C, H, W = content.size(0), content.size(1), content.size(2), content.size(3)

        # compute target mean and standard deviation from the style input
        styleView = style.view((N, C, -1))
        styleStd = styleView.std(2).view(-1) + self.eps
        styleMean = styleView.mean(2).view(-1)
        styleStd = styleStd.view(N, C, 1, 1).expand(N, C, H, W)
        styleMean = styleMean.view(N, C, 1, 1).expand(N, C, H, W)

        contentView = content.view((N, C, -1))
        contentStd = contentView.std(2).view(-1) + self.eps
        contentMean = contentView.mean(2).view(-1)
        contentStd = contentStd.view(N, C, 1, 1).expand(N, C, H, W)
        contentMean = contentMean.view(N, C, 1, 1).expand(N, C, H, W)

        # compute the final output
        normalizedView = (content - contentMean) / contentStd
        out = normalizedView * styleStd + styleMean
        return out

# AdaptiveInstanceNorm2d receives one feature map and one linear vector
class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, eps = 1e-8):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.eps = eps
    
    def IN_noWeight(self, x):
        x = x - torch.mean(x, dim = (2, 3), keepdim = True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, dim = (2, 3), keepdim = True) + self.eps)
        return x * tmp
    
    def Apply_style(self, content, style):
        style = style.contiguous().view([-1, 2, content.size(1), 1, 1])     # N * 2 * C * 1 * 1
        # we assume the first dimension is variance, and second dimension is mean
        content = content * style[:, 0] + style[:, 1]
        return content

    def forward(self, content, style):
        # content is feature maps from one layer (4D Tensor)                # N * C * H * W
        # style is a linear vector from FC layer (2D Tensor)                # N * 2C
        normalized_content = self.IN_noWeight(content)
        stylized_content = self.Apply_style(normalized_content, style)
        return stylized_content
