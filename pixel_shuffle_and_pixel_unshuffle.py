import torch
import torch.nn as nn
import torch.nn.functional as F

### implementation 1

# F.pixel_shuffle(x, 2)   F.pixel_shuffle(x, 4)

def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    downscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]
    kernel = torch.zeros(size = [downscale_factor * downscale_factor * c, 1, downscale_factor, downscale_factor],
                        device = input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor * downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride = downscale_factor, groups = c)

class PixelUnShuffle(nn.Module):
    def __init__(self, downscale_factor):
        super(PixelUnShuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, input):
        '''
        input: batchSize * c * k*w * k*h
        downscale_factor: k
        batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
        '''
        return pixel_unshuffle(input, self.downscale_factor)

### implementation 2

class PixelShuffleAlign(nn.Module):
    def __init__(self, upscale_factor: int = 1, mode: str = 'caffe'):
        """
        :param upscale_factor: upsample scale
        :param mode: caffe, pytorch
        """
        super(PixelShuffleAlign, self).__init__()
        self.upscale_factor = upscale_factor
        self.mode = mode

    def forward(self, x):
        # assert len(x.size()) == 4, "Received input tensor shape is {}".format(
        #     x.size())
        if len(x.size()) != 4:
            raise ValueError("input tensor shape {} is not supported.".format(x.size()))
        N, C, H, W = x.size()
        c = C // (self.upscale_factor ** 2)
        h, w = H * self.upscale_factor, W * self.upscale_factor

        if self.mode == 'caffe':
            # (N, C, H, W) => (N, r, r, c, H, W)
            x = x.reshape(-1, self.upscale_factor,
                          self.upscale_factor, c, H, W)
            x = x.permute(0, 3, 4, 1, 5, 2)
        elif self.mode == 'pytorch':
            # (N, C, H, W) => (N, r, r, c, H, W)
            x = x.reshape(-1, c, self.upscale_factor,
                          self.upscale_factor, H, W)
            x = x.permute(0, 1, 4, 2, 5, 3)
        else:
            raise NotImplementedError(
                "{} mode is not implemented".format(self.mode))

        x = x.reshape(-1, c, h, w)
        return x

class PixelUnShuffleAlign(nn.Module):

    def __init__(self, downscale_factor: int = 2, mode: str = 'caffe'):
        """
        :param downscale_factor: downsample scale
        :param mode: caffe, pytorch
        """
        super(PixelUnShuffleAlign, self).__init__()
        self.dsf = downscale_factor
        self.mode = mode

    def forward(self, x):
        if len(x.size()) != 4:
            raise ValueError("input tensor shape {} is not supported.".format(x.size()))
        N, C, H, W = x.size()
        c = int(C * (self.dsf ** 2))
        h, w = H // self.dsf, W // self.dsf

        x = x.reshape(-1, C, h, self.dsf, w, self.dsf)
        if self.mode == 'caffe':
            x = x.permute(0, 3, 5, 1, 2, 4)
        elif self.mode == 'pytorch':
            x = x.permute(0, 1, 3, 5, 2, 4)
        else:
            raise NotImplementedError(
                "{} mode is not implemented".format(self.mode))

        x = x.reshape(-1, c, h, w)
        return x

if __name__ == "__main__":
    
    # implementation 1
    
    x = torch.range(start = 0, end = 31).reshape([1, 8, 2, 2])
    print('x:')
    print(x.shape)
    print(x)

    y = F.pixel_shuffle(x, 2)
    print('y:')
    print(y.shape)
    print(y)

    x_ = PixelUnShuffle.pixel_unshuffle(y, 2)
    print('x_:')
    print(x_.shape)
    print(x_)

    s = torch.randn(1, 3, 256, 256)
    ss = PixelUnShuffle.pixel_unshuffle(s, 2)
    ssss = PixelUnShuffle.pixel_unshuffle(s, 4)
    print(ss.shape, ssss.shape)

    # implementation 2
        
    subpixeldown = PixelUnShuffleAlign(downscale_factor = 2, mode = 'pytorch')
    subpixelup = PixelShuffleAlign(upscale_factor = 2, mode = 'pytorch')
    x = torch.randn(1, 3, 64, 64)
    y = subpixeldown(x)
    print(y.shape)
    z = subpixelup(y)
    print(z.shape)
    print(x - z)

    subpixeldown = PixelUnShuffleAlign(downscale_factor = 4, mode = 'pytorch')
    subpixelup = PixelShuffleAlign(upscale_factor = 4, mode = 'pytorch')
    x = torch.randn(1, 3, 64, 64)
    y = subpixeldown(x)
    print(y.shape)
    z = subpixelup(y)
    print(z.shape)
    print(x - z)
