import torch
import torch.nn as nn

# L2 loss --> expectation
criterion_L2 = nn.MSELoss()

# L1 loss --> medium
criterion_L1 = nn.L1Loss()

# L0 loss --> mode seeking (annealed version)
class L0_Loss(nn.Module):
    def __init__(self, total_epoch, reduction = 'mean'):
        super(L0_Loss, self).__init__()
        self.total_epoch = total_epoch
        self.eps = 1e-8
        self.reduction = reduction

    def forward(self, x, y, current_epoch):
        # power is annealed linearly from 2 to 0 during training; current_epoch is from 0 to (total_epoch - 1)
        power = 2 * (self.total_epoch - current_epoch) / self.total_epoch
        # base is the sum of difference and epsilon
        base = torch.abs(x - y) + self.eps
        # compute loss
        ret = base ** power
        if self.reduction != 'none':
            ret = torch.mean(ret) if self.reduction == 'mean' else torch.sum(ret)

        return ret

if __name__ == "__main__":

    a = torch.randn(4, 3, 64, 64).cuda()
    b = torch.randn(4, 3, 64, 64).cuda()
    criterion_L0 = L0_Loss(total_epoch = 100)           # the total_epoch should be pre-defined
    loss = criterion_L0(a, b, 10)
    print(loss)
    print(loss.shape)
    print(loss.item())
