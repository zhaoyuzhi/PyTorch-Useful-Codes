import torch

def contextual_loss(x, y, h = 0.5):
    """Computes contextual loss between x and y.
    code is from https://gist.github.com/yunjey/3105146c736f9c1055463c33b4c989da
    Args:
      x: features of shape (N, C, H, W).
      y: features of shape (N, C, H, W).
    Returns:
      cx_loss = contextual loss between x and y (Eq (1) in the paper)
    """
    assert x.size() == y.size()
    N, C, H, W = x.size()                               # e.g., 10 x 512 x 14 x 14. In this case, the number of points is 196 (14x14).

    y_mu = y.mean(3).mean(2).mean(0).reshape(1, -1, 1, 1)
    x_centered = x - y_mu
    y_centered = y - y_mu
    x_normalized = x_centered / torch.norm(x_centered, p=2, dim=1, keepdim=True)
    y_normalized = y_centered / torch.norm(y_centered, p=2, dim=1, keepdim=True)

    # The equation at the bottom of page 6 in the paper
    # Vectorized computation of cosine similarity for each pair of x_i and y_j
    x_normalized = x_normalized.reshape(N, C, -1)                               # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)                               # (N, C, H*W)
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2), y_normalized)          # (N, H*W, H*W)

    d = 1 - cosine_sim                                                          # (N, H*W, H*W)  d[n, i, j] means d_ij for n-th data 
    d_min, _ = torch.min(d, dim=2, keepdim=True)                                # (N, H*W, 1)

    # Eq (2)
    d_tilde = d / (d_min + 1e-5)

    # Eq (3)
    w = torch.exp((1 - d_tilde) / h)

    # Eq (4)
    cx_ij = w / torch.sum(w, dim=2, keepdim=True)                               # (N, H*W, H*W)

    # Eq (1)
    cx = torch.mean(torch.max(cx_ij, dim=1)[0], dim=1)                          # (N, )
    cx_loss = torch.mean(-torch.log(cx + 1e-5))
    
    return cx_loss

if __name__ == "__main__":

    a = torch.randn(4, 32, 64, 64).cuda()
    b = torch.randn(4, 32, 64, 64).cuda()
    cx = contextual_loss(a, b)
    print(cx)
    print(cx.shape)
    print(cx.item())
