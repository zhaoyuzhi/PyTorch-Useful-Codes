import torch
import torch.nn as nn

class MultiHeadSelfAttn(nn.Module):
    def __init__(self, latent_dim, num_of_head, attn_dropout_ratio = 0.5, out_dropout_ratio = 0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_of_head = num_of_head

        assert latent_dim % num_of_head == 0
        self.head_dim = latent_dim // num_of_head

        self.q = nn.Linear(latent_dim, latent_dim)
        self.k = nn.Linear(latent_dim, latent_dim)
        self.v = nn.Linear(latent_dim, latent_dim)
        self.out = nn.Linear(latent_dim, latent_dim)

        self.attn_dropout = nn.Dropout()