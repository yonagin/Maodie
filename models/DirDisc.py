import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Discriminator(nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.net = nn.Sequential(
            # (B, 1, K) -> (B, 64, K/2)
            nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, True),
            # (B, 64, K/2) -> (B, 128, K/4)
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, True),
            # (B, 128, K/4) -> (B, 256, K/8)
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2, True),
            nn.Flatten(),
            nn.LazyLinear(1) # 使用 LazyLinear 自动推断输入维度
        )

    def forward(self, p):
        # p: (B, K)
        # Conv1d 需要 (B, C_in, L_in) 的输入格式
        p = p.unsqueeze(1) # (B, 1, K)
        return self.net(p).squeeze(-1)


class FisherDiscriminator(torch.nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.net = torch.nn.Sequential(
            spectral_norm(nn.Linear(num_embeddings, 256)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Linear(256, 128)),
            nn.LeakyReLU(0.2, True),
            spectral_norm(nn.Linear(128, 1)),
        )
        
    def forward(self, p):
        """p: (B, K) 概率向量"""
        return self.net(p).squeeze(-1)