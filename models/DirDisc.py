import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(torch.nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(num_embeddings, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, True),
            nn.Linear(64, 1),
        )
        
    def forward(self, p):
        """p: (B, K) 概率向量"""
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