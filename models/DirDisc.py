import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(nn.Module):
    def __init__(self, num_embeddings, projection_dim=256):
        """
        num_embeddings (K): 原始高维空间维度
        projection_dim (D): 投影后的低维空间维度
        """
        super().__init__()
        
        # 1. 创建投影层
        self.projection = nn.Linear(num_embeddings, projection_dim, bias=False)
        
        # 2. 特殊初始化 (例如，标准正态分布)
        # 这里的初始化方法可以根据理论选择，标准正态分布是常用的一种
        nn.init.normal_(self.projection.weight, mean=0, std=1/torch.sqrt(torch.tensor(projection_dim)))
        
        # 3. 冻结参数，使其不参与训练
        self.projection.weight.requires_grad = False
        
        # 4. 后续的可训练判别器网络
        self.net = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1),
        )
        
    def forward(self, p):
        """p: (B, K) 概率向量，可以是稀疏或稠密的"""
        # 使用冻结的投影矩阵进行降维
        projected_p = self.projection(p) # (B, D)
        
        # 将降维后的稠密向量送入后续网络
        return self.net(projected_p).squeeze(-1)


class CNNDiscriminator(nn.Module):
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