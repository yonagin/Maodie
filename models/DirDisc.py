import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class Discriminator(torch.nn.Module):
    def __init__(self, num_embeddings):
        super().__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(num_embeddings, 256),
            nn.LeakyReLU(0.2, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, True),
            nn.Linear(128, 1),
        )
        
    def forward(self, p):
        """p: (B, K) 概率向量"""
        return self.net(p).squeeze(-1)

def extract_lightweight_features(x: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    """
    一个内存和计算都极其高效的统计特征提取器。
    该函数中所有的操作都是 element-wise 或 reduction 操作，不会创建巨大的中间张量。
    
    Args:
        x (torch.Tensor): 输入张量，形状 (batch_size, vector_dim)。
        eps (float): 防止数值计算问题的小常数。

    Returns:
        torch.Tensor: 提取的低维特征张量。
    """
    
    # --- 1. 熵 (Entropy) ---
    # 计算高效，内存友好
    entropy = -torch.sum(x * torch.log(x + eps), dim=1, keepdim=True)

    # --- 2. 极值特征 ---
    # torch.max/min 是非常快的 reduction 操作
    max_val, _ = torch.max(x, dim=1, keepdim=True)
    min_val, _ = torch.min(x, dim=1, keepdim=True)

    # --- 3. Top-K 特征 (比 full sort 高效得多) ---
    # 只取最大的几个值进行分析，这对于稀疏分布非常关键
    # torch.topk 比 torch.sort 内存效率高得多
    k = 10 # 只关注最大的10个值
    topk_vals, _ = torch.topk(x, k, dim=1)
    
    # Top-K 值的和，反映了概率质量的集中度
    topk_sum = torch.sum(topk_vals, dim=1, keepdim=True)
    
    # Top-1 值与 Top-2 值的比率，可以衡量“首位突出度”
    top1_vs_top2_ratio = topk_vals[:, 0:1] / (topk_vals[:, 1:2] + eps)

    # --- 4. 矩 (Moments) ---
    # 均值、方差、偏度、峰度都是高效的 reduction 操作
    mean = torch.mean(x, dim=1, keepdim=True)
    var = torch.var(x, dim=1, unbiased=False, keepdim=True)
    
    # 为了数值稳定性，只在方差足够大时计算偏度和峰度
    std = torch.sqrt(var + eps)
    # is_const = (std < 1e-6).float() # 标记那些方差过小的样本
    
    skew = torch.mean(((x - mean) / std)**3, dim=1, keepdim=True)
    kurt = torch.mean(((x - mean) / std)**4, dim=1, keepdim=True)
    # skew = skew * (1-is_const) # 如果方差过小，偏度峰度设为0
    # kurt = kurt * (1-is_const)

    # 将所有特征拼接
    features = torch.cat([
        entropy, 
        max_val, 
        min_val,
        topk_sum,
        top1_vs_top2_ratio,
        mean,
        var, 
        skew,
        kurt
    ], dim=1)
    
    return features


class StatisticalDiscriminator(nn.Module):
    def __init__(self, num_features=9, hidden_dim=64):
        super(StatisticalDiscriminator, self).__init__()
        # 带批归一化的MLP，以应对不同特征的尺度问题
        self.mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. 提取轻量级特征
        statistical_features = extract_lightweight_features(x)
        # 2. 送入小型MLP
        output = self.mlp(statistical_features)
        return output

class RandomProjectionDiscriminator(nn.Module):
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