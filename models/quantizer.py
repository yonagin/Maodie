import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 码本
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/n_embeddings, 1/n_embeddings)
        
    def forward(self, z_e):
        """
        z_e: (B, D, H, W)
        返回: z_q, loss, perplexity, encoding_indices, distances
        """
        # 转换形状: (B, D, H, W) -> (B, H, W, D) -> (B*H*W, D)
        z_e_permuted = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z_e = z_e_permuted.view(-1, self.embedding_dim)
        
        # 计算距离: (B*H*W, K)
        distances = (torch.sum(flat_z_e**2, dim=1, keepdim=True)
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.einsum('bd,dn->bn', flat_z_e, self.embedding.weight.t()))
        
        # 硬量化
        encoding_indices = torch.argmin(distances, dim=1)
        
        # 使用embedding直接获取量化结果，避免one-hot编码和矩阵乘法
        z_q_flat = self.embedding(encoding_indices)
        z_q = z_q_flat.view(z_e_permuted.shape)
        
        # 损失
        e_latent_loss = F.mse_loss(z_q.detach(), z_e_permuted)
        q_latent_loss = F.mse_loss(z_q, z_e_permuted.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 直通估计器
        z_q = z_e_permuted + (z_q - z_e_permuted).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # 困惑度（衡量码本使用多样性）
        encodings = F.one_hot(encoding_indices, self.n_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, loss, perplexity, encoding_indices, distances


class EMAVectorQuantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, beta=0.25, decay=0.99, eps=1e-5):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps
        
        # 码本
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/n_embeddings, 1/n_embeddings)
        
        # EMA参数
        self.register_buffer('cluster_size', torch.zeros(n_embeddings))
        self.register_buffer('embed_avg', self.embedding.weight.data.clone())
        
    def forward(self, z_e):
        """
        z_e: (B, D, H, W)
        返回: z_q, loss, perplexity, encoding_indices
        """
        # 转换形状: (B, D, H, W) -> (B, H, W, D) -> (B*H*W, D)
        z_e_permuted = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z_e = z_e_permuted.view(-1, self.embedding_dim)
        
        # 计算距离
        distances = (flat_z_e.pow(2).sum(dim=1, keepdim=True) 
                    + self.embedding.weight.pow(2).sum(dim=1) 
                    - 2 * torch.einsum('bd,nd->bn', flat_z_e, self.embedding.weight))
        
        # 硬量化
        encoding_indices = torch.argmin(distances, dim=1)
        encodings = F.one_hot(encoding_indices, self.n_embeddings).float()
        
        # 查表 
        z_q = self.embedding(encoding_indices).view(z_e_permuted.shape)
        
        # EMA更新（仅在训练时）
        if self.training:
            # 更新簇大小
            encodings_sum = encodings.sum(0)
            self.cluster_size.data.mul_(self.decay).add_(encodings_sum, alpha=1 - self.decay)
            
            # 更新嵌入平均值
            embed_sum = encodings.transpose(0, 1) @ flat_z_e
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            
            # 更新码本权重
            n = self.cluster_size.sum()
            smoothed_cluster_size = ((self.cluster_size + self.eps) / (n + self.n_embeddings * self.eps)) * n
            embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
            self.embedding.weight.data.copy_(embed_normalized)
        
        # 损失计算
        loss = self.beta * F.mse_loss(z_q.detach(), z_e_permuted)
        
        # 直通估计器
        z_q = z_e_permuted + (z_q - z_e_permuted).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # 困惑度
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return z_q, loss, perplexity, encoding_indices, distances
