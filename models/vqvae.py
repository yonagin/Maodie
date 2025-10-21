
from os import path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer, EMAVectorQuantizer, SimVQ
from models.decoder import Decoder
from models.DirDisc import Discriminator, FisherDiscriminator


class MaodieVQ(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, dirichlet_alpha=0.1, temperature=1.0, patch_size=4, save_img_embedding_map=False, use_fisher=False, use_wgangp=False, lambda_gp=10.0):
        super(MaodieVQ, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vq = VectorQuantizer(
            n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        self.temperature = temperature
        self.patch_size = patch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.use_fisher = use_fisher
        self.use_wgangp = use_wgangp
        self.lambda_gp = lambda_gp

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None
        
        # 根据训练模式选择判别器类型
        if use_fisher:
            self.register_buffer('lambda_param', torch.zeros(1,))
            self.discriminator = FisherDiscriminator(n_embeddings)
        elif use_wgangp:
            self.discriminator = Discriminator(n_embeddings)
        else:
            self.discriminator = Discriminator(n_embeddings)
        
        self.dirichlet_dist = None

    def forward(self, x, verbose=False, return_loss=False):
        """
        Forward pass with optional loss computation
        
        Args:
            x: input tensor
            verbose: whether to print shapes
            return_loss: whether to return loss components
        """
        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        z_q, embedding_loss, perplexity, _, distances = self.vq(
            z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original x shape:', x.shape)
            print('encoded x shape:', z_e.shape)
            print('recon x shape:', x_hat.shape)
            assert False

        if return_loss:
            p = self.get_p(z_q, distances)
            recon_loss = F.mse_loss(x_hat, x)
            total_loss = recon_loss + embedding_loss
            return total_loss, recon_loss, embedding_loss, perplexity, p
        else:
            return embedding_loss, x_hat, perplexity

    def get_p(self, x, distances):
        p_soft = F.softmax(-distances / self.temperature, dim=-1)
        
        # 引入窗口平均操作：将逐点概率转换为窗口级区域摘要
        # 将 p_soft 从 (B*H*W, K) 重塑为 (B, H, W, K)
        p_spatial = p_soft.view(x.shape[0], x.shape[2], x.shape[3], -1)
        
        # 使用平均池化进行窗口平均
        # 将通道维度移到第二位以适配卷积操作
        p_spatial_permuted = p_spatial.permute(0, 3, 1, 2)  # (B, K, H, W)
        
        # 应用平均池化，窗口大小为 patch_size
        p_patch = F.avg_pool2d(p_spatial_permuted, 
                                kernel_size=self.patch_size, 
                                stride=self.patch_size)
        
        # 重塑为 (B*H'*W', K) 供判别器使用
        p_patch_flat = p_patch.permute(0, 2, 3, 1).reshape(-1, p_spatial.shape[-1])
        return p_patch_flat

    def sample_dirichlet_prior(self, batch_size):
        """从Dirichlet(alpha, ..., alpha)采样"""
        device = next(self.parameters()).device
        
        # 如果缓存不存在或设备不匹配，创建新的Dirichlet分布
        if self.dirichlet_dist is None or self.dirichlet_dist.concentration.device != device:
            alpha = torch.full((self.vq.n_embeddings,), self.dirichlet_alpha, device=device)
            self.dirichlet_dist = torch.distributions.Dirichlet(alpha)
        
        # 从缓存的分布中采样
        samples = self.dirichlet_dist.sample((batch_size,))
        return samples
    
    def compute_gradient_penalty(self, p_real, p_fake):
        """计算WGAN-GP的梯度惩罚项"""
        batch_size = p_real.size(0)
        
        # 生成插值样本
        alpha = torch.rand(batch_size, 1, device=p_real.device)
        alpha = alpha.expand_as(p_real)
        p_interpolated = alpha * p_real + (1 - alpha) * p_fake
        p_interpolated.requires_grad_(True)
        
        # 计算判别器在插值样本上的输出
        d_interpolated = self.discriminator(p_interpolated)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=p_interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 计算梯度范数
        gradients_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        
        # 计算梯度惩罚项
        gradient_penalty = ((gradients_norm - 1) ** 2).mean()
        
        return gradient_penalty
        
    def training_step(self, x, optimizer_G, optimizer_D, rho=1e-6, lambda_adv=1e-4):
        """执行一步对抗训练"""
        self.train()
        # Zero gradients
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()
        total_loss, recon_loss, vq_loss, perplexity, p_fake = self(x,return_loss=True)
        
        # Update discriminator
        self.discriminator.requires_grad_(True)
        
        # 生成真实样本（先验）
        p_real = self.sample_dirichlet_prior(p_fake.size(0))
        p_real = p_real.to(p_fake.dtype)
        D_real = self.discriminator(p_real)
        D_fake = self.discriminator(p_fake.detach())
        
        if self.use_fisher:
            # 二阶矩约束项 Omega(f) = 0.5 * E[f(real)^2] + 0.5 * E[f(fake)^2]
            mean_diff = torch.mean(D_real) - torch.mean(D_fake)
            omega = 0.5 * (torch.mean(D_real**2) + torch.mean(D_fake**2))
            # 约束违反度 g(λ) = 1 - Omega(f)
            constraint_violation = 1.0 - omega
            # 增广拉格朗日损失函数
            # L_D = -E(f) - λ * g(λ) + (ρ/2) * g(λ)^2
            loss_D = -mean_diff - self.lambda_param * constraint_violation + (rho / 2.0) * (constraint_violation**2)
             # --- 手动更新拉格朗日乘子 lambda ---
            # λ ← λ + ρ * g(λ)
            self.lambda_param = self.lambda_param + rho * constraint_violation.detach()

        elif self.use_wgangp:
            # WGAN-GP判别器损失
            loss_D = torch.mean(D_fake) - torch.mean(D_real)
            
            # 计算梯度惩罚项
            gradient_penalty = self.compute_gradient_penalty(p_real, p_fake.detach())
            loss_D = loss_D + self.lambda_gp * gradient_penalty

        else:
            # 标准判别器损失
            loss_D_real = -torch.log(torch.sigmoid(D_real) + 1e-8).mean()
            loss_D_fake = -torch.log(1 - torch.sigmoid(D_fake) + 1e-8).mean()
            loss_D = loss_D_real + loss_D_fake
            
        # 更新判别器参数
        loss_D.backward()
        optimizer_D.step()
    
        # Update generator
        self.discriminator.requires_grad_(False)
        D_fake = self.discriminator(p_fake)

        if self.use_fisher or self.use_wgangp:
            loss_adv = -torch.mean(D_fake)
        else:
            loss_adv = -torch.log(torch.sigmoid(D_fake) + 1e-8).mean()

        # Recompute generator loss
        total_loss_G =  total_loss + lambda_adv * loss_adv
        total_loss_G.backward()
        optimizer_G.step()
        
        return recon_loss.item(), vq_loss.item(), loss_D.item(), perplexity.item()
    
    @torch.no_grad()
    def reconstruct(self, x):
        """
        Reconstruct input
        """
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _, encoding_indices, _ = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, encoding_indices


class SimVQModel(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, commitment_cost=0.25, save_img_embedding_map=False):
        super(SimVQModel, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through SimVQ discretization bottleneck
        self.vq = SimVQ(
            n_embeddings, embedding_dim, commitment_cost)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False, return_loss=False):
        """
        Forward pass with optional loss computation
        
        Args:
            x: input tensor
            verbose: whether to print shapes
            return_loss: whether to return loss components
        """
        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        z_q, embedding_loss, perplexity, _, _ = self.vq(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        if return_loss:
            recon_loss = F.mse_loss(x_hat, x)
            total_loss = recon_loss + embedding_loss
            return total_loss, recon_loss, embedding_loss, perplexity, x_hat
        else:
            return embedding_loss, x_hat, perplexity
    
    def training_step(self, x, optimizer):
        """
        执行一步训练
        
        Args:
            x: 输入张量
            optimizer: 优化器
        """
        self.train()
        optimizer.zero_grad()
        
        # 前向传播
        total_loss, recon_loss, embedding_loss, perplexity, _ = self(x, return_loss=True)
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        return recon_loss.item(), embedding_loss.item(), perplexity.item()
    
    @torch.no_grad()
    def reconstruct(self, x):
        """
        Reconstruct input
        """
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _, encoding_indices, _ = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, encoding_indices


class EMAVQ(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta=0.25, decay=0.99, eps=1e-5, save_img_embedding_map=False):
        super(EMAVQ, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(
            h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through EMA discretization bottleneck
        self.vq = EMAVectorQuantizer(
            n_embeddings, embedding_dim, beta, decay, eps)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x, verbose=False, return_loss=False):
        """
        Forward pass with optional loss computation
        
        Args:
            x: input tensor
            verbose: whether to print shapes
            return_loss: whether to return loss components
        """
        z_e = self.encoder(x)

        z_e = self.pre_quantization_conv(z_e)
        z_q, embedding_loss, perplexity, _, _ = self.vq(z_e)
        x_hat = self.decoder(z_q)

        if verbose:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        if return_loss:
            recon_loss = F.mse_loss(x_hat, x)
            total_loss = recon_loss + embedding_loss
            return total_loss, recon_loss, embedding_loss, perplexity, x_hat
        else:
            return embedding_loss, x_hat, perplexity
    
    def training_step(self, x, optimizer):
        """
        执行一步训练
        
        Args:
            x: 输入张量
            optimizer: 优化器
        """
        self.train()
        optimizer.zero_grad()
        
        # 前向传播
        total_loss, recon_loss, embedding_loss, perplexity, _ = self(x, return_loss=True)
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        return recon_loss.item(), embedding_loss.item(), perplexity.item()
    
    @torch.no_grad()
    def reconstruct(self, x):
        """
        Reconstruct input
        """
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _, encoding_indices, _ = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, encoding_indices
