# 文件名: models/maodie_vqvae_comparison.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import VectorQuantizer
from models.DirDisc import Discriminator


class MaodieVQ_Comparison(nn.Module):
    # 在 __init__ 中增加一个 use_standard_gan 参数
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, dirichlet_alpha=0.1,
                 use_standard_gan=False, **kwargs):
        super(MaodieVQ_Comparison, self).__init__()

        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)

        self.pre_quantization_conv = nn.Conv2d(
            h_dim * 4, embedding_dim, kernel_size=1, stride=1)

        self.vq = VectorQuantizer(n_embeddings, embedding_dim, beta)

        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        self.discriminator = Discriminator(n_embeddings)
        self.dirichlet_alpha = dirichlet_alpha
        self.temperature = kwargs.get('temperature', 1.0)
        self.patch_size = kwargs.get('patch_size', 4)
        self.dirichlet_dist = None

        # 保存 GAN 类型
        self.use_standard_gan = use_standard_gan

    def forward(self, x, return_loss=False):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, vq_loss, perplexity, _, distances = self.vq(z_e)
        x_hat = self.decoder(z_q)

        if return_loss:
            p_fake = self.get_p(z_q, distances)
            recon_loss = F.mse_loss(x_hat, x)
            total_loss = recon_loss + vq_loss
            return total_loss, recon_loss, vq_loss, perplexity, p_fake
        else:
            return x_hat, z_q, vq_loss, perplexity, distances

    def get_p(self, z_q, distances):
        p_soft = F.softmax(-distances / self.temperature, dim=-1)
        p_spatial = p_soft.view(z_q.shape[0], z_q.shape[2], z_q.shape[3], -1)
        p_spatial_permuted = p_spatial.permute(0, 3, 1, 2)
        p_patch = F.avg_pool2d(p_spatial_permuted, kernel_size=self.patch_size, stride=self.patch_size)
        p_patch_flat = p_patch.permute(0, 2, 3, 1).reshape(-1, p_spatial.shape[-1])
        return p_patch_flat

    def sample_dirichlet_prior(self, batch_size):
        device = next(self.parameters()).device
        if self.dirichlet_dist is None or self.dirichlet_dist.concentration.device != device:
            alpha = torch.full((self.vq.n_embeddings,), self.dirichlet_alpha, device=device)
            self.dirichlet_dist = torch.distributions.Dirichlet(alpha)
        samples = self.dirichlet_dist.sample((batch_size,))
        return samples

    # ======================================================================
    # 这是我们修改的核心，training_step 现在支持两种 GAN 模式
    # ======================================================================
    def training_step(self, x, optimizer_G, optimizer_D, lambda_adv=1e-4):
        self.train()
        optimizer_D.zero_grad()
        optimizer_G.zero_grad()

        total_loss, recon_loss, vq_loss, perplexity, p_fake = self(x, return_loss=True)

        # --- Update Discriminator ---
        self.discriminator.requires_grad_(True)
        p_real = self.sample_dirichlet_prior(p_fake.size(0))
        D_real = self.discriminator(p_real)
        D_fake = self.discriminator(p_fake.detach())

        if self.use_standard_gan:
            # 标准 GAN 损失
            loss_D_real = F.binary_cross_entropy_with_logits(D_real, torch.ones_like(D_real))
            loss_D_fake = F.binary_cross_entropy_with_logits(D_fake, torch.zeros_like(D_fake))
            loss_D = (loss_D_real + loss_D_fake) / 2
        else:
            # WGAN 损失 (默认)
            loss_D = torch.mean(D_fake) - torch.mean(D_real)

        loss_D.backward()
        optimizer_D.step()

        # --- Update Generator ---
        self.discriminator.requires_grad_(False)
        D_fake = self.discriminator(p_fake)

        if self.use_standard_gan:
            # 标准 GAN 损失
            loss_adv = F.binary_cross_entropy_with_logits(D_fake, torch.ones_like(D_fake))
        else:
            # WGAN 损失 (默认)
            loss_adv = -torch.mean(D_fake)

        total_loss_G = total_loss + lambda_adv * loss_adv
        total_loss_G.backward()
        optimizer_G.step()

        return recon_loss.item(), vq_loss.item(), loss_D.item(), perplexity.item()

    # ======================================================================

    @torch.no_grad()
    def reconstruct(self, x):
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)
        z_q, _, _, encoding_indices, _ = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, encoding_indices


# 为了兼容性
MaodieVQ = MaodieVQ_Comparison