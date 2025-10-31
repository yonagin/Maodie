%cd / kaggle / working / Maodie
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from models.encoder import Encoder
from models.decoder import Decoder
from models.quantizer import VectorQuantizer
from models.DirDisc import Discriminator


# ==================== 工具函数（保持不变）====================
def get_codebook_size(model):
    return model.vq.n_embeddings


def compute_psnr(mse):
    if mse <= 0:
        return 100.0
    return 10 * np.log10(255.0 * 255.0 / mse)


def compute_codebook_usage(all_indices, n_embeddings):
    unique_codes = len(np.unique(all_indices))
    return unique_codes / n_embeddings


def visualize_reconstructions(data, recon, filename="fig_reconstructions.png"):
    originals = (data.cpu() * 0.5 + 0.5).clamp(0, 1)
    recon = (recon.cpu() * 0.5 + 0.5).clamp(0, 1)
    comparison = torch.cat([originals, recon])
    grid = make_grid(comparison, nrow=data.shape[0])
    fig = plt.figure(figsize=(15, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("重建对比")
    plt.axis('off')
    plt.savefig(filename)
    plt.close()


def evaluate(model, test_loader, device):
    model.eval()
    total_mse, all_indices = 0, []
    codebook_size = get_codebook_size(model)

    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            x_recon, indices = model.reconstruct(data)
            data_denorm = (data * 0.5 + 0.5) * 255.0
            x_recon_denorm = (x_recon * 0.5 + 0.5) * 255.0
            all_indices.append(indices.cpu().numpy())
            total_mse += F.mse_loss(x_recon_denorm, data_denorm, reduction='sum').item()

    avg_mse = total_mse / (len(test_loader.dataset) * 3 * 32 * 32)
    psnr = compute_psnr(avg_mse)
    all_indices = np.concatenate(all_indices)
    codebook_usage = compute_codebook_usage(all_indices, codebook_size)
    return {'psnr': psnr, 'codebook_usage': codebook_usage}


# ==================== 模型类 ====================
class MaodieVQ(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers, n_embeddings, embedding_dim,
                 beta, dirichlet_alpha=0.1, temperature=1.0, patch_size=4, cosine=False):
        super().__init__()
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, 1, 1)
        self.vq = VectorQuantizer(n_embeddings, embedding_dim, beta, cosine=cosine)
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)
        self.discriminator = Discriminator(n_embeddings)

        self.temperature = temperature
        self.patch_size = patch_size
        self.dirichlet_alpha = dirichlet_alpha
        self.n_embeddings = n_embeddings

    def forward(self, x):
        z_e = self.pre_quantization_conv(self.encoder(x))
        z_q, vq_loss, perplexity, _, distances = self.vq(z_e)
        x_recon = self.decoder(z_q)
        recon_loss = F.mse_loss(x_recon, x)

        # 计算分布 p
        p = F.softmax(-distances / self.temperature, dim=-1)
        B, C, H, W = z_q.shape
        p_flat = p.view(B * H * W, -1)

        return recon_loss, vq_loss, perplexity, p_flat

    def sample_dirichlet(self, num_samples, device):
        alpha = torch.full((num_samples, self.n_embeddings), self.dirichlet_alpha, device=device)
        return torch.distributions.Dirichlet(alpha).sample()

    @torch.no_grad()
    def reconstruct(self, x):
        z_e = self.pre_quantization_conv(self.encoder(x))
        z_q, _, _, indices, _ = self.vq(z_e)
        return self.decoder(z_q), indices


# ==================== 🔥 训练函数（标准 GAN）====================
def train_step(model, data, opt_G, opt_D, lambda_adv, accum_step, device):
    """
    标准 GAN 训练步骤
    WGAN: d_loss = D(fake) - D(real), g_loss = -D(fake)
    标准GAN: d_loss = BCE(D(real), 1) + BCE(D(fake), 0), g_loss = BCE(D(fake), 1)
    """
    data = data.to(device)

    # ===== 更新生成器（每步都更新）=====
    opt_G.zero_grad()
    recon_loss, vq_loss, perplexity, p_fake = model(data)

    if lambda_adv > 0:
        D_fake = model.discriminator(p_fake)
        # 🔥 标准 GAN：使用 BCE
        g_adv_loss = F.binary_cross_entropy_with_logits(D_fake, torch.ones_like(D_fake))
    else:
        g_adv_loss = 0.0

    g_loss = recon_loss + vq_loss + lambda_adv * g_adv_loss
    g_loss.backward()
    opt_G.step()

    # ===== 更新判别器（累积 accum_step 次后更新）=====
    d_loss_val = 0.0
    if lambda_adv > 0:
        if accum_step == 0:
            opt_D.zero_grad()

        p_real = model.sample_dirichlet(p_fake.size(0), device)
        D_real = model.discriminator(p_real)
        D_fake = model.discriminator(p_fake.detach())

        # 🔥 标准 GAN：使用 BCE
        d_loss = (F.binary_cross_entropy_with_logits(D_real, torch.ones_like(D_real)) +
                  F.binary_cross_entropy_with_logits(D_fake, torch.zeros_like(D_fake))) / 4  # 除以累积步数
        d_loss.backward()

        if accum_step == 3:  # 累积4次后更新
            opt_D.step()

        d_loss_val = d_loss.item() * 4  # 恢复原始损失值

    return recon_loss.item(), vq_loss.item(), d_loss_val, perplexity.item()


# ==================== 训练循环 ====================
def train_model(model, train_loader, test_loader, total_steps, eval_interval,
                lr, lambda_adv, device):
    opt_G = optim.Adam(list(model.encoder.parameters()) +
                       list(model.pre_quantization_conv.parameters()) +
                       list(model.vq.parameters()) +
                       list(model.decoder.parameters()), lr=lr)
    opt_D = optim.Adam(model.discriminator.parameters(), lr=lr)

    losses = {'recon': [], 'vq': [], 'disc': [], 'ppl': []}

    print(f"\n{'=' * 60}")
    print(f"开始训练 - 标准 GAN (BCE 损失)")
    print(f"{'=' * 60}\n")

    step = 0
    while step < total_steps:
        model.train()
        for data, _ in train_loader:
            if step >= total_steps:
                break

            accum_step = step % 4  # 判别器每4步更新一次
            recon_loss, vq_loss, d_loss, ppl = train_step(
                model, data, opt_G, opt_D, lambda_adv, accum_step, device
            )

            losses['recon'].append(recon_loss)
            losses['vq'].append(vq_loss)
            losses['disc'].append(d_loss)
            losses['ppl'].append(ppl)
            step += 1

            # 评估
            if step % eval_interval == 0:
                start = max(0, len(losses['recon']) - eval_interval)
                print(f"\n步骤 {step}/{total_steps}:")
                print(f"  重构: {np.mean(losses['recon'][start:]):.4f}")
                print(f"  VQ: {np.mean(losses['vq'][start:]):.4f}")
                print(f"  判别器: {np.mean(losses['disc'][start:]):.4f}")
                print(f"  困惑度: {np.mean(losses['ppl'][start:]):.1f}")

                eval_res = evaluate(model, test_loader, device)
                print(f"  PSNR: {eval_res['psnr']:.2f} dB")
                print(f"  码本使用率: {eval_res['codebook_usage']:.2%}")

                test_batch, _ = next(iter(test_loader))
                with torch.no_grad():
                    model.eval()
                    recon, _ = model.reconstruct(test_batch[:8].to(device))
                    visualize_reconstructions(test_batch[:8].to(device), recon,
                                              f"recon_{step}.png")

            if step % 100 == 0:
                print(f"步骤 {step} | PPL: {ppl:.1f}", end='\r')

    return losses


# ==================== 主程序 ====================
print("标准 GAN 版本 - 简化版")

# 参数
batch_size = 128
total_steps = 20000
eval_interval = 1000
lr = 2e-4
lambda_adv = 1e-4  # 🔥 可以调这个参数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = torchvision.datasets.CIFAR100(root='./data', train=True,
                                           download=True, transform=transform)
test_data = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

# 模型
model = MaodieVQ(h_dim=32, res_h_dim=32, n_res_layers=2,
                 n_embeddings=8192, embedding_dim=32, beta=0.25).to(device)

print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")

# 训练
losses = train_model(model, train_loader, test_loader, total_steps,
                     eval_interval, lr, lambda_adv, device)

# 保存
torch.save(model.state_dict(), "model_gan.pth")
print("\n✓ 完成")
