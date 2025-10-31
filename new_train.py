# ======================================================================
# === 最终版 new_train.py (CIFAR-100, 8192码本, 标签平滑, detach) ===
# ======================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import deque
from tqdm import tqdm
from torchvision.utils import make_grid
import types

from models.maodie_vqvae_comparison import MaodieVQ_Comparison
from utils import compute_psnr

def normalization_shift(x):
    return x - 0.5

class LabelSmoothedGAN_MaodieVQ(MaodieVQ_Comparison):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_smoothing = kwargs.get('label_smoothing', 0.1)
    def forward(self, x, return_loss=False):
        z_e = self.encoder(x); z_e = self.pre_quantization_conv(z_e)
        z_q, vq_loss, perplexity, _, distances = self.vq(z_e); x_hat = self.decoder(z_q)
        if return_loss:
            p_fake = self.get_p(z_q, distances); recon_loss = F.mse_loss(x_hat, x)
            total_loss = recon_loss + vq_loss
            return total_loss, recon_loss, vq_loss, perplexity, p_fake
        else: return x_hat, z_q
    def get_p(self, z_q, distances):
        p_soft = F.softmax(-distances / self.temperature, dim=-1); p_spatial = p_soft.view(z_q.shape[0], z_q.shape[2], z_q.shape[3], -1)
        p_spatial_permuted = p_spatial.permute(0, 3, 1, 2); p_patch = F.avg_pool2d(p_spatial_permuted, kernel_size=self.patch_size, stride=self.patch_size)
        p_patch_flat = p_patch.permute(0, 2, 3, 1).reshape(-1, p_spatial.shape[-1]); return p_patch_flat
    def sample_dirichlet_prior(self, batch_size):
        device = next(self.parameters()).device
        if self.dirichlet_dist is None or self.dirichlet_dist.concentration.device != device:
            alpha = torch.full((self.vq.n_embeddings,), self.dirichlet_alpha, device=device); self.dirichlet_dist = torch.distributions.Dirichlet(alpha)
        return self.dirichlet_dist.sample((batch_size,))
    def training_step(self, x, optimizer_G, optimizer_D, lambda_adv=1e-4):
        self.train(); optimizer_D.zero_grad(); self.discriminator.requires_grad_(True)
        with torch.no_grad(): _, _, _, _, p_fake = self.forward(x, return_loss=True)
        p_real = self.sample_dirichlet_prior(p_fake.size(0)); D_real = self.discriminator(p_real); D_fake = self.discriminator(p_fake.detach())
        real_labels = torch.full_like(D_real, 1.0 - self.label_smoothing)
        fake_labels = torch.full_like(D_fake, self.label_smoothing)
        loss_D_real = F.binary_cross_entropy_with_logits(D_real, real_labels); loss_D_fake = F.binary_cross_entropy_with_logits(D_fake, fake_labels)
        loss_D = (loss_D_real + loss_D_fake) / 2; loss_D.backward(); optimizer_D.step()
        optimizer_G.zero_grad(); self.discriminator.requires_grad_(False)
        total_loss, recon_loss, vq_loss, perplexity, p_fake = self.forward(x, return_loss=True)
        D_fake_for_G = self.discriminator(p_fake); loss_adv = F.binary_cross_entropy_with_logits(D_fake_for_G, torch.ones_like(D_fake_for_G))
        total_loss_G = total_loss + lambda_adv * loss_adv; total_loss_G.backward(); optimizer_G.step()
        return recon_loss.item(), vq_loss.item(), loss_D.item(), perplexity.item(), loss_adv.item()

MaodieVQ = LabelSmoothedGAN_MaodieVQ


def get_codebook_size(model):
    if hasattr(model, 'vq') and hasattr(model.vq, 'n_embeddings'): return model.vq.n_embeddings
    else: raise AttributeError("Could not determine codebook size.")
def compute_codebook_usage(all_indices, n_embeddings):
    unique_codes = len(np.unique(all_indices)); return unique_codes / n_embeddings
def evaluate(model, test_loader, device):
    model.eval(); total_mse, all_indices = 0, []; codebook_size = get_codebook_size(model)
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device); x_recon, indices = model.reconstruct(data)
            data_denorm = (data + 0.5) * 255.0; x_recon_denorm = (x_recon + 0.5) * 255.0
            data_denorm = data_denorm.clamp(0, 255); x_recon_denorm = x_recon_denorm.clamp(0, 255)
            all_indices.append(indices.cpu().numpy()); total_mse += F.mse_loss(x_recon_denorm, data_denorm, reduction='sum').item()
    avg_mse = total_mse / (len(test_loader.dataset) * data.size(1) * data.size(2) * data.size(3))
    psnr = compute_psnr(avg_mse); all_indices = np.concatenate(all_indices).flatten()
    codebook_usage = compute_codebook_usage(all_indices, codebook_size)
    return {'psnr': psnr, 'mse': avg_mse, 'codebook_usage': codebook_usage}
def visualize_reconstructions(data, recon, filename="fig_reconstructions"):
    originals = (data.cpu() + 0.5).clamp(0, 1); recon = (recon.cpu() + 0.5).clamp(0, 1)
    comparison = torch.cat([originals, recon]); grid = make_grid(comparison, nrow=data.shape[0])
    plt.figure(figsize=(15, 6)); plt.imshow(grid.permute(1, 2, 0)); plt.title("Reconstruction Comparison"); plt.yticks([]); plt.xticks([]); plt.tight_layout(); plt.savefig(filename); plt.close()

# --- 核心修改点 1: 数据加载函数 ---
def load_cifar100_dataset():
    print("Loading CIFAR-100 dataset...");
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(normalization_shift)])
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Training set size: {len(trainset)}, Test set size: {len(testset)}")
    return train_loader, test_loader

def setup_optimizers(model):
    optimizer_G = optim.Adam(list(model.encoder.parameters()) + list(model.pre_quantization_conv.parameters()) + list(model.vq.parameters()) + list(model.decoder.parameters()), lr=lr)
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr)
    return optimizer_G, optimizer_D

def train_model(model, train_loader, test_loader, output_dir):
    optimizer_G, optimizer_D = setup_optimizers(model)
    recon_losses, vq_losses, disc_losses, perplexities, adv_losses = [deque(maxlen=eval_interval) for _ in range(5)]
    print(f"Starting training with Label Smoothed GAN loss...");
    step, start_time = 0, time.time()
    while step < total_training_steps:
        for data, _ in train_loader:
            if step >= total_training_steps: break
            data = data.to(device)
            recon_loss, vq_loss, disc_loss, perplexity, adv_loss = model.training_step(data, optimizer_G, optimizer_D, lambda_adv)
            if np.isnan(recon_loss) or np.isnan(vq_loss):
                print(f"\nERROR: NaN detected at step {step}. Stopping training."); return
            recon_losses.append(recon_loss); vq_losses.append(vq_loss); disc_losses.append(disc_loss); perplexities.append(perplexity); adv_losses.append(adv_loss)
            step += 1
            if step > 0 and step % eval_interval == 0:
                elapsed_time = time.time() - start_time; steps_per_sec = eval_interval / elapsed_time
                print(f"\n--- Step {step}/{total_training_steps} (Speed: {steps_per_sec:.2f} step/s) ---")
                with torch.no_grad():
                    model.eval(); eval_results = evaluate(model, test_loader, device)
                    print(f"  Avg Losses (train): Recon={np.mean(recon_losses):.4f}, VQ={np.mean(vq_losses):.4f}, Adv={np.mean(adv_losses):.4f}, Disc={np.mean(disc_losses):.4f}")
                    print(f"  Avg PPL (train): {np.mean(perplexities):.2f}")
                    print(f"  Test Metrics: PSNR={eval_results['psnr']:.2f} dB, Codebook Usage={eval_results['codebook_usage']:.2%}")
                    test_batch, _ = next(iter(test_loader)); test_batch = test_batch[:8].to(device)
                    recon_batch, _ = model.reconstruct(test_batch)
                    visualize_reconstructions(test_batch, recon_batch, os.path.join(output_dir, f"reconstructions_step_{step}.png"))
                    print(f"  > Saved reconstruction image.")
                start_time = time.time(); model.train()
    print(f"\nTraining completed!")

if __name__ == "__main__":
    # --- 核心修改点 2: 实验参数 ---
    batch_size = 128;
    total_training_steps = 60000;
    eval_interval = 1000
    lr = 2e-4;
    n_embeddings = 8192; # <--- 改回 8192
    embedding_dim = 32;   # <--- 改回 32
    commitment_cost = 0.25;
    lambda_adv = 1e-4
    h_dim = 128;
    res_h_dim = 128;
    n_res_layers = 2
    label_smoothing = 0.1;
    temperature = 1.0;
    dirichlet_alpha = 0.1;
    patch_size = 4
    num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    print(f"Using device: {device}")

    print("\n=== MaodieVQ on CIFAR-100 (8192 Codebook, Label Smoothing GAN, .detach() trick) ===")
    output_dir = "./results_cifar100_ls_detach_8192"; # <--- 新的专属文件夹
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: {output_dir}")

    # --- 核心修改点 3: 加载数据集 ---
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(normalization_shift)])
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader, test_loader = load_cifar100_dataset()

    model = MaodieVQ(h_dim=h_dim, res_h_dim=res_h_dim, n_res_layers=n_res_layers, n_embeddings=n_embeddings,
                     embedding_dim=embedding_dim,
                     beta=commitment_cost, dirichlet_alpha=dirichlet_alpha, temperature=temperature,
                     patch_size=patch_size, label_smoothing=label_smoothing
                     ).to(device)

    print("--- Applying .detach() trick to the VectorQuantizer ---")
    def patched_vq_forward(self, z_e):
        z_e_permuted = z_e.permute(0, 2, 3, 1).contiguous(); flat_z_e = z_e_permuted.view(-1, self.embedding_dim)
        distances = (torch.sum(flat_z_e.detach() ** 2, dim=1, keepdim=True) + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.einsum('bd,dn->bn', flat_z_e.detach(), self.embedding.weight.t()))
        encoding_indices = torch.argmin(distances, dim=1); z_q_flat = self.embedding(encoding_indices)
        z_q = z_q_flat.view(z_e_permuted.shape)
        commitment_loss = F.mse_loss(z_q.detach(), z_e_permuted)
        codebook_loss = F.mse_loss(z_q, z_e_permuted.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        z_q = z_e_permuted + (z_q - z_e_permuted).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous();
        encodings = F.one_hot(encoding_indices, self.n_embeddings).float()
        avg_probs = torch.mean(encodings, dim=0); perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        return z_q, loss, perplexity, encoding_indices, distances
    model.vq.forward = types.MethodType(patched_vq_forward, model.vq)

    train_model(model, train_loader, test_loader, output_dir)

    print("\n=== Final Evaluation ===");
    eval_results = evaluate(model, test_loader, device)
    print(f"Final PSNR: {eval_results['psnr']:.2f} dB, Codebook Usage: {eval_results['codebook_usage']:.2%}")
    model_save_path = os.path.join(output_dir, f"maodie_model_cifar100_ls_detach_{n_embeddings}.pth")
    torch.save(model.state_dict(), model_save_path);
    print(f"Model saved to: {model_save_path}")
    print("\n=== Battle Completed ===")