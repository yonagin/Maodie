# ======================================================================
# === train_gumbel_vqvae.py (V34 - 修复 NameError 的最终完整版) ===
# ======================================================================

import os, torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchvision, torchvision.transforms as transforms
import numpy as np, time, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import deque
from tqdm import tqdm
from torchvision.utils import make_grid
import types

from models.maodie_vqvae_comparison import MaodieVQ_Comparison
from utils import compute_psnr


def normalization_shift(x): return x - 0.5


# ======================================================================
# 1. 全新的、正确的 GumbelQuantizer
# ======================================================================
class GumbelQuantizer(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost, temperature=1.0):
        super().__init__()
        self.n_embeddings = n_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / n_embeddings, 1 / n_embeddings)
        self.temperature = temperature

    def forward(self, z_e):
        z_e_permuted = z_e.permute(0, 2, 3, 1).contiguous()
        flat_z_e = z_e_permuted.view(-1, self.embedding_dim)

        distances = (torch.sum(flat_z_e ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_z_e, self.embedding.weight.t()))
        logits = -distances

        # --- 核心修正：使用 hard=True ---
        # Gumbel-Softmax (hard=True) 会输出 one-hot 向量，但梯度会像 soft 版本一样传播
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=1)

        # 使用 one-hot 向量进行加权平均，得到离散的 z_q
        z_q_flat = torch.matmul(soft_one_hot, self.embedding.weight)
        z_q = z_q_flat.view(z_e_permuted.shape)

        # VQ 损失保持不变
        vq_loss = F.mse_loss(z_q, z_e_permuted.detach()) + self.commitment_cost * F.mse_loss(z_q.detach(), z_e_permuted)

        # 直通估计器 (STE for decoder)
        z_q_ste = z_e_permuted + (z_q - z_e_permuted).detach()
        z_q_ste = z_q_ste.permute(0, 3, 1, 2).contiguous()

        # 计算 Perplexity 和 encoding_indices
        encoding_indices = soft_one_hot.argmax(dim=1)
        avg_probs = soft_one_hot.mean(dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return z_q_ste, vq_loss, perplexity, encoding_indices


# ======================================================================
# 2. 模型和训练逻辑 (与之前版本基本相同，但现在会正确工作)
# ======================================================================
class Gumbel_VQVAE(MaodieVQ_Comparison):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("--- Replacing VectorQuantizer with CORRECT GumbelQuantizer ---")
        self.vq = GumbelQuantizer(
            n_embeddings=kwargs.get('n_embeddings'), embedding_dim=kwargs.get('embedding_dim'),
            commitment_cost=kwargs.get('beta'), temperature=kwargs.get('temperature')
        )

    def training_step(self, x, optimizer):
        self.train();
        optimizer.zero_grad()
        z_e = self.encoder(x);
        z_e_prequant = self.pre_quantization_conv(z_e)
        z_q, vq_loss, perplexity, _ = self.vq(z_e_prequant)
        x_hat = self.decoder(z_q)
        recon_loss = F.mse_loss(x_hat, x)
        total_loss = recon_loss + vq_loss
        total_loss.backward();
        optimizer.step()
        return recon_loss.item(), vq_loss.item(), perplexity.item()

    def reconstruct(self, x):
        z_e = self.encoder(x);
        z_e_prequant = self.pre_quantization_conv(z_e)
        z_q, _, _, indices = self.vq(z_e_prequant)
        x_recon = self.decoder(z_q)
        return x_recon, indices


MaodieVQ = Gumbel_VQVAE


# (为简洁，将上面省略的代码补充完整)
def get_codebook_size(model):
    if hasattr(model, 'vq') and hasattr(model.vq, 'n_embeddings'):
        return model.vq.n_embeddings
    else:
        raise AttributeError("Could not determine codebook size.")


def compute_codebook_usage(all_indices, n_embeddings):
    unique_codes = len(np.unique(all_indices));
    return unique_codes / n_embeddings


def evaluate(model, test_loader, device):
    model.eval();
    total_mse, all_indices = 0, [];
    codebook_size = get_codebook_size(model)
    with torch.no_grad():
        for data, _ in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device);
            x_recon, indices = model.reconstruct(data)
            data_denorm = (data + 0.5) * 255.0;
            x_recon_denorm = (x_recon + 0.5) * 255.0
            data_denorm = data_denorm.clamp(0, 255);
            x_recon_denorm = x_recon_denorm.clamp(0, 255)
            if indices is not None: all_indices.append(indices.cpu().numpy())
            total_mse += F.mse_loss(x_recon_denorm, data_denorm, reduction='sum').item()
    avg_mse = total_mse / (len(test_loader.dataset) * data.size(1) * data.size(2) * data.size(3))
    psnr = compute_psnr(avg_mse)
    if len(all_indices) > 0:
        all_indices = np.concatenate(all_indices).flatten();
        codebook_usage = compute_codebook_usage(all_indices, codebook_size)
    else:
        codebook_usage = 0
    return {'psnr': psnr, 'mse': avg_mse, 'codebook_usage': codebook_usage}


def visualize_reconstructions(model, test_loader, device, filename):
    model.eval()
    with torch.no_grad():
        data, _ = next(iter(test_loader));
        data = data[:8].to(device)
        # --- 核心 BUG 修复！ ---
        x_recon, _ = model.reconstruct(data)
    originals = (data.cpu() + 0.5).clamp(0, 1);
    recon = (x_recon.cpu() + 0.5).clamp(0, 1)  # 使用 x_recon
    comparison = torch.cat([originals, recon]);
    grid = make_grid(comparison, nrow=data.shape[0])
    plt.figure(figsize=(15, 6));
    plt.imshow(grid.permute(1, 2, 0));
    plt.title("Reconstruction Comparison");
    plt.yticks([]);
    plt.xticks([]);
    plt.tight_layout();
    plt.savefig(filename);
    plt.close()

# ======================================================================
# 5. 主执行逻辑
# ======================================================================
if __name__ == "__main__":
    batch_size = 128;
    total_training_steps = 150000;
    eval_interval = 1000
    lr = 1e-3;
    n_embeddings = 512;
    embedding_dim = 16
    commitment_cost = 1.0
    h_dim = 128;
    res_h_dim = 128;
    n_res_layers = 2
    temp_init = 1.0;
    temp_final = 0.1;
    temp_anneal_steps = 50000
    num_workers = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    print(f"Using device: {device}")

    print("\n=== Gumbel-Softmax VQ-VAE Experiment ===")
    output_dir = "./results_gumbel_vqvae"
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(normalization_shift)])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    model = MaodieVQ(
        h_dim=h_dim, res_h_dim=res_h_dim, n_res_layers=n_res_layers,
        n_embeddings=n_embeddings, embedding_dim=embedding_dim,
        beta=commitment_cost, temperature=temp_init
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training for Gumbel-Softmax VQ-VAE...");
    step, start_time = 0, time.time()
    recon_losses, vq_losses, perplexities = [deque(maxlen=eval_interval) for _ in range(3)]

    while step < total_training_steps:
        for data, _ in train_loader:
            if step >= total_training_steps: break
            data = data.to(device)

            if step < temp_anneal_steps:
                model.vq.temperature = temp_init * ((temp_final / temp_init) ** (step / temp_anneal_steps))

            recon_loss, vq_loss, perplexity = model.training_step(data, optimizer)

            recon_losses.append(recon_loss);
            vq_losses.append(vq_loss);
            perplexities.append(perplexity)
            step += 1

            if step > 0 and step % eval_interval == 0:
                elapsed_time = time.time() - start_time;
                steps_per_sec = eval_interval / elapsed_time
                print(
                    f"\n--- Step {step}/{total_training_steps} (Speed: {steps_per_sec:.2f} step/s, Temp: {model.vq.temperature:.4f}) ---")
                print(f"  Avg Losses (train): Recon={np.mean(recon_losses):.4f}, VQ={np.mean(vq_losses):.4f}")
                print(f"  Avg PPL (train): {np.mean(perplexities):.2f}")
                with torch.no_grad():
                    model.eval();
                    eval_results = evaluate(model, test_loader, device)
                    print(
                        f"  Test Metrics: PSNR={eval_results['psnr']:.2f} dB, Codebook Usage={eval_results['codebook_usage']:.2%}")
                    visualize_reconstructions(model, test_loader, device,
                                              os.path.join(output_dir, f"reconstructions_step_{step}.png"))
                    print(f"  > Saved reconstruction image.")
                start_time = time.time();
                model.train()

    print("\n=== Final Evaluation ===");
    eval_results = evaluate(model, test_loader, device)
    print(f"Final PSNR: {eval_results['psnr']:.2f} dB, Codebook Usage: {eval_results['codebook_usage']:.2%}")
    model_save_path = os.path.join(output_dir, f"gumbel_vqvae_model_{n_embeddings}.pth")
    torch.save(model.state_dict(), model_save_path);
    print(f"Model saved to: {model_save_path}")
    print("\n=== Experiment Completed ===")