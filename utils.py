import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
from tqdm import tqdm  # 你的文件里有这个，我就保留了
import math


# --- 新增的辅助函数 ---
def get_codebook_size(model):
    """智能地从不同类型的模型中获取码本大小。"""
    if hasattr(model, 'vq') and hasattr(model.vq, 'n_embeddings'):
        return model.vq.n_embeddings  # For VQ-VAE based models
    elif hasattr(model, 'num_codes'):
        return model.num_codes  # For our FSQ_VQVAE model
    else:
        raise AttributeError("Could not determine codebook size for the given model.")


# --- 修改后的核心函数 ---

def compute_psnr(mse):
    """计算 PSNR (假设 mse 是基于 [0, 255] 范围计算的)。"""
    if mse <= 0:  # 避免 log(0)
        return 100.0
    return 10 * np.log10(255.0 * 255.0 / mse)


def compute_codebook_usage(all_indices, n_embeddings):
    """计算码本利用率。"""
    unique_codes = len(np.unique(all_indices))
    return unique_codes / n_embeddings


def visualize_reconstructions(data, recon, filename="fig_reconstructions"):
    """可视化重建对比 (无需修改)。"""
    print(f"\nGenerating reconstruction comparison: {filename}")
    originals = (data.cpu() * 0.5 + 0.5).clamp(0, 1)
    recon = (recon.cpu() * 0.5 + 0.5).clamp(0, 1)
    comparison = torch.cat([originals, recon])
    grid = make_grid(comparison, nrow=data.shape[0])
    fig = plt.figure(figsize=(15, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Reconstruction Comparison", fontsize=16)
    plt.yticks([originals.shape[2] * 0.5, originals.shape[2] * 1.5], ["Original", "Recon"], rotation=90, va='center',
               fontsize=12)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved '{filename}'")


def evaluate(model, test_loader, device):
    """评估模型性能 (已修改为通用)。"""
    model.eval()
    total_mse, all_indices = 0, []
    codebook_size = get_codebook_size(model)  # 使用辅助函数

    with torch.no_grad():
        print("Running evaluation on test set...")
        for data, _ in tqdm(test_loader, desc="Evaluating"):
            data = data.to(device)
            x_recon, indices = model.reconstruct(data)

            data_denorm = (data * 0.5 + 0.5) * 255.0
            x_recon_denorm = (x_recon * 0.5 + 0.5) * 255.0

            data_denorm = data_denorm.clamp(0, 255)
            x_recon_denorm = x_recon_denorm.clamp(0, 255)

            all_indices.append(indices.cpu().numpy())
            total_mse += F.mse_loss(x_recon_denorm, data_denorm, reduction='sum').item()

    avg_mse = total_mse / (len(test_loader.dataset) * 3 * 32 * 32)
    psnr = compute_psnr(avg_mse)

    all_indices = np.concatenate(all_indices)
    codebook_usage = compute_codebook_usage(all_indices, codebook_size)

    return {'psnr': psnr, 'mse': avg_mse, 'codebook_usage': codebook_usage}


# --- visualize_latent_space 对于 FSQ 不适用，我们先注释掉或简化 ---
# FSQ 没有一个显式的 codebook 可以画出来。为了让脚本能跑通，
# 一个简单的处理是让这个函数只在模型是 VQ-VAE 时才执行。

def visualize_latent_space(model, data_loader, device, title, filename, n_batches=5):
    """可视化潜空间 (已修改，对 FSQ 模型会跳过)。"""

    # 检查模型是否是标准的 VQ-VAE 类型
    if not (hasattr(model, 'vq') and hasattr(model.vq, 'embedding')):
        print(f"\nSkipping latent space visualization for {title}: Model is not a standard VQ-VAE (e.g., FSQ).")
        return

    print(f"\nGenerating latent space visualization for {title}...")
    # ... (原始的 visualize_latent_space 代码保持不变)
    model.eval()
    all_z_e_flat = []
    used_indices = set()
    with torch.no_grad():
        for i, (data, _) in enumerate(tqdm(data_loader, total=n_batches, desc="Visualizing latent space")):
            if i >= n_batches:
                break
            data = data.to(device)
            z_e = model.encoder(data)
            z_e = model.pre_quantization_conv(z_e)
            all_z_e_flat.append(z_e.permute(0, 2, 3, 1).reshape(-1, model.vq.embedding_dim))
            _, _, _, indices, _ = model.vq(z_e)
            used_indices.update(indices.cpu().numpy().flatten())

    z_e_flat = torch.cat(all_z_e_flat, dim=0).cpu().numpy()
    codebook = model.vq.embedding.weight.data.cpu().numpy()
    tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
    combined = np.vstack([z_e_flat, codebook])
    tsne_results = tsne.fit_transform(combined)
    z_e_tsne, codebook_tsne = tsne_results[:len(z_e_flat)], tsne_results[len(z_e_flat):]

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(z_e_tsne[:, 0], z_e_tsne[:, 1], alpha=0.1, label='Encoder Outputs (z_e)')
    used_code_vectors = np.array([c for i, c in enumerate(codebook_tsne) if i in used_indices])
    unused_code_vectors = np.array([c for i, c in enumerate(codebook_tsne) if i not in used_indices])

    if len(used_code_vectors) > 0:
        plt.scatter(used_code_vectors[:, 0], used_code_vectors[:, 1], color='orange', edgecolor='black', s=80,
                    label='Used Codebook Vectors')
    if len(unused_code_vectors) > 0:
        plt.scatter(unused_code_vectors[:, 0], unused_code_vectors[:, 1], color='gray', marker='x', s=80,
                    label='Unused Codebook Vectors')

    plt.title(f"Latent Space and Codebook Coverage: {title}", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved '{filename}'")


def visualize_codebook_usage(model, data_loader, device, title, filename):
    """可视化码本使用频率 (已修改为通用)。"""
    print(f"\nGenerating codebook usage histogram for {title}...")
    model.eval()
    all_indices = []
    codebook_size = get_codebook_size(model)  # 使用辅助函数

    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc=f"Collecting indices for {title}"):
            data = data.to(device)
            _, indices = model.reconstruct(data)
            all_indices.append(indices.cpu().numpy())

    all_indices = np.concatenate(all_indices).flatten()
    counts = np.bincount(all_indices, minlength=codebook_size)
    usage_rate = len(np.unique(all_indices)) / codebook_size

    fig = plt.figure(figsize=(10, 5))
    plt.bar(range(codebook_size), counts, width=1.0)
    plt.title(f"Codebook Usage: {title} (Usage: {usage_rate:.2%})", fontsize=14)
    plt.xlabel("Codebook Index")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved '{filename}'")