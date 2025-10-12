import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
from tqdm import tqdm


def compute_psnr(mse):
    if mse == 0:
        return 100
    return 10 * np.log10(4.0 / mse)

def compute_codebook_usage(all_indices, num_embeddings):
    unique_codes = len(np.unique(all_indices))
    return unique_codes / num_embeddings

def visualize_reconstructions(data, recon):
    """Visualize reconstruction comparison"""
    print("\nGenerating reconstruction comparison...")
    # un-normalize: (img * 0.5) + 0.5
    originals = (data.cpu() * 0.5 + 0.5).clamp(0, 1)
    recon = (recon.cpu() * 0.5 + 0.5).clamp(0, 1)

    # Concatenate images together
    comparison = torch.cat([originals, recon])
    grid = make_grid(comparison, nrow=data.shape[0])
    
    plt.figure(figsize=(15, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("Reconstruction Comparison", fontsize=16)
    plt.yticks([32*0.5, 32*1.0], ["Original", "Maodie (Ours)"], rotation=90, va='center', fontsize=12)
    plt.xticks([])
    plt.tight_layout()
    plt.savefig("fig_reconstructions.png")
    print("Saved 'fig_reconstructions.png'")


def evaluate(model, test_loader, device):
    """Evaluate model performance"""
    model.eval()
    total_mse, all_indices = 0, []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            x_recon, indices= model.reconstruct(data)
            all_indices.append(indices.cpu().numpy())
            total_mse += F.mse_loss(x_recon, data, reduction='sum').item()
            all_indices.append(indices.cpu().numpy())
            
    avg_mse = total_mse / (len(test_loader.dataset) * 3 * 32 * 32)
    psnr = compute_psnr(avg_mse)
    codebook_usage = compute_codebook_usage(np.concatenate(all_indices), model.vq.num_embeddings)
    
    return {'psnr': psnr, 'mse': avg_mse, 'codebook_usage': codebook_usage}


def visualize_latent_space(model, data_loader, device, title, filename, n_batches=5):
    """Visualize latent space and codebook using t-SNE"""
    print(f"\nGenerating latent space visualization for {title}...")
    model.eval()
    
    all_z_e_flat = []
    used_indices = set()
    
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
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
    
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    combined = np.vstack([z_e_flat, codebook])
    tsne_results = tsne.fit_transform(combined)
    
    z_e_tsne = tsne_results[:len(z_e_flat)]
    codebook_tsne = tsne_results[len(z_e_flat):]
    
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(z_e_tsne[:, 0], z_e_tsne[:, 1], alpha=0.1, label='Encoder Outputs (z_e)')
    
    used_code_vectors = [c for i, c in enumerate(codebook_tsne) if i in used_indices]
    unused_code_vectors = [c for i, c in enumerate(codebook_tsne) if i not in used_indices]
    
    if used_code_vectors:
        used_code_vectors = np.array(used_code_vectors)
        plt.scatter(used_code_vectors[:, 0], used_code_vectors[:, 1], 
                    color='orange', edgecolor='black', s=80, label='Used Codebook Vectors')
    if unused_code_vectors:
        unused_code_vectors = np.array(unused_code_vectors)
        plt.scatter(unused_code_vectors[:, 0], unused_code_vectors[:, 1], 
                    color='gray', marker='x', s=80, label='Unused Codebook Vectors')

    plt.title(f"Latent Space and Codebook Coverage: {title}", fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved '{filename}'")


def visualize_codebook_usage(model, data_loader, device, title, filename):
    """Visualize codebook usage frequency"""
    print(f"\nGenerating codebook usage histogram for {title}...")
    model.eval()
    all_indices = []
    
    with torch.no_grad():
        for data, _ in tqdm(data_loader, desc=f"Collecting indices for {title}"):
            data = data.to(device)
            z_e = model.encoder(data)
            _, _, _, indices, _ = model.vq(z_e)
            all_indices.append(indices.cpu().numpy().flatten())
            
    all_indices = np.concatenate(all_indices)
    counts = np.bincount(all_indices, minlength=model.vq.num_embeddings)
    
    usage_rate = np.count_nonzero(counts) / model.vq.num_embeddings
    
    plt.figure(figsize=(10, 5))
    plt.bar(range(model.vq.num_embeddings), counts, width=1.0)
    plt.title(f"Codebook Usage: {title} (Usage: {usage_rate:.2%})", fontsize=14)
    plt.xlabel("Codebook Index")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved '{filename}'")