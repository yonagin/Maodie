%cd maodie

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from sklearn.manifold import TSNE
from tqdm import tqdm

# Import models and utility functions
from models.vqvae import MaodieVQ
from utils import (
    compute_psnr, compute_codebook_usage, visualize_reconstructions,
    evaluate, visualize_latent_space, visualize_codebook_usage
)


def load_cifar100_dataset():
    """Load CIFAR-10 dataset"""
    print("Loading CIFAR-10 dataset...")
    
    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Download training set
    trainset = torchvision.datasets.CIFAR100(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Download test set
    testset = torchvision.datasets.CIFAR100(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")
    print(f"Classes: {trainset.classes}")
    
    return train_loader, test_loader


def setup_optimizers(model):
    """Setup optimizers"""
    optimizer_G = optim.Adam(
        list(model.encoder.parameters()) + 
        list(model.pre_quantization_conv.parameters()) +
        list(model.vq.parameters()) + 
        list(model.decoder.parameters()),
        lr=lr
    )
    
    optimizer_D = optim.Adam(
        model.discriminator.parameters(),
        lr=lr
    )
    
    return optimizer_G, optimizer_D

def reset_discriminator(model, optimizer_D, lr):
    """重置判别器参数和优化器"""
    print("\n[INFO]resetting discriminator...")
    
    # 重新初始化判别器参数
    for layer in model.discriminator.modules():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            nn.init.normal_(layer.weight, 0.0, 0.02)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
    
    # 重置优化器
    optimizer_D = optim.Adam(
        model.discriminator.parameters(),
        lr=lr
    )
    
    return optimizer_D


def train_model(model, train_loader, test_loader):
    """Train model with discriminator reset strategy"""
    optimizer_G, optimizer_D = setup_optimizers(model)
    
    # Training statistics
    train_losses = []
    recon_losses = []
    vq_losses = []
    disc_losses = []
    perplexities = []
    
    # Discriminator reset tracking
    disc_reset_threshold = 1e-4
    disc_reset_window = 500  # 检查最近100步的平均损失
    disc_reset_count = 0
    
    print("Starting training...")
    print(f"Discriminator reset threshold: {disc_reset_threshold}")
    print(f"Discriminator reset window: {disc_reset_window} steps")
    
    step = 0
    while step < total_training_steps:
        model.train()
        
        for batch_idx, (data, _) in enumerate(train_loader):
            if step >= total_training_steps:
                break
                
            data = data.to(device)
            
            # Training step
            recon_loss, vq_loss, disc_loss, perplexity = model.training_step(
                data, optimizer_G, optimizer_D, lambda_adv=lambda_adv
            )
            
            # Record statistics
            train_losses.append(recon_loss + vq_loss)
            recon_losses.append(recon_loss)
            vq_losses.append(vq_loss)
            disc_losses.append(disc_loss)
            perplexities.append(perplexity)
            step += 1
            
            # Check discriminator loss and reset if needed
            if step > disc_reset_window and step % disc_reset_window == 0:
                recent_disc_loss = np.mean(disc_losses[-disc_reset_window:])
                
                if recent_disc_loss < disc_reset_threshold:
                    print(f"\n[Step {step}] Recent avg discriminator loss: {recent_disc_loss:.6f}")
                    optimizer_D = reset_discriminator(model, optimizer_D, lr)
                    disc_reset_count += 1
                    print(f"Discriminator has been reset {disc_reset_count} times")
            
            # Periodic evaluation
            if step % eval_interval == 0:
                # Calculate average of recent eval_interval steps
                recent_start = max(0, len(train_losses) - eval_interval)
                recent_recon_loss = np.mean(recon_losses[recent_start:])
                recent_vq_loss = np.mean(vq_losses[recent_start:])
                recent_disc_loss = np.mean(disc_losses[recent_start:])
                recent_perplexity = np.mean(perplexities[recent_start:])
                
                print(f"\nStep {step}/{total_training_steps}:")
                print(f"Reconstruction loss: {recent_recon_loss:.4f}")
                print(f"VQ loss: {recent_vq_loss:.4f}")
                print(f"Discriminator loss: {recent_disc_loss:.4f}")
                print(f"Perplexity: {recent_perplexity:.2f}")
                print(f"Discriminator resets: {disc_reset_count}")
                
                # Evaluate model
                eval_results = evaluate(model, test_loader, device)
                print(f"PSNR: {eval_results['psnr']:.2f}")
                print(f"Codebook usage: {eval_results['codebook_usage']:.2%}")
                
                # Visualize reconstruction results
                with torch.no_grad():
                    model.eval()
                    test_batch, _ = next(iter(test_loader))
                    test_batch = test_batch[:8].to(device)
                    recon_batch, _ = model.reconstruct(test_batch)
                    visualize_reconstructions(test_batch, recon_batch, f"maodie_reconstructions_step_{step}")
            
            if step % 100 == 0:
                print(f"Progress: {step}/{total_training_steps} | D-resets: {disc_reset_count}", end="\r")
    
    print(f"\n\nTraining completed with {disc_reset_count} discriminator resets")
    return train_losses, recon_losses, vq_losses, disc_losses, perplexities

def visualize_training_results(model, test_loader, train_losses, recon_losses, vq_losses, disc_losses, perplexities):
    """Visualize training results"""
    print("\nGenerating training results visualization...")
    
    # Loss curves
    fig = plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(train_losses)
    plt.title(f"Total Loss Curve (Codebook Size: {n_embeddings})")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(recon_losses)
    plt.title(f"Reconstruction Loss Curve (Codebook Size: {n_embeddings})")
    plt.xlabel("Training Steps")
    plt.ylabel("Reconstruction Loss")
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(vq_losses)
    plt.title(f"VQ Loss Curve (Codebook Size: {n_embeddings})")
    plt.xlabel("Training Steps")
    plt.ylabel("VQ Loss")
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(disc_losses)
    plt.title(f"Discriminator Loss Curve (Codebook Size: {n_embeddings})")
    plt.xlabel("Training Steps")
    plt.ylabel("Discriminator Loss")
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(perplexities)
    plt.title(f"Perplexity Curve (Codebook Size: {n_embeddings})")
    plt.xlabel("Training Steps")
    plt.ylabel("Perplexity")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"maodie_training_curves_step_{total_training_steps}.png")
    plt.close(fig)  # 关闭图形释放内存
    print(f"Saved training curves: maodie_training_curves_step_{total_training_steps}.png")
    
    # Latent space visualization
    visualize_latent_space(
        model, test_loader, device, 
        f"Maodie VQ-VAE (Codebook Size: {n_embeddings})",
        f"maodie_latent_space_step_{total_training_steps}.png"
    )
    
    # Codebook usage visualization
    visualize_codebook_usage(
        model, test_loader, device,
        f"Maodie VQ-VAE (Codebook Size: {n_embeddings})",
        f"maodie_codebook_usage_step_{total_training_steps}.png"
    )

def create_model():
    """Create Maodie model"""
    print("Creating Maodie model...")
    print(f"Codebook size: {n_embeddings}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Adversarial weight: {lambda_adv}")
    print(f"Temperature parameter: {temperature}")
    print(f"Dirichlet parameter: {dirichlet_alpha}")
    print(f"Use cosine similarity: {cosine}")
    
    model = MaodieVQ(
        h_dim=h_dim,
        res_h_dim=res_h_dim,
        n_res_layers=n_res_layers,
        n_embeddings=n_embeddings,
        embedding_dim=embedding_dim,
        beta=commitment_cost,
        dirichlet_alpha=dirichlet_alpha,
        temperature=temperature,
        patch_size=patch_size,
        use_fisher=fisher,
        cosine=cosine
    ).to(device)
    
    return model
 

if __name__ == "__main__":
    # Training parameters
    batch_size = 128
    total_training_steps = 50000
    eval_interval = 1000
    lr = 2e-4
    n_embeddings = 8192
    embedding_dim = 32
    commitment_cost = 0.25
    lambda_adv = 1e-4
    temperature = 1.0
    dirichlet_alpha = 0.1
    patch_size = 4
    fisher = False
    cosine = False  # 是否使用余弦相似度计算距离

    # Model parameters
    h_dim = 32
    res_h_dim = 32
    n_res_layers = 2

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("=== Maodie VQ-VAE Training Script ===")
    print(f"Codebook size: {n_embeddings}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Batch size: {batch_size}")
    print(f"Total training steps: {total_training_steps}")
    
    # Create output directory
    os.makedirs("./results", exist_ok=True)
    
    # Load data
    train_loader, test_loader = load_cifar100_dataset()
    
    # Create model
    model = create_model()

    # Train model
    train_losses, recon_losses, vq_losses, disc_losses, perplexities = train_model(
        model, train_loader, test_loader
    )
    
    # Final evaluation
    print("\n=== Final Evaluation ===")
    eval_results = evaluate(model, test_loader, device)
    print(f"Final PSNR: {eval_results['psnr']:.2f}")
    print(f"Final codebook usage: {eval_results['codebook_usage']:.2%}")
    
    # Visualize results
    visualize_training_results(
        model, test_loader, train_losses, recon_losses, 
        vq_losses, disc_losses, perplexities
    )
    
    # Save model
    torch.save(model.state_dict(), f"maodie_model_{n_embeddings}.pth")
    print(f"Model saved: maodie_model_{n_embeddings}.pth")
    
    print("\n=== Training Completed ===")
    print("Generated visualization files:")
    print(f"- maodie_reconstructions_step_*.png: Reconstruction comparisons at different steps")
    print(f"- maodie_training_curves_step_{total_training_steps}.png: Training loss curves")
    print(f"- maodie_latent_space_step_{total_training_steps}.png: Latent space visualization")
    print(f"- maodie_codebook_usage_step_{total_training_steps}.png: Codebook usage")