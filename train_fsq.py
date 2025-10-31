# train_fsq.py
# FSQ-VQVAE Training Script (Paper's Method)
# Compare with train_maodie.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import shared utilities
from utils import (
    compute_psnr, compute_codebook_usage, visualize_reconstructions,
    evaluate, visualize_latent_space, visualize_codebook_usage
)


# ============================================================================
# FSQ Implementation (from the paper) - FIXED
# ============================================================================

def round_ste(z):
    """Round with straight-through estimator."""
    zhat = torch.round(z)
    return z + (zhat - z).detach()


class FSQ(nn.Module):
    """
    Finite Scalar Quantization from the paper:
    "Finite Scalar Quantization: VQ-VAE Made Simple"

    Key differences from VQ:
    - No learnable codebook
    - No commitment loss needed
    - 100% codebook usage by design
    """

    def __init__(self, levels):
        super().__init__()
        self._levels = np.asarray(levels)
        self._levels_tensor = torch.tensor(levels, dtype=torch.float32)
        self._basis = np.concatenate(([1], np.cumprod(self._levels[:-1]))).astype(np.int64)

        # For compatibility with utils.py
        self.n_embeddings = int(np.prod(levels))
        self.embedding_dim = len(levels)

        print(f"[FSQ] Initialized with levels {levels}")
        print(f"[FSQ] Codebook size: {self.n_embeddings}")
        print(f"[FSQ] No learnable parameters in quantizer!")

    def bound(self, z):
        """
        Bound z to the range of each level using tanh.

        Args:
            z: (B, C, H, W) where C = len(levels)
        Returns:
            bounded z in range appropriate for each level
        """
        eps = 1e-3
        # Get level parameters - shape (C,)
        levels = self._levels_tensor.to(z.device)
        half_l = (levels - 1) * (1 - eps) / 2
        offset = torch.where(levels % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half_l)

        # Reshape for broadcasting: (1, C, 1, 1) to match (B, C, H, W)
        half_l = half_l.view(1, -1, 1, 1)
        offset = offset.view(1, -1, 1, 1)
        shift = shift.view(1, -1, 1, 1)

        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z):
        """
        Quantize z to the nearest integer in the bounded range.

        Args:
            z: (B, C, H, W)
        Returns:
            quantized z, normalized to [-1, 1]
        """
        quantized = round_ste(self.bound(z))
        # Normalize to [-1, 1]
        half_width = self._levels_tensor.to(z.device) // 2
        half_width = half_width.view(1, -1, 1, 1)  # Broadcast shape
        return quantized / half_width

    def forward(self, z):
        """
        Args:
            z: (B, C, H, W) where C = len(levels)
        Returns:
            z_q: quantized representation
            indices: codebook indices for tracking
            None: placeholders for compatibility
        """
        # Quantize
        z_q = self.quantize(z)

        # Compute indices for codebook usage tracking
        half_width = self._levels_tensor.to(z.device) // 2
        half_width = half_width.view(1, -1, 1, 1)
        z_int = (z_q * half_width + half_width).long()

        B, C, H, W = z_int.shape
        indices = torch.zeros(B, H, W, dtype=torch.long, device=z.device)
        basis_tensor = torch.tensor(self._basis, dtype=torch.long, device=z.device)

        for i in range(C):
            indices += z_int[:, i, :, :] * basis_tensor[i]

        # Return format compatible with VQ
        return z_q, None, None, indices, None


# ============================================================================
# Encoder and Decoder (matching your architecture)
# ============================================================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, h_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(h_dim, out_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers):
        super().__init__()
        self.conv1 = nn.Conv2d(3, h_dim // 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(h_dim // 2, h_dim, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(h_dim, h_dim, kernel_size=3, stride=1, padding=1)
        self.res_stack = nn.ModuleList(
            [ResidualBlock(h_dim, h_dim, res_h_dim) for _ in range(n_res_layers)]
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        for layer in self.res_stack:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, h_dim, res_h_dim, n_res_layers):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=1, padding=1)
        self.res_stack = nn.ModuleList(
            [ResidualBlock(h_dim, h_dim, res_h_dim) for _ in range(n_res_layers)]
        )
        self.deconv1 = nn.ConvTranspose2d(h_dim, h_dim // 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(h_dim // 2, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        for layer in self.res_stack:
            x = layer(x)
        x = F.relu(self.deconv1(x))
        x = torch.tanh(self.deconv2(x))
        return x


# ============================================================================
# FSQ-VQVAE Model
# ============================================================================

class FSQ_VQVAE(nn.Module):
    """
    FSQ-VQVAE: VQ-VAE with Finite Scalar Quantization

    Key simplifications compared to standard VQ-VAE:
    1. NO commitment loss
    2. NO codebook loss
    3. NO EMA updates
    4. NO codebook collapse issues
    5. NO discriminator (optional, using L1 loss only)
    """

    def __init__(self, levels, h_dim, res_h_dim, n_res_layers):
        super().__init__()

        # Encoder
        self.encoder = Encoder(h_dim, res_h_dim, n_res_layers)

        # Project to FSQ dimensions
        fsq_dim = len(levels)
        self.pre_quantization_conv = nn.Conv2d(h_dim, fsq_dim, 1)

        # FSQ quantizer (NO learnable parameters!)
        self.vq = FSQ(levels)

        # Project back to decoder dimensions
        self.post_quantization_conv = nn.Conv2d(fsq_dim, h_dim, 1)

        # Decoder
        self.decoder = Decoder(h_dim, h_dim, res_h_dim, n_res_layers)

        print(f"[FSQ-VQVAE] Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        print(f"[FSQ-VQVAE] Quantizer parameters: 0 (FSQ has no learnable codebook!)")

    def forward(self, x):
        """Forward pass."""
        # Encode
        z_e = self.encoder(x)
        z_e = self.pre_quantization_conv(z_e)

        # Quantize (NO gradients needed for FSQ!)
        z_q, _, _, indices, _ = self.vq(z_e)

        # Decode
        z_q = self.post_quantization_conv(z_q)
        x_recon = self.decoder(z_q)

        return x_recon, z_e, z_q, indices

    def reconstruct(self, x):
        """Reconstruct input (for evaluation)."""
        with torch.no_grad():
            z_e = self.encoder(x)
            z_e = self.pre_quantization_conv(z_e)
            z_q, _, _, indices, _ = self.vq(z_e)
            z_q = self.post_quantization_conv(z_q)
            x_recon = self.decoder(z_q)
        return x_recon, indices.flatten(start_dim=1)

    def training_step(self, x, optimizer):
        """
        Training step for FSQ-VQVAE.

        NOTE: Much simpler than VQ-VAE!
        - Only reconstruction loss
        - No commitment loss
        - No codebook loss
        - No discriminator (optional)
        """
        optimizer.zero_grad()

        # Forward pass
        x_recon, z_e, z_q, indices = self(x)

        # ONLY reconstruction loss (L1)
        recon_loss = F.l1_loss(x_recon, x)

        # That's it! No other losses needed
        total_loss = recon_loss

        # Backward
        total_loss.backward()
        optimizer.step()

        # Calculate perplexity for logging (even though FSQ doesn't optimize for it)
        with torch.no_grad():
            unique_codes = torch.unique(indices).numel()
            perplexity = unique_codes  # Simple approximation

        return recon_loss.item(), 0.0, 0.0, perplexity


# ============================================================================
# Training Functions
# ============================================================================

def load_cifar100_dataset(batch_size):
    """Load CIFAR-100 dataset."""
    print("Loading CIFAR-100 dataset...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    testset = torchvision.datasets.CIFAR100(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        trainset, batch_size=batch_size,
        shuffle=True, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        testset, batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True
    )

    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")

    return train_loader, test_loader


def train_model(model, train_loader, test_loader, device, config):
    """Train FSQ-VQVAE model."""

    # Simple optimizer (no discriminator!)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Training statistics
    train_losses = []
    recon_losses = []
    perplexities = []

    print("\n" + "=" * 80)
    print("Starting FSQ-VQVAE Training")
    print("=" * 80)
    print(f"Key differences from standard VQ-VAE:")
    print("  ✓ NO commitment loss")
    print("  ✓ NO codebook loss")
    print("  ✓ NO discriminator")
    print("  ✓ NO codebook collapse issues")
    print("  ✓ Simple L1 reconstruction loss only")
    print("=" * 80 + "\n")

    step = 0
    while step < config['total_steps']:
        model.train()

        for batch_idx, (data, _) in enumerate(train_loader):
            if step >= config['total_steps']:
                break

            data = data.to(device)

            # Training step (simple!)
            recon_loss, vq_loss, disc_loss, perplexity = model.training_step(data, optimizer)

            # Record statistics
            train_losses.append(recon_loss)
            recon_losses.append(recon_loss)
            perplexities.append(perplexity)

            step += 1

            # Periodic evaluation
            if step % config['eval_interval'] == 0:
                # Calculate average of recent steps
                recent_start = max(0, len(train_losses) - config['eval_interval'])
                recent_recon_loss = np.mean(recon_losses[recent_start:])
                recent_perplexity = np.mean(perplexities[recent_start:])

                print(f"\nStep {step}/{config['total_steps']}:")
                print(f"Reconstruction loss: {recent_recon_loss:.4f}")
                print(f"VQ loss: {vq_loss:.4f} (should be 0 for FSQ)")
                print(f"Discriminator loss: {disc_loss:.4f} (no discriminator)")
                print(f"Unique codes used: {recent_perplexity:.0f}/{model.vq.n_embeddings}")

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
                    visualize_reconstructions(
                        test_batch, recon_batch,
                        f"fsq_reconstructions_step_{step}.png"
                    )

            if step % 100 == 0:
                print(f"Progress: {step}/{config['total_steps']} | "
                      f"Recon Loss: {recon_loss:.4f}", end="\r")

    print("\n\nTraining completed!")
    return train_losses, recon_losses, perplexities


def visualize_training_results(model, test_loader, train_losses, recon_losses,
                               perplexities, config):
    """Visualize training results."""
    print("\nGenerating training results visualization...")

    # Loss curves
    fig = plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses)
    plt.title(f"Total Loss (FSQ, Codebook: {config['n_embeddings']})")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(recon_losses)
    plt.title("Reconstruction Loss (L1)")
    plt.xlabel("Training Steps")
    plt.ylabel("Reconstruction Loss")
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(perplexities)
    plt.axhline(y=config['n_embeddings'], color='r', linestyle='--',
                label=f'Max ({config["n_embeddings"]})')
    plt.title("Unique Codes Used")
    plt.xlabel("Training Steps")
    plt.ylabel("Number of Codes")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"fsq_training_curves_step_{config['total_steps']}.png", dpi=150)
    plt.close(fig)
    print(f"Saved: fsq_training_curves_step_{config['total_steps']}.png")

    # Codebook usage visualization
    visualize_codebook_usage(
        model, test_loader, device,
        f"FSQ-VQVAE (Codebook: {config['n_embeddings']})",
        f"fsq_codebook_usage_step_{config['total_steps']}.png"
    )


def create_model(config):
    """Create FSQ-VQVAE model."""
    print("\n" + "=" * 80)
    print("Creating FSQ-VQVAE Model")
    print("=" * 80)
    print(f"Target codebook size: {config['n_embeddings']}")
    print(f"FSQ levels: {config['fsq_levels']}")
    print(f"Actual codebook size: {np.prod(config['fsq_levels'])}")
    print(f"FSQ dimensions: {len(config['fsq_levels'])}")
    print("=" * 80 + "\n")

    model = FSQ_VQVAE(
        levels=config['fsq_levels'],
        h_dim=config['h_dim'],
        res_h_dim=config['res_h_dim'],
        n_res_layers=config['n_res_layers']
    ).to(device)

    return model


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Training configuration (matching your Maodie setup)
    config = {
        'batch_size': 128,
        'total_steps': 50000,
        'eval_interval': 5000,
        'lr': 2e-4,
        'n_embeddings': 8192,  # Target size

        # FSQ specific: levels that give ~8192 codes
        # Paper recommends levels >= 5 for good performance
        # [8,8,8,4,4] = 8192 exactly
        'fsq_levels': [8, 8, 8, 4, 4],

        # Model architecture (matching Maodie)
        'h_dim': 32,
        'res_h_dim': 32,
        'n_res_layers': 2,
    }

    # Verify codebook size
    actual_size = np.prod(config['fsq_levels'])
    assert actual_size == config['n_embeddings'], \
        f"FSQ levels {config['fsq_levels']} give {actual_size}, not {config['n_embeddings']}"

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 80)
    print("FSQ-VQVAE Training Script (Paper's Method)")
    print("=" * 80)
    print(f"Codebook size: {config['n_embeddings']}")
    print(f"FSQ levels: {config['fsq_levels']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"Total training steps: {config['total_steps']}")
    print("\nKey advantages over standard VQ-VAE:")
    print("  1. No commitment loss - simpler training")
    print("  2. No codebook collapse - 100% usage by design")
    print("  3. No complex tricks needed (EMA, reseeding, etc.)")
    print("  4. Fewer parameters (no learnable codebook)")
    print("=" * 80 + "\n")

    # Create output directory
    os.makedirs("./results", exist_ok=True)

    # Load data
    train_loader, test_loader = load_cifar100_dataset(config['batch_size'])

    # Create model
    model = create_model(config)

    # Train model
    train_losses, recon_losses, perplexities = train_model(
        model, train_loader, test_loader, device, config
    )

    # Final evaluation
    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    eval_results = evaluate(model, test_loader, device)
    print(f"Final PSNR: {eval_results['psnr']:.2f} dB")
    print(f"Final codebook usage: {eval_results['codebook_usage']:.2%}")
    print(f"Target: ~100% (FSQ should use almost all codes)")
    print("=" * 80 + "\n")

    # Visualize results
    visualize_training_results(
        model, test_loader, train_losses, recon_losses,
        perplexities, config
    )

    # Save model
    model_path = f"fsq_model_{config['n_embeddings']}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED")
    print("=" * 80)
    print("Generated files:")
    print(f"  - fsq_reconstructions_step_*.png: Reconstruction comparisons")
    print(f"  - fsq_training_curves_step_{config['total_steps']}.png: Training curves")
    print(f"  - fsq_codebook_usage_step_{config['total_steps']}.png: Codebook usage")
    print(f"  - {model_path}: Model weights")
    print("\nCompare these results with your Maodie method!")
    print("Expected FSQ advantages:")
    print("  ✓ Similar or slightly lower PSNR")
    print("  ✓ ~100% codebook usage (vs Maodie's 99.94%)")
    print("  ✓ Simpler training (no discriminator resets)")
    print("  ✓ Faster training (fewer losses to compute)")
    print("=" * 80)
