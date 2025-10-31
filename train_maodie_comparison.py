# 文件名: train_maodie_comparison.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
from torch.utils.data import DataLoader

# 导入我们修正后的主模型
from models.maodie_vqvae_comparison import MaodieVQ
# 导入您的工具函数
from utils import evaluate, visualize_reconstructions


def load_cifar10_dataset():
    """Load CIFAR-10 dataset"""
    print("Loading CIFAR-10 dataset...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Training set size: {len(trainset)}, Test set size: {len(testset)}")
    return train_loader, test_loader


def setup_optimizers(model):
    """Setup optimizers"""
    optimizer_G = optim.Adam(
        list(model.encoder.parameters()) + list(model.pre_quantization_conv.parameters()) +
        list(model.vq.parameters()) + list(model.decoder.parameters()), lr=lr)
    optimizer_D = optim.Adam(model.discriminator.parameters(), lr=lr)
    return optimizer_G, optimizer_D


def train_model(model, train_loader, test_loader):
    """Train model by calling the model's own training_step method."""
    optimizer_G, optimizer_D = setup_optimizers(model)
    print(f"Starting training with {'Standard GAN' if use_standard_gan else 'WGAN'} loss...")
    step = 0
    start_time = time.time()

    while step < total_training_steps:
        for batch_idx, (data, _) in enumerate(train_loader):
            if step >= total_training_steps: break
            data = data.to(device)

            # --- 简洁的调用！所有逻辑都封装在模型内部 ---
            recon_loss, vq_loss, disc_loss, perplexity = model.training_step(
                data, optimizer_G, optimizer_D, lambda_adv
            )

            step += 1

            if step > 0 and step % eval_interval == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = eval_interval / elapsed_time
                print(f"\n--- Step {step}/{total_training_steps} (Speed: {steps_per_sec:.2f} step/s) ---")

                with torch.no_grad():
                    model.eval()
                    eval_results = evaluate(model, test_loader, device)
                    print(
                        f"  Test PSNR: {eval_results['psnr']:.2f} dB, Codebook Usage: {eval_results['codebook_usage']:.2%}")
                    test_batch, _ = next(iter(test_loader))
                    test_batch = test_batch[:8].to(device)
                    recon_batch, _ = model.reconstruct(test_batch)
                    visualize_reconstructions(test_batch, recon_batch,
                                              f"./results_battle/maodie_reconstructions_step_{step}")
                    print(f"  > Saved reconstruction image for step {step}.")

                start_time = time.time()
                model.train()  # 切换回训练模式

    print(f"\nTraining completed!")


# ======================================================================
# 主执行逻辑
# ======================================================================
if __name__ == "__main__":
    # --- 终极对决实验参数 ---
    batch_size = 128
    total_training_steps = 50000
    eval_interval = 1000
    lr = 2e-4
    n_embeddings = 512
    embedding_dim = 16
    commitment_cost = 0.25
    lambda_adv = 1e-4

    # --- GAN 类型开关 ---
    use_standard_gan = True  # True 使用标准GAN, False 使用WGAN

    # 模型架构参数
    h_dim = 128
    res_h_dim = 128
    n_res_layers = 2

    # 其他辅助参数
    temperature = 1.0;
    dirichlet_alpha = 0.1;
    patch_size = 4

    # --- 设备设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n=== FINAL BATTLE: MaodieVQ-GAN vs EdVAE-style on CIFAR-10 ===")
    print(f"Codebook: {n_embeddings}x{embedding_dim}, Model Width: {h_dim}, LR: {lr}")
    print(f"Total steps: {total_training_steps}, GAN Type: {'Standard GAN' if use_standard_gan else 'WGAN'}")

    os.makedirs("./results_battle", exist_ok=True)

    # --- 开始执行 ---
    train_loader, test_loader = load_cifar10_dataset()

    model = MaodieVQ(
        h_dim=h_dim, res_h_dim=res_h_dim, n_res_layers=n_res_layers,
        n_embeddings=n_embeddings, embedding_dim=embedding_dim,
        beta=commitment_cost, dirichlet_alpha=dirichlet_alpha,
        temperature=temperature, patch_size=patch_size,
        use_standard_gan=use_standard_gan  # 将开关传递给模型
    ).to(device)

    train_model(model, train_loader, test_loader)

    print("\n=== Final Evaluation ===")
    eval_results = evaluate(model, test_loader, device)
    print(f"Final PSNR: {eval_results['psnr']:.2f} dB")
    print(f"Final Codebook Usage: {eval_results['codebook_usage']:.2%}")

    model_save_path = f"./results_battle/maodie_model_battle_{'stdGAN' if use_standard_gan else 'wgan'}_{n_embeddings}.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

    print("\n=== Battle Completed ===")