#!/usr/bin/env python3
"""
Maodie VQ-VAE 训练脚本
支持对抗训练和可视化功能
"""

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

# 导入模型和工具函数
from models.vqvae import MaodieVQ
from utils import (
    compute_psnr, compute_codebook_usage, visualize_reconstructions,
    evaluate, visualize_latent_space, visualize_codebook_usage
)

# 训练参数
batch_size = 128
total_training_steps = 10000
eval_interval = 1000
lr = 2e-4
num_embeddings = 1024
embedding_dim = 64
commitment_cost = 0.25
lambda_adv = 1e-4
temperature = 1.0
dirichlet_alpha = 0.1

# 模型参数
h_dim = 64
res_h_dim = 32
n_res_layers = 2

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def load_cifar10_dataset():
    """加载CIFAR-10数据集"""
    print("正在加载CIFAR-10数据集...")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 下载训练集
    trainset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # 下载测试集
    testset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # 创建数据加载器
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"训练集大小: {len(trainset)}")
    print(f"测试集大小: {len(testset)}")
    print(f"类别: {trainset.classes}")
    
    return train_loader, test_loader

def create_model():
    """创建Maodie模型"""
    print("正在创建Maodie模型...")
    print(f"码本大小: {num_embeddings}")
    print(f"嵌入维度: {embedding_dim}")
    print(f"对抗权重: {lambda_adv}")
    print(f"温度参数: {temperature}")
    print(f"Dirichlet参数: {dirichlet_alpha}")
    
    model = MaodieVQ(
        h_dim=h_dim,
        res_h_dim=res_h_dim,
        n_res_layers=n_res_layers,
        n_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        beta=commitment_cost,
        dirichlet_alpha=dirichlet_alpha,
        temperature=temperature,
        patch_size=2,
    ).to(device)
    
    return model

def setup_optimizers(model):
    """设置优化器"""
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

def train_model(model, train_loader, test_loader):
    """训练模型"""
    optimizer_G, optimizer_D = setup_optimizers(model)
    
    # 训练统计
    train_losses = []
    recon_losses = []
    vq_losses = []
    disc_losses = []
    perplexities = []
    
    print("开始训练...")
    
    step = 0
    while step < total_training_steps:
        model.train()
        
        for batch_idx, (data, _) in enumerate(train_loader):
            if step >= total_training_steps:
                break
                
            data = data.to(device)
            
            # 训练步骤
            recon_loss, vq_loss, disc_loss, perplexity = model.training_step(
                data, optimizer_G, optimizer_D, lambda_adv=lambda_adv
            )
            
            # 记录统计信息
            train_losses.append(recon_loss + vq_loss)
            recon_losses.append(recon_loss)
            vq_losses.append(vq_loss)
            disc_losses.append(disc_loss)
            perplexities.append(perplexity)
            
            # 定期评估
            if step % eval_interval == 0:
                # 计算最近eval_interval步的平均值
                recent_start = max(0, len(train_losses) - eval_interval)
                recent_recon_loss = np.mean(recon_losses[recent_start:])
                recent_vq_loss = np.mean(vq_losses[recent_start:])
                recent_disc_loss = np.mean(disc_losses[recent_start:])
                recent_perplexity = np.mean(perplexities[recent_start:])
                
                print(f"\n步骤 {step}/{total_training_steps}:")
                print(f"重建损失: {recent_recon_loss:.4f}")
                print(f"VQ损失: {recent_vq_loss:.4f}")
                print(f"判别器损失: {recent_disc_loss:.4f}")
                print(f"困惑度: {recent_perplexity:.2f}")
                
                # 评估模型
                eval_results = evaluate(model, test_loader, device)
                print(f"PSNR: {eval_results['psnr']:.2f}")
                print(f"码本使用率: {eval_results['codebook_usage']:.2%}")
                
                # 可视化重建效果
                with torch.no_grad():
                    model.eval()
                    test_batch, _ = next(iter(test_loader))
                    test_batch = test_batch[:8].to(device)
                    recon_batch, _ = model.reconstruct(test_batch)
                    visualize_reconstructions(test_batch, recon_batch)
            
            step += 1
            
            if step % 100 == 0:
                print(f"进度: {step}/{total_training_steps}", end="\r")
    
    return train_losses, recon_losses, vq_losses, disc_losses, perplexities

def visualize_training_results(model, test_loader, train_losses, recon_losses, vq_losses, disc_losses, perplexities):
    """可视化训练结果"""
    print("\n正在生成训练结果可视化...")
    
    # 损失曲线
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(train_losses)
    plt.title(f"总损失曲线 (码本大小: {num_embeddings})")
    plt.xlabel("训练步数")
    plt.ylabel("损失")
    plt.grid(True)
    
    plt.subplot(2, 3, 2)
    plt.plot(recon_losses)
    plt.title(f"重建损失曲线 (码本大小: {num_embeddings})")
    plt.xlabel("训练步数")
    plt.ylabel("重建损失")
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(vq_losses)
    plt.title(f"VQ损失曲线 (码本大小: {num_embeddings})")
    plt.xlabel("训练步数")
    plt.ylabel("VQ损失")
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(disc_losses)
    plt.title(f"判别器损失曲线 (码本大小: {num_embeddings})")
    plt.xlabel("训练步数")
    plt.ylabel("判别器损失")
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    plt.plot(perplexities)
    plt.title(f"困惑度曲线 (码本大小: {num_embeddings})")
    plt.xlabel("训练步数")
    plt.ylabel("困惑度")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("training_curves.png")
    print("保存训练曲线图: training_curves.png")
    
    # 潜在空间可视化
    visualize_latent_space(
        model, test_loader, device, 
        f"Maodie VQ-VAE (码本大小: {num_embeddings})",
        "latent_space_visualization.png"
    )
    
    # 码本使用情况可视化
    visualize_codebook_usage(
        model, test_loader, device,
        f"Maodie VQ-VAE (码本大小: {num_embeddings})",
        "codebook_usage.png"
    )

def main():
    """主函数"""
    print("=== Maodie VQ-VAE 训练脚本 ===")
    print(f"码本大小: {num_embeddings}")
    print(f"嵌入维度: {embedding_dim}")
    print(f"批次大小: {batch_size}")
    print(f"总训练步数: {total_training_steps}")
    
    # 创建输出目录
    os.makedirs("./results", exist_ok=True)
    
    # 加载数据
    train_loader, test_loader = load_cifar10_dataset()
    
    # 创建模型
    model = create_model()
    
    # 训练模型
    train_losses, recon_losses, vq_losses, disc_losses, perplexities = train_model(
        model, train_loader, test_loader
    )
    
    # 最终评估
    print("\n=== 最终评估 ===")
    eval_results = evaluate(model, test_loader, device)
    print(f"最终 PSNR: {eval_results['psnr']:.2f}")
    print(f"最终码本使用率: {eval_results['codebook_usage']:.2%}")
    
    # 可视化结果
    visualize_training_results(
        model, test_loader, train_losses, recon_losses, 
        vq_losses, disc_losses, perplexities
    )
    
    # 保存模型
    torch.save(model.state_dict(), f"maodie_model_{num_embeddings}.pth")
    print(f"模型已保存: maodie_model_{num_embeddings}.pth")
    
    print("\n=== 训练完成 ===")
    print("生成的可视化文件:")
    print("- fig_reconstructions.png: 重建效果对比")
    print("- training_curves.png: 训练损失曲线")
    print("- latent_space_visualization.png: 潜在空间可视化")
    print("- codebook_usage.png: 码本使用情况")

if __name__ == "__main__":
    main()