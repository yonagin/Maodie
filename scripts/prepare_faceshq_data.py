#!/usr/bin/env python3
"""
一键准备人脸数据集脚本
用于自动下载和准备CelebA-HQ和FFHQ数据集
"""

import os
import sys
import argparse
import urllib.request
import tarfile
import zipfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob


def download_file(url, filename, target_dir):
    """下载文件并显示进度条"""
    os.makedirs(target_dir, exist_ok=True)
    filepath = os.path.join(target_dir, filename)
    
    if os.path.exists(filepath):
        print(f"文件已存在: {filepath}")
        return filepath
    
    print(f"正在下载: {filename}")
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, downloaded * 100 / total_size)
            sys.stdout.write(f'\r进度: {percent:.1f}% ({downloaded}/{total_size} bytes)')
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, filepath, progress_hook)
        print(f"\n下载完成: {filename}")
        return filepath
    except Exception as e:
        print(f"下载失败: {e}")
        return None


def extract_tar(filepath, extract_dir):
    """解压tar文件"""
    print(f"正在解压: {filepath}")
    try:
        with tarfile.open(filepath, 'r') as tar:
            tar.extractall(extract_dir)
        print(f"解压完成: {filepath}")
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False


def extract_zip(filepath, extract_dir):
    """解压zip文件"""
    print(f"正在解压: {filepath}")
    try:
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        print(f"解压完成: {filepath}")
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False


def resize_images(input_dir, output_dir, size=256):
    """调整图像大小为指定尺寸"""
    os.makedirs(output_dir, exist_ok=True)
    
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    if not image_files:
        print(f"在 {input_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件，正在调整大小为 {size}x{size}")
    
    for img_path in tqdm(image_files):
        try:
            # 保持相对路径结构
            rel_path = os.path.relpath(img_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 调整大小
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                img_resized = img.resize((size, size), Image.Resampling.LANCZOS)
                img_resized.save(output_path, 'JPEG', quality=95)
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")


def create_file_list(image_dir, output_file, train_ratio=0.8):
    """创建训练和验证文件列表"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
    
    # 按文件名排序
    image_files.sort()
    
    if not image_files:
        print(f"在 {image_dir} 中未找到图像文件")
        return
    
    # 分割训练集和验证集
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # 写入训练集文件列表
    train_output = output_file.replace('.txt', 'train.txt')
    with open(train_output, 'w') as f:
        for img_path in train_files:
            # 使用相对于数据目录的路径
            rel_path = os.path.relpath(img_path, os.path.dirname(image_dir))
            f.write(rel_path + '\n')
    
    # 写入验证集文件列表
    val_output = output_file.replace('.txt', 'validation.txt')
    with open(val_output, 'w') as f:
        for img_path in val_files:
            rel_path = os.path.relpath(img_path, os.path.dirname(image_dir))
            f.write(rel_path + '\n')
    
    print(f"创建文件列表: {train_output} ({len(train_files)} 个文件)")
    print(f"创建文件列表: {val_output} ({len(val_files)} 个文件)")


def prepare_celebahq_data(data_dir='data'):
    """准备CelebA-HQ数据集"""
    print("=== 准备CelebA-HQ数据集 ===")
    
    celebahq_dir = os.path.join(data_dir, 'celebahq')
    os.makedirs(celebahq_dir, exist_ok=True)
    
    print("CelebA-HQ数据集需要手动准备:")
    print("1. 下载CelebA-HQ数据集 (通常需要申请权限)")
    print("2. 将图像文件放置在 data/celebahq/ 目录下")
    print("3. 图像应为高分辨率人脸图像")
    
    # 检查是否已有图像文件
    image_files = glob.glob(os.path.join(celebahq_dir, '*.jpg')) + glob.glob(os.path.join(celebahq_dir, '*.png'))
    
    if image_files:
        print(f"发现 {len(image_files)} 个图像文件")
        
        # 调整图像大小
        resized_dir = os.path.join(data_dir, 'celebahq_resized')
        resize_images(celebahq_dir, resized_dir, size=256)
        
        # 创建文件列表
        create_file_list(resized_dir, os.path.join(data_dir, 'celebahq.txt'))
        
        return True
    else:
        print("未找到CelebA-HQ图像文件，请手动准备数据集")
        return False


def prepare_ffhq_data(data_dir='data'):
    """准备FFHQ数据集"""
    print("=== 准备FFHQ数据集 ===")
    
    ffhq_dir = os.path.join(data_dir, 'ffhq')
    os.makedirs(ffhq_dir, exist_ok=True)
    
    print("FFHQ数据集需要手动准备:")
    print("1. 从NVIDIA官方下载FFHQ数据集")
    print("2. 将图像文件放置在 data/ffhq/ 目录下")
    print("3. 图像应为1024x1024分辨率的人脸图像")
    
    # 检查是否已有图像文件
    image_files = glob.glob(os.path.join(ffhq_dir, '*.png')) + glob.glob(os.path.join(ffhq_dir, '*.jpg'))
    
    if image_files:
        print(f"发现 {len(image_files)} 个图像文件")
        
        # 调整图像大小
        resized_dir = os.path.join(data_dir, 'ffhq_resized')
        resize_images(ffhq_dir, resized_dir, size=256)
        
        # 创建文件列表
        create_file_list(resized_dir, os.path.join(data_dir, 'ffhq.txt'))
        
        return True
    else:
        print("未找到FFHQ图像文件，请手动准备数据集")
        return False


def prepare_custom_face_data(data_dir='data', custom_image_dir=None):
    """准备自定义人脸数据集"""
    print("=== 准备自定义人脸数据集 ===")
    
    if custom_image_dir is None:
        custom_image_dir = os.path.join(data_dir, 'custom_faces')
    
    if not os.path.exists(custom_image_dir):
        print(f"自定义图像目录不存在: {custom_image_dir}")
        return False
    
    image_files = glob.glob(os.path.join(custom_image_dir, '*.jpg')) + glob.glob(os.path.join(custom_image_dir, '*.png'))
    
    if not image_files:
        print("在自定义目录中未找到图像文件")
        return False
    
    print(f"发现 {len(image_files)} 个自定义图像文件")
    
    # 调整图像大小
    celebahq_resized = os.path.join(data_dir, 'celebahq_resized')
    ffhq_resized = os.path.join(data_dir, 'ffhq_resized')
    
    resize_images(custom_image_dir, celebahq_resized, size=256)
    resize_images(custom_image_dir, ffhq_resized, size=256)
    
    # 创建文件列表（同时用于训练和验证）
    create_file_list(celebahq_resized, os.path.join(data_dir, 'celebahq.txt'))
    create_file_list(ffhq_resized, os.path.join(data_dir, 'ffhq.txt'))
    
    print("自定义数据集准备完成！")
    return True


def create_symlinks_for_faceshq(data_dir='data'):
    """为FacesHQ数据集创建符号链接"""
    print("=== 创建FacesHQ数据集符号链接 ===")
    
    # 检查必要的文件列表是否存在
    required_files = [
        'celebahqtrain.txt', 'celebahqvalidation.txt',
        'ffhqtrain.txt', 'ffhqvalidation.txt'
    ]
    
    # 如果文件列表不存在，尝试从现有的txt文件创建
    for file in required_files:
        if not os.path.exists(os.path.join(data_dir, file)):
            base_name = file.replace('train.txt', '.txt').replace('validation.txt', '.txt')
            if os.path.exists(os.path.join(data_dir, base_name)):
                # 复制文件
                shutil.copy2(os.path.join(data_dir, base_name), os.path.join(data_dir, file))
                print(f"创建文件: {file}")
    
    print("符号链接创建完成！")


def main():
    parser = argparse.ArgumentParser(description='一键准备人脸数据集')
    parser.add_argument('--data-dir', default='data', help='数据目录路径')
    parser.add_argument('--custom-dir', help='自定义图像目录路径')
    parser.add_argument('--only-custom', action='store_true', help='仅使用自定义数据集')
    
    args = parser.parse_args()
    
    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)
    
    print("开始准备人脸数据集...")
    
    success_count = 0
    
    if args.custom_dir or args.only_custom:
        # 使用自定义数据集
        if prepare_custom_face_data(data_dir, args.custom_dir):
            success_count += 1
    else:
        # 尝试准备标准数据集
        if prepare_celebahq_data(data_dir):
            success_count += 1
        
        if prepare_ffhq_data(data_dir):
            success_count += 1
    
    # 创建必要的符号链接
    create_symlinks_for_faceshq(data_dir)
    
    if success_count > 0:
        print(f"\n数据集准备完成！成功准备了 {success_count} 个数据集")
        print("数据目录结构:")
        print(f"{data_dir}/")
        print("├── celebahq_resized/       # CelebA-HQ调整后图像")
        print("├── ffhq_resized/           # FFHQ调整后图像")
        print("├── celebahqtrain.txt       # CelebA-HQ训练集列表")
        print("├── celebahqvalidation.txt  # CelebA-HQ验证集列表")
        print("├── ffhqtrain.txt           # FFHQ训练集列表")
        print("└── ffhqvalidation.txt      # FFHQ验证集列表")
        print("\n现在可以使用 faceshq_vqgan.yaml 配置文件进行训练了！")
    else:
        print("\n数据集准备失败，请检查错误信息或使用自定义数据集")
        print("使用方法:")
        print("1. 准备自定义图像: python prepare_faceshq_data.py --custom-dir /path/to/your/images")
        print("2. 使用标准数据集: 手动下载CelebA-HQ和FFHQ数据集到data目录")


if __name__ == '__main__':
    main()