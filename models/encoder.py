# 文件名: models/encoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# 导入你原来的残差块组件
from models.residual import ResidualStack
# 从我们新建的文件中导入 EdVAE 的残差块组件
from models.encoder_block_edvae import EncoderBlock as EdVAE_EncoderBlock


# ======================================================================
# 1. 这是你原来的 Encoder，我们把它命名为 SimpleEncoder
#    它代表了“漏斗式”架构，并且被完整地保留了下来
# ======================================================================
class SimpleEncoder(nn.Module):
    """
    This is the original q_theta (z|x) network, a "funnel-style" encoder.
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(SimpleEncoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel - 1,
                      stride=stride - 1, padding=1),
            ResidualStack(
                h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)


# ======================================================================
# 2. 这是我们“补充”进去的新 Encoder，忠实地模仿 EdVAE 的架构
#    我们叫它 EdVAE_Encoder
# ======================================================================
class EdVAE_Encoder(nn.Module):
    """
    An encoder that strictly follows the architecture of EdVAE.
    Note: n_res_layers and res_h_dim are kept for compatibility but are not directly used in the same way.
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(EdVAE_Encoder, self).__init__()

        self.channel = h_dim
        self.group_count = 3
        self.n_blk_per_group = 2
        self.n_layers = self.group_count * self.n_blk_per_group

        self.encoder = nn.Sequential(OrderedDict([
            ('conv_in', nn.Conv2d(in_channels=in_dim, out_channels=self.channel, kernel_size=3, padding=1)),

            ('group_1', nn.Sequential(OrderedDict([
                *[(f'block_{i + 1}', EdVAE_EncoderBlock(self.channel, self.channel, self.n_layers)) for i in
                  range(self.n_blk_per_group)],
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),

            ('group_2', nn.Sequential(OrderedDict([
                ('block_1', EdVAE_EncoderBlock(self.channel, self.channel * 2, self.n_layers)),
                ('block_2', EdVAE_EncoderBlock(self.channel * 2, self.channel * 2, self.n_layers)),
                ('pool', nn.MaxPool2d(kernel_size=2)),
            ]))),

            ('group_last', nn.Sequential(OrderedDict([
                ('block_1', EdVAE_EncoderBlock(self.channel * 2, self.channel * 4, self.n_layers)),
                ('block_2', EdVAE_EncoderBlock(self.channel * 4, self.channel * 4, self.n_layers)),
            ]))),

            ('output_relu', nn.ReLU()),
        ]))

    def forward(self, x):
        return self.encoder(x)


# ======================================================================
# 3. 切换开关：决定 MaodieVQ 使用哪个 Encoder
#    你的主模型 (MaodieVQ) 会导入名为 'Encoder' 的类
# ======================================================================

# --- 当前设置为使用 EdVAE_Encoder 进行对比实验 ---
Encoder = EdVAE_Encoder

# --- 如果你想换回原来的版本，注释掉上面一行，然后取消下面一行的注释 ---
# Encoder = SimpleEncoder