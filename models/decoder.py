# 文件名: models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

# 导入你原来的残差块组件
from models.residual import ResidualStack
# 从我们新建的文件中导入 EdVAE 的解码器残差块组件
from models.decoder_block_edvae import DecoderBlock as EdVAE_DecoderBlock


# ======================================================================
# 1. 这是你原来的 Decoder，我们把它重命名为 SimpleDecoder
#    它被完整地保留了下来
# ======================================================================
class SimpleDecoder(nn.Module):
    """
    This is the original p_phi (x|z) network, a "funnel-style" decoder.
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(SimpleDecoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel - 1, stride=stride - 1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                               kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim // 2, 3, kernel_size=kernel,
                               stride=stride, padding=1)
        )

    def forward(self, x):
        return self.inverse_conv_stack(x)


# ======================================================================
# 2. 这是我们“补充”进去的新 Decoder，忠实地模仿 EdVAE 的架构
#    我们叫它 EdVAE_Decoder
# ======================================================================
class EdVAE_Decoder(nn.Module):
    """
    A decoder that strictly follows the architecture of EdVAE.
    It's the mirror image of the EdVAE_Encoder.
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(EdVAE_Decoder, self).__init__()

        self.channel = h_dim  # 使用 h_dim 作为基础通道数
        self.embedding_dim = in_dim  # 解码器的输入维度是量化后的维度
        self.group_count = 3
        self.n_blk_per_group = 2
        self.n_layers = self.group_count * self.n_blk_per_group

        self.decoder = nn.Sequential(OrderedDict([
            ('group_1', nn.Sequential(OrderedDict([
                ('block_1', EdVAE_DecoderBlock(self.embedding_dim, self.channel * 4, self.n_layers)),
                ('block_2', EdVAE_DecoderBlock(self.channel * 4, self.channel * 4, self.n_layers)),
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),

            ('group_2', nn.Sequential(OrderedDict([
                ('block_1', EdVAE_DecoderBlock(self.channel * 4, self.channel * 2, self.n_layers)),
                ('block_2', EdVAE_DecoderBlock(self.channel * 2, self.channel * 2, self.n_layers)),
                ('upsample', nn.Upsample(scale_factor=2, mode='nearest')),
            ]))),

            ('group_last', nn.Sequential(OrderedDict([
                ('block_1', EdVAE_DecoderBlock(self.channel * 2, self.channel, self.n_layers)),
                ('block_2', EdVAE_DecoderBlock(self.channel, self.channel, self.n_layers)),
            ]))),

            ('output', nn.Sequential(OrderedDict([
                ('relu', nn.ReLU()),
                ('conv', nn.Conv2d(self.channel, 3, 1)),
            ]))),
        ]))

    def forward(self, x):
        return self.decoder(x)


# ======================================================================
# 3. 切换开关：决定 MaodieVQ 使用哪个 Decoder
# ======================================================================

# --- 当前设置为使用 EdVAE_Decoder 进行对比实验 ---
Decoder = EdVAE_Decoder

# --- 如果你想换回原来的版本，注释掉上面一行，然后取消下面一行的注释 ---
# Decoder = SimpleDecoder