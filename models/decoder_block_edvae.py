# 文件名: models/decoder_block_edvae.py

import torch.nn as nn
from collections import OrderedDict


# 注意：您没有提供 DecoderBlock 的代码，我将使用与 EncoderBlock 对应的结构来推断
# 它很可能与 EncoderBlock 结构类似，但可能没有 post_gain
# 如果您有 DecoderBlock 的确切代码，请替换掉下面的内容
class DecoderBlock(nn.Module):
    def __init__(self, n_in, n_out, n_layers):
        super(DecoderBlock, self).__init__()
        self.n_in = n_in
        self.n_hid = n_out // 4
        self.n_out = n_out
        self.n_layers = n_layers

        self.id_path = nn.Conv2d(in_channels=n_in, out_channels=n_out,
                                 kernel_size=1) if self.n_in != n_out else nn.Identity()
        self.res_path = nn.Sequential(OrderedDict([
            ('relu_1', nn.ReLU()),
            ('conv_1', nn.Conv2d(self.n_in, self.n_hid, kernel_size=3, padding=1)),
            ('relu_2', nn.ReLU()),
            ('conv_2', nn.Conv2d(self.n_hid, self.n_hid, kernel_size=3, padding=1)),
            ('relu_3', nn.ReLU()),
            ('conv_3', nn.Conv2d(self.n_hid, self.n_hid, kernel_size=3, padding=1)),
            ('relu_4', nn.ReLU()),
            ('conv_4', nn.Conv2d(self.n_hid, self.n_out, kernel_size=1)), ])
        )

    def forward(self, x):
        return self.id_path(x) + self.res_path(x)