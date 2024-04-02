"""
定义了通道自注意力模块、空间自注意力模块，以及各自的编码器
"""
import torch
import torch.nn as nn
from models.basic import Conv2d
from models.basic.attention import ChannelAttention, SpatialAttention, CSAM, SSAM, MPCA


class CFSAB(nn.Module):
    """ Channel Fuse Series Attention Block """

    def __init__(self, c1, attention_type=['ca', ]):  # ch_in, kernels
        super().__init__()
        self.attention_type = attention_type
        if 'ca' in self.attention_type:
            self.ca = CSAM()
        if 'sa' in self.attention_type:
            self.sa = SSAM()
        if 'mpca' in self.attention_type:
            self.mpca = MPCA(c1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        """Applies the forward pass through C1 module."""
        for attention_type in self.attention_type:
            if attention_type == 'ca':
                x = self.ca(x)
            elif attention_type == 'sa':
                x = self.sa(x)
            elif attention_type == 'mpca':
                x = self.mpca(x)
        return x


class CFPAB(nn.Module):
    """ Channel Fuse Parallel Attention Block """

    def __init__(self, c1, attention_type=['ca', 'sa']):  # ch_in, kernels
        super().__init__()
        self.attention_type = attention_type
        if 'ca' in self.attention_type:
            self.ca = CSAM()
        if 'sa' in self.attention_type:
            self.sa = SSAM()
        if 'mpca' in self.attention_type:
            self.mpca = MPCA(c1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        """Applies the forward pass through C1 module."""
        x_list = []
        for attention_type in self.attention_type:
            if attention_type == 'ca':
                x_list.append(self.ca(x))
            elif attention_type == 'sa':
                x_list.append(self.sa(x))
            elif attention_type == 'mpca':
                x_list.append(self.mpca(x))
        x = sum(x_list)/len(x_list)
        return x


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):  # ch_in, kernels
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B x C x C
        """
        """Applies the forward pass through C1 module."""
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='', norm_type='', encoder_type='CFSAB', attention_type=['ca', ]):
        super().__init__()
        if encoder_type == 'CBAB':
            self.fuse_convs = nn.Sequential(
                Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
                Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
                CBAM(c1=out_dim),  # Convolutional Block Attention Module 卷积块自注意力模块，返回(B，C，H，W)
                Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
                nn.Dropout(0.1, inplace=False),
                nn.Conv2d(out_dim, out_dim, kernel_size=1)
            )
        elif encoder_type == 'CFSAB':
            self.fuse_convs = nn.Sequential(
                Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
                Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
                CFSAB(c1=out_dim, attention_type=attention_type),  # 自注意力模块，返回(B，C，H，W) 顺序可变
                Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
                nn.Dropout(0.1, inplace=False),
                nn.Conv2d(out_dim, out_dim, kernel_size=1)
            )
        elif encoder_type == 'CFPAB':
            self.fuse_convs = nn.Sequential(
                Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
                Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
                CFPAB(c1=out_dim, attention_type=attention_type),  # 自注意力模块，返回(B，C，H，W) 内部可变
                Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
                nn.Dropout(0.1, inplace=False),
                nn.Conv2d(out_dim, out_dim, kernel_size=1)
            )
        elif encoder_type == 'PASS':  # 简单卷积融合的方式 是否需要这么多层
            self.fuse_convs = nn.Sequential(
                Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
                Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
                Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
                nn.Dropout(0.1, inplace=False),
                nn.Conv2d(out_dim, out_dim, kernel_size=1)
            )

    def forward(self, x1, x2):
        """
            x1和x2分别来自2D分支和3D分支
            x: [B, C, H, W]
        """
        x = torch.cat([x1, x2], dim=1)
        # [B, CN, H, W] -> [B, C, H, W]
        x = self.fuse_convs(x)

        return x


def build_encoder(cfg, in_dim, out_dim, attention_type):
    """

    :param cfg:
    :param in_dim:
    :param out_dim:
    :param attention_type: ['ca',]
    :return: encoder
    """

    encoder = Encoder(
        in_dim=in_dim,
        out_dim=out_dim,
        act_type=cfg['head_act'],
        norm_type=cfg['head_norm'],
        encoder_type=cfg['encoder_type'],
        attention_type=attention_type
    )

    return encoder
