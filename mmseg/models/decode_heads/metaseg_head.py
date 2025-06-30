import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
import math


class DWConv(nn.Module):
    """Depthwise Convolution module for channel-wise feature processing."""
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # handle various input dimensions
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    """MLP module with 1x1 convolutions.

    Args:
        in_features (int): Input feature channels.
        hidden_features (int, optional): Hidden feature channels.
        out_features (int, optional): Output feature channels.
        act_layer (nn.Module): Activation layer.
        drop_path (float): Drop path rate.
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop_path=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop_path)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        x = x.flatten(2).transpose(1, 2)
        return x


class ChannelReductionAttention(nn.Module):
    """Channel Reduction Attention.
    
    Args:
        dim1 (int): Input channel dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
        qk_scale (float, optional): Override default qk scale.
        attn_drop (float): Attention dropout rate.
        proj_drop (float): Output dropout rate.
        pool_ratio (int): Pooling ratio to reduce spatial dimensions.
    """
    def __init__(self, dim1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., pool_ratio=16):
        super(ChannelReductionAttention, self).__init__()
        assert dim1 % num_heads == 0, f"dim {dim1} should be divided by num_heads {num_heads}."

        self.dim1 = dim1
        self.pool_ratio = pool_ratio
        self.num_heads = num_heads
        head_dim = dim1 // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim1, self.num_heads, bias=qkv_bias)
        self.k = nn.Linear(dim1, self.num_heads, bias=qkv_bias)  
        self.v = nn.Linear(dim1, dim1, bias=qkv_bias)  
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim1, dim1)  
        self.proj_drop = nn.Dropout(proj_drop)

        self.pool = nn.AvgPool2d(pool_ratio, pool_ratio)
        self.sr = nn.Conv2d(dim1, dim1, kernel_size=1, stride=1)
        self.norm = nn.LayerNorm(dim1)
        self.act = nn.GELU()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, h, w):
        B, N, C = x.shape  
        q = self.q(x).reshape(B, N, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        x_ = x.permute(0, 2, 1).reshape(B, C, h, w)
        x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.norm(x_)
        x_ = self.act(x_)

        k = self.k(x_).reshape(B, -1, self.num_heads).permute(0, 2, 1).unsqueeze(-1)
        v = self.v(x_).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class GlobalMetaBlock(nn.Module):
    """Global Meta Block with attention and MLP.
    
    Args:
        dim1 (int): Input channel dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): MLP hidden dim expansion ratio.
        qkv_bias (bool): If True, add a learnable bias to q, k, v.
        qk_scale (float, optional): Override default qk scale.
        drop (float): Dropout rate.
        attn_drop (float): Attention dropout rate.
        drop_path (float): Drop path rate.
        act_layer (nn.Module): Activation layer.
        norm_layer (nn.Module): Normalization layer.
        pool_ratio (int): Pooling ratio for spatial reduction.
    """
    def __init__(self, dim1, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, pool_ratio=16):
        super(GlobalMetaBlock, self).__init__()
        self.norm1 = norm_layer(dim1)
        self.norm3 = norm_layer(dim1)

        self.attn = ChannelReductionAttention(dim1=dim1, num_heads=num_heads, pool_ratio=pool_ratio)

        # Drop path for stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim1 * mlp_ratio)
        self.mlp = Mlp(in_features=dim1, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop_path=drop_path)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, h, w):
        x = x + self.drop_path(self.attn(self.norm1(x), h, w))
        x = x + self.drop_path(self.mlp(self.norm3(x), h, w))
        return x


@MODELS.register_module()
class MetaSegHead(BaseDecodeHead):
    """MetaSeg Decode Head.
    
    This decode head uses global meta blocks with channel-wise attention
    to process multi-scale features.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
        decoder_params (dict): The parameters for decoder configuration.
            - embed_dim (int): The embedding dimension for decoder.
    """
    def __init__(self, feature_strides, **kwargs):
        super(MetaSegHead, self).__init__(input_transform='multiple_select', **kwargs)
        _, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels 

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        pool_ratio = 2
        mlp_ratio = 2

        self.attn4 = GlobalMetaBlock(dim1=c4_in_channels, num_heads=8, mlp_ratio=mlp_ratio,
                                     drop_path=0.1, pool_ratio=pool_ratio)

        self.attn3 = GlobalMetaBlock(dim1=c3_in_channels, num_heads=4, mlp_ratio=mlp_ratio, drop_path=0.1,
                                    pool_ratio=pool_ratio * 2)

        self.attn2 = GlobalMetaBlock(dim1=c2_in_channels, num_heads=2, mlp_ratio=mlp_ratio, drop_path=0.1,
                                    pool_ratio=pool_ratio * 4)


        self.linear_fuse = ConvModule(
            in_channels=(c2_in_channels + c3_in_channels + c4_in_channels),
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        """Forward function."""
        x = self._transform_inputs(inputs)  
        c1, c2, c3, c4 = x

        n, _, h4, w4 = c4.shape
        _, _, h3, w3 = c3.shape
        _, _, h2, w2 = c2.shape

        c2 = c2.flatten(2).transpose(1, 2)
        c3 = c3.flatten(2).transpose(1, 2)
        c4 = c4.flatten(2).transpose(1, 2)

        _c4 = self.attn4(c4, h4, w4)
        _c4 = _c4.permute(0, 2, 1).reshape(n, -1, h4, w4)
        _c4 = resize(_c4, size=(h2, w2), mode='bilinear', align_corners=False)

        _c3 = self.attn3(c3, h3, w3)
        _c3 = _c3.permute(0, 2, 1).reshape(n, -1, h3, w3)
        _c3 = resize(_c3, size=(h2, w2), mode='bilinear', align_corners=False)

        _c2 = self.attn2(c2, h2, w2)
        _c2 = _c2.permute(0, 2, 1).reshape(n, -1, h2, w2)
        _c2 = resize(_c2, size=(h2, w2), mode='bilinear', align_corners=False)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x 