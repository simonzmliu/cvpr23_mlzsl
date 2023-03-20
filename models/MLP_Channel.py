import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import BasicConv
from torch.hub import load_state_dict_from_url
import os
from config import opt
from einops import rearrange


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class MlpBlock(nn.Module):

    def __init__(self, hidden_dim, reduction_ratio):
        super().__init__()
        self.dense_1 = nn.Linear(hidden_dim, hidden_dim // reduction_ratio)
        self.gelu = nn.GELU()
        self.dense_2 = nn.Linear(hidden_dim // reduction_ratio, hidden_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dense_2(x)
        return x


class MixerBlock(nn.Module):
    def __init__(self, hidden_dim, token_dim, reduction_ratio=5):
        super().__init__()
        self.mlp_token = MlpBlock(token_dim, reduction_ratio)
        self.mlp_channel = MlpBlock(hidden_dim, reduction_ratio)
        self.layer_norm_1 = nn.LayerNorm(hidden_dim)
        self.layer_norm_2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        y = self.layer_norm_1(x)
        y = y.permute(0, 1, 3, 2)
        y = self.mlp_token(y)
        y = y.permute(0, 1, 3, 2)
        x = x + y
        y = self.layer_norm_2(x)
        x = x + self.mlp_channel(y)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super(Attention, self).__init__()
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.qkv(x)
        qkv = self.qkv_dwconv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=opt.num_rows)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=opt.num_rows)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=opt.num_rows)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        return out


class MLP_Channel(nn.Module):
    def __init__(self, reduction_ratio=5, num_block=3, pool_types=['avg', 'max']):
        super(MLP_Channel, self).__init__()
        # self.channel_mlp = nn.ModuleList([MixerBlock(hidden_dim=opt.wordvec_dim, token_dim=49, reduction_ratio=reduction_ratio) for _ in range(num_block)])
        self.channel_mlp = MixerBlock(hidden_dim=opt.wordvec_dim, token_dim=49, reduction_ratio=reduction_ratio)
        self.pool_types = pool_types
        self.channel_attn = Attention(dim=opt.wordvec_dim*opt.num_rows, num_heads=opt.num_rows)
        self.groups = opt.num_rows
        self.avg = nn.AdaptiveAvgPool2d((300, 1))
        self.max = nn.AdaptiveMaxPool2d((300, 1))

    def forward(self, x):

        x = self.channel_attn(x)
        b, g, c, s = x.shape
        x = rearrange(x, 'b g c s -> b g s c')
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                x = self.channel_mlp(x)
                x = rearrange(x, 'b g s c -> b g c s')
                channel_att_raw = self.avg(x)
            elif pool_type == 'max':
                x = self.channel_mlp(x)
                x = rearrange(x, 'b g s c -> b g c s')
                channel_att_raw = self.max(x)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        # scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        # x = x * scale
        # out = x.reshape(b, self.groups, c)
        out = F.sigmoid(channel_att_sum)
        out = out.reshape(b, self.groups, c)
        return out


class ChannelPool(nn.Module):

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
