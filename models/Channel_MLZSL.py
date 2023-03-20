import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from config import opt
from models.vgg import vgg_bn
from models.resnet import BasicConv, resnet101, resnet50
from models.MLP_Channel import MLP_Channel
from models.Group_Attention import Group_Attention
import einops


class BasicMaxPooling(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False, max_kernel_size=4, max_stride=4):
        super(BasicMaxPooling, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None
        self.maxpool = nn.MaxPool2d(kernel_size=max_kernel_size, stride=max_stride)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        x = self.maxpool(x)

        return x


def feature_transform_module(name):

    if name == 'vgg19':
        layers = []
        layers += [BasicMaxPooling(256, 512, kernel_size=1, padding=0, max_kernel_size=4, max_stride=4)]
        layers += [BasicMaxPooling(512, 512, kernel_size=1, padding=0, max_kernel_size=2, max_stride=2)]
        layers += [BasicConv(512, 1024, kernel_size=1, padding=0)]
    else:
        layers = []
        layers += [BasicMaxPooling(512, 512, kernel_size=1, padding=0, max_kernel_size=4, max_stride=4)]
        layers += [BasicMaxPooling(1024, 512, kernel_size=1, padding=0, max_kernel_size=2, max_stride=2)]
        layers += [BasicConv(2048, 1024, kernel_size=1, padding=0)]

    return nn.Sequential(*layers)


class Channel_MLZSL(nn.Module):
    def __init__(self, size, base, ft_module, in_dim=2048, state='train'):
        super(Channel_MLZSL, self).__init__()
        self.size = size
        self.group = opt.num_rows
        # Backbone
        self.vgg = base
        self.ft_module = ft_module
        self.conv_embedding = nn.Conv2d(in_dim, 300*self.group, stride=1, kernel_size=1, padding=0)
        self.state = state
        self.Channel_MLP = MLP_Channel(reduction_ratio=5, num_block=3, pool_types=['max'])
        self.Group_Attention = Group_Attention(hidden_dim=300, mlp_dim=300)
        self.max = nn.AdaptiveMaxPool3d((300, 1, 1))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        transformed_features = list()
        sources = self.vgg(x)
        assert len(self.ft_module) == len(sources)
        for k, v in enumerate(self.ft_module):
            transformed_features.append(v(sources[k]))
        x = torch.cat(transformed_features, 1)
        x = self.conv_embedding(x)
        x = self.channel_shuffle(x, self.group)

        # b, c, h, w = x.shape
        # x = x.reshape(b, self.group, -1, h, w)
        # x = self.max(x)
        x = self.Channel_MLP(x)
        x = self.Group_Attention(x)
        # x = torch.squeeze(x)
        return x

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


def build_GroupChannel(size=224, name='vgg19', state='train'):

    if name == 'vgg19':
        return Channel_MLZSL(size, vgg_bn(name), feature_transform_module(name), state=state)
    else:
        return Channel_MLZSL(size, resnet50(pretrained=True, progress=True), feature_transform_module(name), state=state)
