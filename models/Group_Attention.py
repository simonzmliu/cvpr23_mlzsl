import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class MlpBlock(nn.Module):

    def __init__(self, hidden_dim, mlp_dim):
        super().__init__()
        self.mlp_dim = mlp_dim
        self.dense_1 = nn.Linear(hidden_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.dense_2 = nn.Linear(mlp_dim, hidden_dim)

    def forward(self, x):
        x = self.dense_1(x)
        x = self.gelu(x)
        x = self.dense_2(x)
        return x


class Group_Attention(nn.Module):

    def __init__(self, hidden_dim, mlp_dim):
        super(Group_Attention, self).__init__()
        self.Linear1 = MlpBlock(hidden_dim, mlp_dim)
        self.Linear2 = MlpBlock(hidden_dim, mlp_dim)


    def forward(self, x):
        q = self.Linear1(x)
        k = self.Linear2(x)
        v = x

        k = k.permute(0, 2, 1)
        group_att = torch.matmul(q, k)
        S = torch.matmul(group_att, v)

        return S
