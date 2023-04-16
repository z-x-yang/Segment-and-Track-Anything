import torch
import torch.nn.functional as F
from torch import nn


class GroupNorm1D(nn.Module):
    def __init__(self, indim, groups=8):
        super().__init__()
        self.gn = nn.GroupNorm(groups, indim)

    def forward(self, x):
        return self.gn(x.permute(1, 2, 0)).permute(2, 0, 1)


class GNActDWConv2d(nn.Module):
    def __init__(self, indim, gn_groups=32):
        super().__init__()
        self.gn = nn.GroupNorm(gn_groups, indim)
        self.conv = nn.Conv2d(indim,
                              indim,
                              5,
                              dilation=1,
                              padding=2,
                              groups=indim,
                              bias=False)

    def forward(self, x, size_2d):
        h, w = size_2d
        _, bs, c = x.size()
        x = x.view(h, w, bs, c).permute(2, 3, 0, 1)
        x = self.gn(x)
        x = F.gelu(x)
        x = self.conv(x)
        x = x.view(bs, c, h * w).permute(2, 0, 1)
        return x


class DWConv2d(nn.Module):
    def __init__(self, indim, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv2d(indim,
                              indim,
                              5,
                              dilation=1,
                              padding=2,
                              groups=indim,
                              bias=False)
        self.dropout = nn.Dropout2d(p=dropout, inplace=True)

    def forward(self, x, size_2d):
        h, w = size_2d
        _, bs, c = x.size()
        x = x.view(h, w, bs, c).permute(2, 3, 0, 1)
        x = self.conv(x)
        x = self.dropout(x)
        x = x.view(bs, c, h * w).permute(2, 0, 1)
        return x


class ScaleOffset(nn.Module):
    def __init__(self, indim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(indim))
        # torch.nn.init.normal_(self.gamma, std=0.02)
        self.beta = nn.Parameter(torch.zeros(indim))

    def forward(self, x):
        if len(x.size()) == 3:
            return x * self.gamma + self.beta
        else:
            return x * self.gamma.view(1, -1, 1, 1) + self.beta.view(
                1, -1, 1, 1)


class ConvGN(nn.Module):
    def __init__(self, indim, outdim, kernel_size, gn_groups=8):
        super().__init__()
        self.conv = nn.Conv2d(indim,
                              outdim,
                              kernel_size,
                              padding=kernel_size // 2)
        self.gn = nn.GroupNorm(gn_groups, outdim)

    def forward(self, x):
        return self.gn(self.conv(x))


def seq_to_2d(tensor, size_2d):
    h, w = size_2d
    _, n, c = tensor.size()
    tensor = tensor.view(h, w, n, c).permute(2, 3, 0, 1).contiguous()
    return tensor


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (
        x.shape[0],
        x.shape[1],
    ) + (1, ) * (x.ndim - 2
                 )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


def mask_out(x, y, mask_rate=0.15, training=False):
    if mask_rate == 0. or not training:
        return x

    keep_prob = 1 - mask_rate
    shape = (
        x.shape[0],
        x.shape[1],
    ) + (1, ) * (x.ndim - 2
                 )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x * random_tensor + y * (1 - random_tensor)

    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None, batch_dim=0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.batch_dim = batch_dim

    def forward(self, x):
        return self.drop_path(x, self.drop_prob)

    def drop_path(self, x, drop_prob):
        if drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - drop_prob
        shape = [1 for _ in range(x.ndim)]
        shape[self.batch_dim] = x.shape[self.batch_dim]
        random_tensor = keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class DropOutLogit(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropOutLogit, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_logit(x, self.drop_prob)

    def drop_logit(self, x, drop_prob):
        if drop_prob == 0. or not self.training:
            return x
        random_tensor = drop_prob + torch.rand(
            x.shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        mask = random_tensor * 1e+8 if (
            x.dtype == torch.float32) else random_tensor * 1e+4
        output = x - mask
        return output
