import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.math import truncated_normal_


class Downsample2D(nn.Module):
    def __init__(self, mode='nearest', scale=4):
        super().__init__()
        self.mode = mode
        self.scale = scale

    def forward(self, x):
        n, c, h, w = x.size()
        x = F.interpolate(x,
                          size=(h // self.scale + 1, w // self.scale + 1),
                          mode=self.mode)
        return x


def generate_coord(x):
    _, _, h, w = x.size()
    device = x.device
    col = torch.arange(0, h, device=device)
    row = torch.arange(0, w, device=device)
    grid_h, grid_w = torch.meshgrid(col, row)
    return grid_h, grid_w


class PositionEmbeddingSine(nn.Module):
    def __init__(self,
                 num_pos_feats=64,
                 temperature=10000,
                 normalize=False,
                 scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        grid_y, grid_x = generate_coord(x)

        y_embed = grid_y.unsqueeze(0).float()
        x_embed = grid_x.unsqueeze(0).float()

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats,
                             dtype=torch.float32,
                             device=x.device)
        dim_t = self.temperature**(2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    def __init__(self, num_pos_feats=64, H=30, W=30):
        super().__init__()
        self.H = H
        self.W = W
        self.pos_emb = nn.Parameter(
            truncated_normal_(torch.zeros(1, num_pos_feats, H, W)))

    def forward(self, x):
        bs, _, h, w = x.size()
        pos_emb = self.pos_emb
        if h != self.H or w != self.W:
            pos_emb = F.interpolate(pos_emb, size=(h, w), mode="bilinear")
        return pos_emb
