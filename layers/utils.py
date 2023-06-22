import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, n, epsilon=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n) - epsilon)
        self.epsilon = epsilon

    def forward(self, x):
        if x.requires_grad:
            scale = self.weight * (self.running_var + self.epsilon).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            out_dtype = x.dtype
            return x * scale.to(out_dtype) + bias.to(out_dtype)
        else:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.epsilon,
            )
