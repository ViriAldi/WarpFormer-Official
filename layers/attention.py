import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import GNActDWConv2d, PositionEmbeddingSine


def multiply_by_ychunks(x, y, chunks=1):
    if chunks <= 1:
        return x @ y
    else:
        return torch.cat([x @ _y for _y in y.chunk(chunks, dim=-1)], dim=-1)


def multiply_by_xchunks(x, y, chunks=1):
    if chunks <= 1:
        return x @ y
    else:
        return torch.cat([_x @ y for _x in x.chunk(chunks, dim=-2)], dim=-2)


class MultiheadAttention(nn.Module):
    def __init__(self,
                 d_model,
                 num_head=8,
                 dropout=0.,
                 use_linear=True,
                 d_att=None,
                 use_dis=False,
                 qk_chunks=1,
                 max_mem_len_ratio=-1,
                 top_k=-1):
        super().__init__()
        self.d_model = d_model
        self.num_head = num_head
        self.use_dis = use_dis
        self.qk_chunks = qk_chunks
        self.max_mem_len_ratio = float(max_mem_len_ratio)
        self.top_k = top_k

        self.hidden_dim = d_model // num_head
        self.d_att = self.hidden_dim if d_att is None else d_att
        self.T = self.d_att**0.5
        self.use_linear = use_linear

        if use_linear:
            self.linear_Q = nn.Linear(d_model, d_model)
            self.linear_K = nn.Linear(d_model, d_model)
            self.linear_V = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.drop_prob = dropout
        self.projection = nn.Linear(d_model, d_model)
        self._init_weight()

    def forward(self, Q, K, V):
        """
        :param Q: A 3d tensor with shape of [T_q, bs, C_q]
        :param K: A 3d tensor with shape of [T_k, bs, C_k]
        :param V: A 3d tensor with shape of [T_v, bs, C_v]
        """
        num_head = self.num_head
        hidden_dim = self.hidden_dim

        bs = Q.size()[1]

        # Linear projections
        if self.use_linear:
            Q = self.linear_Q(Q)
            K = self.linear_K(K)
            V = self.linear_V(V)

        # Scale
        Q = Q / self.T

        if not self.training and self.max_mem_len_ratio > 0:
            mem_len_ratio = float(K.size(0)) / Q.size(0)
            if mem_len_ratio > self.max_mem_len_ratio:
                scaling_ratio = math.log(mem_len_ratio) / math.log(
                    self.max_mem_len_ratio)
                Q = Q * scaling_ratio

        # Multi-head
        Q = Q.view(-1, bs, num_head, self.d_att).permute(1, 2, 0, 3)
        K = K.view(-1, bs, num_head, self.d_att).permute(1, 2, 3, 0)
        V = V.view(-1, bs, num_head, hidden_dim).permute(1, 2, 0, 3)

        # Multiplication
        QK = multiply_by_ychunks(Q, K, self.qk_chunks)
        if self.use_dis:
            QK = 2 * QK - K.pow(2).sum(dim=-2, keepdim=True)

        # Activation
        if not self.training and self.top_k > 0 and self.top_k < QK.size()[-1]:
            top_QK, indices = torch.topk(QK, k=self.top_k, dim=-1)
            top_attn = torch.softmax(top_QK, dim=-1)
            attn = torch.zeros_like(QK).scatter_(-1, indices, top_attn)
        else:
            attn = torch.softmax(QK, dim=-1)

        # Dropouts
        attn = self.dropout(attn)

        # Weighted sum
        outputs = multiply_by_xchunks(attn, V,
                                      self.qk_chunks).permute(2, 0, 1, 3)

        # Restore shape
        outputs = outputs.reshape(-1, bs, self.d_model)

        outputs = self.projection(outputs)

        return outputs, attn

    def _init_weight(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class MultiheadLocalAttention(nn.Module):
    def __init__(self,
                 d_model,
                 num_head,
                 max_dis=7,
                 dilation=1,
                 use_linear=True):
        super().__init__()
        self.dilation = dilation
        self.window_size = 2 * max_dis + 1
        self.max_dis = max_dis
        self.num_head = num_head
        self.T = ((d_model / num_head)**0.5)

        self.use_linear = use_linear
        if use_linear:
            self.linear_Q = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_K = nn.Conv2d(d_model, d_model, kernel_size=1)
            self.linear_V = nn.Conv2d(d_model, d_model, kernel_size=1)

        self.relative_emb_k = nn.Conv2d(d_model,
                                        num_head * self.window_size *
                                        self.window_size,
                                        kernel_size=1,
                                        groups=num_head)
        self.relative_emb_v = nn.Parameter(
            torch.zeros([
                self.num_head, d_model // self.num_head,
                self.window_size * self.window_size
            ]))

        self.projection = nn.Linear(d_model, d_model)

        self.padded_local_mask = None
        self.local_mask = None
        self.last_size_2d = None
        self.qk_mask = None

    def forward(self, q, k, v):
        n, c, h, w = q.size()

        if self.use_linear:
            q = self.linear_Q(q)
            k = self.linear_K(k)
            v = self.linear_V(v)

        hidden_dim = c // self.num_head

        relative_emb = self.relative_emb_k(q)
        relative_emb = relative_emb.view(n, self.num_head,
                                         self.window_size * self.window_size,
                                         h * w)
        padded_local_mask, local_mask = self.compute_mask(h,
                                                          w,
                                                          device=q.device)
        qk_mask = (~padded_local_mask).float()

        # Scale
        q = q / self.T

        q = q.view(-1, self.num_head, hidden_dim, h * w)
        k = k.view(-1, self.num_head, hidden_dim, h * w)
        v = v.view(-1, self.num_head, hidden_dim, h * w)

        qk = q.transpose(-1, -2) @ k  # [B, nH, kL, qL]

        pad_pixel = self.max_dis * self.dilation

        padded_qk = F.pad(qk.view(-1, self.num_head, h * w, h, w),
                          (pad_pixel, pad_pixel, pad_pixel, pad_pixel),
                          mode='constant',
                          value=-1e+8 if qk.dtype == torch.float32 else -1e+4)

        qk_mask = qk_mask * 1e+8 if (padded_qk.dtype
                                     == torch.float32) else qk_mask * 1e+4
        padded_qk = padded_qk - qk_mask

        padded_qk[padded_local_mask.expand(n, self.num_head, -1, -1,
                                           -1)] += relative_emb.transpose(
                                               -1, -2).reshape(-1)

        local_qk = padded_qk[padded_local_mask.expand(n, self.num_head, -1, -1,
                                                      -1)]

        global_qk = padded_qk[:, :, :, self.max_dis:-self.max_dis,
                              self.max_dis:-self.max_dis].reshape(
                                  n, self.num_head, h * w, h * w)

        local_attn = torch.softmax(local_qk.reshape(
            n, self.num_head, h * w, self.window_size * self.window_size),
                                   dim=3)
        global_attn = torch.softmax(global_qk, dim=3)

        agg_bias = torch.einsum('bhnw,hcw->nbhc', local_attn,
                                self.relative_emb_v).reshape(h * w, n, c)

        agg_value = (global_attn @ v.transpose(-2, -1)).permute(2, 0, 1, 3).reshape(h * w, n, c)

        output = agg_value + agg_bias

        output = self.projection(output)

        self.last_size_2d = (h, w)
        return output, local_attn

    def compute_mask(self, height, width, device=None):
        pad_height = height + 2 * self.max_dis
        pad_width = width + 2 * self.max_dis

        if self.padded_local_mask is not None and (height,
                                                   width) == self.last_size_2d:
            padded_local_mask = self.padded_local_mask
            local_mask = self.local_mask

        else:
            ky, kx = torch.meshgrid([
                torch.arange(0, pad_height, device=device),
                torch.arange(0, pad_width, device=device)
            ])
            qy, qx = torch.meshgrid([
                torch.arange(0, height, device=device),
                torch.arange(0, width, device=device)
            ])

            qy = qy.reshape(-1, 1)
            qx = qx.reshape(-1, 1)
            offset_y = qy - ky.reshape(1, -1) + self.max_dis
            offset_x = qx - kx.reshape(1, -1) + self.max_dis
            padded_local_mask = (offset_y.abs() <= self.max_dis) & (
                offset_x.abs() <= self.max_dis)
            padded_local_mask = padded_local_mask.view(1, 1, height * width,
                                                       pad_height, pad_width)
            local_mask = padded_local_mask[:, :, :, self.max_dis:-self.max_dis,
                                           self.max_dis:-self.max_dis]
            pad_pixel = self.max_dis * self.dilation
            local_mask = F.pad(local_mask.float(),
                               (pad_pixel, pad_pixel, pad_pixel, pad_pixel),
                               mode='constant',
                               value=0).view(1, 1, height * width, pad_height,
                                             pad_width)
            self.padded_local_mask = padded_local_mask
            self.local_mask = local_mask

        return padded_local_mask, local_mask

class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_size, pos_emb=False):
        super(SelfAttentionBlock, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiheadAttention(d_model, n_heads)

        self.pos_emb = pos_emb
        if pos_emb:
            self.pos_generator = PositionEmbeddingSine(d_model // 2, normalize=True)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, hidden_size)
        self.activation = GNActDWConv2d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, d_model)

    def forward(self, x):
        B, C, H, W = x.shape

        if self.pos_emb:
            pos_emb = self.pos_generator(x).expand(x.shape[0], -1, -1, -1).view(x.shape[0], -1, H*W).permute(2, 0, 1)

        x = x.view(B, C, H*W).permute(2, 0, 1)
        _x = self.norm1(x)

        q = _x
        k = _x
        if self.pos_emb:
            q = q + pos_emb
            k = k + pos_emb
        v = _x

        _x = self.self_attn(q, k, v)[0]
        x = x + _x

        _x = self.norm2(x)
        _x = self.linear2(self.activation(self.linear1(_x), (H, W)))
        x = x + _x
        x = x.permute(1, 2, 0).view(B, C, H, W)

        return x
