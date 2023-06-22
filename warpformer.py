import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders.SwinTransformer import SwinTransformer
from encoders.MobileNetV2 import MobileNetV2

from layers.attention import MultiheadAttention, MultiheadLocalAttention, SelfAttentionBlock
from layers.fpn import FPNSegmentationHead
from layers.utils import GNActDWConv2d, FrozenBatchNorm2d

from flow_estimators.GMA.core.network import RAFTGMA
from flow_estimators.FlowFormer.core.FlowFormer import build_flowformer
from flow_estimators.FlowFormer.configs.submission import get_cfg

from utils.utils import load_network


def build_flow_estimator(cfg):
    if cfg.FLOW_ESTIMATOR_MODEL == "GMA":
        flow_estimator = RAFTGMA()
        flow_estimator.load_state_dict({k[7:]: v for k, v in torch.load("flow_estimators/GMA/checkpoints/gma-sintel.pth").items()})
    elif cfg.FLOW_ESTIMATOR_MODEL == "FlowFormer":
        flow_estimator = build_flowformer(get_cfg())
        flow_estimator.load_state_dict({k[7:]: v for k, v in torch.load("flow_estimators/FlowFormer/checkpoints/sintel.pth").items()})
    else:
        raise NotImplementedError
    return flow_estimator

def build_encoder(cfg):

    if cfg.ENCODER_MODEL == "MobileNetV2":
        encoder = MobileNetV2(
            output_stride=16, 
            norm_layer=FrozenBatchNorm2d, 
            freeze_at=cfg.ENCODER_FREEZE_STAGES
            )
        load_network(encoder, "pretrained/mobilenet_v2-b0353104.pth")

    elif cfg.ENCODER_MODEL == "SwinB":
        encoder = SwinTransformer(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            drop_path_rate=0.3,
            out_indices=(0, 1, 2),
            ape=False,
            patch_norm=True,
            frozen_stages=cfg.ENCODER_FREEZE_STAGES,
            use_checkpoint=False
            )
        load_network(encoder, "pretrained/swin_base_patch4_window7_224_22k.pth")

    else:
        raise NotImplementedError

    return encoder


class WarpFormerBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_size, max_dist=7):
        super(WarpFormerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiheadLocalAttention(d_model, n_heads, max_dis=max_dist)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, hidden_size)
        self.activation = GNActDWConv2d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, d_model)

    def forward(self, img, img_warp, mask):
        B, C, H, W = img.shape

        img = img.view(B, C, H*W).permute(2, 0, 1)
        img_warp = img_warp.view(B, C, H*W).permute(2, 0, 1)
        mask = mask.view(B, C, H*W).permute(2, 0, 1)

        x = img

        q = self.norm1(x)
        k = self.norm1(img_warp)
        v = self.norm1(img_warp + mask)

        q = q.permute(1, 2, 0).view(B, C, H, W)
        k = k.permute(1, 2, 0).view(B, C, H, W)
        v = v.permute(1, 2, 0).view(B, C, H, W)
        _x = self.cross_attn(q, k, v)[0]
        
        x = x + _x

        _x = self.norm2(x)
        _x = self.linear2(self.activation(self.linear1(_x), (H, W)))
        x = x + _x
        x = x.permute(1, 2, 0).reshape(B, C, H, W)

        return x

class MemFormerBlock(nn.Module):
    def __init__(self, d_model, n_heads, hidden_size):
        super(MemFormerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiheadAttention(d_model, n_heads)

        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, hidden_size)
        self.activation = GNActDWConv2d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, d_model)

    def forward(self, img, memory_imgs, memory_masks):
        B, C, H, W = img.shape

        img = img.view(B, C, H*W).permute(2, 0, 1)

        x = img

        q = self.norm1(x)
        k = self.norm1(memory_imgs)
        v = self.norm1(memory_imgs + memory_masks)

        _x = self.cross_attn(q, k, v)[0]
        
        x = x + _x

        _x = self.norm2(x)
        _x = self.linear2(self.activation(self.linear1(_x), (H, W)))
        x = x + _x
        x = x.permute(1, 2, 0).reshape(B, C, H, W)

        return x

class WarpFormer(nn.Module):
    def __init__(self, cfg):
        super(WarpFormer, self).__init__()

        self.cfg = cfg
        self.n_classes = cfg.NUM_CLASSES

        if cfg.ENCODER_MODEL == "MobileNetV2":
            self.shortcut_dims = [24, 32, 96, 1280]
        elif cfg.ENCODER_MODEL == "SwinB":
            self.shortcut_dims = [128, 256, 512, 512]
        else:
            raise NotImplementedError

        self.encoder = build_encoder(cfg)
        self.flow_estimator = build_flow_estimator(cfg)

        for p in self.flow_estimator.parameters():
            p.requires_grad = False

        self.decoder = FPNSegmentationHead(
            in_dim=cfg.WARPFORMER_DIM,
            out_dim=cfg.NUM_CLASSES,
            decode_intermediate_input=False,
            hidden_dim=cfg.WARPFORMER_DIM,
            shortcut_dims=self.shortcut_dims,
            align_corners=False,
        )

        self.id_bank = nn.Conv2d(
            cfg.NUM_CLASSES,
            cfg.WARPFORMER_DIM,
            kernel_size=16,
            stride=16,
            padding=0
        )

        self.encoder_projector = nn.Conv2d(self.shortcut_dims[-1], cfg.WARPFORMER_DIM, kernel_size=1)

        self.encoder_self_attn = SelfAttentionBlock(cfg.WARPFORMER_DIM, cfg.WARPFORMER_ATTENTION_HEADS, cfg.WARPFORMER_HIDDEN_SIZE, pos_emb=True)

        self.cross_attn_short_term = WarpFormerBlock(cfg.WARPFORMER_DIM, cfg.WARPFORMER_ATTENTION_HEADS, cfg.WARPFORMER_HIDDEN_SIZE, max_dist=7)
        self.cross_attn_long_term = MemFormerBlock(cfg.WARPFORMER_DIM, cfg.WARPFORMER_ATTENTION_HEADS, cfg.WARPFORMER_HIDDEN_SIZE)

        self.self_attn = SelfAttentionBlock(cfg.WARPFORMER_DIM, cfg.WARPFORMER_ATTENTION_HEADS, cfg.WARPFORMER_HIDDEN_SIZE, pos_emb=False)

        self._init_weight()

    def convert_flow(self, flow):
        _, _, H, W = flow.shape

        flow[:, 0, :, :] += torch.arange(W).view(1, 1, -1).cuda()
        flow[:, 1, :, :] += torch.arange(H).view(1, -1, 1).cuda()
        flow[:, 0, :, :] /= W
        flow[:, 1, :, :] /= H
        flow = 2 * flow - 1
        return flow
    
    def imagenet2imageflow(self, img):
        img *= torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()
        img += torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
        img = img * 255.0
        return img

    def add_reference_img(self, img):
        img_emb = self.encoder_self_attn(self.encoder_projector(self.encoder(img)[-1]))
        self.add_reference_img_emb(img_emb)

    def add_reference_img_emb(self, img_emb):
        B, C, H, W = img_emb.shape

        img_emb = img_emb.view(B, C, H*W).permute(2, 0, 1)

        if self.long_term_memory_img is None:
            self.long_term_memory_img = img_emb
        else:
            self.long_term_memory_img = torch.cat([self.long_term_memory_img, img_emb], dim=0)

    def add_reference_mask(self, mask):
        mask_emb = self.id_bank(mask)

        B, C, H, W = mask_emb.shape
        mask_emb = mask_emb.view(B, C, H*W).permute(2, 0, 1)

        if self.long_term_memory_mask is None:
            self.long_term_memory_mask = mask_emb
        else:
            self.long_term_memory_mask = torch.cat([self.long_term_memory_mask, mask_emb], dim=0)

    def forward(self, img, img_prev, mask_prev, add_to_memory=False):

        with torch.no_grad():
            img_flow = self.imagenet2imageflow(img.clone())
            img_prev_flow = self.imagenet2imageflow(img_prev.clone())

            if self.cfg.FLOW_ESTIMATOR_MODEL == "GMA":
                flow = self.flow_estimator(img_flow, img_prev_flow, iters=12)[-1]
            elif self.cfg.FLOW_ESTIMATOR_MODEL == "FlowFormer":
                flow = self.flow_estimator(img_flow, img_prev_flow)[0]

        flow = self.convert_flow(flow)
        mask_prev_warped = F.grid_sample(mask_prev, flow.permute(0, 2, 3, 1), mode="bilinear")
        img_prev_warped = F.grid_sample(img_prev, flow.permute(0, 2, 3, 1), mode="bilinear")

        mask_prev_emb = self.id_bank(mask_prev_warped)
        img_prev_emb = self.encoder_self_attn(self.encoder_projector(self.encoder(img_prev_warped)[-1]))

        xs = self.encoder(img)
        x = self.encoder_self_attn(self.encoder_projector(xs[-1]))

        if add_to_memory:
            x_mem = x.clone()

        mask_prev_emb = self.cross_attn_short_term(x, img_prev_emb, mask_prev_emb)
        mask_ref_emb = self.cross_attn_long_term(x, self.long_term_memory_img, self.long_term_memory_mask)

        if add_to_memory:
            self.add_reference_img_emb(x_mem)

        x = x + mask_prev_emb + mask_ref_emb
        x = self.self_attn(x)

        x = self.decoder([x], xs)
        x = F.interpolate(x, size=img.shape[2:], mode="bilinear")

        return x
    
    def reset(self):
        self.long_term_memory_img = None
        self.long_term_memory_mask = None

    def _init_weight(self):
        nn.init.xavier_uniform_(self.encoder_projector.weight)
        nn.init.orthogonal_(
            self.id_bank.weight.view(256, -1).permute(0, 1),
            gain=16**-2)
