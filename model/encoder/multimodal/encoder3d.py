import einops
import torch
from torch import nn
from torch.nn import functional as F

from ...utils.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from ...utils.layers import AttentionModule
from ..vision.fpn import EfficientFeaturePyramidNetwork
from .base_encoder import Encoder as BaseEncoder


class Encoder(BaseEncoder):

    def __init__(self,
                 backbone="clip",
                 embedding_dim=60,
                 nhist=1,
                 num_attn_heads=9,
                 num_vis_instr_attn_layers=2,
                 fps_subsampling_factor=5,
                 finetune_backbone=False,
                 finetune_text_encoder=False):
        super().__init__(
            backbone=backbone,
            embedding_dim=embedding_dim,
            nhist=nhist,
            num_attn_heads=num_attn_heads,
            num_vis_instr_attn_layers=num_vis_instr_attn_layers,
            fps_subsampling_factor=fps_subsampling_factor,
            finetune_backbone=finetune_backbone,
            finetune_text_encoder=finetune_text_encoder
        )

        # Postprocess scene features
        if self._backbone_name == 'clip':
            self.output_level = "res3"
            self.feature_pyramid = EfficientFeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048],
                embedding_dim, output_level="res3"
            )
            self.rgb2d_proj = nn.Linear(1024, embedding_dim)

        # 3D relative positional embeddings
        self.relative_pe_layer = RotaryPositionEncoding3D(embedding_dim)

        # Proprioception learnable encoding if 3D is used
        self.proprio_embed = nn.Embedding(nhist, embedding_dim)
        
        self.gripper_context_head = AttentionModule(
            num_layers=3, d_model=embedding_dim, dim_fw=embedding_dim,
            n_heads=num_attn_heads, rotary_pe=True, use_adaln=False,
            pre_norm=False
        )

        # Camera IDs for the 2D cameras
        # self.camera_ids = nn.Embedding(2, embedding_dim)
        # self.pos_embed_2d = SinusoidalPosEmb(embedding_dim)

    def encode_proprio(self, proprio, context_feats, context_pos):
        """
        Compute proprioception features.

        Args:
            - proprio: (B, nhist, 3+) 
            - context_feats: (B, npt, C) (rgb feats)
            - context_pos: (B, npt, 3)

        Returns:
            - gripper_feats: (B, nhist, F)
        """
        # Learnable embedding for proprioception
        proprio_feats = self.proprio_embed.weight.unsqueeze(0).repeat(
            len(proprio), 1, 1
        )

        # Rotary positional encoding
        proprio_pos = self.relative_pe_layer(proprio[..., :3])
        context_pos = self.relative_pe_layer(context_pos)

        # Attention to scene tokens
        proprio_feats = self.gripper_context_head(
            proprio_feats, context_feats,
            seq1_pos=proprio_pos, seq2_pos=context_pos
        )[-1]

        return proprio_feats

    def encode_clip(self, rgb, pcd, text):
        """
        Compute visual features/pos embeddings.

        Args:
            - rgb: (B, ncam, 3, H, W), rgb obs of cameras
            - pcd: (B, ncam, 3, H, W)
            - text: [str] of len=B, text instruction

        Returns:
            - rgb_feats: (B, Np, F)
            - pcd: (B, Np, 3)
            - instr_feats: (B, L, F)
        """
        # Encode language
        instruction = self.text_encoder(text)
        instr_feats = self.instruction_encoder(instruction)

        # 3D camera features
        num_cameras = rgb.shape[1]
        # Pass each view independently through backbone
        rgb = einops.rearrange(rgb, "bt ncam c h w -> (bt ncam) c h w")
        rgb = self.normalize(rgb)
        rgb_feats = self.backbone(rgb)
        # Pass visual features through feature pyramid network
        rgb_feats = self.feature_pyramid(rgb_feats)[self.output_level]
        feat_h, feat_w = rgb_feats.shape[-2:]
        # Merge different cameras
        rgb_feats = einops.rearrange(
            rgb_feats,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )
        # Attention from vision to language
        rgb_feats = self.vl_attention(seq1=rgb_feats, seq2=instr_feats)[-1]

        # Point cloud
        num_cameras = pcd.shape[1]
        # Interpolate point cloud to get the corresponding locations
        pcd = F.interpolate(
            einops.rearrange(pcd, "bt ncam c h w -> (bt ncam) c h w"),
            (feat_h, feat_w),
            mode='bilinear'
        )
        # Merge different cameras
        pcd = einops.rearrange(
            pcd,
            "(bt ncam) c h w -> bt (ncam h w) c", ncam=num_cameras
        )

        return rgb_feats, pcd, instr_feats
