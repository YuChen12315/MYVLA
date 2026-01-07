import torch
import torch.nn as nn
from typing import Optional
from typing_extensions import override
import einops
from base_model import BaseModel
from model.config.policy import PolicyConfig
from ..encoder.multimodal.vltencoder import Encoder
from ..utils.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
from ..utils.layers import AttentionModule


class VLTM(BaseModel):

    def __init__(self,config: PolicyConfig):
        super().__init__(action_dim=config.action_dim, action_horizon=config.action_horizon)
        # Vision-language encoder, runs only once
        self.encoder = None
        # Action decoder, runs at every denoising timestep
        self.action_expert = TransformerHead(
            embedding_dim=config.transformer.embedding_dim,
            num_attn_heads=config.transformer.num_attn_heads,
            num_shared_attn_layers=config.transformer.num_shared_attn_layers,
            nhist=config.input_history_length,
            rotary_pe=config.transformer.rotary_pe,
            arm_dim=config.arm_dim,
            hand_dim=config.hand_dim
        )
        
    @override
    def compute_loss(
        self,
        observation,
        actions,
        *,
        train: bool = False,
    ) -> torch.Tensor:
        pass
    
    @override
    def sample_actions(self, observation, **kwargs) -> torch.Tensor:
        pass

class TransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=60,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=3,
                 rotary_pe=True,
                 arm_dim=7,
                 hand_dim=22):
        super().__init__()

        # Different embeddings
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.curr_gripper_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.traj_time_emb = SinusoidalPosEmb(embedding_dim)
        self.hand_embed = nn.Embedding(2, embedding_dim)

        # Attention from trajectory queries to language
        self.traj_lang_attention = AttentionModule(
            num_layers=1,
            d_model=embedding_dim,
            dim_fw=4 * embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=False,
            use_adaln=False,
            is_self=False
        )

        # Estimate attends to context (no subsampling)
        self.cross_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=False
        )

        # Shared attention layers
        self.self_attn = AttentionModule(
            num_layers=num_shared_attn_layers,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )

        # Specific (non-shared) Output layers:
        # 1. Rotation
        # self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
        # self.rotation_self_attn = AttentionModule(
        #     num_layers=2,
        #     d_model=embedding_dim,
        #     dim_fw=embedding_dim,
        #     dropout=0.1,
        #     n_heads=num_attn_heads,
        #     pre_norm=False,
        #     rotary_pe=rotary_pe,
        #     use_adaln=True,
        #     is_self=True
        # )
        # self.rotation_predictor = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, rot_dim)
        # )

        # # 2. Position
        # self.position_proj = nn.Linear(embedding_dim, embedding_dim)
        # self.position_self_attn = AttentionModule(
        #     num_layers=2,
        #     d_model=embedding_dim,
        #     dim_fw=embedding_dim,
        #     dropout=0.1,
        #     n_heads=num_attn_heads,
        #     pre_norm=False,
        #     rotary_pe=rotary_pe,
        #     use_adaln=True,
        #     is_self=True
        # )
        # self.position_predictor = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, 3)
        # )

        # # 3. Openess
        # self.openess_predictor = nn.Sequential(
        #     nn.Linear(embedding_dim, embedding_dim),
        #     nn.ReLU(),
        #     nn.Linear(embedding_dim, 1)
        # )

    def forward(self, traj_feats, trajectory, timesteps,
                rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos):
        """
        Arguments:
            traj_feats: (B, trajectory_length, nhand, F)
            trajectory: (B, trajectory_length, nhand, 3+6+X)
            timesteps: (B, 1)
            rgb3d_feats: (B, N, F)
            rgb3d_pos: (B, N, 3)
            rgb2d_feats: (B, N2d, F)
            rgb2d_pos: (B, N2d, 3)
            instr_feats: (B, L, F)
            instr_pos: (B, L, 3)
            proprio_feats: (B, nhist*nhand, F)
            fps_scene_feats: (B, M, F), M < N
            fps_scene_pos: (B, M, 3)

        Returns:
            list of (B, trajectory_length, nhand, 3+6+X)
        """
        _, traj_len, nhand, _ = trajectory.shape

        # Trajectory features
        if nhand > 1:
            traj_feats = traj_feats + self.hand_embed.weight[None, None]
        traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
        trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

        # Trajectory features cross-attend to context features
        traj_time_pos = self.traj_time_emb(
            torch.arange(0, traj_len, device=traj_feats.device)
        )[None, None].repeat(len(traj_feats), 1, nhand, 1)
        traj_time_pos = einops.rearrange(traj_time_pos, 'b l h c -> b (l h) c')
        traj_feats = self.traj_lang_attention(
            seq1=traj_feats,
            seq2=instr_feats,
            seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
        )[-1]
        traj_feats = traj_feats + traj_time_pos
        traj_xyz = trajectory[..., :3]

        # Denoising timesteps' embeddings
        time_embs = self.encode_denoising_timestep(
            timesteps, proprio_feats
        )

        # Positional embeddings
        rel_traj_pos, rel_scene_pos, rel_pos = self.get_positional_embeddings(
            traj_xyz, traj_feats,
            rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
            timesteps, proprio_feats,
            fps_scene_feats, fps_scene_pos,
            instr_feats, instr_pos
        )

        # Cross attention from gripper to full context
        traj_feats = self.cross_attn(
            seq1=traj_feats,
            seq2=rgb3d_feats,
            seq1_pos=rel_traj_pos,
            seq2_pos=rel_scene_pos,
            ada_sgnl=time_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = self.get_sa_feature_sequence(
            traj_feats, fps_scene_feats,
            rgb3d_feats, rgb2d_feats, instr_feats
        )
        features = self.self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=rel_pos,
            seq2_pos=rel_pos,
            ada_sgnl=time_embs
        )[-1]

        # Rotation head
        rotation = self.predict_rot(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )

        # Position head
        position, position_features = self.predict_pos(
            features, rel_pos, time_embs, traj_feats.shape[1]
        )

        # Openess head from position head
        openess = self.openess_predictor(position_features)

        return [
            torch.cat((position, rotation, openess), -1)
                 .unflatten(1, (traj_len, nhand))
        ]

    def encode_denoising_timestep(self, timestep, proprio_feats):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        proprio_feats = proprio_feats.flatten(1)
        curr_gripper_feats = self.curr_gripper_emb(proprio_feats)
        return time_feats + curr_gripper_feats

    def get_positional_embeddings(
        self,
        traj_xyz, traj_feats,
        rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
        timesteps, proprio_feats,
        fps_scene_feats, fps_scene_pos,
        instr_feats, instr_pos
    ):
        return None, None, None

    def get_sa_feature_sequence(
        self,
        traj_feats, fps_scene_feats,
        rgb3d_feats, rgb2d_feats, instr_feats
    ):
        return torch.cat([traj_feats, fps_scene_feats], 1)

    def predict_pos(self, features, pos, time_embs, traj_len):
        position_features = self.position_self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=pos,
            seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        position_features = position_features[:, :traj_len]
        position_features = self.position_proj(position_features)  # (B, N, C)
        position = self.position_predictor(position_features)
        return position, position_features

    def predict_rot(self, features, pos, time_embs, traj_len):
        rotation_features = self.rotation_self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=pos,
            seq2_pos=pos,
            ada_sgnl=time_embs
        )[-1]
        rotation_features = rotation_features[:, :traj_len]
        rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
        rotation = self.rotation_predictor(rotation_features)
        return rotation
