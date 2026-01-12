import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional
from typing_extensions import override
from base_model import BaseModel
from model.config.policy import PolicyConfig
from ..encoder.multimodal.vltencoder import Encoder
from ..noise_scheduler import fetch_schedulers
from ..encoder.text import fetch_tokenizers

# import einops
from ..utils.position_encodings import RotaryPositionEncoding3D, SinusoidalPosEmb
# from ..utils.layers import AttentionModule


class VLTM(BaseModel):

    def __init__(self,config: PolicyConfig):
        super().__init__(action_dim=config.action_dim, action_horizon=config.action_horizon)
        # Vision-language encoder, runs only once
        self.config = config    
        self.encoder = Encoder(config.encoder_config)
        self.arm_scheduler, self.hand_scheduler = fetch_schedulers(
            config.denoise_model, config.denoise_timesteps
        )
        self.language_tokenizer = fetch_tokenizers(
            config.encoder_config.vl_backbone
        )
        # Action decoder, runs at every denoising timestep
        self.action_expert = self._build_action_expert(self.encoder.VL_encoder)
        
        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)
        
    def _prepare_observations(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        observations = {}
        images = batch['images']
        device = images.device
        # Preprocess image features present in the batch
        prepared_images = []
        img_masks = []
        present_img_keys = [key for key in images]
        for key in present_img_keys:
            img = images[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0
            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            prepared_images.append(img)
            img_masks.append(mask)
        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)
        
        observations = {
            'images': prepared_images,
            # 'point_clouds': batch['point_clouds'],
            'touchs': batch['touchs'],
            'popriotions': batch['states'],
            'img_masks': img_masks,
            'lang_tokens': lang_tokens,
            'lang_masks': lang_masks,
        }
        return observations

    def _prepare_actions(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        #TODO: modify according to action space
        return batch['actions']
    
    def _build_action_expert(self, VL_encoder):
        return VL_encoder
    
    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = SinusoidalPosEmb(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks
    
    @override
    def compute_loss(
        self,
        batch: dict[str: torch.Tensor],
        *,
        train: bool = False,
    ) -> torch.Tensor:

        observation = self._prepare_observations(batch)
        states = observation['propriotions']
        actions = self._prepare_actions(batch)     
        
        # Sample noise
        noise = torch.randn(actions.shape, device=actions.device)

        # Sample a random timestep
        timesteps = self.position_scheduler.sample_noise_step(
            num_noise=len(noise), device=noise.device
        )

        # Add noise to the clean trajectories
        arm = self.arm_scheduler.add_noise(
            actions[..., 3:], noise[..., 3:],
            timesteps
        )
        
        hand = self.hand_scheduler.add_noise(
            actions[..., :3], noise[..., :3],
            timesteps
        )
        noisy_xt = torch.cat((arm, hand), -1)
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.encoder(observation)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(states, noisy_xt, timesteps)
        
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        # Predict the noise residual
        (_, suffix_out), _ = self.action_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        suffix_out = suffix_out[:, -self.config.n_action_steps :]
        # reflect the action projection
        suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out = self.action_out_proj(suffix_out)
        # Compute loss
    
        denoise_target = self.arm_scheduler.prepare_target(
            noise, actions
        )
        losses = F.mse_loss(denoise_target, suffix_out, reduction="none")
        return losses
    
    @override
    def sample_actions(self, observation, **kwargs) -> torch.Tensor:
        pass
    
    

# class TransformerHead(nn.Module):

#     def __init__(self,
#                  embedding_dim=60,
#                  num_attn_heads=8,
#                  num_shared_attn_layers=4,
#                  nhist=3,
#                  rotary_pe=True,):
#         super().__init__()
#         self.arm_dim=7,
#         self.hand_dim=22
#         # Different embeddings
#         self.time_emb = nn.Sequential(
#             SinusoidalPosEmb(embedding_dim),
#             nn.Linear(embedding_dim, embedding_dim),
#             nn.SiLU(),
#             nn.Linear(embedding_dim, embedding_dim)
#         )
#         self.curr_gripper_emb = nn.Sequential(
#             nn.Linear(embedding_dim * nhist, embedding_dim),
#             nn.SiLU(),
#             nn.Linear(embedding_dim, embedding_dim)
#         )
#         self.traj_time_emb = SinusoidalPosEmb(embedding_dim)
#         self.hand_embed = nn.Embedding(2, embedding_dim)        #表示双臂的嵌入

#         # Attention from trajectory queries to language
#         self.traj_lang_attention = AttentionModule(
#             num_layers=1,
#             d_model=embedding_dim,
#             dim_fw=4 * embedding_dim,
#             dropout=0.1,
#             n_heads=num_attn_heads,
#             pre_norm=False,
#             rotary_pe=False,
#             use_adaln=False,
#             is_self=False
#         )

#         # Estimate attends to context (no subsampling)
#         self.cross_attn = AttentionModule(
#             num_layers=2,
#             d_model=embedding_dim,
#             dim_fw=embedding_dim,
#             dropout=0.1,
#             n_heads=num_attn_heads,
#             pre_norm=False,
#             rotary_pe=rotary_pe,
#             use_adaln=True,
#             is_self=False
#         )

#         # Shared attention layers
#         self.self_attn = AttentionModule(
#             num_layers=num_shared_attn_layers,
#             d_model=embedding_dim,
#             dim_fw=embedding_dim,
#             dropout=0.1,
#             n_heads=num_attn_heads,
#             pre_norm=False,
#             rotary_pe=rotary_pe,
#             use_adaln=True,
#             is_self=True
#         )

#         # Specific (non-shared) Output layers:
#         # 1. Rotation
#         # self.rotation_proj = nn.Linear(embedding_dim, embedding_dim)
#         # self.rotation_self_attn = AttentionModule(
#         #     num_layers=2,
#         #     d_model=embedding_dim,
#         #     dim_fw=embedding_dim,
#         #     dropout=0.1,
#         #     n_heads=num_attn_heads,
#         #     pre_norm=False,
#         #     rotary_pe=rotary_pe,
#         #     use_adaln=True,
#         #     is_self=True
#         # )
#         # self.rotation_predictor = nn.Sequential(
#         #     nn.Linear(embedding_dim, embedding_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(embedding_dim, rot_dim)
#         # )

#         # # 2. Position
#         # self.position_proj = nn.Linear(embedding_dim, embedding_dim)
#         # self.position_self_attn = AttentionModule(
#         #     num_layers=2,
#         #     d_model=embedding_dim,
#         #     dim_fw=embedding_dim,
#         #     dropout=0.1,
#         #     n_heads=num_attn_heads,
#         #     pre_norm=False,
#         #     rotary_pe=rotary_pe,
#         #     use_adaln=True,
#         #     is_self=True
#         # )
#         # self.position_predictor = nn.Sequential(
#         #     nn.Linear(embedding_dim, embedding_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(embedding_dim, 3)
#         # )

#         # # 3. Openess
#         # self.openess_predictor = nn.Sequential(
#         #     nn.Linear(embedding_dim, embedding_dim),
#         #     nn.ReLU(),
#         #     nn.Linear(embedding_dim, 1)
#         # )

#     def forward(self, traj_feats, trajectory, timesteps,
#                 rgb3d_feats, rgb3d_pos, rgb2d_feats, rgb2d_pos,
#                 instr_feats, instr_pos, proprio_feats,
#                 fps_scene_feats, fps_scene_pos):
#         """
#         Arguments:
#             traj_feats: (B, trajectory_length, nhand, F)
#             trajectory: (B, trajectory_length, nhand, 3+6+X)
#             timesteps: (B, 1)
#             rgb3d_feats: (B, N, F)
#             rgb3d_pos: (B, N, 3)
#             rgb2d_feats: (B, N2d, F)
#             rgb2d_pos: (B, N2d, 3)
#             instr_feats: (B, L, F)
#             instr_pos: (B, L, 3)
#             proprio_feats: (B, nhist*nhand, F)
#             fps_scene_feats: (B, M, F), M < N
#             fps_scene_pos: (B, M, 3)

#         Returns:
#             list of (B, trajectory_length, nhand, 3+6+X)
#         """
#         _, traj_len, nhand, _ = trajectory.shape

#         # Trajectory features
#         if nhand > 1:
#             traj_feats = traj_feats + self.hand_embed.weight[None, None]
#         traj_feats = einops.rearrange(traj_feats, 'b l h c -> b (l h) c')
#         trajectory = einops.rearrange(trajectory, 'b l h c -> b (l h) c')

#         # Trajectory features cross-attend to context features
#         traj_time_pos = self.traj_time_emb(
#             torch.arange(0, traj_len, device=traj_feats.device)
#         )[None, None].repeat(len(traj_feats), 1, nhand, 1)
#         traj_time_pos = einops.rearrange(traj_time_pos, 'b l h c -> b (l h) c')
#         traj_feats = self.traj_lang_attention(
#             seq1=traj_feats,
#             seq2=instr_feats,
#             seq1_sem_pos=traj_time_pos, seq2_sem_pos=None
#         )[-1]
#         traj_feats = traj_feats + traj_time_pos
#         traj_xyz = trajectory[..., :3]

#         # Denoising timesteps' embeddings
#         time_embs = self.encode_denoising_timestep(
#             timesteps, proprio_feats
#         )

#         # Positional embeddings
#         rel_traj_pos, rel_scene_pos, rel_pos = self.get_positional_embeddings(
#             traj_xyz, traj_feats,
#             rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
#             timesteps, proprio_feats,
#             fps_scene_feats, fps_scene_pos,
#             instr_feats, instr_pos
#         )

#         # Cross attention from gripper to full context
#         traj_feats = self.cross_attn(
#             seq1=traj_feats,
#             seq2=rgb3d_feats,
#             seq1_pos=rel_traj_pos,
#             seq2_pos=rel_scene_pos,
#             ada_sgnl=time_embs
#         )[-1]

#         # Self attention among gripper and sampled context
#         features = self.get_sa_feature_sequence(
#             traj_feats, fps_scene_feats,
#             rgb3d_feats, rgb2d_feats, instr_feats
#         )
#         features = self.self_attn(
#             seq1=features,
#             seq2=features,
#             seq1_pos=rel_pos,
#             seq2_pos=rel_pos,
#             ada_sgnl=time_embs
#         )[-1]

#         # Rotation head
#         rotation = self.predict_rot(
#             features, rel_pos, time_embs, traj_feats.shape[1]
#         )

#         # Position head
#         position, position_features = self.predict_pos(
#             features, rel_pos, time_embs, traj_feats.shape[1]
#         )

#         # Openess head from position head
#         openess = self.openess_predictor(position_features)

#         return [
#             torch.cat((position, rotation, openess), -1)
#                  .unflatten(1, (traj_len, nhand))
#         ]

#     def encode_denoising_timestep(self, timestep, proprio_feats):
#         """
#         Compute denoising timestep features and positional embeddings.

#         Args:
#             - timestep: (B,)

#         Returns:
#             - time_feats: (B, F)
#         """
#         time_feats = self.time_emb(timestep)
#         proprio_feats = proprio_feats.flatten(1)
#         curr_gripper_feats = self.curr_gripper_emb(proprio_feats)
#         return time_feats + curr_gripper_feats

#     def get_positional_embeddings(
#         self,
#         traj_xyz, traj_feats,
#         rgb3d_pos, rgb3d_feats, rgb2d_feats, rgb2d_pos,
#         timesteps, proprio_feats,
#         fps_scene_feats, fps_scene_pos,
#         instr_feats, instr_pos
#     ):
#         return None, None, None

#     def get_sa_feature_sequence(
#         self,
#         traj_feats, fps_scene_feats,
#         rgb3d_feats, rgb2d_feats, instr_feats
#     ):
#         return torch.cat([traj_feats, fps_scene_feats], 1)

#     def predict_pos(self, features, pos, time_embs, traj_len):
#         position_features = self.position_self_attn(
#             seq1=features,
#             seq2=features,
#             seq1_pos=pos,
#             seq2_pos=pos,
#             ada_sgnl=time_embs
#         )[-1]
#         position_features = position_features[:, :traj_len]
#         position_features = self.position_proj(position_features)  # (B, N, C)
#         position = self.position_predictor(position_features)
#         return position, position_features

#     def predict_rot(self, features, pos, time_embs, traj_len):
#         rotation_features = self.rotation_self_attn(
#             seq1=features,
#             seq2=features,
#             seq1_pos=pos,
#             seq2_pos=pos,
#             ada_sgnl=time_embs
#         )[-1]
#         rotation_features = rotation_features[:, :traj_len]
#         rotation_features = self.rotation_proj(rotation_features)  # (B, N, C)
#         rotation = self.rotation_predictor(rotation_features)
#         return rotation

def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks

def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img

def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector

def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

