import torch
from torch import nn
from torch.nn import functional as F
import einops

from ..noise_scheduler import fetch_schedulers
from ..utils.layers import AttentionModule
from ..utils.position_encodings import SinusoidalPosEmb
from ..utils.utils import (
    compute_rotation_matrix_from_ortho6d,
    get_ortho6d_from_rotation_matrix,
    normalise_quat,
    matrix_to_quaternion,
    quaternion_to_matrix
)


class DenoiseActor(nn.Module):

    def __init__(self,
                 # Encoder and decoder arguments
                 embedding_dim=60,
                 num_attn_heads=8,
                 nhist=3,
                 nhand=1,
                 action_dim=7,
                 action_chunk = 10,
                 # Decoder arguments
                 num_shared_attn_layers=4,
                 # Denoising arguments
                 denoise_timesteps=5,
                 denoise_model="ddpm",):
        super().__init__()
        # Arguments to be accessed by the main class

        # Vision-language encoder, runs only once
        self.encoder = None  # Implement this!

        # Action decoder, runs at every denoising timestep
        self.act_encoder = nn.Linear(
            action_dim,
            embedding_dim
        )
        self.prediction_head = TransformerHead(
            embedding_dim=embedding_dim,
            nhist=nhist * nhand,
            num_attn_heads=num_attn_heads,
            num_shared_attn_layers=num_shared_attn_layers)

        # Noise/denoise schedulers and hyperparameters
        self.action_scheduler = fetch_schedulers(denoise_model, denoise_timesteps)
        self.n_steps = denoise_timesteps

        # Normalization for the 3D space, will be loaded in the main process

        self.workspace_normalizer = nn.Parameter(
            torch.Tensor([[0.]*action_dim, [1.]*action_dim]),
            requires_grad=False
        )
        self.action_dim = action_dim
        self.action_chunk = action_chunk

    def encode_inputs(self, rgb, pcd, instruction, proprio):
        (instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos
        ) = self.encoder(rgb, pcd, instruction, proprio.flatten(1, 2))
        
        return (instr_feats, instr_pos, proprio_feats, fps_scene_feats, fps_scene_pos)

    def policy_forward_pass(self, action, timestep, fixed_inputs):
        # Parse inputs
        (   rgb3d_feats, pcd,
            instr_feats, instr_pos,
            proprio_feats,
            fps_scene_feats, fps_scene_pos) = fixed_inputs

        # Get features from normalized (relative) action
        action_feats = self.act_encoder(action)

        # But use actions from unnormalized absolute action
        action = self.unnormalize_action(action)[..., :-1]

        return self.prediction_head(
            action_feats,
            action,
            timestep,
            rgb3d_feats=rgb3d_feats,
            rgb3d_pos=pcd,
            instr_feats=instr_feats,
            instr_pos=instr_pos,
            proprio_feats=proprio_feats,
            fps_scene_feats=fps_scene_feats,
            fps_scene_pos=fps_scene_pos
        )

    def conditional_sample(self, action, device, fixed_inputs):
        # Set schedulers
        self.action_scheduler.set_timesteps(self.n_steps, device=device)

        # Iterative denoising
        timesteps = self.action_scheduler.timesteps
        for t_ind, t in enumerate(timesteps):
            out = self.policy_forward_pass(
                action,
                t * torch.ones(len(action)).to(device).long(),
                fixed_inputs
            )
            action = self.action_scheduler.step(out, t_ind, action).prev_sample
            
        return torch.cat((action, out[..., -1:]), -1)

    def compute_action(self, action_mask,
                           rgb, pcd, instruction, proprio):
        # Encode observations, states, instructions
        fixed_inputs = self.encode_inputs(rgb, pcd, instruction, proprio)

        # Sample from learned model starting from noise
        noise = torch.randn(
            size=tuple(action_mask.shape) + (self.action_dim,),
            device=action_mask.device
        )
        action = self.conditional_sample(
            noise,
            device=action_mask.device,
            fixed_inputs=fixed_inputs
        )

        # unnormalize action
        action = self.unnormalize_action(action[..., :-1])
        return action

    def compute_loss(self, action,
                     rgb, pcd, instruction, proprio):
        # Encode observations, states, instructions
        fixed_inputs = self.encode_inputs(rgb, pcd, instruction, proprio)

        # Process action
        gt_openess = action[..., -1:]
        action = action[..., :-1]
        # Normalize all pos
        action = self.normalize_action(action)
        # Sample noise
        noise = torch.randn(action.shape, device=action.device)

        # Sample a random timestep
        timesteps = self.action_scheduler.sample_noise_step(num_noise=len(noise), device=noise.device)

        # Add noise to the clean trajectories
        x_t = self.action_scheduler.add_noise(
            action, noise,
            timesteps
        )

        # Predict the noise residual
        pred = self.policy_forward_pass(
            x_t,
            timesteps, fixed_inputs
        )

        # Compute loss
        u_t = pred[..., :-1]
        openess = pred[..., -1:]
        v_t = self.action_scheduler.prepare_target(
            noise, action
        )
        loss = (
            F.l1_loss(u_t, v_t[..., :-1], reduction='mean')
            + F.binary_cross_entropy_with_logits(openess, gt_openess)
        )
        return loss

    def normalize_action(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min

        out = signal.clone()
        out[..., :self.action_dim] = (
            (signal[..., :self.action_dim] - _min) / diff * 2.0
            - 1.0
        )
        return out

    def unnormalize_action(self, signal):
        _min = self.workspace_normalizer[0]
        _max = self.workspace_normalizer[1]
        diff = _max - _min

        out = signal.clone()
        out[..., :self.action_dim] = (
            (signal[..., :self.action_dims] + 1.0) / 2.0 * diff
            + _min
        )
        return out

    # def convert_rot(self, signal):
    #     # If Euler then no conversion
    #     if self._rotation_format == 'euler':
    #         return signal
    #     # Else assume quaternion
    #     rot = normalise_quat(signal[..., 3:7])
    #     res = signal[..., 7:] if signal.size(-1) > 7 else None
    #     # The following code expects wxyz quaternion format!
    #     if self._rotation_format == 'quat_xyzw':
    #         rot = rot[..., (3, 0, 1, 2)]
    #     # Convert to rotation matrix
    #     rot = quaternion_to_matrix(rot)
    #     # Convert to 6D
    #     if len(rot.shape) == 4:
    #         B, L, D1, D2 = rot.shape
    #         rot = rot.reshape(B * L, D1, D2)
    #         rot = get_ortho6d_from_rotation_matrix(rot)
    #         rot = rot.reshape(B, L, 6)
    #     else:
    #         rot = get_ortho6d_from_rotation_matrix(rot)
    #     # Concatenate pos, rot, other state info
    #     signal = torch.cat([signal[..., :3], rot], dim=-1)
    #     if res is not None:
    #         signal = torch.cat((signal, res), -1)
    #     return signal

    # def unconvert_rot(self, signal):
    #     # If Euler then no conversion
    #     if self._rotation_format == 'euler':
    #         return signal
    #     # Else assume quaternion
    #     res = signal[..., 9:] if signal.size(-1) > 9 else None
    #     if len(signal.shape) == 3:
    #         B, L, _ = signal.shape
    #         rot = signal[..., 3:9].reshape(B * L, 6)
    #         mat = compute_rotation_matrix_from_ortho6d(rot)
    #         quat = matrix_to_quaternion(mat)
    #         quat = quat.reshape(B, L, 4)
    #     else:
    #         rot = signal[..., 3:9]
    #         mat = compute_rotation_matrix_from_ortho6d(rot)
    #         quat = matrix_to_quaternion(mat)
    #     # The above code handled wxyz quaternion format!
    #     if self._rotation_format == 'quat_xyzw':
    #         quat = quat[..., (1, 2, 3, 0)]
    #     signal = torch.cat([signal[..., :3], quat], dim=-1)
    #     if res is not None:
    #         signal = torch.cat((signal, res), -1)
    #     return signal

    def forward(
        self,
        batch,
        run_inference=False
    ):
        """
        Arguments:
            action: (B, action_length, nhand, 3+4+X)
            action_mask: (B, action_length, nhand)
            rgb3d: (B, num_3d_cameras, 3, H, W) in [0, 1]
            pcd: (B, num_3d_cameras, 3, H, W) in world coordinates
            instruction: tokenized text instruction
            proprio: (B, nhist, nhand, 3+4+X)

        Note:
            The input rotation is expressed either as:
                a) quaternion (4D), then the model converts it to 6D internally.
                b) Euler angles (3D).

        Returns:
            - loss: scalar, if run_inference is False
            - action: (B, action_length, nhand, 3+rot+1), at inference
        """
        
        rgb = batch["rgb"]
        pcd = batch["pcd"]
        instruction = batch["task"]
        proprio = batch["state"]
        # Inference, don't use action
        if run_inference:
            action_mask = (rgb.shape[0], self.action_chunk, self.nhand)
            return self.compute_action(
                action_mask,
                rgb, pcd, instruction, proprio
            )

        # Training, use action to compute loss
        action = batch["action"]
        return self.compute_loss(
            action,
            rgb, pcd, instruction, proprio
        )


class TransformerHead(nn.Module):

    def __init__(self,
                 embedding_dim=120,
                 num_attn_heads=8,
                 num_shared_attn_layers=4,
                 nhist=3*2,
                 rotary_pe=True,
                 action_dim=7):
        super().__init__()

        # Different embeddings
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(embedding_dim),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.action_emb = nn.Sequential(
            nn.Linear(embedding_dim * nhist, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.action_time_mlp = nn.Sequential(
            nn.Linear(embedding_dim*2, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        self.action_time_emb = SinusoidalPosEmb(embedding_dim)
        self.hand_embed = nn.Embedding(2, embedding_dim)

        # Attention from action queries to language
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
        self.action_proj = nn.Linear(embedding_dim, embedding_dim)
        self.action_self_attn = AttentionModule(
            num_layers=2,
            d_model=embedding_dim,
            dim_fw=embedding_dim,
            dropout=0.1,
            n_heads=num_attn_heads,
            pre_norm=False,
            rotary_pe=rotary_pe,
            use_adaln=True,
            is_self=True
        )
        self.action_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, action_dim)
        )

        self.openess_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, action_feats, action, timesteps,
                rgb_feats, rgb_pos, 
                instr_feats, instr_pos, proprio_feats,
                fps_scene_feats, fps_scene_pos):
        """
        Arguments:
            traj_feats: (B, action_length, nhand, F)
            action: (B, action_length, nhand, action_dim+X)
            timesteps: (B, 1)
            rgb_feats: (B, N, F)
            rgb_pos: (B, N, 3)
            instr_feats: (B, L, F)
            instr_pos: (B, L, 3)
            proprio_feats: (B, nhist*nhand, F)
            fps_scene_feats: (B, M, F), M < N
            fps_scene_pos: (B, M, 3)

        Returns:
            list of (B, action_length, nhand, action_dim+X)
        """
        _, traj_len, nhand, _ = action.shape

        # action features
        if nhand > 1:
            action_feats = action_feats + self.hand_embed.weight[None, None]
        action_feats = einops.rearrange(action_feats, 'b l h c -> b (l h) c')
        action = einops.rearrange(action, 'b l h c -> b (l h) c')

        # action features cross-attend to context features
        action_time_pos = self.action_time_emb(
            torch.arange(0, traj_len, device=action_feats.device)
        )[None, None].repeat(len(action_feats), 1, nhand, 1)
        
        action_time_pos = einops.rearrange(action_time_pos, 'b l h c -> b (l h) c')
        
        action_feats = self.traj_lang_attention(
            seq1=action_feats,
            seq2=instr_feats,
            seq1_sem_pos=action_time_pos, seq2_sem_pos=None
        )[-1]
        action_feats = action_feats + action_time_pos

        # Denoising timesteps' embeddings
        time_action_embs = self.encode_denoising_timestep(
            timesteps, action_feats
        )

        # Positional embeddings
        rel_act_pos, rel_scene_pos, rel_pos = self.get_positional_embeddings(
            action,
            rgb_pos,
            timesteps,
            fps_scene_pos,
            instr_pos
        )

        # Cross attention from gripper to full context
        action_feats = self.cross_attn(
            seq1=action_feats,
            seq2=rgb_feats,
            seq1_pos=rel_act_pos,
            seq2_pos=rel_scene_pos,
            ada_sgnl=time_action_embs
        )[-1]

        # Self attention among gripper and sampled context
        features = self.get_sa_feature_sequence(
            action_feats, fps_scene_feats,
            rgb_feats, instr_feats
        )
        features = self.self_attn(
            seq1=features,
            seq2=features,
            seq1_pos=rel_pos,
            seq2_pos=rel_pos,
            ada_sgnl=time_action_embs
        )[-1]
        # action head
        
        action_features = features[:, :traj_len]
        action_features = self.action_proj(action_features)  # (B, N, C)
        action = self.action_predictor(action_features)

        # Openess head from action head
        openess = self.openess_predictor(action_features)
        openess = (openess>0.5).float()

        return torch.cat((action, openess), -1).unflatten(1, (traj_len, nhand))

    def encode_denoising_timestep(self, timestep, action_feats):
        """
        Compute denoising timestep features and positional embeddings.

        Args:
            - timestep: (B,)

        Returns:
            - time_feats: (B, F)
        """
        time_feats = self.time_emb(timestep)
        action_feats = action_feats.flatten(1)
        action_feats = self.action_emb(action_feats)
        action_time_emb = torch.cat(time_feats, action_feats, dim=2)
        action_time_emb = self.action_time_mlp(action_time_emb)
        return action_time_emb

    def get_positional_embeddings(
        self,
        traj_xyz,
        rgb_pos,
        timesteps,
        fps_scene_pos,
        instr_pos
    ):
        return None, None, None

    def get_sa_feature_sequence(
        self,
        act_feats, fps_scene_feats,
        rgb_feats, instr_feats
    ):
        return torch.cat([act_feats, fps_scene_feats], 1)

    # def predict(self, features, pos, time_embs, traj_len):
    #     action_features = self.action_self_attn(
    #         seq1=features,
    #         seq2=features,
    #         seq1_pos=pos,
    #         seq2_pos=pos,
    #         ada_sgnl=time_embs
    #     )[-1]
    #     action_features = action_features[:, :traj_len]
    #     action_features = self.action_proj(action_features)  # (B, N, C)
    #     action = self.action_predictor(action_features)
    #     return action, action_features
