from dataclasses import dataclass, field
from policy import FeatureType, NormalizationMode, PolicyConfig
import draccus

CONFIG_NAME = "config.json"

@dataclass
class FlowMatch3DConfig(PolicyConfig):
    """Configuration class for DiffusionPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        
    """

    # Inputs / output structure.
    n_obs_steps: int = 3
    horizon: int = 10
    n_action_steps: int = 10

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Encoder arguments
    backbone="clip"
    finetune_backbone=False
    finetune_text_encoder=False
    num_vis_instr_attn_layers=2
    fps_subsampling_factor=4
    # Encoder and decoder arguments
    embedding_dim=120
    num_attn_heads=8 
    nhist=3
    nhand=2
    body_extra_dim=1
    # Decoder arguments
    num_shared_attn_layers=4
    relative=False
    rotation_format='quat_xyzw'
    # Denoising arguments
    denoise_timesteps=5
    denoise_model="rectified_flow"
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0
    # Training arguments
    lv2_batch_size=1
    dexhand_dim = 6  # dexhand has 6 DoF
    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 5e-4
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if not self.backbone.startswith("clip"):
            raise ValueError(
                f"`backbone` must be one of the clip variants. Got {self.backbone}."
            )
            
def main():
    config = FlowMatch3DConfig()
    print(config)
            
if __name__ == "__main__":
    import argparse, sys

    cli = argparse.ArgumentParser(...)
    cli.add_argument("--n_obs_steps", type=int, default=1, help="Number of observation steps")
    args = cli.parse_args()

    # 重新构造 sys.argv，使 wrapper 使用这些值（不要忘了保留程序名在 sys.argv[0]）
    new_argv = [sys.argv[0]]
    new_argv.append(f"--n_obs_steps={args.n_obs_steps}")

    sys.argv = new_argv
    main()