from dataclasses import dataclass, field
import draccus
from typing import Optional

from .policy import PolicyConfig
CONFIG_NAME = "config.json"
@dataclass
class EncoderConfig:
    """Encoder 配置"""
    vl_backbone: Optional[str] = "PaliGemma"
<<<<<<< HEAD
    # touch_encoder: str = "moevt"
    touch_encoder: str = None
=======
    touch_encoder: str = "moevt"
>>>>>>> 6e963d0 (v-0.0.1  pi0改动 touch encoder 和 2D 视觉编码器，修改动作预测头以适应新的动作维度)
    finetune_visual_backbone: bool = False
    finetune_llm_backbone: bool = False
    finetune_touch_encoder: bool = True
    
@dataclass
class VLTMConfig(PolicyConfig):
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
<<<<<<< HEAD
    
=======

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )
    
    resize_imgs_with_padding: Optional[tuple[int, int]] = (224, 224)
    tokenizer_max_length: int = 77
>>>>>>> 6e963d0 (v-0.0.1  pi0改动 touch encoder 和 2D 视觉编码器，修改动作预测头以适应新的动作维度)
    # Encoder arguments
    encoder_config: EncoderConfig = field(default_factory=EncoderConfig)
    
    # Encoder and decoder arguments
    embedding_dim=120
    
    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (224, 224)

    # Add empty images. Used by pi0_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Tokenizer
    tokenizer_max_length: int = 48

    # Projector
    proj_width: int = 1024

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True
    attention_implementation: str = "eager"  # or fa2, flex

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = False
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    # Denoising arguments
    num_steps: int = 10

    def __post_init__(self):
        super().__post_init__()
            
def main(args):
    config = VLTMConfig(args)
    print(config)
            
if __name__ == "__main__":
    import argparse, sys

    cli = argparse.ArgumentParser(...)
    cli.add_argument("--n_obs_steps", type=int, default=1, help="Number of observation steps")
    args = cli.parse_args()
    main(args)