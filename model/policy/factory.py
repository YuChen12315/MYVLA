import torch.nn as nn
from model.config.policy import PolicyConfig
from model.config.flowmatch_2d import FlowMatch2DConfig
from model.config.flowmatch_3d import FlowMatch3DConfig

def fetch_model_class(model_type: str)-> nn.Module:
    if model_type == 'denoise3d':  # standard 3DFA
        from .denoise_actor_3d import DenoiseActor as DenoiseActor3D
        return DenoiseActor3D
    elif model_type == 'denoise2d':  # standard 2DFA
        from .denoise_actor_2d import DenoiseActor as DenoiseActor2D
        return DenoiseActor2D
    elif model_type == 'VLTM':  # myvla
        from .VLTM import VLTM as VLTM
        return VLTM
    else:
        raise ValueError(f"model type '{model_type}' is not available.")
    return None

def make_model_config(model_type: str, **kwargs) -> PolicyConfig:

    if model_type == "denoise2d":
        return FlowMatch2DConfig(**kwargs)
    elif model_type == "denoise3d":
        return FlowMatch3DConfig(**kwargs)
    else:
        raise ValueError(f"model type '{model_type}' is not available.")
    
    
def make_model(
    model_type: str,
    config: PolicyConfig
) -> nn.Module:
    """Make an instance of a model class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a model for that dataset or environment.

    Args:
    Returns: nn.Module
    """

    model_cls = fetch_model_class(model_type)
    cfg = config
    if cfg.pretrained_path:
        # Load a pretrained model and override the config if needed (for example, if there are inference-time
        # hyperparameters that we want to vary).
        model = model_cls.from_pretrained(cfg)
    else:
        # Make a fresh model.
        model = model_cls(cfg)

    model.to(cfg.device)
    assert isinstance(model, nn.Module)
    return model