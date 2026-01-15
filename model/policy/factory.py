import torch.nn as nn
from model.config.policy import PolicyConfig

def fetch_model_class(model_type: str)-> nn.Module:
    if model_type.upper() == 'VLTM':  # myvla
        from .VLTM import VLTM as VLTM
        return VLTM
    else:
        raise ValueError(f"model type '{model_type}' is not available.")
    
def make_policy_config(policy_type: str, args)-> PolicyConfig:
    if policy_type.upper() == 'VLTM':
        from model.config.VLTM import VLTMConfig
        return VLTMConfig(args)
    else:
        raise ValueError(f"policy type '{policy_type}' is not available.")
    
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