from abc import ABC, abstractmethod
import torch
from torch import nn

from ....config.define import Backbone2DConfig, ImageTransform


class Backbone2D(nn.Module, ABC):
    config: Backbone2DConfig
    image_transform: ImageTransform

    def __init__(self, config: Backbone2DConfig) -> None:
        super().__init__()
        self.config = config

    @property
    @abstractmethod
    def feature_dim(self) -> int: ...

    @abstractmethod
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def init(name) -> "Backbone2D":
        if name == "dinosiglip":
            from .dinosiglip_vit import DinoSigLIPViTBackbone
            config = Backbone2DConfig(name="dinosiglip", image_size=224)
            return DinoSigLIPViTBackbone(config)
        elif name == "dinosiglip_stereo_vcp":
            from .dinosiglip_stereo import DINOSigLIPStereoBackbone
            config = Backbone2DConfig(name="dinosiglip_stereo_vcp", image_size=224)
            return DINOSigLIPStereoBackbone(config, stereo_feature="vcp")
        else:
            raise NotImplementedError
