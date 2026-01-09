from dataclasses import dataclass, field
from utils import is_torch_device_available, auto_select_torch_device
from enum import Enum
from typing import Any, Protocol
import os
from pathlib import Path
import draccus
import yaml
import inspect
import tempfile

CONFIG_NAME = "config.json"
class FeatureType(str, Enum):
    STATE = "STATE"
    VISUAL = "VISUAL"
    ENV = "ENV"
    ACTION = "ACTION"
    NO_NORM = "NO_NORM"

class NormalizationMode(str, Enum):
    MIN_MAX = "MIN_MAX"
    MEAN_STD = "MEAN_STD"
    IDENTITY = "IDENTITY"


class DictLike(Protocol):
    def __getitem__(self, key: Any) -> Any: ...


@dataclass
class PolicyFeature:
    type: FeatureType
    shape: tuple
    
@dataclass
class PolicyConfig(draccus.ChoiceRegistry):
    """
    Base configuration class for policy models.

    Args:
        device: The device to use for the policy. Options are 'cuda', 'cpu', or 'mp' (multi-processing).
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        
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

    device: str | None = None  # cuda | cpu | mp

    def __post_init__(self):
        self.pretrained_path = None
        if not self.device or not is_torch_device_available(self.device):
            auto_device = auto_select_torch_device()
            print(f"Device '{self.device}' is not available. Switching to '{auto_device}'.")
            self.device = auto_device.type
            

    @property
    def robot_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.STATE:
                return ft
        return None
    @property
    def arm_state_feature(self) -> PolicyFeature | None:
        for state_name, ft in self.input_features.items():
            if  'arm_state' in state_name:
                return ft
        return None
    
    @property
    def hand_state_feature(self) -> PolicyFeature | None:
        for state_name, ft in self.input_features.items():
            if  'hand_state' in state_name:
                return ft
        return None
    
    @property
    def touch_state_feature(self) -> PolicyFeature | None:
        for state_name, ft in self.input_features.items():
            if  'touch_state' in state_name:
                return ft
        return None

    @property
    def env_state_feature(self) -> PolicyFeature | None:
        for _, ft in self.input_features.items():
            if ft.type is FeatureType.ENV:
                return ft
        return None

    @property
    def image_features(self) -> dict[str, PolicyFeature]:
        return {key: ft for key, ft in self.input_features.items() if ft.type is FeatureType.VISUAL}

    @property
    def action_feature(self) -> PolicyFeature | None:
        for _, ft in self.output_features.items():
            if ft.type is FeatureType.ACTION:
                return ft
        return None
    
    def _save_config_pretrained(self, save_directory: Path) -> None:
        with open(save_directory / CONFIG_NAME, "w") as f, draccus.config_type("json"):
            draccus.dump(self, f, indent=4)
            
    @classmethod
    def get_config_from_pretrained(cls, pretrained_name_or_path: str | Path, **policy_kwargs):
        model_id = str(pretrained_name_or_path)
        config_file: str | None = None
        if Path(model_id).is_dir():
            if CONFIG_NAME in os.listdir(model_id):
                config_file = os.path.join(model_id, CONFIG_NAME)
            else:
                raise FileNotFoundError(f"{CONFIG_NAME} not found in {Path(model_id).resolve()}")
        else:
            raise FileNotFoundError(f"{model_id} is not a dir") 
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)

        # 获取数据类所有有效的字段名
        cls_fields = set(inspect.signature(cls).parameters)
        # 创建一个新的配置字典，只保留类中存在的字段
        filtered_config = {k: v for k, v in config_data.items() if k in cls_fields}

        cli_overrides = policy_kwargs.pop("cli_overrides", [])
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            # 将过滤后的配置字典写入临时文件
            yaml.dump(filtered_config, temp_file)
            temp_file_path = temp_file.name
        
        try:
            # 解析临时配置文件
            return draccus.parse(cls, temp_file_path, args=cli_overrides)
        finally:
            # 清理：删除临时文件
            os.unlink(temp_file_path)