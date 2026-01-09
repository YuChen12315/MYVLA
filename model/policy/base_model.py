import abc
import dataclasses
import torch.nn as nn
from torch import Tensor

@dataclasses.dataclass
class BaseModel(nn.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """
    def __init__(self,):
        super().__init__() 
        
    @abc.abstractmethod
    def compute_loss(
        self,
        observation,
        actions,
        *,
        train: bool = False,
    ) -> Tensor: ...

    @abc.abstractmethod
    def sample_actions(self, observation, **kwargs) -> Tensor: ...
    
    # @abc.abstractmethod
    # def _save_to_safetensor(self):...