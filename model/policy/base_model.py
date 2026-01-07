import abc
import dataclasses
import torch.nn as nn
from torch import Tensor

@dataclasses.dataclass
class BaseModel(nn.Module, abc.ABC):
    """Base class for all model implementations. Specific models should inherit from this class. They should call
    super().__init__() to initialize the shared attributes (action_dim, action_horizon, and max_token_len).
    """

    action_dim: int
    action_horizon: int

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