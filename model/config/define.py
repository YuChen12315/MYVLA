from typing import Optional, List, Type, Union, Dict
from pydantic import BaseModel, Field
from PIL import Image
import torch


def optional_str(x: Union[str, None]) -> Union[str, None]:
    if x is None or x == "none" or x == "None":
        return None
    else:
        return x


class ImageTransform:
    def __call__(
        self, img: Image, **kwargs: str
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]: ...

class MixDatasetConfig(BaseModel):
    datasets: Optional[List[str]] = Field(default=None)
    datasets_type: Optional[List[str]] = Field(default=None)
    datasets_weight: Optional[List[float]] = Field(default=None)

    @staticmethod
    def from_str(string: str) -> "MixDatasetConfig":
        datasets = []
        datasets_type = []
        datasets_weight = []
        for item in string.split(";"):
            d, dt, dw = item.split(',')
            datasets.append(d)
            datasets_type.append(dt)
            datasets_weight.append(float(dw))
        return MixDatasetConfig(
            datasets=datasets,
            datasets_type=datasets_type,
            datasets_weight=datasets_weight,
        )

class Backbone2DConfig(BaseModel):
    name: str
    image_size: int


class ActionExpertConfig(BaseModel):
    hidden_size_scale: Optional[int] = Field(default=None)
    intermediate_size_scale: Optional[int] = Field(default=None)
    hidden_size: Optional[int] = Field(init=False, default=None)
    intermediate_size: Optional[int] = Field(init=False, default=None)
    hidden_act: Optional[str] = Field(init=False, default=None)
            

