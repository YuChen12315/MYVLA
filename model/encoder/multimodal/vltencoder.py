import torch
from torch import nn
from model.config.VLTM import EncoderConfig
from ..vision import fetch_visual_encoders
from ..text import fetch_text_encoders
from ..touch import fetch_touch_encoders

class Encoder(nn.Module):
    def __init__(self, config: EncoderConfig):  # 直接传入配置对象
        super().__init__()
        self.visual_backbone = fetch_visual_encoders(config.visual_backbone)
        self.llm_backbone = fetch_text_encoders(config.llm_backbone)
        self.touch_encoder = fetch_touch_encoders(config.touch_encoder)
        
        # 设置参数是否需要梯度
        for p in self.visual_backbone.parameters():
            p.requires_grad = config.finetune_visual_backbone
        for p in self.llm_backbone.parameters():
            p.requires_grad = config.finetune_llm_backbone
        for p in self.touch_encoder.parameters():
            p.requires_grad = config.finetune_touch_encoder
        
    def forward(self, observations: dict):
        vision = observations['vision']
        vision_features = self.visual_backbone(vision)
        