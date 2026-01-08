import torch
from torch import nn
from ..vision import fetch_visual_encoders
from ..text import fetch_text_encoders
from ..touch import fetch_touch_encoders

class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.visual_backbone = fetch_visual_encoders(kwargs.get("visual_backbone","clip"))
        self.llm_backbone = fetch_text_encoders(kwargs.get("llm_backbone","gpt2"))
        self.touch_encoder = fetch_touch_encoders(kwargs.get("touch_encoder","moevt"))
        for p in self.visual_backbone.parameters():
            p.requires_grad = kwargs.get("finetune_visual_backbone",False)
        for p in self.llm_backbone.parameters():
            p.requires_grad = kwargs.get("finetune_llm_backbone",False)
        for p in self.touch_encoder.parameters():
            p.requires_grad = kwargs.get("finetune_touch_encoder",True)
        
    def forward(self, observations: dict):
        vision = observations['vision']
        vision_features = self.visual_backbone(vision)
        