import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from ..vision.fpn import EfficientFeaturePyramidNetwork


class Moevt(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        num_image = kwargs.get("num_image",20)
        embedding_dim = kwargs.get("embedding_dim",120)
        self.cnn = Conv2dNormActivation(3,embedding_dim,kernel_size=7,stride=2,padding=3)
        self.feature_pyramid = EfficientFeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048],
                embedding_dim, output_level="res3"
            )
        
    def forward(self, touch):
        """
        Args:
            - touch: (B, C, H, W)
        Returns:
            - touch_feats: (B, N, F)
            - touch_pos: (B, N, 3)
        """
        B,C,H,W = touch.shape
        x = self.cnn(touch)  # (B, F, H', W')
        x = self.feature_pyramid(x)  # (B, F, H'', W'')
        x = x.flatten(2).transpose(1,2)  # (B, N, F)
        pos = torch.zeros(B,x.shape[1],3,device=x.device)  #TODO:# Dummy positional encodings
        return x, pos