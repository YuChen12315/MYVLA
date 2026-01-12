import torch
from torch import nn
from torchvision import resnets

class NResNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        model_depth = kwargs.get("model_depth", 18)
        embedding_dim = kwargs.get("embedding_dim", 120)

        if model_depth == 18:
            base_model = resnets.resnet18(weights=None)
            out_channels = 512
        elif model_depth == 34:
            base_model = resnets.resnet34(weights=None)
            out_channels = 512
        elif model_depth == 50:
            base_model = resnets.resnet50(weights=None)
            out_channels = 2048
        elif model_depth == 101:
            base_model = resnets.resnet101(weights=None)
            out_channels = 2048
        else:
            raise ValueError(f"Unsupported model depth: {model_depth}")

        # Remove the fully connected layer and avgpool
        self.cnn = nn.Sequential(*list(base_model.children())[:-2])
        self.conv1x1 = nn.Conv2d(out_channels, embedding_dim, kernel_size=1)

    def forward(self, images: torch.Tensor):
        """
        Args:
            - x: (B, N, C, H, W)
        Returns:
            - feats: (B, N, F)
        """
        b, n, _, *chw = images.shape
        # 1. 特征提取
        y = self.cnn(images.reshape(b * n, *chw))
        # 2. 通道调整
        y = self.conv1x1(y)  # (B*N, F, H', W')
        
        # 3. 空间全局聚合（如全局平均池化）
        y = y.mean(dim=[2, 3])  # (B*N, F)
        
        # 4. 恢复原始维度结构
        feats = y.reshape(b, n, -1)  # (B, N, F)
        return feats