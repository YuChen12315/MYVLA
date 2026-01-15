import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, resnet34, resnet50, ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class NResNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        model = kwargs.get("model", 18)
        embedding_dim = kwargs.get("embedding_dim", 120)

        # 可选：添加是否使用预训练权重的选项
        pretrained = kwargs.get("pretrained", True)
        weights = "DEFAULT" if pretrained else None
        
        # 选择基础模型
        if model == 'resnet18':
            base_model = resnet18(weights=weights)
            out_channels = 512
        elif model == 'resnet34':
            base_model = resnet34(weights=weights)
            out_channels = 512
        elif model == 'resnet50':
            base_model = resnet50(weights=weights)
            out_channels = 2048
        else:
            raise ValueError(f"Unsupported model: {model}")
        # Remove the fully connected layer and avgpool
        self.cnn = nn.Sequential(*list(base_model.children())[:-2])
        self.conv1x1 = nn.Conv2d(out_channels, embedding_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(embedding_dim)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, images: torch.Tensor):
        """
        Args:
            - x: (B, N, C, H, W)
        Returns:
            - feats: (B, N, F)
        """
        b, n, c, h, w = images.shape
        # 1. 特征提取
        y = self.cnn(images.reshape(b * n, c, h, w))
        # 2. 通道调整
        y = self.conv1x1(y)  # (B*N, F, H', W')
        y = self.bn(y)
        y = self.activation(y)
        avg_feat = self.avg_pool(y)  # (B*N, embedding_dim, 1, 1)
        max_feat = self.max_pool(y)  # (B*N, embedding_dim, 1, 1)
        y = (avg_feat + max_feat) / 2
        y = y.view(b * n, -1)  # (B*N, embedding_dim)
        y = F.normalize(y, p=2, dim=1)
        feats = y.reshape(b, n, -1)  # (B, N, embedding_dim)
        return feats