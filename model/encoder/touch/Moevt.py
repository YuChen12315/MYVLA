import torch
from torch import nn
from torchvision.ops import Conv2dNormActivation
from ..vision.fpn import EfficientFeaturePyramidNetwork


class Moevt(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        num_image = kwargs.get("num_image",20)
        embedding_dim = kwargs.get("embedding_dim",120)
        self.moe = MoeExpert(num_experts=num_image, embedding_dim=embedding_dim)
        self.feature_pyramid = EfficientFeaturePyramidNetwork(
                [64, 256, 512, 1024, 2048],
                embedding_dim, output_level="res3"
            )
        
    def forward(self, touch, touch_img):
        """
        Args:
            - touch: (B, C, H, W)
        Returns:
            - touch_feats: (B, N, F)
            - touch_pos: (B, N, 3)
        """
        B,C,H,W = touch.shape
        x = self.cnn(touch_img)  # (B, F, H', W')
        x = self.feature_pyramid(x)  # (B, F, H'', W'')
        x = x.flatten(2).transpose(1,2)  # (B, N, F)
        pos = torch.zeros(B,x.shape[1],3,device=x.device)  #TODO:# Dummy positional encodings
        return x, pos
    
class GatingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 计算各专家权重（需经过softmax归一化）
        logits = x
        weights = torch.softmax(logits, dim=-1)
        return weights
    
class MoeExpert(nn.Module):
    def __init__(self, num_experts, embedding_dim):
        super().__init__()
        self.num_experts = num_experts
        self.embedding_dim = embedding_dim
        self.gating_network = GatingNetwork()
        cnn = nn.Sequential(
            # 第1阶段：提取低级特征
            Conv2dNormActivation(3, 64, kernel_size=3, stride=2, padding=1),
            
            # 第2阶段：进一步提取特征
            Conv2dNormActivation(64, 128, kernel_size=3, stride=2, padding=1),
            
            # 第3阶段：准备输入Transformer
            Conv2dNormActivation(128, 256, kernel_size=3, stride=1, padding=1),
            
            # 全局平均池化或展平
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.experts = nn.ModuleList([
            cnn for _ in range(num_experts)
        ])

    def forward(self, x):
        # 计算各专家的输出
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # (B, F, H, W)
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, F, H, W)

        #TODO: 需要设计动态路由机制，实现专家选择功能并且最终拼接输出而不是融合，融合会破坏不同手指的独立性
        # # 计算各专家的权重
        # gating_weights = self.gating_network(x.mean(dim=[2, 3]))  # (B, num_experts)

        # # 加权求和得到最终输出
        # output = torch.einsum('be,befhw->bfhw', gating_weights, expert_outputs)  # (B, F, H, W)
        return None