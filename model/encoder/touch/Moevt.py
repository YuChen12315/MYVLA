import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet50, ResNet50_Weights


class Moevt(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        num_image = kwargs.get("num_image",20)
        embedding_dim = kwargs.get("embedding_dim",120)
        self.threshold = kwargs.get("threshold",0.1)
        self.num_experts = num_image
        self.embedding_dim = embedding_dim
        self.gate_network = GatingNetwork()
        # 同构专家网络
        self.expert = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.expert.fc = nn.Linear(self.expert.fc.in_features, embedding_dim)  #视觉特征投影到嵌入维度
        # 只冻结前面的卷积层，保持最后一层可训练
        for param in self.expert.parameters():
            param.requires_grad = False  # 先冻结所有
        
        # 解冻最后一层（新加的fc层默认requires_grad=True）
        for param in self.expert.fc.parameters():
            param.requires_grad = True
  
        
    def forward(self, touch, x):
        """
        Args:
            - touch: (B, num_experts) touch value inputs
            - x: (B, N, C, H, W)
        Returns:
            - touch_feats: (B, N, F)
            - touch_pos: (B, N, 3)
        """
        B,N,C,H,W = x.shape
        # 计算各专家的输出
        expert_outputs = []
        weights = self.gate_network(touch)  # (B, num_experts)
        weights_mask = weights > self.threshold  # 设置阈值，选择活跃专家
    
        # 对触觉图像处理进行稀疏激活
        for i in range(self.num_experts):
            expert_mask = weights_mask[:, i]  # (B, num_experts)
            expert_output = torch.zeros(B, 1, self.embedding_dim, device=x.device)  # 默认输出为零张量
            if expert_mask.any():
                # 获取需要当前专家的样本
                batch_indices = torch.nonzero(expert_mask, as_tuple=True)[0]
                x_selected = x[batch_indices]
                # 通过专家处理选中的样本
                expert_output = self.expert(x_selected)
                expert_output = F.normalize(expert_output, p=2, dim=1)  # L2归一化
                
            expert_outputs.append(expert_output)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, F)
        pos = torch.zeros(B,x.shape[1],3,device=x.device)  #TODO:# Dummy positional encodings
        return expert_outputs, pos

class GatingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 计算各专家权重（需经过softmax归一化）
        logits = x
        weights = torch.softmax(logits, dim=-1)
        return weights
    