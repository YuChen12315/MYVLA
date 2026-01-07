import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()