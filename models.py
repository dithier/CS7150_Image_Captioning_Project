import torch
import torch.nn as nn
from torch import Tensor

"""
Pretrained RESNET encoder
LSTM decoder
"""
class BaselineModel(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

"""
Other possibilities:
1) Prertrained ResNet with LSTM with attention
2) Pretrained ResNet with Transformer
3) custom CNN with the best of the above decoders
"""