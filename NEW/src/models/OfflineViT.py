import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vit_b_16, ViT_B_16_Weights

class OfflineViT(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        vit.conv_proj = nn.Conv2d(1, 768, kernel_size=16, stride=16)
        vit.heads = nn.Identity()

        self.vit = vit
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):

        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)

        x = self.vit(x)  # (B*T, 768)

        x = x.view(B, T, 768)

        # pooling temporal
        x = x.mean(dim=1)  # (B, 768)

        return self.fc(x)