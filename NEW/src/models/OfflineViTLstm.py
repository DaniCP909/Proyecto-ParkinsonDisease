import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import vit_b_16, ViT_B_16_Weights


class OfflineViTLstm(nn.Module):

    def __init__(self, lstm_hidden=256, num_classes=2):
        super().__init__()

        vit = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # adaptar a grayscale
        vit.conv_proj = nn.Conv2d(
            1,
            768,
            kernel_size=16,
            stride=16
        )

        # quitar head
        vit.heads = nn.Identity()

        self.vit = vit

        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(lstm_hidden, num_classes)

    def forward(self, x):

        B, T, C, H, W = x.shape

        x = x.view(B*T, C, H, W)

        # ViT features
        x = self.vit(x)  # (B*T, 768)

        # reconstruir secuencia
        x = x.view(B, T, 768)

        out, (h_n, _) = self.lstm(x)

        return self.fc(h_n[-1])