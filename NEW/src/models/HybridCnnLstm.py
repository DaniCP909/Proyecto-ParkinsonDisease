import torch
import torch.nn as nn

class HybridPatchLSTM(nn.Module):
    """
    Modelo híbrido: por-patch CNN (compartida) -> proyección -> LSTM sobre la secuencia de patches -> FC
    Input: x of shape (B, T, C, H, W)
    """
    def __init__(self,
                 in_channels=4,       # e.g. 1(trazo) + 3 atributos
                 patch_size=(32,32),  # H, W of each patch
                 cnn_feature_dim=64,  # final per-patch feature dim
                 lstm_hidden=128,
                 lstm_layers=2,
                 num_classes=2):
        super().__init__()

        H, W = patch_size
        # Simple CNN: ajusta filtros/kernels a tu necesidad
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),   # reduce espacialmente
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),                              # resultado (64,4,4)
        )

        self.cnn_proj = nn.Linear(64 * 4 * 4, cnn_feature_dim)
        self.feature_dim = cnn_feature_dim

        self.lstm = nn.LSTM(input_size=cnn_feature_dim,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=0.0,
                            bidirectional=False)

        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)           # (B*T, C, H, W)
        x = self.cnn(x)                      # (B*T, 64, 4, 4)
        x = x.view(B * T, -1)                # (B*T, 64*4*4)
        x = self.cnn_proj(x)                 # (B*T, feature_dim)
        x = x.view(B, T, -1)                 # (B, T, feature_dim)
        outputs, (h_n, c_n) = self.lstm(x)   # outputs: (B,T,hidden)
        last_hidden = h_n[-1]                # (B, hidden)
        logits = self.fc(last_hidden)        # (B, num_classes)
        return logits