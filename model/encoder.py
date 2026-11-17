import torch
from torch import nn

from contracts import D_EMBED, WINDOW

# hidden units between the 100-sample window and the 16-number embedding
ENCODER_HIDDEN = 64  

class ChannelEncoder(nn.Module):
    """(batch, channels, WINDOW) to (batch, channels, D_EMBED). One small MLP shared by every channel."""
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WINDOW, ENCODER_HIDDEN),
            nn.GELU(),
            nn.Linear(ENCODER_HIDDEN, D_EMBED)
        )

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return self.net(windows)