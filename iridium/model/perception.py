"""Small trainable modality-specific residual encoders inside the shared model."""
from torch import nn


class PerceptualEncoder(nn.Module):
    def __init__(self, projection, width, rank, depth):
        super().__init__()
        self.projection = projection
        self.blocks = nn.ModuleList(nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, rank), nn.GELU(), nn.Linear(rank, width)
        ) for _ in range(depth))
        self.scale = (2 * depth) ** -.5

    def forward(self, payload):
        hidden = self.projection(payload)
        for block in self.blocks:
            hidden = hidden + self.scale * block(hidden)
        return hidden
