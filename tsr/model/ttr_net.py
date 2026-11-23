"""Set-based TTR-Net: effector keypoints + object tokens -> per-frame density label.

Status: stub for P4. Input is embodiment-agnostic (no joint dims).
"""

from __future__ import annotations

import torch
from torch import nn


class TTRNet(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 128, n_layers: int = 4, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=d_model * 4
        )
        self.temporal = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head_y = nn.Linear(d_model, 1)
        self.head_future = nn.Linear(d_model, 1)
        self.head_contact = nn.Linear(d_model, 1)

    def forward(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """tokens: (B, T, N, in_dim) mean-pooled over N, or (B, T, in_dim)."""
        x = self.input_proj(tokens)
        if x.ndim == 4:
            x = x.mean(dim=2)
        h = self.temporal(x)
        return {
            "y": self.head_y(h).squeeze(-1),
            "y_future": self.head_future(h).squeeze(-1),
            "contact_logit": self.head_contact(h).squeeze(-1),
        }
