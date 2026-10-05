"""TempSpec Encoder: dual-branch (temporal + spectral) identity encoder."""
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """h = GELU(x + IN(Conv(GELU(IN(Conv(x))))))."""

    def __init__(self, channels, kernel_size=7, dropout=0.3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.InstanceNorm1d(channels, affine=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.InstanceNorm1d(channels, affine=True),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class TemporalBranch(nn.Module):
    """Three Conv-IN-GELU-AvgPool stages (64/128/256 channels), each followed by
    two residual blocks, then global average pooling and a linear projection."""

    def __init__(self, in_channels, embed_dim=256, channels=(64, 128, 256),
                 kernel_size=7, dropout=0.3):
        super().__init__()
        layers, prev = [], in_channels
        for ch in channels:
            layers += [
                nn.Sequential(
                    nn.Conv1d(prev, ch, kernel_size, padding=kernel_size // 2),
                    nn.InstanceNorm1d(ch, affine=True),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.AvgPool1d(2),
                ),
                ResidualBlock(ch, kernel_size, dropout),
                ResidualBlock(ch, kernel_size, dropout),
            ]
            prev = ch
        self.layers = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(prev, embed_dim)

    def forward(self, x):
        return self.fc(self.pool(self.layers(x)).squeeze(-1))


class SpectralBranch(nn.Module):
    """Channel-averaged log-magnitude spectrum -> two-layer LN-MLP (Eq. 1)."""

    def __init__(self, window_length, embed_dim=256):
        super().__init__()
        spec_dim = window_length // 2 + 1
        self.mlp = nn.Sequential(
            nn.Linear(spec_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        spec = torch.log1p(torch.fft.rfft(x, dim=-1).abs()).mean(dim=1)
        return self.mlp(spec)


class TempSpecEncoder(nn.Module):
    """Fuses temporal feature t and spectral feature s: e = LN(W2 GELU(W1 [t; s])) (Eq. 2)."""

    def __init__(self, in_channels, window_length, embed_dim=256,
                 channels=(64, 128, 256), kernel_size=7, dropout=0.3):
        super().__init__()
        self.temporal = TemporalBranch(in_channels, embed_dim, channels, kernel_size, dropout)
        self.spectral = SpectralBranch(window_length, embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        return self.fusion(torch.cat([self.temporal(x), self.spectral(x)], dim=-1))
