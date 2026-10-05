"""ViPLink: modality-specific TempSpec encoders for video and contact PPG."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .tempspec import TempSpecEncoder


class ProjectionHead(nn.Module):
    """Two-layer MLP projection used only by the cross-modal alignment loss."""

    def __init__(self, embed_dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, proj_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class ViPLink(nn.Module):
    """Video encoder E_v and PPG encoder E_p share the architecture but not weights.

    Args:
        window_length: samples per window (window_sec * fs), 45 s x 30 Hz = 1350.
        with_projection: build projection heads (training only; discarded at inference).
    """

    VIDEO_CHANNELS = 3   # green, CHROM, red-green difference
    PPG_CHANNELS = 1

    def __init__(self, window_length=1350, embed_dim=256, proj_dim=128,
                 dropout=0.3, with_projection=False):
        super().__init__()
        self.window_length = window_length
        self.embed_dim = embed_dim
        self.video_encoder = TempSpecEncoder(self.VIDEO_CHANNELS, window_length, embed_dim,
                                             dropout=dropout)
        self.ppg_encoder = TempSpecEncoder(self.PPG_CHANNELS, window_length, embed_dim,
                                           dropout=dropout)
        if with_projection:
            self.video_proj = ProjectionHead(embed_dim, proj_dim)
            self.ppg_proj = ProjectionHead(embed_dim, proj_dim)

    def encode_video(self, video):
        """video: (B, 3, T) -> (B, embed_dim)"""
        return self.video_encoder(video)

    def encode_ppg(self, ppg):
        """ppg: (B, T) or (B, 1, T) -> (B, embed_dim)"""
        if ppg.dim() == 2:
            ppg = ppg.unsqueeze(1)
        return self.ppg_encoder(ppg)

    def forward(self, video, ppg):
        e_v, e_p = self.encode_video(video), self.encode_ppg(ppg)
        if not hasattr(self, 'video_proj'):
            return e_v, e_p
        return e_v, e_p, self.video_proj(e_v), self.ppg_proj(e_p)

    def encoder_state_dict(self):
        return {k: v for k, v in self.state_dict().items()
                if k.startswith(('video_encoder.', 'ppg_encoder.'))}

    @classmethod
    def from_pretrained(cls, path, map_location='cpu'):
        ckpt = torch.load(path, map_location=map_location)
        model = cls(window_length=ckpt['window_length'], embed_dim=ckpt['embed_dim'])
        model.load_state_dict(ckpt['state_dict'])
        return model.eval()
