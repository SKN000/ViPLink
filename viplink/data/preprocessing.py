"""Modality-specific preprocessing (Sec. 3.2)."""
import numpy as np


def extract_video_traces(face_frames):
    """Build the three-channel video trace from face crops.

    Args:
        face_frames: (T, H, W, 3) RGB face-crop frames.
    Returns:
        (3, T) float32 array of [green, CHROM, red-green difference].
    """
    T, H, W, _ = face_frames.shape
    center = face_frames[:, H // 4:3 * H // 4, W // 4:3 * W // 4, :].astype(np.float64)
    rgb = center.mean(axis=(1, 2))                       # (T, 3)
    for c in range(3):
        col = rgb[:, c]
        fill = np.nanmean(col) if not np.all(np.isnan(col)) else 0.0
        rgb[:, c] = np.nan_to_num(col, nan=fill)

    r, g, b = (rgb / (rgb.mean(axis=0) + 1e-10)).T     # mean-normalized traces
    x = 3 * r - 2 * g
    y = 1.5 * r + g - 1.5 * b
    chrom = x - (np.std(x) / (np.std(y) + 1e-10)) * y
    rg = r - g
    return np.stack([g, chrom, rg], axis=0).astype(np.float32)


def normalize_video_window(window):
    """Per-channel z-normalization of a (3, T) window."""
    w = window.astype(np.float32).copy()
    w = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + 1e-10)
    return np.nan_to_num(w, nan=0.0)


def normalize_ppg_window(window):
    """Z-normalization of a raw, unfiltered (T,) cPPG window."""
    w = window.astype(np.float32)
    return (w - w.mean()) / (w.std() + 1e-6)
