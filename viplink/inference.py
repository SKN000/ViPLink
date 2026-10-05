"""Window-level encoding of whole recordings."""
import numpy as np
import torch

from .data.preprocessing import normalize_video_window, normalize_ppg_window


def sliding_windows(signal, window, stride):
    """Yield windows along the last axis."""
    T = signal.shape[-1]
    for t0 in range(0, T - window + 1, stride):
        yield signal[..., t0:t0 + window]


@torch.no_grad()
def encode_windows(model, windows, modality, device='cpu', batch_size=256):
    """Encode pre-normalized windows. modality: 'video' -> (N, 3, T), 'ppg' -> (N, T)."""
    encode = model.encode_video if modality == 'video' else model.encode_ppg
    out = []
    for i in range(0, len(windows), batch_size):
        x = torch.as_tensor(np.stack(windows[i:i + batch_size]), dtype=torch.float32, device=device)
        out.append(encode(x).cpu().numpy())
    return np.concatenate(out) if out else np.empty((0, model.embed_dim), np.float32)


def encode_video(model, traces, fs=30, window_sec=45.0, stride_sec=10.0, device='cpu',
                 reduce=True):
    """(3, T) video traces -> mean embedding (or per-window embeddings if reduce=False)."""
    win, stride = int(window_sec * fs), int(stride_sec * fs)
    windows = [normalize_video_window(w) for w in sliding_windows(traces, win, stride)]
    emb = encode_windows(model, windows, 'video', device)
    return emb.mean(0) if reduce and len(emb) else emb


def encode_ppg(model, ppg, fs=30, window_sec=45.0, stride_sec=10.0, device='cpu',
               reduce=True):
    """(T,) raw cPPG resampled to `fs` -> mean embedding (or per-window embeddings)."""
    win, stride = int(window_sec * fs), int(stride_sec * fs)
    windows = [normalize_ppg_window(w) for w in sliding_windows(ppg, win, stride)]
    emb = encode_windows(model, windows, 'ppg', device)
    return emb.mean(0) if reduce and len(emb) else emb


def cosine_similarity_matrix(video_emb, ppg_emb):
    v = video_emb / (np.linalg.norm(video_emb, axis=-1, keepdims=True) + 1e-8)
    p = ppg_emb / (np.linalg.norm(ppg_emb, axis=-1, keepdims=True) + 1e-8)
    return v @ p.T
