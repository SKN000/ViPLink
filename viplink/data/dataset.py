"""MCD-rPPG window datasets.

Expected layout (both resampled to the same frame rate and synchronized):
    <video_dir>/<sid>_<camera>_<session>.npy   (3, T) traces from extract_video_traces
    <ppg_dir>/<sid>_<camera>_<session>.txt     raw cPPG in the first column
where session is 'before' (rest) or 'after' (post-exercise).
"""
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .preprocessing import normalize_video_window, normalize_ppg_window

SESSIONS = ('before', 'after')


def index_recordings(video_dir, ppg_dir, camera, subjects):
    """Return {sid: {session: (video_path, ppg_path)}} for recordings with both modalities."""
    recordings = defaultdict(dict)
    for sid in subjects:
        for session in SESSIONS:
            v = os.path.join(video_dir, f'{sid}_{camera}_{session}.npy')
            p = os.path.join(ppg_dir, f'{sid}_{camera}_{session}.txt')
            if os.path.exists(v) and os.path.exists(p):
                recordings[sid][session] = (v, p)
    return recordings


def list_subjects(video_dir, ppg_dir, camera):
    """Sorted IDs of subjects with at least one recording that has both modalities."""
    sids = {os.path.basename(f).split('_')[0] for f in os.listdir(video_dir) if f.endswith('.npy')}
    return sorted(index_recordings(video_dir, ppg_dir, camera, sids))


def make_folds(subjects, n_folds=4, seed=42):
    """Subject-independent k-fold split: shuffle once with `seed`, cut into equal folds.

    Returns a list of (train_subjects, test_subjects) tuples, one per fold."""
    shuffled = sorted(subjects)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    bounds = [i * n // n_folds for i in range(n_folds + 1)]
    folds = [sorted(shuffled[bounds[i]:bounds[i + 1]]) for i in range(n_folds)]
    return [(sorted(s for j, f in enumerate(folds) if j != i for s in f), folds[i])
            for i in range(n_folds)]


def split_validation(subjects, fraction, seed=42):
    """Hold out a random `fraction` of training subjects for model selection."""
    shuffled = sorted(subjects)
    random.Random(seed).shuffle(shuffled)
    n_val = int(round(len(shuffled) * fraction))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


class RecordingCache:
    """Lazily loads video traces and PPG signals, truncated to their common length."""

    def __init__(self):
        self._cache = {}

    def get(self, video_path, ppg_path):
        key = (video_path, ppg_path)
        if key not in self._cache:
            video = np.load(video_path).astype(np.float32)
            ppg = np.loadtxt(ppg_path, ndmin=2)[:, 0].astype(np.float32)
            n = min(video.shape[1], len(ppg))
            self._cache[key] = (video[:, :n], ppg[:n])
        return self._cache[key]


class MCDTrainDataset(Dataset):
    """Sliding-window training pairs with asynchronous sampling and augmentation.

    With probability `async_ratio`, the PPG window is drawn from a random offset of a
    random session of the same subject instead of the co-located window. The ratio is
    annealed from 0 to 1 by the training script (curriculum)."""

    def __init__(self, video_dir, ppg_dir, subjects, camera='FullHDwebcam', fs=30,
                 window_sec=45.0, stride_sec=22.5, augment=True):
        self.win = int(window_sec * fs)
        self.stride = int(stride_sec * fs)
        self.augment = augment
        self.async_ratio = 0.0
        self.cache = RecordingCache()

        recordings = index_recordings(video_dir, ppg_dir, camera, subjects)
        self.subjects = sorted(recordings)
        self.label_of = {sid: i for i, sid in enumerate(self.subjects)}
        self.n_classes = len(self.subjects)

        self.sessions_of = defaultdict(list)
        self.segments = []
        for sid in self.subjects:
            for session, (v, p) in recordings[sid].items():
                video, _ = self.cache.get(v, p)
                T = video.shape[1]
                if T < self.win:
                    continue
                self.sessions_of[sid].append((v, p, T))
                for t0 in range(0, T - self.win + 1, self.stride):
                    self.segments.append((sid, session, v, p, t0))

    def set_async_ratio(self, ratio):
        self.async_ratio = float(np.clip(ratio, 0.0, 1.0))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        sid, session, v, p, t0 = self.segments[idx]
        video, ppg_full = self.cache.get(v, p)
        x_v = normalize_video_window(video[:, t0:t0 + self.win])
        if self.augment:
            x_v = self._augment_video(x_v)

        if random.random() < self.async_ratio:
            v2, p2, T2 = random.choice(self.sessions_of[sid])
            t1 = random.randint(0, max(T2 - self.win, 0))
            x_p = self.cache.get(v2, p2)[1][t1:t1 + self.win].copy()
        else:
            x_p = ppg_full[t0:t0 + self.win].copy()
        if self.augment:
            x_p = self._augment_ppg(x_p)
        x_p = normalize_ppg_window(x_p)

        return (torch.from_numpy(x_v), torch.from_numpy(x_p),
                self.label_of[sid], session)

    def _augment_video(self, x):
        if random.random() < 0.5:                      # amplitude compression
            m = x.mean(axis=1, keepdims=True)
            x = m + (x - m) * random.uniform(0.05, 0.5)
        if random.random() < 0.3:
            x = np.roll(x, random.randint(-15, 15), axis=1)
        if random.random() < 0.5:
            x = x * np.random.uniform(0.8, 1.2, size=(x.shape[0], 1)).astype(np.float32)
        if random.random() < 0.5:
            x = x + np.random.randn(*x.shape).astype(np.float32) * 0.03
        if random.random() < 0.3:                      # random crop, zero-padded
            n = x.shape[1]
            length = int(n * random.uniform(0.7, 1.0))
            start = random.randint(0, n - length)
            out = np.zeros_like(x)
            out[:, :length] = x[:, start:start + length]
            x = out
        return x.astype(np.float32)

    def _augment_ppg(self, x):
        if random.random() < 0.5:
            x = x * np.random.uniform(0.8, 1.2)
        if random.random() < 0.5:
            x = x + np.random.randn(len(x)).astype(np.float32) * 0.05
        if random.random() < 0.3:                      # time warping
            n = len(x)
            idx = np.clip(np.linspace(0, n - 1, int(n * random.uniform(0.9, 1.1))), 0, n - 1).astype(int)
            w = x[idx]
            x = w[:n] if len(w) >= n else np.pad(w, (0, n - len(w)), mode='edge')
        if random.random() < 0.3:                      # baseline wander
            t = np.linspace(0, 2 * np.pi * random.uniform(0.5, 3), len(x))
            x = x + 0.1 * np.sin(t).astype(np.float32)
        if random.random() < 0.3:                      # dynamic-range clipping
            x = np.clip(x, np.percentile(x, 5), np.percentile(x, 95))
        return x.astype(np.float32)


class PairedSessionSampler(Sampler):
    """Each batch holds batch_size/2 subjects, with one rest and one post-exercise window each."""

    def __init__(self, dataset, batch_size):
        self.per_batch = batch_size // 2
        index = defaultdict(lambda: defaultdict(list))
        for i, (sid, session, *_rest) in enumerate(dataset.segments):
            index[sid][session].append(i)
        self.index = index
        self.subjects = [s for s in index if all(index[s][x] for x in SESSIONS)]
        self.n_batches = len(dataset) // batch_size

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            for sid in random.sample(self.subjects, min(self.per_batch, len(self.subjects))):
                batch += [random.choice(self.index[sid]['before']),
                          random.choice(self.index[sid]['after'])]
            yield batch

    def __len__(self):
        return self.n_batches


class MCDWindowDataset(Dataset):
    """Synchronous, non-augmented windows for validation."""

    def __init__(self, video_dir, ppg_dir, subjects, camera='FullHDwebcam', fs=30,
                 window_sec=45.0, stride_sec=22.5, session='before'):
        self.win = int(window_sec * fs)
        stride = int(stride_sec * fs)
        self.cache = RecordingCache()
        self.items = []
        for sid, sess in sorted(index_recordings(video_dir, ppg_dir, camera, subjects).items()):
            if session not in sess:
                continue
            v, p = sess[session]
            T = self.cache.get(v, p)[0].shape[1]
            for t0 in range(0, T - self.win + 1, stride):
                self.items.append((sid, v, p, t0))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sid, v, p, t0 = self.items[idx]
        video, ppg = self.cache.get(v, p)
        return (torch.from_numpy(normalize_video_window(video[:, t0:t0 + self.win])),
                torch.from_numpy(normalize_ppg_window(ppg[t0:t0 + self.win])),
                sid)
