"""Fit the multi-scale CCA alignment on training-subject embeddings.

    python scripts/fit_cca.py --config configs/mcd_rppg.yaml \
        --weights weights/viplink_mcd.pt --out weights/viplink_mcd_cca.npz
"""
import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viplink import ViPLink, MultiScaleCCA
from viplink.data import RecordingCache, index_recordings, list_subjects, make_folds
from viplink.inference import encode_video, encode_ppg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mcd_rppg.yaml')
    ap.add_argument('--weights', default='weights/viplink_mcd.pt')
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--out', default='weights/viplink_mcd_cca.npz')
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d, e = cfg['data'], cfg['eval']
    fold = d['fold'] if args.fold is None else args.fold
    all_subjects = list_subjects(d['video_dir'], d['ppg_dir'], d['camera'])
    subjects = make_folds(all_subjects, d['n_folds'], d['split_seed'])[fold][0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ViPLink.from_pretrained(args.weights).to(device)
    window_sec = model.window_length / d['fs']

    cache = RecordingCache()
    video_emb, ppg_emb = [], []
    recordings = index_recordings(d['video_dir'], d['ppg_dir'], d['camera'], subjects)
    for i, sid in enumerate(sorted(recordings), 1):
        for v_path, p_path in recordings[sid].values():
            video, ppg = cache.get(v_path, p_path)
            v = encode_video(model, video, d['fs'], window_sec, e['cca_fit_stride_sec'], device, reduce=False)
            p = encode_ppg(model, ppg, d['fs'], window_sec, e['cca_fit_stride_sec'], device, reduce=False)
            video_emb.append(v)
            ppg_emb.append(p)
        if i % 50 == 0:
            print(f'{i}/{len(recordings)} subjects')

    video_emb, ppg_emb = np.concatenate(video_emb), np.concatenate(ppg_emb)
    cca = MultiScaleCCA(e['cca_components']).fit(video_emb, ppg_emb)
    cca.save(args.out)
    print(f'CCA fitted on {len(video_emb)} windows from {len(recordings)} subjects -> {args.out}')


if __name__ == '__main__':
    main()
