"""Score facial-video recordings against contact-PPG recordings.

    python scripts/inference.py --weights weights/viplink_mcd.pt \
        --video a_traces.npy b_traces.npy --ppg x.txt y.txt [--out scores.csv]

Video inputs are (3, T) traces from scripts/extract_video_traces.py; PPG inputs are raw
cPPG (first column) resampled to the same rate. Each recording must be >= 45 s.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viplink import ViPLink, MultiScaleCCA, encode_video, encode_ppg, cosine_similarity_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default='weights/viplink_mcd.pt')
    ap.add_argument('--video', nargs='+', required=True)
    ap.add_argument('--ppg', nargs='+', required=True)
    ap.add_argument('--cca', default=None, help='CCA alignment from scripts/fit_cca.py')
    ap.add_argument('--stride_sec', type=float, default=10.0)
    ap.add_argument('--out', default=None, help='optional CSV for the similarity matrix')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.weights, map_location='cpu')
    model = ViPLink.from_pretrained(args.weights).to(device)
    fs, window_sec = ckpt['fs'], ckpt['window_length'] / ckpt['fs']

    v = np.stack([encode_video(model, np.load(p), fs, window_sec, args.stride_sec, device)
                  for p in args.video])
    p = np.stack([encode_ppg(model, np.loadtxt(f, ndmin=2)[:, 0], fs, window_sec, args.stride_sec, device)
                  for f in args.ppg])
    if args.cca:
        scores = MultiScaleCCA.load(args.cca).similarity(v, p)
    else:
        scores = cosine_similarity_matrix(v, p)

    names_v = [os.path.basename(x) for x in args.video]
    names_p = [os.path.basename(x) for x in args.ppg]
    width = max(len(n) for n in names_v)
    print(' ' * width + '  ' + '  '.join(f'{n:>12.12}' for n in names_p))
    for name, row in zip(names_v, scores):
        print(f'{name:<{width}}  ' + '  '.join(f'{s:12.4f}' for s in row))

    if args.out:
        with open(args.out, 'w') as f:
            f.write('video,' + ','.join(names_p) + '\n')
            for name, row in zip(names_v, scores):
                f.write(name + ',' + ','.join(f'{s:.6f}' for s in row) + '\n')


if __name__ == '__main__':
    main()
