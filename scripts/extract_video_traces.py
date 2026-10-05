"""Convert face-crop frame arrays into three-channel video traces.

    python scripts/extract_video_traces.py --input_dir faces/ --output_dir data/mcd_rppg/video_traces

Each input is a (T, H, W, 3) RGB .npy of per-frame face crops at 30 fps; the output
keeps the same file stem and stores a (3, T) float32 array.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viplink.data import extract_video_traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, '*.npy')))
    for i, path in enumerate(files, 1):
        stem = os.path.basename(path).split('.')[0]
        out = os.path.join(args.output_dir, f'{stem}.npy')
        if os.path.exists(out) and not args.overwrite:
            continue
        np.save(out, extract_video_traces(np.load(path)))
        if i % 50 == 0 or i == len(files):
            print(f'{i}/{len(files)}')


if __name__ == '__main__':
    main()
