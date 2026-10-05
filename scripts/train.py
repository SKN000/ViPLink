"""Train ViPLink on one MCD-rPPG fold.

    python scripts/train.py --config configs/mcd_rppg.yaml [--fold 0]
"""
import argparse
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viplink.models import ViPLink
from viplink.losses import ViPLinkLoss
from viplink.data import (MCDTrainDataset, MCDWindowDataset, PairedSessionSampler,
                          list_subjects, make_folds, split_validation)
from viplink.metrics import verification_metrics, split_scores


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate(batch):
    video, ppg, labels, sessions = zip(*batch)
    return torch.stack(video), torch.stack(ppg), torch.tensor(labels), list(sessions)


@torch.no_grad()
def validate(model, loader, device):
    """Subject-level cross-modal EER on the projection space."""
    model.eval()
    zv, zp = defaultdict(list), defaultdict(list)
    for video, ppg, sids in loader:
        _, _, v, p = model(video.to(device), ppg.to(device))
        for sid, a, b in zip(sids, v.cpu(), p.cpu()):
            zv[sid].append(a)
            zp[sid].append(b)
    sids = sorted(zv)
    V = torch.stack([F.normalize(torch.stack(zv[s]).mean(0), dim=0) for s in sids]).numpy()
    P = torch.stack([F.normalize(torch.stack(zp[s]).mean(0), dim=0) for s in sids]).numpy()
    return verification_metrics(*split_scores(V @ P.T, sids, sids))['eer']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/mcd_rppg.yaml')
    ap.add_argument('--fold', type=int, default=None)
    ap.add_argument('--output_dir', default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    d, m, l, t = cfg['data'], cfg['model'], cfg['loss'], cfg['train']
    fold = d['fold'] if args.fold is None else args.fold
    out_dir = args.output_dir or t['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    set_seed(t['seed'] + fold)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    subjects = list_subjects(d['video_dir'], d['ppg_dir'], d['camera'])
    fold_train, fold_test = make_folds(subjects, d['n_folds'], d['split_seed'])[fold]
    train_subjects, val_subjects = split_validation(fold_train, d['val_fraction'], d['split_seed'])
    print(f'fold {fold}/{d["n_folds"]}: train={len(train_subjects)} val={len(val_subjects)} '
          f'test={len(fold_test)} (test subjects are not used during training)')

    common = dict(video_dir=d['video_dir'], ppg_dir=d['ppg_dir'], camera=d['camera'],
                  fs=d['fs'], window_sec=d['window_sec'], stride_sec=d['train_stride_sec'])
    train_set = MCDTrainDataset(subjects=train_subjects, augment=True, **common)
    val_set = MCDWindowDataset(subjects=val_subjects, session='before', **common)
    train_loader = DataLoader(train_set, batch_sampler=PairedSessionSampler(train_set, t['batch_size']),
                              num_workers=t['num_workers'], collate_fn=collate, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=t['num_workers'],
                            collate_fn=lambda b: (torch.stack([x[0] for x in b]),
                                                  torch.stack([x[1] for x in b]),
                                                  [x[2] for x in b]))

    model = ViPLink(window_length=train_set.win, embed_dim=m['embed_dim'], proj_dim=m['proj_dim'],
                    dropout=m['dropout'], with_projection=True).to(device)
    criterion = ViPLinkLoss(train_set.n_classes, m['embed_dim'], l['weights'], l['arc_scale'],
                            l['arc_margin'], l['center_momentum']).to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params, lr=t['lr'], weight_decay=t['weight_decay'])

    def lr_lambda(ep):
        if ep < t['warmup_epochs']:
            return ep / t['warmup_epochs']
        return 0.5 * (1 + math.cos(math.pi * (ep - t['warmup_epochs']) / (t['epochs'] - t['warmup_epochs'])))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_eer, stale = float('inf'), 0
    for epoch in range(1, t['epochs'] + 1):
        model.train()
        criterion.train()
        train_set.set_async_ratio(epoch / t['async_warmup_epochs'])
        temperature = l['temperature_start'] - (l['temperature_start'] - l['temperature_end']) * \
            min(epoch / l['temperature_anneal_epochs'], 1.0)

        running, n = defaultdict(float), 0
        for video, ppg, labels, _ in train_loader:
            video, ppg, labels = video.to(device), ppg.to(device), labels.to(device)
            e_v, e_p, z_v, z_p = model(video, ppg)
            loss, terms = criterion(e_v, e_p, z_v, z_p, labels, temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, t['grad_clip'])
            optimizer.step()
            running['total'] += loss.item()
            for k, v in terms.items():
                running[k] += v
            n += 1
        scheduler.step()
        log = ' '.join(f'{k}={v / max(n, 1):.4f}' for k, v in running.items())

        if epoch % t['eval_every'] == 0 or epoch <= 3:
            val_eer = validate(model, val_loader, device)
            print(f'epoch {epoch:3d}  {log}  tau={temperature:.3f}  val_eer={val_eer * 100:.2f}%', flush=True)
            if val_eer < best_eer:
                best_eer, stale = val_eer, 0
                torch.save({'epoch': epoch, 'model': model.state_dict(), 'criterion': criterion.state_dict(),
                            'config': cfg, 'fold': fold, 'val_eer': val_eer},
                           os.path.join(out_dir, 'checkpoint.pt'))
                torch.save({'state_dict': model.encoder_state_dict(), 'window_length': train_set.win,
                            'embed_dim': m['embed_dim'], 'fs': d['fs']},
                           os.path.join(out_dir, 'viplink.pt'))
            else:
                stale += 1
            if stale >= t['patience']:
                print(f'early stopping at epoch {epoch}')
                break
        else:
            print(f'epoch {epoch:3d}  {log}  tau={temperature:.3f}', flush=True)

    print(f'best val EER {best_eer * 100:.2f}%  ->  {out_dir}/viplink.pt')


if __name__ == '__main__':
    main()
