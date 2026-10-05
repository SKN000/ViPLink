"""Verification metrics: EER, AUC and TAR@FAR."""
import numpy as np


def verification_metrics(genuine, impostor):
    """Return dict(eer, auc, tar_far1, tar_far10) from genuine/impostor score arrays."""
    genuine, impostor = np.asarray(genuine), np.asarray(impostor)
    g, i = np.sort(genuine), np.sort(impostor)
    thr = np.sort(np.concatenate([genuine, impostor]))
    fnr = np.searchsorted(g, thr, 'left') / len(g)
    fpr = 1.0 - np.searchsorted(i, thr, 'left') / len(i)
    k = np.argmin(np.abs(fnr - fpr))
    order = np.argsort(fpr)
    trapezoid = getattr(np, 'trapezoid', None) or np.trapz
    auc = float(trapezoid((1.0 - fnr)[order], fpr[order]))
    return {
        'eer': float((fnr[k] + fpr[k]) / 2),
        'auc': auc,
        'tar_far1': float(np.mean(genuine >= np.percentile(impostor, 99))),
        'tar_far10': float(np.mean(genuine >= np.percentile(impostor, 90))),
    }


def split_scores(scores, row_ids, col_ids):
    """Split a score matrix into genuine and impostor scores by identity labels."""
    same = np.asarray(row_ids)[:, None] == np.asarray(col_ids)[None, :]
    return scores[same].ravel(), scores[~same].ravel()
