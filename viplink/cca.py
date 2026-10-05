"""Global correlation-aware cross-modal alignment (Sec. 3.4)."""
import numpy as np

DEFAULT_COMPONENTS = (1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24)


def znorm_rows(s):
    return (s - s.mean(axis=1, keepdims=True)) / np.maximum(s.std(axis=1, keepdims=True), 1e-8)


def _l2(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)


class MultiScaleCCA:
    """Regularized CCA fitted once on training embeddings and evaluated at several
    numbers of components; per-scale similarity matrices are z-normalized and averaged."""

    def __init__(self, components=DEFAULT_COMPONENTS, reg=1e-4):
        self.components = tuple(components)
        self.reg = reg

    def fit(self, video_emb, ppg_emb):
        self.x_mean, self.y_mean = video_emb.mean(0), ppg_emb.mean(0)
        X, Y = video_emb - self.x_mean, ppg_emb - self.y_mean
        n = X.shape[0]
        cxx = X.T @ X / n + self.reg * np.eye(X.shape[1])
        cyy = Y.T @ Y / n + self.reg * np.eye(Y.shape[1])
        cxy = X.T @ Y / n
        kx, ky = self._inv_sqrt(cxx), self._inv_sqrt(cyy)
        u, _, vt = np.linalg.svd(kx @ cxy @ ky, full_matrices=False)
        self.wx, self.wy = kx @ u, ky @ vt.T
        return self

    @staticmethod
    def _inv_sqrt(c):
        lam, q = np.linalg.eigh(c)
        return q @ np.diag(1.0 / np.sqrt(np.maximum(lam, 1e-8))) @ q.T

    def similarity(self, video_emb, ppg_emb):
        """Fused (n_video, n_ppg) similarity matrix."""
        sims = []
        for nc in self.components:
            v = _l2((video_emb - self.x_mean) @ self.wx[:, :nc])
            p = _l2((ppg_emb - self.y_mean) @ self.wy[:, :nc])
            sims.append(znorm_rows(v @ p.T))
        return znorm_rows(np.mean(sims, axis=0))

    def save(self, path):
        np.savez(path, components=np.array(self.components), reg=self.reg,
                 x_mean=self.x_mean, y_mean=self.y_mean, wx=self.wx, wy=self.wy)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        obj = cls(tuple(int(c) for c in d['components']), float(d['reg']))
        obj.x_mean, obj.y_mean, obj.wx, obj.wy = d['x_mean'], d['y_mean'], d['wx'], d['wy']
        return obj
