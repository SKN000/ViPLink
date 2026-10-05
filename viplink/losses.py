"""The five training objectives of ViPLink (Sec. 3.3.2)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_modal_alignment_loss(z_v, z_p, labels, temperature):
    """Symmetric supervised contrastive loss on L2-normalized projections (Eq. 3).
    Positives for each anchor are all samples of the other modality with the same identity."""
    sim = z_v @ z_p.T / temperature
    pos = (labels[:, None] == labels[None, :]).float()

    def one_direction(s, mask):
        n_pos = mask.sum(1).clamp(min=1)
        return (torch.logsumexp(s, dim=1) - (s * mask).sum(1) / n_pos).mean()

    return 0.5 * (one_direction(sim, pos) + one_direction(sim.T, pos.T))


class ArcFaceHead(nn.Module):
    """Additive angular margin classifier (Eq. 4)."""

    def __init__(self, embed_dim, n_classes, scale=30.0, margin=0.3):
        super().__init__()
        self.scale, self.margin = scale, margin
        self.weight = nn.Parameter(torch.empty(n_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, labels):
        cos = F.normalize(x, dim=1) @ F.normalize(self.weight, dim=1).T
        if self.training:
            one_hot = F.one_hot(labels, cos.shape[1]).float()
            theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
            cos = cos * (1 - one_hot) + torch.cos(theta + self.margin * one_hot) * one_hot
        return F.cross_entropy(cos * self.scale, labels)


class CenterLoss(nn.Module):
    """Intra-class compactness with EMA-updated class centers (Eq. 5)."""

    def __init__(self, n_classes, embed_dim, momentum=0.01):
        super().__init__()
        self.momentum = momentum
        self.register_buffer('centers', torch.zeros(n_classes, embed_dim))

    def forward(self, emb, labels):
        loss = 0.5 * ((emb - self.centers[labels]) ** 2).sum(1).mean()
        with torch.no_grad():
            for e, y in zip(emb.detach(), labels.tolist()):
                self.centers[y].mul_(1 - self.momentum).add_(self.momentum * e)
        return loss


def modality_bridge_loss(e_v, e_p):
    """Cosine distance between paired video and PPG embeddings (Eq. 6)."""
    return (1 - F.cosine_similarity(e_v, e_p, dim=-1)).mean()


def decorrelation_loss(emb):
    """Mean squared off-diagonal Pearson correlation of embedding dimensions (Eq. 7)."""
    x = emb - emb.mean(0, keepdim=True)
    cov = x.T @ x / (emb.shape[0] - 1)
    std = torch.sqrt(torch.diag(cov) + 1e-8)
    corr = cov / (std[:, None] * std[None, :] + 1e-8)
    off = ~torch.eye(corr.shape[0], dtype=torch.bool, device=corr.device)
    return (corr[off] ** 2).mean()


class ViPLinkLoss(nn.Module):
    """L = w_align L_align + w_arc L_arc + w_compact L_compact + w_bridge L_bridge + w_reg L_reg (Eq. 8)."""

    def __init__(self, n_classes, embed_dim=256, weights=None,
                 arc_scale=30.0, arc_margin=0.3, center_momentum=0.01):
        super().__init__()
        self.weights = weights or dict(align=0.40, arc=0.30, compact=0.15, bridge=0.10, reg=0.05)
        self.video_arc = ArcFaceHead(embed_dim, n_classes, arc_scale, arc_margin)
        self.ppg_arc = ArcFaceHead(embed_dim, n_classes, arc_scale, arc_margin)
        self.video_center = CenterLoss(n_classes, embed_dim, center_momentum)
        self.ppg_center = CenterLoss(n_classes, embed_dim, center_momentum)

    def forward(self, e_v, e_p, z_v, z_p, labels, temperature):
        terms = {
            'align': cross_modal_alignment_loss(z_v, z_p, labels, temperature),
            'arc': 0.5 * (self.video_arc(e_v, labels) + self.ppg_arc(e_p, labels)),
            'compact': 0.5 * (self.video_center(e_v, labels) + self.ppg_center(e_p, labels)),
            'bridge': modality_bridge_loss(e_v, e_p),
            'reg': 0.5 * (decorrelation_loss(e_v) + decorrelation_loss(e_p)),
        }
        total = sum(self.weights[k] * v for k, v in terms.items())
        return total, {k: v.item() for k, v in terms.items()}
