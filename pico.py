"""PiCO / PiCO+ state for one fold: candidate sets, pseudo-targets, prototypes,
FIFO key queue and the per-bag clean/noisy split. One copy per OSATS criterion."""
import torch
import torch.nn.functional as F


def soft_ce(logits, target):
    if logits.numel() == 0:
        return logits.sum() * 0.0
    return -(target * F.log_softmax(logits, 1)).sum(1).mean()


def smoothed_nll(p, y, eps=0.1):
    """Cross-entropy on a probability vector with label smoothing."""
    logp = p.clamp_min(1e-8).log()
    return -(1 - eps) * logp[y] - eps / p.numel() * logp.sum()


def restricted_argmax(logits, cand):
    return logits.masked_fill(~cand, float("-inf")).argmax(1)


class Queue:
    def __init__(self, size, dim, device):
        self.feats = torch.zeros(size, dim, device=device)
        self.labels = torch.full((size,), -1, dtype=torch.long, device=device)
        self.clean = torch.ones(size, dtype=torch.bool, device=device)
        self.video = torch.full((size,), -1, dtype=torch.long, device=device)
        self.ptr, self.n, self.size = 0, 0, size

    def push(self, k, y, clean, video):
        i = (torch.arange(k.size(0), device=k.device) + self.ptr) % self.size
        self.feats[i], self.labels[i], self.clean[i], self.video[i] = k, y, clean, video
        self.ptr = (self.ptr + k.size(0)) % self.size
        self.n = min(self.size, self.n + k.size(0))

    def get(self):
        return self.feats[:self.n], self.labels[:self.n], self.clean[:self.n], self.video[:self.n]


def contrastive(q, k, y, clean, video, queue, tau, anchors, knn=0):
    """MoCo/SupCon loss over the pool A_m = (V_q u V_k u Q) minus the anchor.

    Positives: keys sharing the anchor's label (only clean keys when `clean`
    is given), or, with knn > 0, the knn nearest keys from other videos."""
    if anchors.numel() == 0:
        return q.sum() * 0.0
    qf, ql, qc, qv = queue.get()
    M = q.size(0)
    keys = torch.cat([q, k, qf])
    labels = torch.cat([y, y, ql])
    sim = q[anchors] @ keys.t() / tau
    A = torch.arange(anchors.numel(), device=q.device)
    self_mask = torch.zeros_like(sim, dtype=torch.bool)
    self_mask[A, anchors] = True
    logp = sim - torch.logsumexp(sim.masked_fill(self_mask, float("-inf")), 1, True)
    if knn > 0:
        banned = self_mask.clone()
        banned[:, :2 * M] = True                         # same video
        banned[:, 2 * M:] |= (qv == video)[None]
        score = sim.detach().masked_fill(banned, float("-inf"))
        val, idx = score.topk(min(knn, score.size(1)), dim=1)
        w = torch.zeros_like(sim).scatter_(1, idx, torch.isfinite(val).float())
    else:
        w = (labels[None] == y[anchors][:, None]) & ~self_mask
        if clean is not None:                            # clean keys only
            w = w & torch.cat([clean, clean, qc])[None]
        w = w.float()
    has = w.sum(1) > 0
    if not has.any():
        return q.sum() * 0.0
    return (-(logp * w).sum(1) / w.sum(1).clamp_min(1))[has].mean()


class PiCO:
    def __init__(self, ds, cfg, n_criteria, K, dim, device):
        self.cfg, self.E, self.K = cfg, n_criteria, K
        cand = torch.as_tensor(ds.clip_table("cand"), device=device)     # [N, E, K]
        s0 = torch.as_tensor(ds.clip_table("s0"), device=device).float()
        self.cand = cand.permute(1, 0, 2).bool()                         # [E, N, K]
        self.s = s0.permute(1, 0, 2).contiguous()
        self.video = torch.as_tensor(ds.clip_video(), device=device)
        N = self.video.numel()
        self.protos = torch.zeros(n_criteria, K, dim, device=device)
        self.queues = [Queue(cfg.queue, dim, device) for _ in range(n_criteria)]
        self.clean = torch.ones(n_criteria, N, dtype=torch.bool, device=device)
        self.bank_q = torch.zeros(n_criteria, N, dim, device=device)
        self.bank_y = torch.zeros(n_criteria, N, dtype=torch.long, device=device)
        self.seen = torch.zeros(N, dtype=torch.bool, device=device)

    def phi(self, epoch):
        c = self.cfg
        t = min(1.0, max(0, epoch - c.warmup) / max(1, c.epochs - c.warmup))
        return c.phi_start + (c.phi_end - c.phi_start) * t

    @torch.no_grad()
    def split(self):
        """Eq. (clean split): per bag, keep the delta fraction of clips closest
        to the prototype of their candidate-restricted prediction."""
        self.clean.fill_(True)
        for e in range(self.E):
            sim = (self.bank_q[e] * self.protos[e][self.bank_y[e]]).sum(1)
            for v in self.video.unique():
                rows = ((self.video == v) & self.seen).nonzero()[:, 0]
                n_keep = max(1, round(self.cfg.delta * rows.numel()))
                order = sim[rows].argsort(descending=True)
                self.clean[e, rows[order[n_keep:]]] = False

    @torch.no_grad()
    def update(self, q, k, y, rows, video, epoch):
        """Queue, prototype (clean clips only) and pseudo-target updates."""
        for e in range(self.E):
            clean = self.clean[e, rows]
            self.bank_q[e, rows], self.bank_y[e, rows] = q[e], y[e]
            self.queues[e].push(k[e], y[e], clean, video)
            for c in y[e][clean].unique():
                sel = q[e][clean & (y[e] == c)]
                g = self.cfg.proto_m ** sel.size(0)
                self.protos[e, c] = F.normalize(
                    g * self.protos[e, c] + (1 - g) * sel.mean(0), dim=0)
            if epoch >= self.cfg.warmup:
                z = restricted_argmax(q[e] @ self.protos[e].t(), self.cand[e, rows])
                phi = self.phi(epoch)
                self.s[e, rows] = phi * self.s[e, rows] + (1 - phi) * F.one_hot(z, self.K)
        self.seen[rows] = True
