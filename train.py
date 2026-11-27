"""Surg-Pico: training and cross-validated evaluation."""
import argparse
import copy
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from augment import key_view, normalize, query_view
from data import Bags, attach_labels, folds, loader, read_aixsuture, read_jigsaws
from model import Encoder
from pico import PiCO, contrastive, restricted_argmax, smoothed_nll, soft_ce

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["jigsaws", "aixsuture"], required=True)
    p.add_argument("--labels", required=True, help="JigsawOSATS.txt or OSATS.xlsx")
    p.add_argument("--clips", required=True, help="directory of <video>.npy files")
    p.add_argument("--protocol", choices=["loso", "louo", "lmuo", "official"],
                   required=True)
    p.add_argument("--candidates", choices=["collapsed", "hull"], default="hull")
    p.add_argument("--readout", choices=["centroid", "recap"], default="centroid")
    p.add_argument("--pll", choices=["none", "pico", "picoplus"], default="picoplus")
    p.add_argument("--delta", type=float, default=0.8)
    p.add_argument("--level_cap", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--lr_warmup", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--clip_drop", type=float, default=0.25)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--lam", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--knn", type=int, default=32)
    p.add_argument("--knn_start", type=int, default=12)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--phi_start", type=float, default=0.95)
    p.add_argument("--phi_end", type=float, default=0.80)
    p.add_argument("--proto_m", type=float, default=0.99)
    p.add_argument("--ema_m", type=float, default=0.999)
    p.add_argument("--queue", type=int, default=8192)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--report_last", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default="results")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(enc, a):
    params = [p for p in enc.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.ndim >= 2]},
              {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]
    opt = torch.optim.SGD(groups, lr=a.lr, momentum=0.9, weight_decay=a.weight_decay)

    def schedule(ep):
        if ep < a.lr_warmup:
            return (ep + 1) / a.lr_warmup
        t = (ep - a.lr_warmup) / max(1, a.epochs - a.lr_warmup)
        return 0.5 * (1 + math.cos(math.pi * t))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, schedule)


def bag_loss(enc, ema, pico, x, rows, v, y, epoch, a):
    """Loss of one bag, averaged over OSATS criteria."""
    h = enc.latent(normalize(query_view(x)))
    logits = enc.classify(h)
    E = len(logits)
    if pico is None:                                   # ReCAP read-out only
        loss = sum(smoothed_nll(F.softmax(l, 1).mean(0), int(y[e]), a.smoothing)
                   for e, l in enumerate(logits))
        return loss / E, None

    q = enc.project(h)
    with torch.no_grad():
        k = ema.project(ema.latent(normalize(key_view(x))))
    w_cont = a.lam * min(1.0, max(0, epoch - 1) / 10)
    split_on = a.pll == "picoplus" and epoch >= a.warmup
    all_rows = torch.arange(rows.numel(), device=DEV)
    total, y_tilde = 0.0, []
    for e in range(E):
        queue, clean, s = pico.queues[e], pico.clean[e, rows], pico.s[e, rows]
        yt = restricted_argmax(logits[e].detach(), pico.cand[e, rows])
        y_tilde.append(yt)
        l_cls = soft_ce(logits[e][clean], s[clean])
        l_cont = contrastive(q[e], k[e], yt, clean, v, queue, a.tau,
                             clean.nonzero()[:, 0])
        if a.pll == "pico":
            loss = l_cls + w_cont * l_cont
        else:
            loss = a.alpha * (l_cls + w_cont * l_cont)
            if split_on:
                noisy = ~clean
                yh = torch.where(noisy, logits[e].detach().argmax(1), yt)
                l_ncont = contrastive(q[e], k[e], yh, None, v, queue, a.tau, all_rows)
                l_knn = (contrastive(q[e], k[e], yh, None, v, queue, a.tau,
                                     noisy.nonzero()[:, 0], knn=a.knn)
                         if epoch >= a.knn_start else 0.0)
                target = F.softmax(q[e][noisy].detach() @ pico.protos[e].t() / a.tau, 1)
                l_ncls = soft_ce(logits[e][noisy], target)
                loss = loss + a.beta * (l_ncls + w_cont * (l_ncont + l_knn))
        if a.readout == "recap":
            loss = loss + smoothed_nll(F.softmax(logits[e], 1).mean(0), int(y[e]),
                                       a.smoothing)
        elif epoch >= a.warmup:
            c = F.normalize(q[e].mean(0), dim=0)
            loss = loss + smoothed_nll(F.softmax(pico.protos[e] @ c / a.tau, 0),
                                       int(y[e]), a.smoothing)
        total = total + loss
    return total / E, ([t.detach() for t in q], k, y_tilde, rows, v)


def train_epoch(enc, ema, pico, dl, opt, ds, epoch, a):
    enc.train()
    for xs, ids in dl:
        opt.zero_grad(set_to_none=True)
        stash = []
        for x, v in zip(xs, ids):
            n = x.size(0)
            keep = max(4, round(n * (1 - a.clip_drop)))
            keep = torch.randperm(n)[:keep].sort().values if keep < n else torch.arange(n)
            rows = int(ds.offsets[v]) + keep.to(DEV)
            x = x[keep].to(DEV).float().div_(255)
            loss, state = bag_loss(enc, ema, pico, x, rows, v,
                                   ds.videos[v]["level"], epoch, a)
            (loss / len(xs)).backward()
            stash.append(state)
        opt.step()
        if pico is not None:
            for state in stash:
                pico.update(*state, epoch)
            with torch.no_grad():
                for p, pm in zip(enc.parameters(), ema.parameters()):
                    if p.requires_grad:
                        pm.mul_(a.ema_m).add_(p.detach(), alpha=1 - a.ema_m)


@torch.no_grad()
def predict(enc, pico, dl, a):
    """[V, E, K] video-level distributions."""
    enc.eval()
    out = []
    for xs, _ in dl:
        x = xs[0]
        h = torch.cat([enc.latent(normalize(c.to(DEV).float().div_(255)))
                       for c in x.split(32)])
        if a.readout == "centroid":
            q = enc.project(h)
            p = [F.softmax(pico.protos[e] @ F.normalize(q[e].mean(0), dim=0) / a.tau, 0)
                 for e in range(len(q))]
        else:
            p = [F.softmax(l, 1).mean(0) for l in enc.classify(h)]
        out.append(torch.stack(p).cpu())
    return torch.stack(out).numpy()


def macro_f1(pred, true):
    f1 = []
    for c in np.union1d(pred, true):
        tp = ((pred == c) & (true == c)).sum()
        f1.append(2 * tp / (2 * tp + ((pred == c) != (true == c)).sum()))
    return float(np.mean(f1))


def rho(a, b):
    return float(spearmanr(a, b)[0])


def metrics(P, videos, names, by_task):
    K = P.shape[-1]
    exp, arg = (P * np.arange(K)).sum(-1), P.argmax(-1)
    target = np.stack([v["target"] for v in videos])
    level = np.stack([v["level"] for v in videos])
    m = {f"rho_{n}": rho(exp[:, e], target[:, e]) for e, n in enumerate(names)}
    m["rho_osats"] = float(np.nanmean([m[f"rho_{n}"] for n in names]))
    m["rho_grs"] = rho(exp.sum(1), target.sum(1))
    m["f1"] = float(np.mean([macro_f1(arg[:, e], level[:, e]) for e in range(len(names))]))
    m["ec"] = float(np.abs(arg - level).mean() / (K - 1))
    if by_task:
        for t in sorted({v["task"] for v in videos}):
            i = [j for j, v in enumerate(videos) if v["task"] == t]
            m[f"rho_osats_{t}"] = float(np.nanmean(
                [rho(exp[i, e], target[i, e]) for e in range(len(names))]))
    return m


def mean_dicts(ds):
    return {k: float(np.nanmean([d[k] for d in ds])) for k in ds[0]}


def run_fold(a, train_v, val_v, test_v, names, K):
    E = len(names)
    train_ds = Bags(train_v, a.clips)
    enc = Encoder(E, K).to(DEV)
    ema = pico = None
    if a.pll != "none":
        ema = copy.deepcopy(enc).eval().requires_grad_(False)
        pico = PiCO(train_ds, a, E, K, 128, DEV)
    opt, sched = build_optimizer(enc, a)
    train_dl = loader(train_ds, True, a.batch_size, a.workers, a.seed)
    held = {"test": test_v} if val_v is None else {"val": val_v, "test": test_v}
    held = {s: Bags(v, a.clips) for s, v in held.items()}
    held = {s: (ds, loader(ds, False, 1, a.workers)) for s, ds in held.items()}
    history = []
    for epoch in range(a.epochs):
        if a.pll == "picoplus" and epoch >= a.warmup:
            pico.split()
        train_epoch(enc, ema, pico, train_dl, opt, train_ds, epoch, a)
        sched.step()
        if val_v is not None or epoch >= a.epochs - a.report_last:
            rec = {s: metrics(predict(enc, pico, dl, a), ds.videos, names,
                              a.dataset == "jigsaws") for s, (ds, dl) in held.items()}
            history.append(rec)
            print(f"  epoch {epoch:3d} " + " ".join(
                f"{s}: rho={r['rho_osats']:.3f} f1={r['f1']:.3f}" for s, r in rec.items()),
                flush=True)
    if val_v is not None:                     # best validation macro-F1 epoch
        return max(history, key=lambda r: r["val"]["f1"])["test"]
    return mean_dicts([r["test"] for r in history])


def main():
    a = get_args()
    assert not (a.readout == "centroid" and a.pll == "none"), \
        "the centroid read-out decodes against PiCO prototypes"
    K = a.level_cap
    if a.dataset == "jigsaws":
        videos, names = read_jigsaws(a.labels)
    else:
        videos, names = read_aixsuture(a.labels, K)
    videos = [v for v in videos
              if os.path.exists(os.path.join(a.clips, v["name"] + ".npy"))]
    attach_labels(videos, K, a.candidates)
    print(f"{len(videos)} videos | {vars(a)}")
    results = {}
    for f, (name, tr, va, te) in enumerate(folds(videos, a.protocol, a.seed)):
        seed_everything(a.seed + f)
        print(f"fold {name}: {len(tr)} train / {len(te)} test videos", flush=True)
        pick = lambda ix: None if ix is None else [videos[i] for i in ix]
        results[name] = run_fold(a, pick(tr), pick(va), pick(te), names, K)
        print(f"fold {name}: " + json.dumps(results[name]), flush=True)
    summary = mean_dicts(list(results.values()))
    print("mean over folds: " + json.dumps(summary, indent=1))
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.dataset}_{a.protocol}_{a.candidates}_{a.readout}_{a.pll}_seed{a.seed}"
    with open(os.path.join(a.out, tag + ".json"), "w") as fh:
        json.dump({"args": vars(a), "folds": results, "mean": summary}, fh, indent=1)


if __name__ == "__main__":
    main()
