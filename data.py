"""Labels, candidate sets, bag datasets and cross-validation folds."""
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from splits import OFFICIAL

# JigsawOSATS.txt columns 3..8
JIGSAWS_CRITERIA = ["respect", "suture_needle", "time_motion", "flow",
                    "overall", "quality"]
AIX_CRITERIA = {"respect": "OSATS_RESPECT", "time_motion": "OSATS_MOTION",
                "instrument": "OSATS_INSTRUMENT", "suture": "OSATS_SUTURE",
                "flow": "OSATS_FLOW", "knowledge": "OSATS_KNOWLEDGE",
                "overall": "OSATS_PERFORMANCE", "quality": "OSATS_FINAL_QUALITY"}


def read_jigsaws(path):
    tab = pd.read_csv(path, header=None, sep=r"\s+")
    videos = []
    for _, r in tab.iterrows():
        name = str(r[0])
        task, trial = name.rsplit("_", 1)
        votes = np.array([[int(r[c])] for c in range(3, 9)])          # [E, 1]
        videos.append(dict(name=name, task=task, user=trial[0],
                           rep=int(trial[1:]), votes=votes))
    return videos, JIGSAWS_CRITERIA


def read_aixsuture(path, level_cap=5):
    df = pd.read_excel(path) if path.endswith(".xlsx") else pd.read_csv(path)
    cols = list(AIX_CRITERIA.values())
    videos = []
    for name, g in df.groupby("VIDEO", sort=True):
        g = g.sort_values("INVESTIGATOR")
        votes = np.floor(g[cols].to_numpy(float).T + 0.5).astype(int)  # [E, R]
        videos.append(dict(name=str(name), task=str(g["GROUP"].iloc[0]),
                           user=str(g["STUDENT"].iloc[0]),
                           rep=str(g["TIME"].iloc[0]),
                           votes=np.clip(votes, 1, level_cap)))
    return videos, list(AIX_CRITERIA)


def attach_labels(videos, K, candidates):
    """0-based video level, evaluation target, candidate set and s^(0)."""
    for v in videos:
        votes = v["votes"] - 1                                    # [E, R]
        mean = votes.mean(1)
        v["target"] = mean                                        # unrounded
        v["level"] = np.floor(mean + 0.5).astype(int)             # rounded
        levels = np.arange(K)
        if candidates == "hull":
            cand = (levels >= votes.min(1, keepdims=True)) & \
                   (levels <= votes.max(1, keepdims=True))
            s0 = (votes[:, :, None] == levels).mean(1)
        else:
            cand = levels == v["level"][:, None]
            s0 = cand.astype(float)
        v["cand"], v["s0"] = cand, s0


class Bags(Dataset):
    """One item = every clip of one video, uint8 [M, 3, T, H, W]."""

    def __init__(self, videos, root):
        self.videos, self.root = videos, root
        n = [np.load(self.path(v), mmap_mode="r").shape[0] for v in videos]
        self.offsets = np.concatenate([[0], np.cumsum(n)])

    def path(self, v):
        return os.path.join(self.root, v["name"] + ".npy")

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, i):
        return torch.from_numpy(np.load(self.path(self.videos[i]))), i

    def clip_table(self, key):
        """Per-clip array of a per-video field (candidate sets, s^(0))."""
        return np.concatenate([np.repeat(v[key][None], self.offsets[i + 1]
                                         - self.offsets[i], 0)
                               for i, v in enumerate(self.videos)])

    def clip_video(self):
        return np.repeat(np.arange(len(self)), np.diff(self.offsets))


def _collate(batch):
    return list(zip(*batch))


def loader(ds, shuffle, batch_size=2, workers=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g,
                      num_workers=workers, collate_fn=_collate,
                      persistent_workers=workers > 0)


def folds(videos, protocol, seed=0, k=4):
    """[(name, train, val, test)] as lists of video indices."""
    idx = range(len(videos))
    if protocol == "official":
        split = {s: [i for i in idx if videos[i]["name"] in set(OFFICIAL[s])]
                 for s in OFFICIAL}
        return [("official", split["train"], split["val"], split["test"])]
    if protocol in ("loso", "louo"):
        key = "rep" if protocol == "loso" else "user"
        out = []
        for g in sorted({videos[i][key] for i in idx}):
            test = [i for i in idx if videos[i][key] == g]
            out.append((str(g), [i for i in idx if i not in test], None, test))
        return out
    if protocol == "lmuo":                     # participant-disjoint k-fold
        users = sorted({v["user"] for v in videos})
        random.Random(seed).shuffle(users)
        out = []
        for f in range(k):
            held = set(users[f::k])
            test = [i for i in idx if videos[i]["user"] in held]
            out.append((f"fold{f}", [i for i in idx if i not in test], None,
                        test))
        return out
    raise ValueError(protocol)
