"""Decode videos into non-overlapping clips of 10 frames at 2 fps, 224x224.

Writes <out>/<video>.npy, uint8 [n_clips, 3, 10, 224, 224]. JIGSAWS uses the
capture1 view (<trial>_capture1.avi -> <trial>.npy); AIxSuture <video>.mp4.

    python prepare.py --src <dir with videos> --out <clips dir>
"""
import argparse
import os
import re
import subprocess

import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--out", required=True)
p.add_argument("--fps", type=float, default=2)
p.add_argument("--frames", type=int, default=10)
p.add_argument("--size", type=int, default=224)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)

for root, _, files in os.walk(a.src):
    for f in sorted(files):
        stem, ext = os.path.splitext(f)
        if ext.lower() not in (".avi", ".mp4") or "capture2" in stem:
            continue
        name = re.sub(r"_capture\d+$", "", stem)
        raw = subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-i", os.path.join(root, f),
             "-vf", f"fps={a.fps},scale={a.size}:{a.size}",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            check=True, capture_output=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, a.size, a.size, 3)
        n = len(frames) // a.frames
        clips = frames[:n * a.frames].reshape(n, a.frames, a.size, a.size, 3)
        np.save(os.path.join(a.out, name + ".npy"), clips.transpose(0, 4, 1, 2, 3))
        print(f"{name}: {n} clips", flush=True)
