# Surg-Pico

Code for the Surg-Pico paper: partial-label multiple-instance learning for OSATS skill assessment from surgical video.

| File | Contents |
|---|---|
| `prepare.py` | decodes videos into 10-frame clips at 2 fps, 224×224 |
| `data.py` | labels, candidate sets, $s^{(0)}$, cross-validation folds |
| `splits.py` | official AIxSuture train/val/test split |
| `augment.py` | query and key augmentations |
| `model.py` | Video Swin-T + LoRA, trunk, instance classifiers, projector |
| `pico.py` | prototypes, FIFO queue, contrastive losses, clean/noisy split, pseudo-label updates |
| `train.py` | objective, training loop, evaluation, metrics |

## Setup

```bash
pip install -r requirements.txt
```

`ffmpeg` is needed for `prepare.py`. The Kinetics-400 Swin-T weights are downloaded by torchvision.

## Data

- JIGSAWS: the `capture1` videos, plus `JigsawOSATS.txt`, the three `meta_file_*.txt` of the release concatenated (one line per trial: `trial  skill  GRS  RT SNH TM FO OP QFP`).
- AIxSuture: the videos and `OSATS.xlsx` from the dataset release.

```bash
python prepare.py --src /path/to/JIGSAWS --out clips/jigsaws
python prepare.py --src /path/to/AIxSuture --out clips/aixsuture
```

## Experiments

Run each command with `--seed 0`, `1` and `2`. Results are written to `results/*.json`.

**JIGSAWS (LOSO / LOUO)**

```bash
python train.py --dataset jigsaws --labels JigsawOSATS.txt --clips clips/jigsaws \
  --protocol loso --candidates collapsed --delta 0.5
python train.py --dataset jigsaws --labels JigsawOSATS.txt --clips clips/jigsaws \
  --protocol louo --candidates collapsed --delta 0.5
```

**AIxSuture (official split with labels capped at 4 / 4-fold LMUO)**

```bash
python train.py --dataset aixsuture --labels OSATS.xlsx --clips clips/aixsuture \
  --protocol official --level_cap 4
python train.py --dataset aixsuture --labels OSATS.xlsx --clips clips/aixsuture \
  --protocol lmuo
```

**Ablation**: add these options to the JIGSAWS LOSO or AIxSuture LMUO command.

| Candidate set | Read-out | PLL module | Options |
|---|---|---|---|
| Collapsed | ReCAP | None | `--candidates collapsed --readout recap --pll none` |
| Collapsed | ReCAP | PiCO | `--candidates collapsed --readout recap --pll pico` |
| Collapsed | ReCAP | PiCO+ | `--candidates collapsed --readout recap --pll picoplus` |
| Collapsed | Centroid | PiCO+ | `--candidates collapsed --readout centroid --pll picoplus` |
| Rater hull | Centroid | PiCO | `--candidates hull --readout centroid --pll pico` |
| Rater hull | Centroid | PiCO+ | `--candidates hull --readout centroid --pll picoplus` |
