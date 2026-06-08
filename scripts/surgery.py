"""
surgery.py — Phase 2: load cached activations, compute directions, apply surgery.
Run this after collect_acts.py has finished.

Usage:
    python surgery.py [--alpha 0.5] [--n_components 1]
"""

import os, warnings, logging, argparse, gc
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import torch
from pathlib import Path
from tribev2.demo_utils import TribeModel

# ── Config ────────────────────────────────────────────────────────────────────

STUDY_ROOT    = Path("./tribe_study")
MASK_DIR      = STUDY_ROOT / "masks"
CACHE_DIR     = Path("./cache")
OUT_DIR       = Path("./abliterated")

GORE_MASK_FILE = MASK_DIR / "gore_strict_bicontrast_strict.npy"
PORN_MASK_FILE = MASK_DIR / "porn_no_food_strict.npy"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORIES = {
    "gore": [f"gore{i}.mp4" for i in range(1, 49)],
    "porn": [f"porn{i}.mp4" for i in range(1, 49)],
}

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",        type=float, default=0.2,
                    help="Suppression strength 0–1 (default 0.2)")
parser.add_argument("--n_components", type=int,   default=1,
                    help="PCA components per category (default 1)")
parser.add_argument("--gore_only",    action="store_true",
                    help="Apply gore direction only (skip porn)")
args = parser.parse_args()

print(f"alpha={args.alpha}  n_components={args.n_components}  "
      f"gore_only={args.gore_only}")

# ── Load saved activations ────────────────────────────────────────────────────

def load_category(category, filenames):
    acts_dir = OUT_DIR / f"acts_{category}"
    X_list, y_list = [], []
    missing = []
    nan_files = []
    for fname in filenames:
        stem     = Path(fname).stem
        act_path = acts_dir / f"{stem}_acts.npy"
        y_path   = acts_dir / f"{stem}_y.npy"
        if act_path.exists() and y_path.exists():
            X_val = np.load(act_path)
            y_val = np.load(y_path)
            if np.isnan(y_val).any():
                nan_files.append(y_path.name)
            X_list.append(X_val)
            y_list.append(y_val)
        else:
            missing.append(fname)
    if nan_files:
        print(f"  [{category}] WARNING: {len(nan_files)} target files contain NaNs:")
        print(f"    {nan_files[:5]}{'...' if len(nan_files)>5 else ''}")
        raise ValueError(
            f"Target variable y contains NaNs in category {category}. "
            f"Please delete the corrupted files and rerun collect_acts.py."
        )
    if missing:
        print(f"  [{category}] WARNING: {len(missing)} files missing — "
              f"run collect_acts.py first")
        print(f"    {missing[:5]}{'...' if len(missing)>5 else ''}")
    if not X_list:
        raise FileNotFoundError(
            f"No activations found for {category}. Run collect_acts.py first.")
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    print(f"  [{category}] loaded {X.shape[0]} clips, dim={X.shape[1]}, "
          f"y∈[{y.min():.3f}, {y.max():.3f}]")
    return X, y


print("\nLoading activations from disk...")
X_gore, y_gore = load_category("gore", CATEGORIES["gore"])
X_porn, y_porn = load_category("porn", CATEGORIES["porn"])
gc.collect()

# ── Weighted PCA ──────────────────────────────────────────────────────────────

def find_directions(X, y, n_components, label):
    if np.isnan(y).any():
        raise ValueError(f"[{label}] target variable y contains NaNs! SVD will fail.")
    
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    if y_range < 1e-9:
        print(f"  [{label}] y is constant (range < 1e-9), using uniform weights")
        weights = np.ones_like(y) / len(y)
    else:
        weights  = (y - y_min) / (y_range + 1e-9)
        weights /= weights.sum()
        
    X_mean   = (X * weights[:, None]).sum(axis=0, keepdims=True)
    X_c      = (X - X_mean) * np.sqrt(weights[:, None])
    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    print(f"  [{label}] singular values: {S[:5].round(4)}")
    print(f"  [{label}] explained variance ratio: "
          f"{(S[:n_components+2]**2 / (S**2).sum()).round(3)}")
    directions = Vt[:n_components].copy()
    for i in range(n_components):
        proj = X @ directions[i]
        corr = float(np.corrcoef(proj, y)[0, 1])
        if corr < 0:
            directions[i] *= -1
            print(f"  [{label}] flipped direction {i}")
    return directions


print("\nComputing directions...")
gore_dirs = find_directions(X_gore, y_gore, args.n_components, "gore")
porn_dirs = find_directions(X_porn, y_porn, args.n_components, "porn")

np.save(OUT_DIR / "gore_directions.npy", gore_dirs)
np.save(OUT_DIR / "porn_directions.npy", porn_dirs)
print(f"Directions saved → {OUT_DIR}")

# Free activation arrays before loading the model
del X_gore, y_gore, X_porn, y_porn
gc.collect()

# ── Offline selectivity printout ──────────────────────────────────────────────

gore_mask = np.load(GORE_MASK_FILE)
porn_mask = np.load(PORN_MASK_FILE)

# Save masks to OUT_DIR for validation.py
np.save(OUT_DIR / "gore_mask.npy", gore_mask)
np.save(OUT_DIR / "porn_mask.npy", porn_mask)
print(f"Masks saved → {OUT_DIR}")

print(f"\nGore mask: {gore_mask.sum()} vertices")
print(f"Porn mask: {porn_mask.sum()} vertices")

print(f"\n{'Category':12s}  {'gore_mask':>10}  {'porn_mask':>10}")
print("-" * 40)
for cat in ["porn", "gore", "cute", "nature", "food", "kissing", "chase", "fight"]:
    paths = sorted((STUDY_ROOT / cat).glob("*/preds.npy"))
    if not paths:
        continue
    cat_mean = np.stack([np.load(p)[:30].mean(axis=0) for p in paths]).mean(axis=0)
    gm = float(cat_mean[gore_mask].mean())
    pm = float(cat_mean[porn_mask].mean())
    print(f"  {cat:12s}  {gm:10.4f}  {pm:10.4f}")

# ── Load model for surgery ────────────────────────────────────────────────────

print("\nLoading model for weight surgery...")
model          = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
vjepa2_module  = model.data.video_feature.image.model.model
encoder_blocks = vjepa2_module.encoder.layer
N_LAYERS       = len(encoder_blocks)
TARGET_IDX     = int(N_LAYERS * 0.75)
print(f"Target: encoder.layer[{TARGET_IDX}] / {N_LAYERS}")

# ── Weight surgery ────────────────────────────────────────────────────────────

def apply_weight_surgery(gore_dirs, porn_dirs, alpha, gore_only):
    if gore_only:
        print("  Gore-only mode — skipping porn directions")
        all_dirs = gore_dirs
    else:
        all_dirs = np.concatenate([gore_dirs, porn_dirs], axis=0)

    dirs_t = torch.tensor(all_dirs, dtype=torch.float32).to(DEVICE)

    # Gram-Schmidt orthogonalization
    ortho = []
    for d in dirs_t:
        for q in ortho:
            d = d - (d @ q) * q
        if d.norm() > 1e-6:
            ortho.append(d / d.norm())

    if not ortho:
        raise ValueError("No valid directions after Gram-Schmidt")

    ortho = torch.stack(ortho)
    print(f"  Applying {len(ortho)} orthogonal directions with alpha={alpha}")

    block = encoder_blocks[TARGET_IDX]

    for layer_name in ["attention.value", "attention.proj"]:
        mod = block
        for part in layer_name.split("."):
            mod = getattr(mod, part)
        W = mod.weight.data.clone()
        print(f"  Surgery on block[{TARGET_IDX}].{layer_name}  {tuple(W.shape)}")
        for q in ortho:
            W -= alpha * (W @ q).unsqueeze(-1) * q
        mod.weight.data = W

    out_path = OUT_DIR / f"vjepa2_abliterated_a{alpha}_c{len(ortho)}.pt"
    torch.save(vjepa2_module.state_dict(), out_path)
    print(f"  Saved → {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB")
    return out_path


out_path = apply_weight_surgery(gore_dirs, porn_dirs, args.alpha, args.gore_only)

# Also save a copy with the canonical name validation.py expects
import shutil
canonical = OUT_DIR / "vjepa2_abliterated.pt"
shutil.copy(out_path, canonical)
print(f"  Copied to canonical path: {canonical}")

print("\nSurgery complete.")
print(f"Next step: run tribe_validation.py")