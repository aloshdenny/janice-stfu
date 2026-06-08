import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import torchvision.io as tvio
from torchvision import transforms
from torchvision.transforms.functional import resize
import gc
import time
from tribev2.demo_utils import TribeModel

# ── Config ────────────────────────────────────────────────────────────────────

STUDY_ROOT = Path("./tribe_study")
MASK_DIR   = STUDY_ROOT / "masks"
CACHE_DIR  = Path("./cache")
DATA_DIR   = Path("./data")
OUT_DIR    = Path("./abliterated")
OUT_DIR.mkdir(exist_ok=True)

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_FRAMES   = 16
CLIP_DURATION = 4
IMG_SIZE      = 256

# Gore: Option 3a (multivariate, food/porn/cute suppressed, 2502 vertices)
GORE_MASK_FILE = MASK_DIR / "gore_strict_bicontrast_strict.npy"
PORN_MASK_FILE = MASK_DIR / "porn_no_food_strict.npy"

CATEGORIES = {
    "gore": [f"gore{i}.mp4" for i in range(1, 49)],
    "porn": [f"porn{i}.mp4" for i in range(1, 49)],
}

# ── Get model config / TARGET_IDX dynamically ────────────────────────────────
print("Loading model config to determine TARGET_IDX...")
model_temp = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
N_LAYERS = len(model_temp.data.video_feature.image.model.model.encoder.layer)
TARGET_IDX = int(N_LAYERS * 0.75)
print(f"Target Layer Index: {TARGET_IDX} (out of {N_LAYERS})")
del model_temp
torch.cuda.empty_cache()
gc.collect()

# ── Load masks ────────────────────────────────────────────────────────────────

gore_mask = np.load(GORE_MASK_FILE)
porn_mask = np.load(PORN_MASK_FILE)
print(f"Gore mask: {gore_mask.sum()} vertices")
print(f"Porn mask: {porn_mask.sum()} vertices")

# ── Subprocess Video Collection Helper ────────────────────────────────────────

import subprocess
import sys

def collect_single_video(category, fname, mask_file, out_dir_str, 
                          study_root_str, data_dir_str, cache_dir_str,
                          device, clip_frames, clip_duration, img_size,
                          target_idx):
    """Run in a subprocess — full process isolation per video."""
    script = f"""
import os, warnings, logging, gc, time
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
import numpy as np
import torch
from pathlib import Path
import torchvision.io as tvio
from torchvision import transforms
from torchvision.transforms.functional import resize
from tribev2.demo_utils import TribeModel

OUT_DIR    = Path("{out_dir_str}")
STUDY_ROOT = Path("{study_root_str}")
DATA_DIR   = Path("{data_dir_str}")
CACHE_DIR  = Path("{cache_dir_str}")
DEVICE     = "{device}"
CLIP_FRAMES   = {clip_frames}
CLIP_DURATION = {clip_duration}
IMG_SIZE      = {img_size}
TARGET_IDX    = {target_idx}
category      = "{category}"
fname         = "{fname}"

acts_dir  = OUT_DIR / f"acts_{{category}}"
acts_dir.mkdir(exist_ok=True)
stem      = Path(fname).stem
act_path  = acts_dir / f"{{stem}}_acts.npy"
y_path    = acts_dir / f"{{stem}}_y.npy"

if act_path.exists() and y_path.exists():
    print(f"  [CACHED] {{fname}}")
    exit(0)

preds_path = STUDY_ROOT / category / stem / "preds.npy"
video_path = (DATA_DIR / fname).resolve()
if not preds_path.exists() or not video_path.exists():
    print(f"  [SKIP] {{fname}}")
    exit(0)

mask = np.load("{mask_file}")
preds = np.load(preds_path)[:30]
y_tr  = preds[:, mask].mean(axis=1)

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
vjepa2_module  = model.data.video_feature.image.model.model
encoder_blocks = vjepa2_module.encoder.layer
for m in vjepa2_module.modules():
    m._forward_hooks.clear()

collected_acts = []
def hook_fn(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    collected_acts.clear()
    collected_acts.append(hidden.mean(dim=1).detach().cpu().float())

hook_handle = encoder_blocks[TARGET_IDX].register_forward_hook(hook_fn)

normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

vframes, _, info = tvio.read_video(str(video_path), pts_unit="sec")
try:
    vframes = vframes.float() / 255.0
    vframes = vframes.permute(0, 3, 1, 2)
    fps     = info.get("video_fps", 30.0)
    total_f = vframes.shape[0]
    spf     = CLIP_DURATION * fps
    n_clips = max(1, int(total_f // spf))

    clip_acts, clip_ys = [], []
    vjepa2_module.eval()
    for c in range(n_clips):
        start = int(c * spf)
        end   = min(start + int(spf), total_f)
        chunk = vframes[start:end]
        idx   = torch.linspace(0, len(chunk) - 1, CLIP_FRAMES).long()
        clip  = chunk[idx]
        clip  = torch.stack([normalize_fn(resize(clip[i], [IMG_SIZE, IMG_SIZE]))
                             for i in range(len(clip))])
        inp = clip.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            vjepa2_module(pixel_values_videos=inp)
        if collected_acts:
            clip_acts.append(collected_acts[-1].squeeze(0).numpy())
            t_start = int(c * CLIP_DURATION)
            t_end   = min(t_start + CLIP_DURATION, 30)
            clip_ys.append(float(y_tr[t_start:t_end].mean()))
        del inp, clip
        torch.cuda.empty_cache()
finally:
    del vframes
    hook_handle.remove()
    gc.collect()

if clip_acts:
    np.save(act_path, np.stack(clip_acts))
    np.save(y_path,   np.array(clip_ys))
    print(f"  {{fname}}: {{len(clip_acts)}} clips saved")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        timeout=300
    )
    return result.returncode == 0

# ── Collect activations + vertex targets ──────────────────────────────────────

def collect_for_category(category, filenames, out_dir):
    acts_dir = out_dir / f"acts_{category}"
    acts_dir.mkdir(exist_ok=True)
    
    X_paths, y_paths = [], []

    for fname in filenames:
        stem     = Path(fname).stem
        act_path = acts_dir / f"{stem}_acts.npy"
        y_path   = acts_dir / f"{stem}_y.npy"

        if act_path.exists() and y_path.exists():
            print(f"  [CACHED] {fname}")
        else:
            mask_file = str(GORE_MASK_FILE if category == "gore" else PORN_MASK_FILE)
            ok = collect_single_video(
                category, fname, mask_file,
                str(OUT_DIR), str(STUDY_ROOT), str(DATA_DIR), str(CACHE_DIR),
                DEVICE, CLIP_FRAMES, CLIP_DURATION, IMG_SIZE, TARGET_IDX
            )
            if not ok:
                print(f"  [FAILED] {fname}")
                continue

        if act_path.exists() and y_path.exists():
            X_paths.append(act_path)
            y_paths.append(y_path)

    if not X_paths:
        return None, None
    
    # Load all at once only for PCA — still fits since it's just numpy
    X_all = np.concatenate([np.load(p) for p in X_paths])
    y_all = np.concatenate([np.load(p) for p in y_paths])
    return X_all, y_all


print("\nCollecting gore activations (strong signal)...")
X_gore, y_gore = collect_for_category("gore", CATEGORIES["gore"], OUT_DIR)

print("\nCollecting porn activations (weak signal — body/skin proxy)...")
X_porn, y_porn = collect_for_category("porn", CATEGORIES["porn"], OUT_DIR)

torch.cuda.empty_cache()
gc.collect()

# ── Weighted PCA ──────────────────────────────────────────────────────────────

def find_directions(X, y, n_components=1, label=""):  # reduce to 1 component
    weights  = (y - y.min()) / (y.max() - y.min() + 1e-9)
    weights /= weights.sum()
    X_mean   = (X * weights[:, None]).sum(axis=0, keepdims=True)
    X_c      = (X - X_mean) * np.sqrt(weights[:, None])
    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    print(f"  [{label}] singular values: {S[:5].round(4)}")
    print(f"  [{label}] explained variance ratio: {(S[:3]**2 / (S**2).sum()).round(3)}")

    directions = Vt[:n_components]

    # Fix sign: ensure direction correlates positively with y
    # i.e. high-y samples should have positive projection onto direction
    for i in range(n_components):
        proj = X @ directions[i]
        corr = float(np.corrcoef(proj, y)[0, 1])
        if corr < 0:
            directions[i] *= -1
            print(f"  [{label}] flipped direction {i} (was negatively correlated)")

    return directions


print("\nComputing directions...")
gore_dirs = find_directions(X_gore, y_gore, n_components=3, label="gore")
porn_dirs = find_directions(X_porn, y_porn, n_components=3, label="porn")

np.save(OUT_DIR / "gore_directions.npy", gore_dirs)
np.save(OUT_DIR / "porn_directions.npy", porn_dirs)
np.save(OUT_DIR / "gore_mask.npy", gore_mask)
np.save(OUT_DIR / "porn_mask.npy", porn_mask)
print(f"Saved → {OUT_DIR}")

# ── Offline selectivity validation ────────────────────────────────────────────

print("\nSelectivity (saved preds, no inference):")
print(f"{'Category':12s}  {'gore_mask':>10}  {'porn_mask':>10}")
print("-" * 40)

for cat in ["porn", "gore", "cute", "nature", "food", "kissing", "chase", "fight"]:
    paths = sorted((STUDY_ROOT / cat).glob("*/preds.npy"))
    if not paths:
        continue
    cat_mean = np.stack([np.load(p)[:30].mean(axis=0) for p in paths]).mean(axis=0)
    gm = float(cat_mean[gore_mask].mean())
    pm = float(cat_mean[porn_mask].mean())
    print(f"  {cat:12s}  {gm:10.4f}  {pm:10.4f}")

# ── Gram-Schmidt hook factory ─────────────────────────────────────────────────

def make_gs_hook(directions_np):
    dirs = torch.tensor(directions_np, dtype=torch.float32).to(DEVICE)
    ortho = []
    for d in dirs:
        for q in ortho:
            d = d - (d @ q) * q
        if d.norm() > 1e-6:
            ortho.append(d / d.norm())
    if not ortho:
        return None
    ortho = torch.stack(ortho)

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        for q in ortho:
            hidden = hidden - (hidden @ q).unsqueeze(-1) * q
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden
    return hook

# ── Weight surgery (permanent) ────────────────────────────────────────────────

def apply_weight_surgery():
    all_dirs = np.concatenate([gore_dirs, porn_dirs], axis=0)
    dirs_t   = torch.tensor(all_dirs, dtype=torch.float32).to(DEVICE)

    ortho = []
    for d in dirs_t:
        for q in ortho:
            d = d - (d @ q) * q
        if d.norm() > 1e-6:
            ortho.append(d / d.norm())
    ortho = torch.stack(ortho)

    block = encoder_blocks[TARGET_IDX]

    # Project out of both value projection (input) and output projection
    for layer_name in ["attention.value", "attention.proj"]:
        mod = block
        for p in layer_name.split("."):
            mod = getattr(mod, p)
        W = mod.weight.data
        print(f"  Surgery on block[{TARGET_IDX}].{layer_name}  {W.shape}")
        for q in ortho:
            alpha = 0.5  # suppression strength
            W -= alpha * (W @ q).unsqueeze(-1) * q
        mod.weight.data = W

    torch.save(vjepa2_module.state_dict(), OUT_DIR / "vjepa2_abliterated.pt")
    print(f"  Saved → {OUT_DIR / 'vjepa2_abliterated.pt'}")
    print(f"  File size: {(OUT_DIR / 'vjepa2_abliterated.pt').stat().st_size / 1e6:.1f} MB")

# ── Load model for surgery ────────────────────────────────────────────────────
print("\nLoading model for weight surgery...")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
vjepa2_module  = model.data.video_feature.image.model.model
encoder_blocks = vjepa2_module.encoder.layer

apply_weight_surgery()