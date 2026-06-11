"""
abliteration.py — Unified abliteration pipeline script.
Runs both activation collection (streaming, process-isolated memory usage)
and weight surgery (SVD/PCA direction computation + permanent projection surgery).

Usage:
    python scripts/abliteration.py --alpha 0.2 --n_components 3 [--gore_only]
"""

import os
import gc
import sys
import time
import shutil
import warnings
import logging
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
import pandas as pd
import torchvision.io as tvio
from torchvision import transforms
from torchvision.transforms.functional import resize
from tribev2.demo_utils import TribeModel

# ── Config ────────────────────────────────────────────────────────────────────

STUDY_ROOT    = Path("./tribe_study")
MASK_DIR      = STUDY_ROOT / "masks"
CACHE_DIR     = Path("./cache")
DATA_DIR      = Path("./data_256")
OUT_DIR       = Path("./abliterated")
OUT_DIR.mkdir(exist_ok=True)

GORE_MASK_FILE = MASK_DIR / "gore_strict_bicontrast_strict.npy"
PORN_MASK_FILE = MASK_DIR / "porn_no_food_strict.npy"

CLIP_FRAMES   = 16
CLIP_DURATION = 4      # seconds per clip
IMG_SIZE      = 256

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Memory Reporting ──────────────────────────────────────────────────────────

def report_mem(tag=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**2
        r = torch.cuda.memory_reserved() / 1024**2
        print(f"  [MEM{(' '+tag) if tag else ''}] VRAM alloc={a:.0f}MB reserved={r:.0f}MB")

def ram_available_mb():
    try:
        import subprocess
        r = subprocess.run(['free', '-m'], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith('Mem:'):
                return int(line.split()[6])  # available column
    except Exception:
        pass
    return 99999

# ── Phase 1: Activation Collection ───────────────────────────────────────────

def process_video(category, fname, mask, model, vjepa2_module, encoder_blocks, target_idx, normalize_fn, hook_buffer):
    stem     = Path(fname).stem
    acts_dir = OUT_DIR / f"acts_{category}"
    acts_dir.mkdir(exist_ok=True)
    act_path = acts_dir / f"{stem}_acts.npy"
    y_path   = acts_dir / f"{stem}_y.npy"

    if act_path.exists() and y_path.exists():
        try:
            cached_y = np.load(y_path)
            if not np.isnan(cached_y).any():
                print(f"  [CACHED] {fname}")
                return True
            else:
                print(f"  [INVALID CACHE] {fname} has NaNs in cached y. Regenerating...")
        except Exception:
            pass

    preds_path = STUDY_ROOT / category / stem / "preds.npy"
    video_path = (DATA_DIR / fname).resolve()

    if not preds_path.exists():
        print(f"  [SKIP] no preds: {fname}")
        return False
    if not video_path.exists():
        print(f"  [SKIP] no video: {fname}")
        return False

    ram = ram_available_mb()
    if ram < 2000:
        print(f"  [WAIT] low RAM ({ram}MB), sleeping 10s...")
        time.sleep(10)
        gc.collect()
        torch.cuda.empty_cache()

    preds = np.load(preds_path)[:30]
    if mask.sum() == 0:
        raise ValueError(f"Mask is empty! Cannot compute y_tr. Check your mask file.")
    y_tr  = preds[:, mask].mean(axis=1)   # (30,)
    if np.isnan(y_tr).any():
        raise ValueError(f"NaNs detected in y_tr for {fname}!")

    # Open VideoReader — stream only the frames that fall in the clip window
    try:
        reader = tvio.VideoReader(str(video_path), "video")
        meta   = reader.get_metadata()
        fps    = meta["video"]["fps"][0]      if meta["video"]["fps"]      else 30.0
        dur    = meta["video"]["duration"][0] if meta["video"]["duration"] else 30.0
    except Exception as e:
        print(f"  [ERROR] open {fname}: {e}")
        del preds, y_tr
        return False

    n_clips          = max(1, int((dur * fps) // (CLIP_DURATION * fps)))
    clip_acts_list = []
    clip_ys_list   = []

    try:
        for c in range(n_clips):
            t_seek  = float(c * CLIP_DURATION)
            t_end_s = t_seek + CLIP_DURATION

            frames = []
            try:
                reader.seek(t_seek)
                for frame_data in reader:
                    if frame_data["pts"] >= t_end_s:
                        break
                    frames.append(frame_data["data"])    # uint8 (C, H, W)
                    if len(frames) >= CLIP_FRAMES * 4:   # safety cap
                        break
            except Exception as e:
                print(f"  [WARN] clip {c}/{n_clips} of {fname}: {e}")
                del frames
                continue

            if len(frames) < 2:
                del frames
                continue

            # Subsample to exactly CLIP_FRAMES, normalize, resize
            frames_t = torch.stack(frames).float() / 255.0  # (T, C, H, W)
            del frames
            idx  = torch.linspace(0, len(frames_t) - 1, CLIP_FRAMES).long()
            clip = frames_t[idx]                              # (CLIP_FRAMES, C, H, W)
            del frames_t
            clip = torch.stack([
                normalize_fn(resize(clip[i], [IMG_SIZE, IMG_SIZE]))
                for i in range(CLIP_FRAMES)
            ])                                                # (CLIP_FRAMES, C, H, W)

            inp = clip.unsqueeze(0).to(DEVICE)                # (1, T, C, H, W)
            del clip
            hook_buffer[0] = None

            with torch.no_grad():
                vjepa2_module(pixel_values_videos=inp)
            del inp

            if hook_buffer[0] is not None:
                clip_acts_list.append(hook_buffer[0].squeeze(0).numpy().copy())
                t_start_tr = int(c * CLIP_DURATION)
                t_end_tr   = min(t_start_tr + CLIP_DURATION, 30)
                clip_ys_list.append(float(y_tr[t_start_tr:t_end_tr].mean()))
            hook_buffer[0] = None

    except Exception as e:
        print(f"  [ERROR] {fname}: {e}")
        return False

    finally:
        try:
            del reader
        except Exception:
            pass
        del preds, y_tr
        torch.cuda.empty_cache()
        gc.collect()

    valid = len(clip_acts_list)
    if valid > 0:
        arr_acts = np.stack(clip_acts_list)
        arr_ys   = np.array(clip_ys_list, dtype=np.float32)
        np.save(act_path, arr_acts)
        np.save(y_path,   arr_ys)
        print(f"  {fname}: {valid} clips  "
              f"y=[{arr_ys.min():.3f}, {arr_ys.max():.3f}]  "
              f"act_dim={arr_acts.shape[1]}")
        del arr_acts, arr_ys
    else:
        print(f"  [WARN] {fname}: no valid clips")

    del clip_acts_list, clip_ys_list
    return valid > 0

def run_activation_collection():
    print("Loading model config to determine TARGET_IDX dynamically...")
    model_temp = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    n_layers = len(model_temp.data.video_feature.image.model.model.encoder.layer)
    target_idx = int(n_layers * 0.75)
    del model_temp
    torch.cuda.empty_cache()
    gc.collect()

    print("\nLoading TribeModel for activation collection...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    vjepa2_module  = model.data.video_feature.image.model.model
    encoder_blocks = vjepa2_module.encoder.layer
    print(f"Encoder layers: {n_layers}, target layer: {target_idx}")

    for m in vjepa2_module.modules():
        m._forward_hooks.clear()
        m._forward_pre_hooks.clear()

    hook_buffer = [None]
    def hook_fn(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hook_buffer[0] = hidden.mean(dim=1).detach().cpu().float()

    hook_handle = encoder_blocks[target_idx].register_forward_hook(hook_fn)
    normalize_fn = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    vjepa2_module.eval()

    categories = ["gore"] if args.gore_only else ["gore", "porn"]
    for cat in categories:
        mask_file = GORE_MASK_FILE if cat == "gore" else PORN_MASK_FILE
        mask      = np.load(mask_file)
        filenames = CATEGORIES[cat]

        print(f"\n=== Collecting {cat.upper()} ({len(filenames)} videos) ===")
        report_mem("start")

        done, failed = 0, 0
        for i, fname in enumerate(filenames):
            ok = process_video(cat, fname, mask, model, vjepa2_module, encoder_blocks, target_idx, normalize_fn, hook_buffer)
            if ok:
                done += 1
            else:
                failed += 1

            if i % 8 == 7:
                torch.cuda.empty_cache()
                gc.collect()
                report_mem(f"after {i+1} videos")

        print(f"\n{cat}: {done} collected, {failed} failed")

    hook_handle.remove()
    del model, vjepa2_module, encoder_blocks
    torch.cuda.empty_cache()
    gc.collect()

# ── Phase 2: Compute PCA & Apply Surgery ─────────────────────────────────────

def load_category_activations(category, filenames):
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
        raise ValueError(
            f"Target variable y contains NaNs in category {category}. "
            f"Please delete the corrupted files and rerun collection."
        )
    if missing:
        print(f"  [{category}] WARNING: {len(missing)} files missing from cache")
    if not X_list:
        raise FileNotFoundError(f"No cached activations found for {category}.")
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    print(f"  [{category}] loaded {X.shape[0]} clips, dim={X.shape[1]}, y∈[{y.min():.3f}, {y.max():.3f}]")
    return X, y

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
    print(f"  [{label}] explained variance ratio: {(S[:n_components+2]**2 / (S**2).sum()).round(3)}")
    
    directions = Vt[:n_components].copy()
    for i in range(n_components):
        proj = X @ directions[i]
        corr = float(np.corrcoef(proj, y)[0, 1])
        if corr < 0:
            directions[i] *= -1
            print(f"  [{label}] flipped direction {i}")
    return directions

def run_surgery_pipeline():
    print("\nLoading cached activations from disk...")
    X_gore, y_gore = load_category_activations("gore", CATEGORIES["gore"])
    if args.gore_only:
        X_porn, y_porn = None, None
    else:
        X_porn, y_porn = load_category_activations("porn", CATEGORIES["porn"])

    print("\nComputing directions...")
    gore_dirs = find_directions(X_gore, y_gore, args.n_components, "gore")
    if args.gore_only:
        porn_dirs = np.zeros((0, X_gore.shape[1]))
    else:
        porn_dirs = find_directions(X_porn, y_porn, args.n_components, "porn")

    np.save(OUT_DIR / "gore_directions.npy", gore_dirs)
    if not args.gore_only:
        np.save(OUT_DIR / "porn_directions.npy", porn_dirs)

    gore_mask = np.load(GORE_MASK_FILE)
    porn_mask = np.load(PORN_MASK_FILE)
    np.save(OUT_DIR / "gore_mask.npy", gore_mask)
    np.save(OUT_DIR / "porn_mask.npy", porn_mask)
    print(f"Directions and masks saved → {OUT_DIR}")

    # Free memory
    del X_gore, y_gore, X_porn, y_porn
    gc.collect()

    # Offline selectivity printout
    print(f"\nGore mask: {gore_mask.sum()} vertices   Porn mask: {porn_mask.sum()} vertices")
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

    # Load model for weight surgery
    print("\nLoading model for weight surgery...")
    model          = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    vjepa2_module  = model.data.video_feature.image.model.model
    encoder_blocks = vjepa2_module.encoder.layer
    n_layers       = len(encoder_blocks)
    target_idx     = int(n_layers * 0.75)
    print(f"Target: encoder.layer[{target_idx}] / {n_layers}")

    if args.gore_only:
        print("  Gore-only mode — skipping porn directions")
        all_dirs = gore_dirs
    else:
        all_dirs = np.concatenate([gore_dirs, porn_dirs], axis=0)

    dirs_t = torch.tensor(all_dirs, dtype=torch.float32).to(DEVICE)

    # Gram-Schmidt
    ortho = []
    for d in dirs_t:
        for q in ortho:
            d = d - (d @ q) * q
        if d.norm() > 1e-6:
            ortho.append(d / d.norm())
    if not ortho:
        raise ValueError("No valid directions after Gram-Schmidt")
    ortho = torch.stack(ortho)
    print(f"  Applying {len(ortho)} orthogonal directions with alpha={args.alpha}")

    block = encoder_blocks[target_idx]

    # Project out of both value projection (input) and output projection
    for layer_name in ["attention.value", "attention.proj"]:
        mod = block
        for part in layer_name.split("."):
            mod = getattr(mod, part)
        W = mod.weight.data.clone()
        print(f"  Surgery on block[{target_idx}].{layer_name}  {tuple(W.shape)}")
        for q in ortho:
            W -= args.alpha * (W @ q).unsqueeze(-1) * q
        mod.weight.data = W

    # Save checkpoints
    out_name = f"vjepa2_abliterated_a{args.alpha}_c{len(ortho)}.pt"
    torch.save(vjepa2_module.state_dict(), OUT_DIR / out_name)
    print(f"  Saved → {OUT_DIR / out_name}")
    print(f"  File size: {(OUT_DIR / out_name).stat().st_size / 1e6:.1f} MB")

    canonical = OUT_DIR / "vjepa2_abliterated.pt"
    torch.save(vjepa2_module.state_dict(), canonical)
    print(f"  Copied to canonical path: {canonical}")
    print("\nSurgery complete.")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running abliteration pipeline (alpha={args.alpha}, n_components={args.n_components}, gore_only={args.gore_only})")
    run_activation_collection()
    run_surgery_pipeline()
    print("\nPipeline execution finished successfully.")
