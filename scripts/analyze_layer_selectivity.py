"""
analyze_layer_selectivity.py — Analyze representation strength and selectivity
across all 40 layers of V-JEPA2 to find the optimal layer(s) to abliterate.

Runs a single forward pass on validation videos, extracts hidden states at all layers,
computes weighted PCA directions, and evaluates their correlation with the target signals.
"""

import os
import gc
import sys
import warnings
import logging
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import torchvision.io as tvio
from torchvision import transforms
from torchvision.transforms.functional import resize
from tribev2.demo_utils import TribeModel

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

VAL_DIR       = Path("./val_data")
STUDY_ROOT    = Path("./tribe_study")
MASK_DIR      = STUDY_ROOT / "masks"
CACHE_DIR     = Path("./cache")
OUT_DIR       = Path("./abliterated")

GORE_MASK_FILE = MASK_DIR / "gore_strict_bicontrast_strict.npy"
PORN_MASK_FILE = MASK_DIR / "porn_no_food_strict.npy"

CLIP_FRAMES   = 16
CLIP_DURATION = 4
IMG_SIZE      = 256
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

def get_duration(video_path):
    import subprocess
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","default=noprint_wrappers=1:nokey=1",str(video_path)],
                       capture_output=True, text=True)
    try: return round(float(r.stdout.strip()) - 0.1, 3)
    except: return 29.9

def make_video_only_df(video_path):
    dur = get_duration(video_path)
    return pd.DataFrame([{"type":"Video","start":0.0,"duration":dur,
        "timeline":"default","subject":"default","session":"","task":"","run":"",
        "filepath":str(video_path.resolve()),"frequency":60.0,"offset":0.0,
        "stop":dur,"context":float("nan")}])

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading model...")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
vjepa2_module = model.data.video_feature.image.model.model
encoder_blocks = vjepa2_module.encoder.layer
N_LAYERS = len(encoder_blocks)

# Load masks
gore_mask = np.load(GORE_MASK_FILE)
porn_mask = np.load(PORN_MASK_FILE)

val_videos = sorted(VAL_DIR.glob("*.mp4"))
print(f"Found {len(val_videos)} validation videos:")
for vp in val_videos:
    print(f"  - {vp.name}")

# ── Setup Multi-Layer Hooks ───────────────────────────────────────────────────

# We hook every 2nd layer to save memory and processing time, or all 40 layers
# Let's target all 40 layers but clear activations per video to be safe.
LAYERS_TO_TEST = list(range(N_LAYERS))

layer_activations = {l: [] for l in LAYERS_TO_TEST}
layer_targets_gore = []
layer_targets_porn = []

def make_hook(layer_idx):
    def hook(module, inp, out):
        hidden = out[0] if isinstance(out, tuple) else out
        # mean pool over tokens, detach and move to CPU immediately
        mean_act = hidden.mean(dim=1).detach().cpu().float().squeeze(0).numpy()
        layer_activations[layer_idx].append(mean_act)
    return hook

hooks = []
for l in LAYERS_TO_TEST:
    h = encoder_blocks[l].register_forward_hook(make_hook(l))
    hooks.append(h)

normalize_fn = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
vjepa2_module.eval()

# ── Forward Pass Loop ─────────────────────────────────────────────────────────

print("\nRunning inference and extracting layer activations...")
for vp in val_videos:
    stem = vp.stem
    
    # Load targets (y) from tribe_study if available
    preds_path = None
    for cat_dir in STUDY_ROOT.iterdir():
        if not cat_dir.is_dir(): continue
        cand = cat_dir / stem / "preds.npy"
        if cand.exists():
            preds_path = cand
            break
            
    if preds_path is None:
        print(f"  [SKIP] {vp.name} (no baseline predictions in tribe_study)")
        continue

    preds = np.load(preds_path)[:30]
    y_gore = preds[:, gore_mask].mean(axis=1)
    y_porn = preds[:, porn_mask].mean(axis=1)

    # Decode video streaming
    try:
        reader = tvio.VideoReader(str(vp), "video")
        meta   = reader.get_metadata()
        fps    = meta["video"]["fps"][0] if meta["video"]["fps"] else 30.0
        dur    = meta["video"]["duration"][0] if meta["video"]["duration"] else 30.0
    except Exception as e:
        print(f"  [ERROR] opening {vp.name}: {e}")
        continue

    n_clips = max(1, int((dur * fps) // (CLIP_DURATION * fps)))
    
    print(f"  Processing {vp.name} ({n_clips} clips)...")
    
    # Temporarily reset activations for this video's clips
    temp_acts = {l: [] for l in LAYERS_TO_TEST}
    
    # Redirect hook outputs to temp_acts
    old_activations = layer_activations
    layer_activations = temp_acts

    try:
        for c in range(n_clips):
            t_seek  = float(c * CLIP_DURATION)
            t_end_s = t_seek + CLIP_DURATION

            frames = []
            reader.seek(t_seek)
            for frame_data in reader:
                if frame_data["pts"] >= t_end_s:
                    break
                frames.append(frame_data["data"])
                if len(frames) >= CLIP_FRAMES * 4:
                    break

            if len(frames) < 2:
                continue

            frames_t = torch.stack(frames).float() / 255.0
            idx  = torch.linspace(0, len(frames_t) - 1, CLIP_FRAMES).long()
            clip = frames_t[idx]
            clip = torch.stack([normalize_fn(resize(clip[i], [IMG_SIZE, IMG_SIZE])) for i in range(CLIP_FRAMES)])

            inp = clip.unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                vjepa2_module(pixel_values_videos=inp)
            
            t_start_tr = int(c * CLIP_DURATION)
            t_end_tr   = min(t_start_tr + CLIP_DURATION, 30)
            layer_targets_gore.append(float(y_gore[t_start_tr:t_end_tr].mean()))
            layer_targets_porn.append(float(y_porn[t_start_tr:t_end_tr].mean()))
            
            del inp, clip, frames_t
            torch.cuda.empty_cache()
            
    finally:
        # Restore and append to old activations
        layer_activations = old_activations
        for l in LAYERS_TO_TEST:
            layer_activations[l].extend(temp_acts[l])
        del temp_acts
        gc.collect()

# Clean up hooks
for h in hooks:
    h.remove()

# Convert collected lists to numpy arrays
y_g_all = np.array(layer_targets_gore)
y_p_all = np.array(layer_targets_porn)

print(f"\nCollected activations over {len(y_g_all)} total clips.")

# ── Selectivity & PCA Analysis per Layer ──────────────────────────────────────

def compute_layer_selectivity(X, y):
    # Normalize y to weights
    y_min, y_max = y.min(), y.max()
    y_range = y_max - y_min
    if y_range < 1e-9:
        weights = np.ones_like(y) / len(y)
    else:
        weights = (y - y_min) / (y_range + 1e-9)
        weights /= weights.sum()

    # Weighted PCA / SVD
    X_mean   = (X * weights[:, None]).sum(axis=0, keepdims=True)
    X_c      = (X - X_mean) * np.sqrt(weights[:, None])
    _, _, Vt = np.linalg.svd(X_c, full_matrices=False)
    
    # First component direction
    direction = Vt[0]
    
    # Calculate projection and correlation
    proj = X @ direction
    corr = float(np.corrcoef(proj, y)[0, 1])
    return abs(corr)

results = []

print("\nAnalyzing layer representation strength...")
for l in LAYERS_TO_TEST:
    X_l = np.stack(layer_activations[l])
    
    # Compute correlation score of primary direction with targets
    gore_corr = compute_layer_selectivity(X_l, y_g_all)
    porn_corr = compute_layer_selectivity(X_l, y_p_all)
    
    # Combine scores (higher means stronger overall representations of both concepts)
    combined = (gore_corr + porn_corr) / 2.0
    
    results.append({
        "layer": l,
        "gore_corr": gore_corr,
        "porn_corr": porn_corr,
        "combined": combined
    })

# ── Report Results ────────────────────────────────────────────────────────────

df_res = pd.DataFrame(results)
df_res = df_res.sort_values(by="combined", ascending=False)

print("\n" + "="*70)
print(f"{'LAYER SELECTIVITY REPORT':^70}")
print("="*70)
print(f"{'Layer':>6}   {'Gore Selectivity (r)':>20}   {'Porn Selectivity (r)':>20}   {'Combined':>12}")
print("-" * 70)
for idx, row in df_res.iterrows():
    print(f"{int(row['layer']):6d}   {row['gore_corr']:20.4f}   {row['porn_corr']:20.4f}   {row['combined']:12.4f}")
print("="*70)

best_layers = df_res.head(5)
print("\nOptimal layers to abliterate (ranked by combined representation strength):")
for i, (_, row) in enumerate(best_layers.iterrows()):
    print(f"  {i+1}. Layer {int(row['layer'])} (Combined r: {row['combined']:.4f}, Gore r: {row['gore_corr']:.4f}, Porn r: {row['porn_corr']:.4f})")

print("\nDone.")
