"""
tribe_validation.py — Validate abliteration by intercepting the exca cache output.

Cache shape: (20, 1408, 60) = (n_cached_layers, hidden_dim, n_clips)
We project out the abliteration direction from the correct layer slice
before it flows to the regression head.
"""

import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tribev2.demo_utils import TribeModel
import gc
import subprocess

VAL_DIR    = Path("./val_data")
OUT_DIR    = Path("./abliterated")
CACHE_BASE = Path("./cache")
STUDY_ROOT = Path("./tribe_study")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

ALPHA = 0.5

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        return round(float(result.stdout.strip()) - 0.1, 3)
    except (ValueError, AttributeError):
        return 29.9

def make_video_only_df(video_path):
    duration = get_duration(video_path)
    return pd.DataFrame([{
        "type":      "Video",
        "start":     0.0,
        "duration":  duration,
        "timeline":  "default",
        "subject":   "default",
        "session":   "",
        "task":      "",
        "run":       "",
        "filepath":  str(video_path.resolve()),
        "frequency": 60.0,
        "offset":    0.0,
        "stop":      duration,
        "context":   float("nan"),
    }])

# ── Load masks ────────────────────────────────────────────────────────────────

gore_mask = np.load(OUT_DIR / "gore_mask.npy")
porn_mask = np.load(OUT_DIR / "porn_mask.npy")
print(f"Gore mask: {gore_mask.sum()} vertices")
print(f"Porn mask: {porn_mask.sum()} vertices")

# ── Load baseline preds from disk ─────────────────────────────────────────────

val_videos = sorted(VAL_DIR.glob("*.mp4"))
print(f"\nFound {len(val_videos)} validation videos")

baseline_preds = {}
for vp in val_videos:
    stem  = vp.stem
    saved = None
    for cat_dir in STUDY_ROOT.iterdir():
        if not cat_dir.is_dir():
            continue
        candidate = cat_dir / stem / "preds.npy"
        if candidate.exists():
            saved = candidate
            break
    if saved:
        baseline_preds[stem] = np.load(saved)[:30]
        print(f"  {vp.name}: baseline from {saved.relative_to(STUDY_ROOT)}")
    else:
        print(f"  [WARN] {vp.name}: no baseline preds found")

# ── Build abliteration directions ────────────────────────────────────────────

print("\nLoading abliteration directions...")
gore_dirs = np.load(OUT_DIR / "gore_directions.npy")
porn_dirs = np.load(OUT_DIR / "porn_directions.npy")

all_dirs = np.concatenate([gore_dirs, porn_dirs], axis=0)
dirs_t   = torch.tensor(all_dirs, dtype=torch.float32)  # keep on CPU, move later

# Gram-Schmidt orthogonalization
ortho = []
for d in dirs_t:
    for q in ortho:
        d = d - (d @ q) * q
    if d.norm() > 1e-6:
        ortho.append(d / d.norm())

if not ortho:
    raise ValueError("No valid directions after Gram-Schmidt")

ortho_cpu = torch.stack(ortho)  # (n_dirs, 1408) on CPU
print(f"  {len(ortho)} orthogonal directions  (alpha={ALPHA})")

# ── Load model ────────────────────────────────────────────────────────────────

print("\nLoading model...")
model         = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
vjepa2_module = model.data.video_feature.image.model.model
hf_wrapper    = model.data.video_feature.image.model   # _HuggingFace nn.Module

encoder_blocks = vjepa2_module.encoder.layer
N_LAYERS       = len(encoder_blocks)
TARGET_IDX     = int(N_LAYERS * 0.75)
print(f"  V-JEPA2 encoder layers: {N_LAYERS}, TARGET_IDX: {TARGET_IDX}")

# ── Determine which cache layer index corresponds to TARGET_IDX ───────────────
# Cache shape: (cache_n_layers=20, 1408, n_clips)
# We need to find which of the 20 cached layers is encoder.layer[TARGET_IDX].
# The HuggingFaceImage config has cache_n_layers=20; by default neuralset
# caches the last N layers. With N_LAYERS=40 and cache_n_layers=20,
# the cached layers are encoder.layer[20..39].
# TARGET_IDX=30 → cache index = 30 - (40 - 20) = 30 - 20 = 10

hf_image       = model.data.video_feature.image
cache_n_layers = getattr(hf_image, 'cache_n_layers', 20)
first_cached   = N_LAYERS - cache_n_layers          # 40 - 20 = 20
CACHE_LAYER_IDX = TARGET_IDX - first_cached         # 30 - 20 = 10

print(f"  cache_n_layers={cache_n_layers}, first_cached={first_cached}")
print(f"  TARGET_IDX={TARGET_IDX} → CACHE_LAYER_IDX={CACHE_LAYER_IDX}")

if not (0 <= CACHE_LAYER_IDX < cache_n_layers):
    raise ValueError(
        f"CACHE_LAYER_IDX={CACHE_LAYER_IDX} is out of range [0, {cache_n_layers}). "
        f"Check cache_n_layers and TARGET_IDX."
    )

# ── Hook _HuggingFace.forward output ─────────────────────────────────────────
# Cache output shape: (n_cached_layers, hidden_dim, n_clips)  OR
# wrapped in a TimedArray. The hook intercepts the raw tensor output of
# _HuggingFace.forward and projects out the abliteration direction from
# the TARGET layer slice.

_hook_calls    = [0]
_hook_shapes   = []

def cache_output_hook(module, inp, output):
    """
    output is whatever _HuggingFace.forward returns.
    We need to find the tensor of shape (..., 1408, ...) and project in place.
    """
    _hook_calls[0] += 1

    # First call: record the output type/shape for diagnostics
    if _hook_calls[0] == 1:
        if isinstance(output, torch.Tensor):
            _hook_shapes.append(f"Tensor{tuple(output.shape)}")
        else:
            _hook_shapes.append(str(type(output)))

    # output may be a Tensor directly, or a tuple/list, or a custom object
    # Try to find and modify the underlying numpy/torch data

    def project_tensor(t):
        """Project abliteration directions out of tensor slice at CACHE_LAYER_IDX.
        t expected shape: (n_layers, hidden_dim, n_clips) or (n_clips, hidden_dim).
        """
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)

        if t.ndim == 3 and t.shape[0] == cache_n_layers and t.shape[1] == 1408:
            # shape: (n_layers, 1408, n_clips) → operate on layer slice
            layer_slice = t[CACHE_LAYER_IDX].float()  # (1408, n_clips)
            q_cpu = ortho_cpu
            for q in q_cpu:
                proj = q @ layer_slice          # (n_clips,)
                layer_slice = layer_slice - ALPHA * q.unsqueeze(1) * proj.unsqueeze(0)
            t[CACHE_LAYER_IDX] = layer_slice.to(t.dtype)
            return t

        elif t.ndim == 2 and t.shape[0] == 1408:
            # shape: (1408, n_clips) — already a single layer
            layer_slice = t.float()
            q_cpu = ortho_cpu
            for q in q_cpu:
                proj = q @ layer_slice
                layer_slice = layer_slice - ALPHA * q.unsqueeze(1) * proj.unsqueeze(0)
            return layer_slice.to(t.dtype)

        return t  # unrecognised shape — leave unchanged

    if isinstance(output, torch.Tensor):
        return project_tensor(output)

    # TimedArray or other wrapper: try .data attribute
    if hasattr(output, 'data'):
        data = output.data
        if isinstance(data, np.ndarray):
            t = torch.from_numpy(data.copy())
            t = project_tensor(t)
            output.data = t.numpy()
        elif isinstance(data, torch.Tensor):
            output.data = project_tensor(data)
        return output

    return output

hook_handle = hf_wrapper.register_forward_hook(cache_output_hook)
print(f"  Hook registered on _HuggingFace wrapper")

# ── Clear cache so _HuggingFace.forward actually runs ────────────────────────
# exca will re-run _HuggingFace.forward for cache misses.
# We need to delete the cache entries for val videos so the hook fires.

print("\nClearing cache entries for validation videos...")
cache_dict = model.data.video_feature.infra.cache_dict
item_uid   = model.data.video_feature.infra.item_uid
helper     = model.data.video_feature._event_types_helper

cleared = 0
for vp in val_videos:
    df     = make_video_only_df(vp)
    events = helper.extract(df)
    for event in events:
        key = item_uid(event)
        if key in cache_dict:
            del cache_dict[key]
            cleared += 1
            print(f"  Deleted in-memory cache: {key}")

print(f"  Cleared {cleared} in-memory cache entries")

# Also delete disk cache entries by removing folders containing val video paths
import shutil, json
val_resolved = {str(vp.resolve()) for vp in val_videos}
val_names    = {vp.name for vp in val_videos}
disk_cleared = 0

for info_file in CACHE_BASE.rglob("*info.jsonl"):
    try:
        text = info_file.read_text()
        if any(name in text for name in val_names):
            shutil.rmtree(info_file.parent)
            disk_cleared += 1
            print(f"  Deleted disk cache dir: {info_file.parent.name[:60]}...")
    except Exception as e:
        pass

print(f"  Cleared {disk_cleared} disk cache directories")

# ── Inference loop ────────────────────────────────────────────────────────────

results = []

try:
    for vp in val_videos:
        stem = vp.stem
        if stem not in baseline_preds:
            print(f"\n[SKIP] no baseline for {vp.name}")
            continue

        print(f"\n{'='*55}")
        print(f"Video: {vp.name}")

        _hook_calls[0]  = 0
        _hook_shapes.clear()

        preds_base = baseline_preds[stem]
        df         = make_video_only_df(vp)
        print(f"  Duration: {df['duration'].values[0]:.3f}s")
        print("  Running abliterated inference...")

        preds_abl, _ = model.predict(events=df)
        preds_abl = preds_abl[:30]

        print(f"\n  [DIAG] _HuggingFace hook fired: {_hook_calls[0]} times")
        if _hook_shapes:
            print(f"  [DIAG] Output type/shape seen: {_hook_shapes[0]}")
        if _hook_calls[0] == 0:
            print(f"  [DIAG] WARNING: hook did not fire — cache was NOT cleared "
                  f"for this video, or exca bypasses _HuggingFace entirely")

        print()
        for mask, mname in [(gore_mask, "gore_mask"), (porn_mask, "porn_mask")]:
            base_val = float(preds_base[:, mask].mean())
            abl_val  = float(preds_abl[:, mask].mean())
            diff     = abl_val - base_val
            pct      = 100 * diff / (abs(base_val) + 1e-9)
            tag      = "✓ suppressed" if diff < -0.005 else (
                       "✗ no change"  if abs(diff) < 0.005 else "↑ increased")
            print(f"  {mname:12s}  base={base_val:.4f}  abl={abl_val:.4f}  "
                  f"Δ={diff:+.4f} ({pct:+.1f}%)  {tag}")

        base_global = float(preds_base.mean())
        abl_global  = float(preds_abl.mean())
        global_pct  = 100 * (abl_global - base_global) / (abs(base_global) + 1e-9)
        print(f"  {'whole_brain':12s}  base={base_global:.4f}  abl={abl_global:.4f}  "
              f"Δ={abl_global-base_global:+.4f} ({global_pct:+.1f}%)")

        torch.cuda.empty_cache()
        gc.collect()

        results.append({
            "video":      vp.name,
            "preds_base": preds_base,
            "preds_abl":  preds_abl,
            "hook_calls": _hook_calls[0],
        })

finally:
    hook_handle.remove()
    print("\nHook removed.")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("SUMMARY")
print("="*55)
for r in results:
    h = r['hook_calls']
    status = f"hook fired {h}x" if h > 0 else "HOOK DID NOT FIRE"
    print(f"  {r['video']:20s}  {status}")

# ── Save ──────────────────────────────────────────────────────────────────────

for r in results:
    s = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{s}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{s}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{s}_base.npy + val_{s}_abl.npy")

print("\nDone.")