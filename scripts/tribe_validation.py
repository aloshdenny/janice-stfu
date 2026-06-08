"""
tribe_validation.py — Validate abliteration using a LIVE hook instead of weight surgery.

This bypasses the exca caching problem entirely. Instead of modifying weights and hoping
exca re-extracts, we register a forward hook that fires during actual inference and
projects out the abliteration direction in the activation stream.

If hook_calls=0 after predict(), exca is caching the full transformer output and
we need to hook at a deeper level — the script will tell you.
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

ALPHA = 0.5   # suppression strength — change freely, no re-surgery needed

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

# ── Load baseline preds from disk (no model needed) ───────────────────────────

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
        print(f"  [WARN] {vp.name}: no baseline preds found in tribe_study")

# ── Build abliteration directions ────────────────────────────────────────────

print("\nLoading abliteration directions...")
gore_dirs = np.load(OUT_DIR / "gore_directions.npy")
porn_dirs = np.load(OUT_DIR / "porn_directions.npy")

all_dirs = np.concatenate([gore_dirs, porn_dirs], axis=0)
dirs_t   = torch.tensor(all_dirs, dtype=torch.float32).to(DEVICE)

# Gram-Schmidt orthogonalization
ortho = []
for d in dirs_t:
    for q in ortho:
        d = d - (d @ q) * q
    if d.norm() > 1e-6:
        ortho.append(d / d.norm())

if not ortho:
    raise ValueError("No valid directions after Gram-Schmidt — check direction files")

ortho = torch.stack(ortho)
print(f"  {len(ortho)} orthogonal directions built (alpha={ALPHA})")

# ── Load model ────────────────────────────────────────────────────────────────

print("\nLoading model...")
model_abl      = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
vjepa2_module  = model_abl.data.video_feature.image.model.model
encoder_blocks = vjepa2_module.encoder.layer
N_LAYERS       = len(encoder_blocks)
TARGET_IDX     = int(N_LAYERS * 0.75)
print(f"  encoder.layer count: {N_LAYERS}, target: {TARGET_IDX}")

# Clear any stale hooks
for m in vjepa2_module.modules():
    m._forward_hooks.clear()
    m._forward_pre_hooks.clear()

# ── Register live abliteration hook ──────────────────────────────────────────
# This fires during every actual forward pass through block TARGET_IDX.
# Exca caching is irrelevant — if the forward pass runs, the hook fires.

_hook_calls = [0]

def abliteration_hook(module, input, output):
    _hook_calls[0] += 1
    hidden = output[0] if isinstance(output, tuple) else output
    for q in ortho:
        hidden = hidden - ALPHA * (hidden @ q).unsqueeze(-1) * q
    return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

hook_handle = encoder_blocks[TARGET_IDX].register_forward_hook(abliteration_hook)
print(f"  Live hook registered on encoder.layer[{TARGET_IDX}]")

# Also register hooks on ALL blocks to find out which ones actually fire
_block_fire_counts = {}
_block_hooks = []

def make_probe_hook(idx):
    def probe(module, input, output):
        _block_fire_counts[idx] = _block_fire_counts.get(idx, 0) + 1
    return probe

for i, block in enumerate(encoder_blocks):
    h = block.register_forward_hook(make_probe_hook(i))
    _block_hooks.append(h)

print("  Probe hooks registered on all encoder blocks")

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

        # Reset counters
        _hook_calls[0] = 0
        _block_fire_counts.clear()

        preds_base = baseline_preds[stem]
        df         = make_video_only_df(vp)
        print(f"  Duration: {df['duration'].values[0]:.3f}s")
        print("  Running abliterated inference...")

        preds_abl, _ = model_abl.predict(events=df)
        preds_abl = preds_abl[:30]

        # Diagnostic: did the hook actually fire?
        print(f"\n  [DIAG] Abliteration hook fired: {_hook_calls[0]} times")
        fired_blocks = sorted(_block_fire_counts.keys())
        if fired_blocks:
            print(f"  [DIAG] Blocks that fired: {fired_blocks[0]}–{fired_blocks[-1]} "
                  f"({len(fired_blocks)} total)")
        else:
            print(f"  [DIAG] WARNING: NO encoder blocks fired — "
                  f"exca is serving fully-cached output, hooks cannot intercept")

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
    for h in _block_hooks:
        h.remove()
    print("\nAll hooks removed.")

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("SUMMARY")
print("="*55)
for r in results:
    hook_status = f"hook fired {r['hook_calls']}x" if r['hook_calls'] > 0 else "HOOK DID NOT FIRE"
    print(f"  {r['video']:20s}  {hook_status}")

all_fired = all(r['hook_calls'] > 0 for r in results)
if not all_fired:
    print("\n  *** DIAGNOSIS: exca is caching full V-JEPA2 output. ***")
    print("  The hook at encoder.layer[TARGET_IDX] never fires because")
    print("  the extractor loads cached activations from disk and skips")
    print("  the transformer forward pass entirely.")
    print()
    print("  NEXT STEP: hook at the exca output level instead.")
    print("  Run this to find the right interception point:")
    print()
    print("    python find_cache_intercept.py")
else:
    print("\n  Hook fired correctly. Abliteration is active.")

# ── Save ──────────────────────────────────────────────────────────────────────

for r in results:
    s = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{s}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{s}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{s}_base.npy + val_{s}_abl.npy")

print("\nDone.")