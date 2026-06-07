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
import shutil
import subprocess

VAL_DIR    = Path("./val_data")
OUT_DIR    = Path("./abliterated")
CACHE_BASE = Path("./cache")
CACHE_ABL  = Path("./cache_abliterated")
STUDY_ROOT = Path("./tribe_study")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

def find_ckpt():
    """Glob all known HF cache locations for best.ckpt."""
    candidates = [
        Path.home() / ".cache" / "huggingface" / "hub",
        Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub",
        Path(os.environ.get("HUGGINGFACE_HUB_CACHE", "/nonexistent")),
    ]
    for hub in candidates:
        hits = sorted(hub.glob("**/best.ckpt")) if hub.exists() else []
        if hits:
            return hits[0]
    return None

# ── Load masks ────────────────────────────────────────────────────────────────

gore_mask = np.load(OUT_DIR / "gore_mask.npy")
porn_mask = np.load(OUT_DIR / "porn_mask.npy")

# ── Phase 1: baseline inference ───────────────────────────────────────────────
# Running predict() here has the side effect of downloading best.ckpt to the
# HF hub cache, which we need in Phase 2 to build the abliterated checkpoint.

val_videos = sorted(VAL_DIR.glob("*.mp4"))
print(f"Found {len(val_videos)} validation videos\n")

print("Loading baseline model...")
model_base = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)

baseline_preds = {}   # stem → np.ndarray (30, n_verts)

for vp in val_videos:
    stem = vp.stem
    # Re-use saved preds from tribe_study if present — avoids re-running inference
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
        print(f"  {vp.name}: baseline loaded from {saved.relative_to(STUDY_ROOT)}")
    else:
        print(f"  {vp.name}: running baseline inference...")
        df = make_video_only_df(vp)
        preds, _ = model_base.predict(events=df)
        baseline_preds[stem] = preds[:30]
        torch.cuda.empty_cache(); gc.collect()

# ── Phase 2: monkeypatch V-JEPA2 class to inject abliterated weights ──────────
# The neuralset extractor ignores the Python model object we patch in memory.
# Instead it creates a FRESH V-JEPA2 instance during each predict() call.
# Solution: patch the CLASS __init__ so every new instance auto-loads our
# abliterated weights immediately after the normal initialisation.

print("\nSetting up V-JEPA2 monkeypatch...")

# Get the exact class the extractor will instantiate (same class used in model_base)
vjepa2_cls = type(model_base.data.video_feature.image.model.model)
print(f"  V-JEPA2 class: {vjepa2_cls.__module__}.{vjepa2_cls.__name__}")

vjepa2_abl_state = torch.load(OUT_DIR / "vjepa2_abliterated.pt", map_location="cpu")

_original_init = vjepa2_cls.__init__

def _abliterated_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    # After normal init, inject abliterated weights
    missing, unexpected = self.load_state_dict(vjepa2_abl_state, strict=False)
    if missing:
        print(f"  [MONKEYPATCH] {len(missing)} keys missing in abliterated state dict")
    if unexpected:
        print(f"  [MONKEYPATCH] {len(unexpected)} unexpected keys")
    print(f"  [MONKEYPATCH] Abliterated weights injected into new {vjepa2_cls.__name__} instance")

vjepa2_cls.__init__ = _abliterated_init
print("  Monkeypatch active — any new V-JEPA2 instance will use abliterated weights")

del model_base
torch.cuda.empty_cache(); gc.collect()

# ── Clear exca cache for val videos before abliterated run ───────────────────

import json

print("\nClearing exca cache for val videos...")

val_resolved = {str(vp.resolve()) for vp in val_videos}

for info_file in CACHE_BASE.rglob("*info.jsonl"):
    try:
        lines = info_file.read_text().strip().splitlines()
        val_lines    = [l for l in lines if any(v in l for v in val_resolved)]
        nonval_lines = [l for l in lines if not any(v in l for v in val_resolved)]

        if not val_lines:
            continue

        print(f"  Found {len(val_lines)} val entries in {info_file.name}")

        for line in val_lines:
            entry     = json.loads(line)
            data_file = info_file.parent / entry["data"]["filename"]
            offset    = entry["data"]["offset"]
            shape     = entry["data"]["shape"]
            n_bytes   = int(np.prod(shape)) * 4

            if data_file.exists():
                mm = np.memmap(data_file, dtype="float32", mode="r+",
                               shape=tuple(shape), offset=offset)
                mm[:] = 0.0
                mm.flush()
                del mm
                print(f"  Zeroed {data_file.name} offset={offset} shape={shape}")

        # Remove val entries from index so exca re-registers and recomputes
        info_file.write_text("\n".join(nonval_lines) + ("\n" if nonval_lines else ""))
        print(f"  Cleaned index: {info_file.name}")

    except Exception as e:
        print(f"  [ERROR] {info_file}: {e}")

print("Cache cleared — abliterated model will recompute features from scratch\n")

# ── Phase 3: load abliterated model + run inference ───────────────────────────

print("\nLoading abliterated model (fresh V-JEPA2 instances will use patched weights)...")
model_abl = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
print("Model loaded.\n")

gore_mask = np.load(OUT_DIR / "gore_mask.npy")
porn_mask = np.load(OUT_DIR / "porn_mask.npy")

results = []

try:
    for vp in val_videos:
        stem = vp.stem
        if stem not in baseline_preds:
            print(f"  [SKIP] no baseline for {vp.name}")
            continue

        print(f"{'='*50}")
        print(f"Video: {vp.name}")

        preds_base = baseline_preds[stem]

        df = make_video_only_df(vp)
        print(f"  Duration: {df['duration'].values[0]:.3f}s")
        print("  Running abliterated inference...")
        preds_abl, _ = model_abl.predict(events=df)
        preds_abl = preds_abl[:30]
        torch.cuda.empty_cache(); gc.collect()

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
        print()

        results.append({
            "video":      vp.name,
            "preds_base": preds_base,
            "preds_abl":  preds_abl,
        })

finally:
    # Always restore the original __init__ so the class is clean for any
    # subsequent code or imports.
    vjepa2_cls.__init__ = _original_init
    print("V-JEPA2 class __init__ restored.")

# ── Save ──────────────────────────────────────────────────────────────────────

for r in results:
    s = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{s}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{s}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{s}_base.npy + val_{s}_abl.npy")

print("\nDone.")
