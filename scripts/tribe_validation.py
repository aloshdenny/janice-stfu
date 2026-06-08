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
# Load baseline preds from disk (no model needed)

val_videos = sorted(VAL_DIR.glob("*.mp4"))
print(f"Found {len(val_videos)} validation videos\n")

baseline_preds = {}   # stem → np.ndarray (30, n_verts)

for vp in val_videos:
    stem = vp.stem
    # Re-use saved preds from tribe_study
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
        print(f"  [WARNING] {vp.name}: baseline preds not found on disk. Skipping.")

# ── Phase 2: Load abliterated model & patch V-JEPA2 ───────────────────────────

print("\nLoading abliterated model...")
model_abl = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
print("Model loaded.\n")

# Get V-JEPA2 class and inject abliterated weights
vjepa2_cls = type(model_abl.data.video_feature.image.model.model)
print(f"  V-JEPA2 class: {vjepa2_cls.__module__}.{vjepa2_cls.__name__}")

vjepa2_abl_state = torch.load(OUT_DIR / "vjepa2_abliterated.pt", map_location="cpu")

# Inject abliterated weights into the active instance
missing, unexpected = model_abl.data.video_feature.image.model.model.load_state_dict(vjepa2_abl_state, strict=False)
if missing:
    print(f"  [ACTIVE INSTANCE] {len(missing)} keys missing in abliterated state dict")
if unexpected:
    print(f"  [ACTIVE INSTANCE] {len(unexpected)} unexpected keys")
print("  [ACTIVE INSTANCE] Abliterated weights injected directly into loaded model instance")

# Set up class monkeypatch in case fresh instances are created during predict()
_original_init = vjepa2_cls.__init__

def _abliterated_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    self.load_state_dict(vjepa2_abl_state, strict=False)
    print(f"  [MONKEYPATCH] Abliterated weights injected into new {vjepa2_cls.__name__} instance")

vjepa2_cls.__init__ = _abliterated_init
print("  Monkeypatch active — any new V-JEPA2 instance will use abliterated weights")

# ── Clear exca cache for val videos on model_abl ──────────────────────────────

print("\nClearing exca cache for val videos...")
cache_dict = model_abl.data.video_feature.infra.cache_dict
item_uid = model_abl.data.video_feature.infra.item_uid
helper = model_abl.data.video_feature._event_types_helper

# Force population of cache keys by calling keys() or __contains__
all_keys = list(cache_dict.keys())
print(f"  Total keys currently in cache_dict: {len(all_keys)}")
if len(all_keys) > 0:
    print(f"  Sample cache keys: {all_keys[:5]}")

for vp in val_videos:
    df = make_video_only_df(vp)
    events = helper.extract(df)
    for event in events:
        key = item_uid(event)
        print(f"  Checking validation key: {key}")
        if key in cache_dict:
            print(f"    -> Found! Deleting cache key: {key}")
            del cache_dict[key]
            # Confirm deletion
            if key not in cache_dict:
                print(f"    -> Verified deleted from cache_dict")
            else:
                print(f"    -> WARNING: Failed to delete from cache_dict!")
        else:
            print(f"    -> Key not found in cache_dict")

print(f"  Total keys in cache_dict after deletion: {len(list(cache_dict.keys()))}")
print("Cache cleared — abliterated model will recompute features from scratch\n")

# ── Phase 3: run inference ────────────────────────────────────────────────────

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
