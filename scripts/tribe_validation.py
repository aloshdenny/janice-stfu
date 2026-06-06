import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tribev2.demo_utils import TribeModel
import gc
import shutil

CACHE_DIR  = Path("./cache")
VAL_DIR    = Path("./val_data")
OUT_DIR    = Path("./abliterated")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_duration(video_path):
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip()) if result.stdout.strip() else 30.0
    return round(duration - 0.1, 3)   # trim 100ms to avoid end-of-clip assertion

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

def clear_video_cache(video_path):
    """
    tribev2 caches extracted video features keyed by filepath hash.
    We must clear those so the abliterated model re-extracts fresh features.
    The cache lives in ./cache — find and remove entries for this video.
    """
    vpath_str = str(video_path.resolve())
    removed = 0
    for cache_file in CACHE_DIR.rglob("*.pkl"):
        try:
            content = cache_file.read_bytes()
            if vpath_str.encode() in content or video_path.name.encode() in content:
                cache_file.unlink()
                removed += 1
        except Exception:
            pass
    # Also clear parquet caches
    for cache_file in CACHE_DIR.rglob("*.parquet"):
        try:
            content = cache_file.read_bytes()
            if vpath_str.encode() in content or video_path.name.encode() in content:
                cache_file.unlink()
                removed += 1
        except Exception:
            pass
    if removed:
        print(f"  Cleared {removed} cache files for {video_path.name}")

# ── Load models ───────────────────────────────────────────────────────────────

print("Loading baseline model...")
model_base = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)

print("Loading abliterated model...")
model_abl  = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)

vjepa2_abl = model_abl.data.video_feature.image.model.model
state_dict  = torch.load(OUT_DIR / "vjepa2_abliterated.pt", map_location=DEVICE)
vjepa2_abl.load_state_dict(state_dict)
print("Abliterated weights loaded.\n")

# ── Load masks ────────────────────────────────────────────────────────────────

gore_mask = np.load(OUT_DIR / "gore_mask.npy")
porn_mask = np.load(OUT_DIR / "porn_mask.npy")

# ── Run validation ────────────────────────────────────────────────────────────

val_videos = sorted(VAL_DIR.glob("*.mp4"))
print(f"Found {len(val_videos)} validation videos\n")

results = []

for vp in val_videos:
    print(f"{'='*50}")
    print(f"Video: {vp.name}")

    df = make_video_only_df(vp)
    print(f"  Duration (trimmed): {df['duration'].values[0]:.3f}s")

    # Run baseline first — this populates the cache
    print("  Running baseline...")
    preds_base, _ = model_base.predict(events=df)
    preds_base = preds_base[:30]
    torch.cuda.empty_cache(); gc.collect()

    # Clear the cache so abliterated model re-runs the feature extractor
    # with the modified weights instead of loading cached baseline features
    print("  Clearing feature cache...")
    clear_video_cache(vp)

    print("  Running abliterated...")
    preds_abl, _  = model_abl.predict(events=df)
    preds_abl = preds_abl[:30]
    torch.cuda.empty_cache(); gc.collect()

    # Repopulate cache with baseline features for future use
    # (optional — comment out if you don't need it)
    # model_base.predict(events=df)

    print()
    for mask, mname in [(gore_mask, "gore_mask"), (porn_mask, "porn_mask")]:
        base_val = float(preds_base[:, mask].mean())
        abl_val  = float(preds_abl[:, mask].mean())
        diff     = abl_val - base_val
        pct      = 100 * diff / (abs(base_val) + 1e-9)
        tag      = "✓ suppressed" if diff < -0.005 else ("✗ no change" if abs(diff) < 0.005 else "↑ increased")
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

# ── Save ──────────────────────────────────────────────────────────────────────

for r in results:
    stem = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{stem}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{stem}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{stem}_base.npy + val_{stem}_abl.npy")

print("\nDone.")