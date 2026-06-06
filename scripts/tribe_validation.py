import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tribev2.demo_utils import TribeModel
import gc

CACHE_DIR  = Path("./cache")
VAL_DIR    = Path("./val_data")
OUT_DIR    = Path("./abliterated")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def make_video_only_df(video_path):
    """Build minimal events dataframe with just the Video row — skips whisperx entirely."""
    import subprocess
    # Get duration via ffprobe
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip()) if result.stdout.strip() else 30.0

    return pd.DataFrame([{
        "type":     "Video",
        "start":    0.0,
        "duration": duration,
        "filepath": str(video_path),
        "text":     None,
        "context":  "",
    }])

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
    print(f"  Duration: {df['duration'].values[0]:.1f}s")

    print("  Running baseline...")
    preds_base, _ = model_base.predict(events=df)
    preds_base = preds_base[:30]
    torch.cuda.empty_cache(); gc.collect()

    print("  Running abliterated...")
    preds_abl, _  = model_abl.predict(events=df)
    preds_abl = preds_abl[:30]
    torch.cuda.empty_cache(); gc.collect()

    print()
    for mask, mname in [(gore_mask, "gore_mask"), (porn_mask, "porn_mask")]:
        base_val = float(preds_base[:, mask].mean())
        abl_val  = float(preds_abl[:, mask].mean())
        diff     = abl_val - base_val
        pct      = 100 * diff / (abs(base_val) + 1e-9)
        suppressed = "✓ suppressed" if diff < -0.01 else "✗ no change"
        print(f"  {mname:12s}  base={base_val:.4f}  abl={abl_val:.4f}  "
              f"Δ={diff:+.4f} ({pct:+.1f}%)  {suppressed}")

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

# ── Save ─────────────────────────────────────────────────────────────────────

for r in results:
    stem = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{stem}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{stem}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{stem}_base.npy + val_{stem}_abl.npy")

print("\nDone.")