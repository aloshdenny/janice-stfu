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
CACHE_ABL  = Path("./cache_abliterated")
STUDY_ROOT = Path("./tribe_study")

# Path to best.ckpt in the HF hub cache — read from the baseline model
# after loading it so we don't hardcode the snapshot hash.
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

# ── Find best.ckpt by loading baseline model once ────────────────────────────

CACHE_BASE = Path("./cache")
print("Loading baseline model to locate best.ckpt...")
model_base = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)

# TribeModel stores the checkpoint path internally; fall back to glob search
ckpt_path = None
for attr in ["ckpt_path", "_ckpt_path", "checkpoint_path"]:
    p = getattr(model_base, attr, None)
    if p is not None:
        ckpt_path = Path(p)
        break
if ckpt_path is None:
    # Glob the HF hub cache for best.ckpt
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    hits = sorted(hub.glob("**/best.ckpt"))
    if hits:
        ckpt_path = hits[0]

if ckpt_path is None or not ckpt_path.exists():
    raise FileNotFoundError(
        "Could not locate best.ckpt. Set ckpt_path manually in the script."
    )
print(f"Found checkpoint: {ckpt_path}")

del model_base
torch.cuda.empty_cache(); gc.collect()

# ── Build abliterated checkpoint ──────────────────────────────────────────────
# Load full best.ckpt, swap in our modified V-JEPA2 weights, save to disk.

print("\nBuilding abliterated checkpoint...")
full_ckpt = torch.load(str(ckpt_path), map_location="cpu")

# The checkpoint state_dict may be nested under "state_dict"
state = full_ckpt.get("state_dict", full_ckpt)

# Load our abliterated V-JEPA2 weights
vjepa2_state = torch.load(OUT_DIR / "vjepa2_abliterated.pt", map_location="cpu")

# Find the prefix used in the checkpoint for V-JEPA2 params
# Strategy: look for any key in state that ends with a key from vjepa2_state
sample_key = next(iter(vjepa2_state))
prefix = None
for ckpt_key in state:
    if ckpt_key.endswith(sample_key):
        prefix = ckpt_key[: -len(sample_key)]
        break

if prefix is None:
    # Fallback: list matching keys to help debug
    print("  Could not auto-detect prefix. Keys containing sample vjepa2 key fragment:")
    frag = sample_key.split(".")[0]
    for k in state:
        if frag in k:
            print(f"    {k}")
    raise RuntimeError(
        f"Cannot find prefix for V-JEPA2 key '{sample_key}' in checkpoint. "
        "Set prefix manually."
    )

print(f"  V-JEPA2 prefix in checkpoint: '{prefix}'")

n_updated = 0
for vkey, vval in vjepa2_state.items():
    full_key = prefix + vkey
    if full_key in state:
        state[full_key] = vval.cpu()
        n_updated += 1
    else:
        print(f"  [WARN] key not found in checkpoint: {full_key}")

print(f"  Updated {n_updated} / {len(vjepa2_state)} V-JEPA2 tensors")

if "state_dict" in full_ckpt:
    full_ckpt["state_dict"] = state
else:
    full_ckpt = state

abl_ckpt_path = OUT_DIR / "best_abliterated.ckpt"
torch.save(full_ckpt, str(abl_ckpt_path))
print(f"  Saved abliterated checkpoint → {abl_ckpt_path}")
print(f"  File size: {abl_ckpt_path.stat().st_size / 1e9:.2f} GB")

# ── Temporarily replace checkpoint, load abliterated model, restore ───────────

ckpt_backup = ckpt_path.with_suffix(".ckpt.bak")

print(f"\nSwapping checkpoint: {ckpt_path.name} → abliterated version")
shutil.copy2(str(ckpt_path), str(ckpt_backup))   # backup
shutil.copy2(str(abl_ckpt_path), str(ckpt_path)) # replace

try:
    # Wipe abliterated cache to ensure no stale feature caches
    if CACHE_ABL.exists():
        shutil.rmtree(CACHE_ABL)
    CACHE_ABL.mkdir(parents=True)

    print("Loading abliterated model (from patched checkpoint)...")
    model_abl = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_ABL)
    print("Abliterated model loaded.\n")

finally:
    # Always restore original checkpoint even if loading crashes
    shutil.copy2(str(ckpt_backup), str(ckpt_path))
    ckpt_backup.unlink()
    print(f"Original checkpoint restored.\n")

# ── Load masks ────────────────────────────────────────────────────────────────

gore_mask = np.load(OUT_DIR / "gore_mask.npy")
porn_mask = np.load(OUT_DIR / "porn_mask.npy")

# ── Validation ────────────────────────────────────────────────────────────────
# Baseline: load from saved tribe_study preds (no re-inference, fast).
# Abliterated: run live inference through the patched model.

val_videos = sorted(VAL_DIR.glob("*.mp4"))
print(f"Found {len(val_videos)} validation videos\n")

results = []

for vp in val_videos:
    print(f"{'='*50}")
    print(f"Video: {vp.name}")
    stem = vp.stem

    # ── Baseline from saved preds if available ───
    # Try to find saved preds from tribe_study for this video
    base_preds_path = None
    for cat_dir in STUDY_ROOT.iterdir():
        candidate = cat_dir / stem / "preds.npy"
        if candidate.exists():
            base_preds_path = candidate
            break

    if base_preds_path:
        preds_base = np.load(base_preds_path)[:30]
        print(f"  Baseline: loaded from {base_preds_path.relative_to(STUDY_ROOT)}")
    else:
        print(f"  Baseline: no saved preds found for {stem}, skipping")
        continue

    # ── Abliterated: live inference ───
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

# ── Save ──────────────────────────────────────────────────────────────────────

for r in results:
    s = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{s}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{s}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{s}_base.npy + val_{s}_abl.npy")

# ── Cleanup ───────────────────────────────────────────────────────────────────
# cache_abliterated is a full copy of cache only needed while model_abl is
# loaded. Remove it now that inference is done.

if CACHE_ABL.exists():
    print(f"\nRemoving temporary cache: {CACHE_ABL} ...")
    shutil.rmtree(CACHE_ABL)
    print("  Done.")

# best_abliterated.ckpt is kept in abliterated/ for reproducibility.
# Delete manually if disk space is tight:
#   rm abliterated/best_abliterated.ckpt
abl_ckpt = OUT_DIR / "best_abliterated.ckpt"
if abl_ckpt.exists():
    size_gb = abl_ckpt.stat().st_size / 1e9
    print(f"\nNote: abliterated/best_abliterated.ckpt is {size_gb:.1f} GB — "
          f"kept for reproducibility.\n"
          f"      Delete it manually if you no longer need to re-run validation.")

print("\nDone.")