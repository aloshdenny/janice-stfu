import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tribev2.demo_utils import TribeModel
import gc
import shutil
import subprocess
import os

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

del model_base
torch.cuda.empty_cache(); gc.collect()

# ── Phase 2: locate best.ckpt (now guaranteed to exist after predict()) ───────

print("\nLocating best.ckpt...")
ckpt_path = find_ckpt()
if ckpt_path is None or not ckpt_path.exists():
    raise FileNotFoundError(
        "Could not locate best.ckpt in HF hub cache after running baseline inference.\n"
        "Set ckpt_path manually at the top of the script:\n"
        "  ckpt_path = Path('/path/to/best.ckpt')"
    )
print(f"  Found: {ckpt_path}")

# ── Phase 3: build abliterated checkpoint ─────────────────────────────────────

abl_ckpt_path = OUT_DIR / "best_abliterated.ckpt"

# Skip rebuild if already done (saves ~5 min of torch.save time)
if abl_ckpt_path.exists():
    print(f"\nAbliterated checkpoint already exists ({abl_ckpt_path.stat().st_size/1e9:.2f} GB) — reusing.")
else:
    print("\nBuilding abliterated checkpoint...")
    full_ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = full_ckpt.get("state_dict", full_ckpt)

    vjepa2_state = torch.load(OUT_DIR / "vjepa2_abliterated.pt", map_location="cpu")

    # Auto-detect the checkpoint key prefix for V-JEPA2 params
    sample_key = next(iter(vjepa2_state))
    prefix = None
    for ckpt_key in state:
        if ckpt_key.endswith(sample_key):
            prefix = ckpt_key[: -len(sample_key)]
            break

    if prefix is None:
        frag = sample_key.split(".")[0]
        print(f"  Could not auto-detect prefix. Keys containing '{frag}':")
        for k in state:
            if frag in k:
                print(f"    {k}")
        raise RuntimeError(
            f"Cannot find V-JEPA2 prefix in checkpoint for key '{sample_key}'.\n"
            "Set prefix manually and re-run."
        )

    print(f"  V-JEPA2 prefix: '{prefix}'")
    n_updated = 0
    for vkey, vval in vjepa2_state.items():
        full_key = prefix + vkey
        if full_key in state:
            state[full_key] = vval.cpu()
            n_updated += 1
        else:
            print(f"  [WARN] key not found: {full_key}")

    print(f"  Updated {n_updated} / {len(vjepa2_state)} V-JEPA2 tensors")

    if "state_dict" in full_ckpt:
        full_ckpt["state_dict"] = state
    else:
        full_ckpt = state

    torch.save(full_ckpt, str(abl_ckpt_path))
    print(f"  Saved → {abl_ckpt_path}  ({abl_ckpt_path.stat().st_size/1e9:.2f} GB)")
    del full_ckpt, state, vjepa2_state
    torch.cuda.empty_cache(); gc.collect()

# ── Phase 4: swap checkpoint, load abliterated model, restore ─────────────────

ckpt_backup = ckpt_path.with_suffix(".ckpt.bak")
print(f"\nSwapping {ckpt_path.name} → abliterated version...")
shutil.copy2(str(ckpt_path), str(ckpt_backup))
shutil.copy2(str(abl_ckpt_path), str(ckpt_path))

try:
    if CACHE_ABL.exists():
        shutil.rmtree(CACHE_ABL)
    CACHE_ABL.mkdir(parents=True)

    print("Loading abliterated model (from patched checkpoint)...")
    model_abl = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_ABL)
    print("Abliterated model loaded.\n")

    # ── Phase 5: abliterated inference + compare ──────────────────────────────

    results = []

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
    # Always restore original checkpoint
    shutil.copy2(str(ckpt_backup), str(ckpt_path))
    ckpt_backup.unlink()
    print("Original checkpoint restored.")

# ── Save ──────────────────────────────────────────────────────────────────────

for r in results:
    s = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{s}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{s}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{s}_base.npy + val_{s}_abl.npy")

# ── Cleanup ───────────────────────────────────────────────────────────────────

if CACHE_ABL.exists():
    print(f"\nRemoving temporary cache_abliterated...")
    shutil.rmtree(CACHE_ABL)

print(f"\nNote: abliterated/best_abliterated.ckpt kept for reproducibility.")
print("      Delete manually if disk space is tight: rm abliterated/best_abliterated.ckpt")
print("\nDone.")