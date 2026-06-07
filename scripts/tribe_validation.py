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

# ── Phase 2: find V-JEPA2 weights file in cache ───────────────────────────────
# best.ckpt is TRIBEv2's regression model (x-transformers keys).
# V-JEPA2 has its own separate weights file — loaded by the neuralset extractor
# from ./cache with 843 tensors. We scan for it by checking .pt/.bin files for
# our known V-JEPA2 keys.

print("\nScanning cache for V-JEPA2 weights file...")
vjepa2_state = torch.load(OUT_DIR / "vjepa2_abliterated.pt", map_location="cpu")
sample_keys  = set(list(vjepa2_state.keys())[:8])  # first 8 keys as fingerprint

vjepa2_cache_file = None
for candidate in sorted(CACHE_BASE.rglob("*.pt")) + sorted(CACHE_BASE.rglob("*.bin")):
    try:
        ckpt = torch.load(str(candidate), map_location="cpu")
        if not isinstance(ckpt, dict):
            continue
        # Use state_dict if nested, otherwise use directly
        state = ckpt.get("state_dict", ckpt)
        if not isinstance(state, dict):
            continue
        # Check how many of our sample keys are present (with any prefix)
        state_suffixes = {k.split(".", 1)[-1] if "." in k else k for k in state}
        matches = sample_keys & state_suffixes
        if len(matches) >= 4:
            vjepa2_cache_file = candidate
            print(f"  Found V-JEPA2 weights: {candidate.relative_to(CACHE_BASE)}  ({len(state)} tensors)")
            break
    except Exception:
        continue

if vjepa2_cache_file is None:
    # Fallback: look for any large .pt file (>500 MB) — V-JEPA2-Large is ~2 GB
    for candidate in sorted(CACHE_BASE.rglob("*.pt")):
        if candidate.stat().st_size > 500_000_000:
            vjepa2_cache_file = candidate
            print(f"  Fallback: using largest .pt file: {candidate.relative_to(CACHE_BASE)}")
            break

if vjepa2_cache_file is None:
    raise FileNotFoundError(
        "Could not find V-JEPA2 weights file in cache/.\n"
        "Set vjepa2_cache_file manually:\n"
        "  vjepa2_cache_file = Path('./cache/path/to/vjepa2.pt')"
    )

# ── Phase 3: build patched V-JEPA2 cache file ────────────────────────────────

abl_vjepa2_cache = OUT_DIR / ("abl_" + vjepa2_cache_file.name)

if abl_vjepa2_cache.exists():
    print(f"\nAbliterated V-JEPA2 cache already exists — reusing: {abl_vjepa2_cache.name}")
else:
    print(f"\nBuilding abliterated V-JEPA2 cache file...")
    full_ckpt = torch.load(str(vjepa2_cache_file), map_location="cpu")
    state     = full_ckpt.get("state_dict", full_ckpt)
    is_nested = "state_dict" in full_ckpt

    # Detect prefix
    sample_key = next(iter(vjepa2_state))
    prefix = None
    for ckpt_key in state:
        if ckpt_key.endswith(sample_key):
            prefix = ckpt_key[: -len(sample_key)]
            break
    if prefix is None:
        # Try with no prefix (direct key match)
        if sample_key in state:
            prefix = ""
    if prefix is None:
        print("  Could not auto-detect prefix. Sample state keys:")
        for k in list(state.keys())[:20]:
            print(f"    {k}")
        raise RuntimeError(
            f"Cannot find V-JEPA2 prefix for key '{sample_key}'.\n"
            "Set prefix manually."
        )

    print(f"  V-JEPA2 prefix in cache file: '{prefix}'")
    n_updated = 0
    for vkey, vval in vjepa2_state.items():
        full_key = prefix + vkey
        if full_key in state:
            state[full_key] = vval.cpu()
            n_updated += 1
        else:
            print(f"  [WARN] key not found: {full_key}")

    print(f"  Updated {n_updated} / {len(vjepa2_state)} tensors")
    if is_nested:
        full_ckpt["state_dict"] = state
    else:
        full_ckpt = state

    torch.save(full_ckpt, str(abl_vjepa2_cache))
    print(f"  Saved → {abl_vjepa2_cache}  ({abl_vjepa2_cache.stat().st_size/1e9:.2f} GB)")
    del full_ckpt, state
    torch.cuda.empty_cache(); gc.collect()

# ── Phase 4: swap V-JEPA2 weights file, run abliterated model, restore ────────

vjepa2_backup = vjepa2_cache_file.with_suffix(".pt.bak")
print(f"\nSwapping V-JEPA2 weights in cache: {vjepa2_cache_file.name} → abliterated")
shutil.copy2(str(vjepa2_cache_file), str(vjepa2_backup))
shutil.copy2(str(abl_vjepa2_cache), str(vjepa2_cache_file))

try:
    # Use CACHE_BASE (not a fresh empty dir) — all other cached features stay;
    # only the V-JEPA2 weights file on disk has been swapped.
    print("Loading abliterated model (V-JEPA2 weights patched on disk)...")
    model_abl = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
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
    # Always restore original V-JEPA2 weights in cache
    shutil.copy2(str(vjepa2_backup), str(vjepa2_cache_file))
    vjepa2_backup.unlink()
    print("Original V-JEPA2 weights restored in cache.")

# ── Save ──────────────────────────────────────────────────────────────────────

for r in results:
    s = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{s}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{s}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{s}_base.npy + val_{s}_abl.npy")

# ── Cleanup ───────────────────────────────────────────────────────────────────

if CACHE_ABL.exists():
    shutil.rmtree(CACHE_ABL)

abl_vjepa2 = OUT_DIR / ("abl_" + vjepa2_cache_file.name)
if abl_vjepa2.exists():
    size_gb = abl_vjepa2.stat().st_size / 1e9
    print(f"\nNote: abliterated/{abl_vjepa2.name} is {size_gb:.1f} GB — kept for reproducibility.")
    print(f"      Delete manually if tight on disk: rm abliterated/{abl_vjepa2.name}")

print("\nDone.")