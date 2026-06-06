import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tribev2.demo_utils import TribeModel
import gc
import shutil
import subprocess

VAL_DIR  = Path("./val_data")
OUT_DIR  = Path("./abliterated")
CACHE_BASE = Path("./cache")
CACHE_ABL  = Path("./cache_abliterated")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True
    )
    duration = float(result.stdout.strip()) if result.stdout.strip() else 30.0
    return round(duration - 0.1, 3)

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

# ── Load baseline model (uses original cache) ─────────────────────────────────

print("Loading baseline model...")
model_base = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)

# ── Load abliterated model with a clean, separate cache ───────────────────────
# Always wipe and re-copy so no stale exca activations from pre-patch runs
# survive into model_abl.predict(). The HF model files themselves are just
# symlinks/small manifests so the copy is fast; the exca feature caches are
# what we need absent so the patched weights actually run a fresh forward pass.

print("Wiping cache_abliterated and re-copying from cache_base...")
if CACHE_ABL.exists():
    shutil.rmtree(CACHE_ABL)
shutil.copytree(CACHE_BASE, CACHE_ABL)

# Remove any exca/feature-cache subdirs that got copied over — we want
# model downloads preserved but all video-feature caches gone so predict()
# re-runs the encoder with the abliterated weights.
for d in list(CACHE_ABL.iterdir()):
    if d.is_dir() and d.name not in ("hub", "models--facebook--tribev2"):
        shutil.rmtree(d)
        print(f"  Cleared: {d.name[:80]}")

print("Loading abliterated model...")
model_abl = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_ABL)

# Patch abliterated V-JEPA2 weights into the in-memory model
vjepa2_abl = model_abl.data.video_feature.image.model.model
vjepa2_base = model_base.data.video_feature.image.model.model
state_dict  = torch.load(OUT_DIR / "vjepa2_abliterated.pt", map_location=DEVICE)
vjepa2_abl.load_state_dict(state_dict)

# ── Diagnostic 1: confirm weights actually differ ─────────────────────────────
print("=== DIAGNOSTIC: weight diff check ===")
total_diff = 0.0
total_params = 0
for (name_b, p_b), (name_a, p_a) in zip(
        vjepa2_base.named_parameters(), vjepa2_abl.named_parameters()):
    diff = (p_b.data - p_a.data).abs().sum().item()
    total_diff += diff
    total_params += p_b.numel()
    if diff > 0:
        print(f"  CHANGED: {name_b}  |Δ|={diff:.6f}")
if total_diff == 0:
    print("  !! ZERO DIFF — vjepa2_abliterated.pt is identical to baseline weights")
    print("     Weight surgery likely failed (block.attention.proj not found)")
    print("     Run abliteration script again and check for AttributeError")
else:
    print(f"  Total |Δ| across {total_params} params: {total_diff:.4f}  ✓ weights differ")
print()

# ── Diagnostic 2: confirm model_abl.predict routes through vjepa2_abl ─────────
print("=== DIAGNOSTIC: forward hook routing check ===")
_abl_hook_fired = []
def _routing_hook(module, input, output):
    _abl_hook_fired.append(True)
_hook = vjepa2_abl.register_forward_hook(_routing_hook)

print("Abliterated weights patched.\n")

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
    print(f"  Duration: {df['duration'].values[0]:.3f}s")

    print("  Running baseline...")
    preds_base, _ = model_base.predict(events=df)
    preds_base = preds_base[:30]
    torch.cuda.empty_cache(); gc.collect()

    _abl_hook_fired.clear()
    print("  Running abliterated...")
    preds_abl, _  = model_abl.predict(events=df)
    preds_abl = preds_abl[:30]
    torch.cuda.empty_cache(); gc.collect()

    if _abl_hook_fired:
        print(f"  [ROUTING ✓] vjepa2_abl.forward() called {len(_abl_hook_fired)}x")
    else:
        print("  [ROUTING ✗] vjepa2_abl.forward() was NEVER called — predict()")
        print("              bypasses the patched model (cached or different code path)")

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
    stem = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{stem}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{stem}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{stem}_base.npy + val_{stem}_abl.npy")

print("\nDone.")