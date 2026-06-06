import numpy as np
import torch
from pathlib import Path
from tribev2.demo_utils import TribeModel
import gc

CACHE_DIR  = Path("./cache")
VAL_DIR    = Path("./val_data")
OUT_DIR    = Path("./abliterated")
STUDY_ROOT = Path("./tribe_study")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load model twice: baseline and abliterated ────────────────────────────────

print("Loading baseline model...")
model_base = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)

print("Loading abliterated model...")
model_abl  = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)

# Patch abliterated weights into model_abl's V-JEPA2
vjepa2_abl = model_abl.data.video_feature.image.model.model
state_dict  = torch.load(OUT_DIR / "vjepa2_abliterated.pt", map_location=DEVICE)
vjepa2_abl.load_state_dict(state_dict)
print("Abliterated weights loaded.")

# ── Load masks ────────────────────────────────────────────────────────────────

gore_mask = np.load(OUT_DIR / "gore_mask.npy")
porn_mask = np.load(OUT_DIR / "porn_mask.npy")

# ── Inference both models on each val video ───────────────────────────────────

val_videos = sorted(VAL_DIR.glob("*.mp4"))
print(f"\nFound {len(val_videos)} validation videos\n")

results = []

for vp in val_videos:
    print(f"{'='*50}")
    print(f"Video: {vp.name}")

    df = model_base.get_events_dataframe(video_path=vp)

    print("  Running baseline...")
    preds_base, _ = model_base.predict(events=df)
    preds_base = preds_base[:30]

    torch.cuda.empty_cache(); gc.collect()

    print("  Running abliterated...")
    preds_abl, _  = model_abl.predict(events=df)
    preds_abl = preds_abl[:30]

    torch.cuda.empty_cache(); gc.collect()

    # Per-mask stats
    for mask, mname in [(gore_mask, "gore_mask"), (porn_mask, "porn_mask")]:
        base_val = float(preds_base[:, mask].mean())
        abl_val  = float(preds_abl[:, mask].mean())
        diff     = abl_val - base_val
        pct      = 100 * diff / (abs(base_val) + 1e-9)
        print(f"  {mname}: baseline={base_val:.4f}  abliterated={abl_val:.4f}  "
              f"Δ={diff:+.4f}  ({pct:+.1f}%)")

    # Whole-brain stats — confirm global activity isn't destroyed
    base_global = float(preds_base.mean())
    abl_global  = float(preds_abl.mean())
    print(f"  whole_brain: baseline={base_global:.4f}  abliterated={abl_global:.4f}  "
          f"Δ={abl_global-base_global:+.4f}")

    results.append({
        "video":        vp.name,
        "preds_base":   preds_base,
        "preds_abl":    preds_abl,
    })
    print()

# ── Save for visualization ────────────────────────────────────────────────────

for r in results:
    stem = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{stem}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{stem}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{stem}_base.npy + val_{stem}_abl.npy")

print("\nDone.")