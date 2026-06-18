"""
validation.py — Validates abliterated model against baseline preds.

Approach: Load the abliterated checkpoint directly into the vjepa2 encoder,
clear all caches, and run inference. Compare against baseline preds from
tribe_study/. No monkeypatching — works for surgery on ANY layer.
"""

import os, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tribev2.demo_utils import TribeModel
import gc, subprocess, shutil

# ── Parse Command Line Arguments ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Validate abliterated model.")
parser.add_argument("--target", type=str, required=True,
                    help="Target category (e.g., 'porn')")
args = parser.parse_args()

VAL_DIR    = Path("./val_data")
OUT_DIR    = Path("./abliterated")
CACHE_BASE = Path("./cache")
STUDY_ROOT = Path("./tribe_study")

# ── Load surgery log ──────────────────────────────────────────────────────────

surgery_log_path = OUT_DIR / "surgery_log.json"
if not surgery_log_path.exists():
    raise FileNotFoundError(f"surgery_log.json not found in {OUT_DIR}. Run abliteration.py first.")

with open(surgery_log_path) as f:
    surgery_log = json.load(f)

OPERATED_LAYERS = sorted([int(k) for k in surgery_log["layers_operated"].keys()])
TOLERANCE = surgery_log["tolerance"]

print(f"Target: {args.target}")
print(f"Tolerance: {TOLERANCE}")
print(f"Operated layers: {OPERATED_LAYERS}")
print(f"Mask: {surgery_log['mask_used']} (score={surgery_log['mask_score']:.6f})")
print(f"Components per layer: {surgery_log['n_components']}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_duration(video_path):
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","default=noprint_wrappers=1:nokey=1",str(video_path)],
                       capture_output=True,text=True)
    try: return round(float(r.stdout.strip())-0.1,3)
    except: return 29.9

def make_video_only_df(video_path):
    dur = get_duration(video_path)
    return pd.DataFrame([{"type":"Video","start":0.0,"duration":dur,
        "timeline":"default","subject":"default","session":"","task":"","run":"",
        "filepath":str(video_path.resolve()),"frequency":60.0,"offset":0.0,
        "stop":dur,"context":float("nan")}])

# ── Load Target Mask + baseline preds ─────────────────────────────────────────

target_mask_path = OUT_DIR / f"{args.target}_mask.npy"
if not target_mask_path.exists():
    raise FileNotFoundError(f"Mask file not found: {target_mask_path}")

target_mask = np.load(target_mask_path)
print(f"\nTarget ({args.target}) mask: {target_mask.sum()} verts")

val_videos = sorted(VAL_DIR.glob("*.mp4"))
print(f"Found {len(val_videos)} validation videos")

baseline_preds = {}
for vp in val_videos:
    for cat_dir in STUDY_ROOT.iterdir():
        if not cat_dir.is_dir(): continue
        cand = cat_dir / vp.stem / "preds.npy"
        if cand.exists():
            baseline_preds[vp.stem] = np.load(cand)[:30]
            print(f"  {vp.name}: baseline from {cand.relative_to(STUDY_ROOT)}")
            break
    else:
        print(f"  [WARN] {vp.name}: no baseline")

# ── Load model + abliterated weights ──────────────────────────────────────────

abliterated_ckpt = OUT_DIR / "vjepa2_abliterated.pt"
if not abliterated_ckpt.exists():
    raise FileNotFoundError(f"Abliterated checkpoint not found: {abliterated_ckpt}")

print(f"\nLoading model...")
model          = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
vjepa2_module  = model.data.video_feature.image.model.model
encoder_blocks = vjepa2_module.encoder.layer
N_LAYERS       = len(encoder_blocks)

# Load the abliterated weights directly into the encoder
print(f"Loading abliterated checkpoint: {abliterated_ckpt.name}")
state_dict = torch.load(abliterated_ckpt, map_location="cpu")
vjepa2_module.load_state_dict(state_dict)
del state_dict
vjepa2_module.eval()
print(f"  Loaded — surgery on layers {OPERATED_LAYERS} is baked into the weights")

# ── Clear ALL caches so inference uses the modified weights ───────────────────

# 1. Clear in-memory cache
cache_dict = model.data.video_feature.infra.cache_dict
item_uid   = model.data.video_feature.infra.item_uid
helper     = model.data.video_feature._event_types_helper

print("\nClearing caches for validation videos...")
for vp in val_videos:
    events = helper.extract(make_video_only_df(vp))
    for ev in events:
        key = item_uid(ev)
        if key in cache_dict:
            del cache_dict[key]

# 2. Clear disk cache
val_names = {vp.name for vp in val_videos}
disk_cleared = 0
info_files = list(CACHE_BASE.rglob("*info.jsonl"))
for info_file in info_files:
    try:
        if info_file.exists() and any(n in info_file.read_text() for n in val_names):
            shutil.rmtree(info_file.parent)
            disk_cleared += 1
    except Exception:
        pass
print(f"  In-memory: cleared entries for {len(val_videos)} videos")
print(f"  Disk: cleared {disk_cleared} cache dirs")

# ── Inference loop ────────────────────────────────────────────────────────────

results = []
for vp in val_videos:
    stem = vp.stem
    if stem not in baseline_preds:
        print(f"\n[SKIP] {vp.name}")
        continue

    print(f"\n{'='*55}\nVideo: {vp.name}")

    preds_base = baseline_preds[stem]
    df = make_video_only_df(vp)
    preds_abl, _ = model.predict(events=df)
    preds_abl = preds_abl[:30]

    # Target mask evaluation
    base_val = float(preds_base[:,target_mask].mean())
    abl_val  = float(preds_abl[:,target_mask].mean())
    diff = abl_val - base_val
    pct  = 100*diff/(abs(base_val)+1e-9)
    tag  = "✓ suppressed" if diff<-0.005 else \
           ("✗ no change" if abs(diff)<0.005 else "↑ increased")
    print(f"  {args.target + '_mask':12s}  base={base_val:.4f}  abl={abl_val:.4f}  "
          f"Δ={diff:+.4f} ({pct:+.1f}%)  {tag}")

    # Whole brain
    bg = float(preds_base.mean()); ag = float(preds_abl.mean())
    print(f"  {'whole_brain':12s}  base={bg:.4f}  abl={ag:.4f}  "
          f"Δ={ag-bg:+.4f} ({100*(ag-bg)/(abs(bg)+1e-9):+.1f}%)")

    torch.cuda.empty_cache(); gc.collect()
    results.append({"video":vp.name,"preds_base":preds_base,
                     "preds_abl":preds_abl})

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n"+"="*55+"\nSUMMARY")
print(f"  Checkpoint: {abliterated_ckpt.name}")
print(f"  Layers operated: {OPERATED_LAYERS}")
print(f"  Mask: {surgery_log['mask_used']} ({int(target_mask.sum())} verts)")
print(f"  Tolerance: {TOLERANCE}")
print()

all_diffs = []
for r in results:
    base_val = float(r["preds_base"][:,target_mask].mean())
    abl_val  = float(r["preds_abl"][:,target_mask].mean())
    diff = abl_val - base_val
    pct = 100*diff/(abs(base_val)+1e-9)
    tag  = "✓" if diff<-0.005 else ("✗" if abs(diff)<0.005 else "↑")
    print(f"  {r['video']:20s}  Δ={diff:+.4f} ({pct:+.1f}%)  {tag}")
    all_diffs.append(diff)

if all_diffs:
    mean_diff = np.mean(all_diffs)
    print(f"\n  Mean Δ across all videos: {mean_diff:+.4f}")

    # Separate by category
    target_diffs = [d for r, d in zip(results, all_diffs)
                    if r["video"].startswith(args.target)]
    other_diffs  = [d for r, d in zip(results, all_diffs)
                    if not r["video"].startswith(args.target)]
    if target_diffs:
        print(f"  Mean Δ on {args.target} videos: {np.mean(target_diffs):+.4f}")
    if other_diffs:
        print(f"  Mean Δ on other videos:  {np.mean(other_diffs):+.4f}")

# Save results
for r in results:
    s = Path(r["video"]).stem
    np.save(OUT_DIR/f"val_{s}_base.npy", r["preds_base"])
    np.save(OUT_DIR/f"val_{s}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{s}_*.npy")

print("\nDone.")