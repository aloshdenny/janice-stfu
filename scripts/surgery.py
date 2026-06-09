import os, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tribev2.demo_utils import TribeModel
import gc, subprocess, shutil

VAL_DIR    = Path("./val_data")
OUT_DIR    = Path("./abliterated")
CACHE_BASE = Path("./cache")
STUDY_ROOT = Path("./tribe_study")

def get_duration(video_path):
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","default=noprint_wrappers=1:nokey=1",str(video_path)],
                       capture_output=True, text=True)
    try: return round(float(r.stdout.strip()) - 0.1, 3)
    except: return 29.9

def make_video_only_df(video_path):
    dur = get_duration(video_path)
    return pd.DataFrame([{"type":"Video","start":0.0,"duration":dur,
        "timeline":"default","subject":"default","session":"","task":"","run":"",
        "filepath":str(video_path.resolve()),"frequency":60.0,"offset":0.0,
        "stop":dur,"context":float("nan")}])

# ── Load masks + baseline preds ───────────────────────────────────────────────

gore_mask = np.load(OUT_DIR / "gore_mask.npy")
porn_mask = np.load(OUT_DIR / "porn_mask.npy")
print(f"Gore mask: {gore_mask.sum()} verts   Porn mask: {porn_mask.sum()} verts")

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
        print(f"  [WARN] {vp.name}: no baseline preds found in tribe_study")

# ── Step 1: load model ────────────────────────────────────────────────────────

print("\nLoading model...")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)

vjepa2     = model.data.video_feature.image.model.model
N_LAYERS   = len(vjepa2.encoder.layer)
TARGET_IDX = int(N_LAYERS * 0.75)
print(f"V-JEPA2: {N_LAYERS} layers, target={TARGET_IDX}")

# ── Step 2: wipe exca activation cache entirely ───────────────────────────────
# exca stores (20, 1408, n_clips) float32 memmap arrays keyed by video path.
# We delete the uid_folder (config-hash-specific subdirectory) so exca is
# forced to recompute all entries from scratch using the current model weights.

infra      = model.data.video_feature.infra
cache_dict = infra.cache_dict
uid_folder = Path(str(infra.uid_folder))

# clear in-memory cache
n_mem = len(list(cache_dict.keys()))
for k in list(cache_dict.keys()):
    del cache_dict[k]
print(f"\nCleared {n_mem} in-memory cache entries")

# clear disk cache
if uid_folder.exists():
    shutil.rmtree(uid_folder)
    print(f"Deleted disk cache: {uid_folder}")
else:
    # fallback: scan for any .data files matching val video names
    val_names  = {vp.name for vp in val_videos}
    n_disk = 0
    for info_file in CACHE_BASE.rglob("*info.jsonl"):
        try:
            if any(n in info_file.read_text() for n in val_names):
                shutil.rmtree(info_file.parent)
                n_disk += 1
        except Exception:
            pass
    print(f"Fallback disk clear: removed {n_disk} cache dirs")

# ── Step 3: load abliterated weights and VERIFY they changed ─────────────────

abl_path = OUT_DIR / "vjepa2_abliterated.pt"
if not abl_path.exists():
    raise FileNotFoundError(
        f"{abl_path} not found. Re-run: python surgery.py --alpha 0.5 --n_components 1"
    )

# snapshot a weight we know was surgically modified
w_before = vjepa2.encoder.layer[TARGET_IDX].attention.value.weight.data.clone()

print(f"\nLoading abliterated weights from {abl_path} ...")
abl_state = torch.load(abl_path, map_location="cpu")
missing, unexpected = vjepa2.load_state_dict(abl_state, strict=False)

if missing:
    print(f"  WARNING: {len(missing)} missing keys  (sample: {missing[:3]})")
if unexpected:
    print(f"  WARNING: {len(unexpected)} unexpected keys (sample: {unexpected[:3]})")

w_after = vjepa2.encoder.layer[TARGET_IDX].attention.value.weight.data
diff    = (w_before - w_after).abs().max().item()
print(f"  Max weight delta at encoder.layer[{TARGET_IDX}].attention.value: {diff:.8f}")

if diff < 1e-8:
    raise RuntimeError(
        "FATAL: abliterated weights did NOT load — weight tensors are identical.\n"
        "Likely cause: state dict keys do not match.\n"
        "Fix: re-run surgery.py which saves from the same vjepa2_module object.\n"
        f"State dict path: {abl_path}\n"
        f"Missing keys: {missing[:5]}"
    )

print(f"  Weights successfully modified (delta={diff:.6f})")

# ── Step 4: run inference — exca recomputes from scratch ─────────────────────
# Cache is empty; exca calls _HuggingFace.forward() which uses the current
# (abliterated) weights. Results are cached and then fed to the regression head.

results = []

for vp in val_videos:
    stem = vp.stem
    if stem not in baseline_preds:
        print(f"\n[SKIP] no baseline for {vp.name}")
        continue

    print(f"\n{'='*55}\nVideo: {vp.name}")

    preds_base = baseline_preds[stem]
    df         = make_video_only_df(vp)
    print(f"  Duration: {df['duration'].values[0]:.3f}s")
    print("  Running abliterated inference...")

    preds_abl, _ = model.predict(events=df)
    preds_abl = preds_abl[:30]

    print()
    for mask, mname in [(gore_mask,"gore_mask"),(porn_mask,"porn_mask")]:
        base_val = float(preds_base[:,mask].mean())
        abl_val  = float(preds_abl[:,mask].mean())
        diff_val = abl_val - base_val
        pct      = 100 * diff_val / (abs(base_val) + 1e-9)
        tag      = "✓ suppressed" if diff_val < -0.005 else \
                   ("✗ no change"  if abs(diff_val) < 0.005 else "↑ increased")
        print(f"  {mname:12s}  base={base_val:.4f}  abl={abl_val:.4f}  "
              f"Δ={diff_val:+.4f} ({pct:+.1f}%)  {tag}")

    bg = float(preds_base.mean())
    ag = float(preds_abl.mean())
    print(f"  {'whole_brain':12s}  base={bg:.4f}  abl={ag:.4f}  "
          f"Δ={ag-bg:+.4f} ({100*(ag-bg)/(abs(bg)+1e-9):+.1f}%)")

    torch.cuda.empty_cache(); gc.collect()
    results.append({"video":vp.name,"preds_base":preds_base,"preds_abl":preds_abl})

# ── Save ──────────────────────────────────────────────────────────────────────

print("\n" + "="*55 + "\nSaving results...")
for r in results:
    s = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{s}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{s}_abl.npy",  r["preds_abl"])
    print(f"  Saved val_{s}_base.npy + val_{s}_abl.npy")

print("\nDone.")