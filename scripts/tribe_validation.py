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
ALPHA      = 0.5   # change freely — no re-surgery needed

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

# ── Load masks + baseline preds ───────────────────────────────────────────────

gore_mask = np.load(OUT_DIR / "gore_mask.npy")
porn_mask = np.load(OUT_DIR / "porn_mask.npy")
print(f"Gore mask: {gore_mask.sum()} verts  Porn mask: {porn_mask.sum()} verts")

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

# ── Build ortho directions ────────────────────────────────────────────────────

gore_dirs = np.load(OUT_DIR / "gore_directions.npy")
porn_dirs = np.load(OUT_DIR / "porn_directions.npy")
all_dirs  = np.concatenate([gore_dirs, porn_dirs], axis=0)
dirs_t    = torch.tensor(all_dirs, dtype=torch.float32)
ortho = []
for d in dirs_t:
    for q in ortho: d = d - (d @ q) * q
    if d.norm() > 1e-6: ortho.append(d / d.norm())
ortho_cpu = torch.stack(ortho)   # (n_dirs, 1408)
print(f"\n{len(ortho)} orthogonal directions  alpha={ALPHA}")

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading model...")
model          = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
vjepa2_module  = model.data.video_feature.image.model.model
encoder_blocks = vjepa2_module.encoder.layer
N_LAYERS       = len(encoder_blocks)
TARGET_IDX     = int(N_LAYERS * 0.75)

hf_image        = model.data.video_feature.image
cache_n_layers  = getattr(hf_image, 'cache_n_layers', 20)
first_cached    = N_LAYERS - cache_n_layers
CACHE_LAYER_IDX = TARGET_IDX - first_cached
print(f"N_LAYERS={N_LAYERS} TARGET_IDX={TARGET_IDX} "
      f"cache_n_layers={cache_n_layers} CACHE_LAYER_IDX={CACHE_LAYER_IDX}")
assert 0 <= CACHE_LAYER_IDX < cache_n_layers

# ── Projection function ───────────────────────────────────────────────────────

def project_cache_array(arr):
    """
    arr: numpy array of shape (n_layers, 1408, n_clips)
    Modifies CACHE_LAYER_IDX slice in-place and returns arr.
    """
    sl = torch.from_numpy(arr[CACHE_LAYER_IDX].copy()).float()  # (1408, n_clips)
    for q in ortho_cpu:
        sl = sl - ALPHA * q.unsqueeze(1) * (q @ sl).unsqueeze(0)
    arr[CACHE_LAYER_IDX] = sl.numpy().astype(arr.dtype)
    return arr

# ── Monkeypatch cache_dict.__getitem__ ────────────────────────────────────────
# exca reads cached values via cache_dict[key]. We intercept that read,
# project the abliteration direction out of the returned array, and return
# the modified value. This fires regardless of whether the hook return
# value is used by the caller.

cache_dict   = model.data.video_feature.infra.cache_dict
_real_getitem = cache_dict.__class__.__getitem__
_intercept_keys = set()   # populated with val video keys before each run
_getitem_calls  = [0]
_getitem_hits   = [0]

def _patched_getitem(self, key):
    val = _real_getitem(self, key)
    _getitem_calls[0] += 1
    if key in _intercept_keys:
        _getitem_hits[0] += 1
        # val is a TimedArray — modify its .data numpy array
        if hasattr(val, 'data') and isinstance(val.data, np.ndarray):
            if val.data.ndim == 3 and val.data.shape[0] == cache_n_layers \
                    and val.data.shape[1] == 1408:
                val.data = project_cache_array(val.data.copy())
            else:
                print(f"  [PATCH] unexpected cache shape {val.data.shape} for key {key!r}")
        elif isinstance(val, np.ndarray):
            if val.ndim == 3 and val.shape[0] == cache_n_layers and val.shape[1] == 1408:
                val = project_cache_array(val.copy())
            else:
                print(f"  [PATCH] unexpected ndarray shape {val.shape}")
        else:
            print(f"  [PATCH] unexpected cache value type {type(val)} for key {key!r}")
    return val

cache_dict.__class__.__getitem__ = _patched_getitem
print("cache_dict.__getitem__ monkeypatched")

# ── Populate intercept keys + clear disk cache ────────────────────────────────

item_uid = model.data.video_feature.infra.item_uid
helper   = model.data.video_feature._event_types_helper

print("\nRegistering intercept keys and clearing disk cache...")
for vp in val_videos:
    events = helper.extract(make_video_only_df(vp))
    for ev in events:
        key = item_uid(ev)
        _intercept_keys.add(key)
        # Also remove from in-memory cache to force re-read from disk
        # (so we know the hook path is cache-read → __getitem__ → our patch)
        if key in cache_dict:
            del cache_dict[key]
    print(f"  {vp.name}: intercept key registered")

# Clear disk cache so exca re-runs _HuggingFace.forward, writes to cache,
# then reads back via our patched __getitem__
val_names = {vp.name for vp in val_videos}
disk_cleared = 0
for info_file in CACHE_BASE.rglob("*info.jsonl"):
    try:
        if any(n in info_file.read_text() for n in val_names):
            shutil.rmtree(info_file.parent)
            disk_cleared += 1
    except: pass
print(f"Cleared {disk_cleared} disk cache dirs")

# ── Inference loop ────────────────────────────────────────────────────────────

results = []
try:
    for vp in val_videos:
        stem = vp.stem
        if stem not in baseline_preds:
            print(f"\n[SKIP] {vp.name}")
            continue

        print(f"\n{'='*55}\nVideo: {vp.name}")
        _getitem_calls[0] = 0
        _getitem_hits[0]  = 0

        preds_base = baseline_preds[stem]
        df = make_video_only_df(vp)
        preds_abl, _ = model.predict(events=df)
        preds_abl = preds_abl[:30]

        print(f"  [DIAG] __getitem__ calls={_getitem_calls[0]}  "
              f"intercept hits={_getitem_hits[0]}")
        if _getitem_hits[0] == 0:
            print(f"  [DIAG] WARNING: patch never hit — "
                  f"exca may not read via __getitem__, or key mismatch")
            print(f"  [DIAG] intercept_keys: {list(_intercept_keys)[:2]}")

        print()
        for mask, mname in [(gore_mask,"gore_mask"),(porn_mask,"porn_mask")]:
            base_val = float(preds_base[:,mask].mean())
            abl_val  = float(preds_abl[:,mask].mean())
            diff = abl_val - base_val
            pct  = 100*diff/(abs(base_val)+1e-9)
            tag  = "✓ suppressed" if diff<-0.005 else \
                   ("✗ no change" if abs(diff)<0.005 else "↑ increased")
            print(f"  {mname:12s}  base={base_val:.4f}  abl={abl_val:.4f}  "
                  f"Δ={diff:+.4f} ({pct:+.1f}%)  {tag}")

        bg = float(preds_base.mean()); ag = float(preds_abl.mean())
        print(f"  {'whole_brain':12s}  base={bg:.4f}  abl={ag:.4f}  "
              f"Δ={ag-bg:+.4f} ({100*(ag-bg)/(abs(bg)+1e-9):+.1f}%)")

        torch.cuda.empty_cache(); gc.collect()
        results.append({"video":vp.name,"preds_base":preds_base,
                         "preds_abl":preds_abl,"hits":_getitem_hits[0]})

finally:
    # Restore original __getitem__
    cache_dict.__class__.__getitem__ = _real_getitem
    print("\ncache_dict.__getitem__ restored.")

print("\n"+"="*55+"\nSUMMARY")
for r in results:
    h = r['hits']
    print(f"  {r['video']:20s}  {'patched '+str(h)+'x' if h>0 else 'PATCH DID NOT HIT'}")

for r in results:
    s = Path(r["video"]).stem
    np.save(OUT_DIR/f"val_{s}_base.npy", r["preds_base"])
    np.save(OUT_DIR/f"val_{s}_abl.npy",  r["preds_abl"])
    print(f"Saved val_{s}_*.npy")

print("\nDone.")