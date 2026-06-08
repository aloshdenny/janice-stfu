"""
find_cache_intercept.py — Find where exca caches V-JEPA2 output.

741 hooks on vjepa2_module fired 0 times → exca returns cached features
before V-JEPA2 ever runs. This script walks the full model object tree
(not just nn.Module) to find the cache read path and the correct hook point.
"""

import os, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tribev2.demo_utils import TribeModel
import subprocess, gc

VAL_DIR    = Path("./val_data")
OUT_DIR    = Path("./abliterated")
CACHE_BASE = Path("./cache")
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

def get_duration(video_path):
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True,
    )
    try:
        return round(float(result.stdout.strip()) - 0.1, 3)
    except Exception:
        return 29.9

def make_video_only_df(video_path):
    duration = get_duration(video_path)
    return pd.DataFrame([{
        "type": "Video", "start": 0.0, "duration": duration,
        "timeline": "default", "subject": "default",
        "session": "", "task": "", "run": "",
        "filepath": str(video_path.resolve()),
        "frequency": 60.0, "offset": 0.0, "stop": duration,
        "context": float("nan"),
    }])

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading model...")
model         = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
vjepa2_module = model.data.video_feature.image.model.model

# ── Walk full object tree to find all nn.Modules ──────────────────────────────

print("\n=== ALL nn.Module instances reachable from model.data ===")

def walk_for_modules(obj, path="model", visited=None, depth=0, max_depth=8):
    if visited is None:
        visited = set()
    obj_id = id(obj)
    if obj_id in visited or depth > max_depth:
        return
    visited.add(obj_id)

    if isinstance(obj, torch.nn.Module):
        params = sum(p.numel() for p in obj.parameters())
        print(f"  {'  '*depth}[nn.Module] {path}  ({type(obj).__name__})  params={params:,}")

    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(v, (int, float, str, bool, bytes, type(None))):
                walk_for_modules(v, f"{path}[{k!r}]", visited, depth+1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            if not isinstance(v, (int, float, str, bool, bytes, type(None))):
                walk_for_modules(v, f"{path}[{i}]", visited, depth+1, max_depth)
    else:
        try:
            attrs = vars(obj) if not isinstance(obj, torch.nn.Module) else {}
        except TypeError:
            attrs = {}
        for k, v in attrs.items():
            if k.startswith("_") and k not in ("_modules", "_parameters"):
                continue
            if callable(v) and not isinstance(v, torch.nn.Module):
                continue
            if not isinstance(v, (int, float, str, bool, bytes, type(None))):
                walk_for_modules(v, f"{path}.{k}", visited, depth+1, max_depth)

walk_for_modules(model.data, path="model.data", max_depth=6)

# ── Inspect the infra / cache objects directly ────────────────────────────────

print("\n=== EXCA INFRA INSPECTION ===")
infra = model.data.video_feature.infra
print(f"infra type: {type(infra)}")
print(f"infra attrs: {[a for a in dir(infra) if not a.startswith('__')]}")

cache_dict = None
try:
    cache_dict = infra.cache_dict
    print(f"\ncache_dict type: {type(cache_dict)}")
    keys = list(cache_dict.keys())
    print(f"cache_dict key count: {len(keys)}")
    if keys:
        k0 = keys[0]
        v0 = cache_dict[k0]
        print(f"\nSample key: {k0!r}")
        print(f"Sample value type: {type(v0)}")
        if isinstance(v0, np.ndarray):
            print(f"  shape={v0.shape}  dtype={v0.dtype}")
        elif isinstance(v0, torch.Tensor):
            print(f"  shape={v0.shape}  dtype={v0.dtype}")
        elif isinstance(v0, dict):
            for kk, vv in v0.items():
                if hasattr(vv, 'shape'):
                    print(f"  [{kk}]: shape={vv.shape}  dtype={vv.dtype}")
                else:
                    print(f"  [{kk}]: {type(vv)}")
        else:
            print(f"  value: {str(v0)[:200]}")
except Exception as e:
    print(f"cache_dict inspection failed: {e}")

# ── Check .data files ────────────────────────────────────────────────────────

print("\n=== EXCA DISK CACHE FILES ===")
data_files = sorted(CACHE_BASE.rglob("*.data"))
print(f"Found {len(data_files)} .data files")
for f in data_files[:10]:
    size_mb = f.stat().st_size / 1024**2
    print(f"  {f.relative_to(CACHE_BASE)}  ({size_mb:.1f} MB)")

# ── Hook all nn.Modules NOT inside vjepa2_module ─────────────────────────────

print("\n=== HOOKING NON-VJEPA2 nn.Modules DURING predict() ===")

vjepa2_ids = {id(m) for m in vjepa2_module.modules()}
fired_outside = {}
hooks = []
seen_ids = set()

def make_hook(name):
    def h(module, inp, out):
        fired_outside[name] = fired_outside.get(name, 0) + 1
        if fired_outside[name] == 1:
            in_shapes = []
            for x in (inp if isinstance(inp, (list, tuple)) else [inp]):
                in_shapes.append(x.shape if isinstance(x, torch.Tensor) else type(x).__name__)
            out_shapes = []
            for x in (out if isinstance(out, (list, tuple)) else [out]):
                out_shapes.append(x.shape if isinstance(x, torch.Tensor) else type(x).__name__)
            print(f"    FIRED: {name}  in={in_shapes}  out={out_shapes}")
    return h

def enqueue_children(obj, path):
    obj_id = id(obj)
    if obj_id in seen_ids:
        return
    seen_ids.add(obj_id)
    if isinstance(obj, torch.nn.Module):
        if id(obj) not in vjepa2_ids:
            h = obj.register_forward_hook(make_hook(path))
            hooks.append(h)
        for name, child in obj.named_children():
            enqueue_children(child, f"{path}.{name}")
    try:
        attrs = vars(obj) if not isinstance(obj, torch.nn.Module) else {}
        for k, v in attrs.items():
            if k.startswith("_"):
                continue
            if isinstance(v, torch.nn.Module):
                enqueue_children(v, f"{path}.{k}")
    except Exception:
        pass

enqueue_children(model, "model")
print(f"Registered hooks on {len(hooks)} non-vjepa2 nn.Modules")

# ── Run one prediction ────────────────────────────────────────────────────────

val_videos = sorted(VAL_DIR.glob("*.mp4"))
if not val_videos:
    raise FileNotFoundError("No .mp4 in val_data/")

vp = val_videos[0]
print(f"\nRunning predict() on {vp.name}...")
df    = make_video_only_df(vp)
preds, _ = model.predict(events=df)
print(f"\npredict() done, output shape: {preds.shape}")

for h in hooks:
    h.remove()

# ── Report ────────────────────────────────────────────────────────────────────

print("\n=== NON-VJEPA2 MODULES THAT FIRED ===")
if not fired_outside:
    print("  NONE")
else:
    for name, count in sorted(fired_outside.items(), key=lambda x: -x[1]):
        print(f"  {count:4d}x  {name}")

# ── Inspect _HuggingFace wrapper ──────────────────────────────────────────────

print("\n=== _HuggingFace WRAPPER ===")
hf_wrapper = model.data.video_feature.image.model
print(f"Type: {type(hf_wrapper)}")
print(f"MRO: {[c.__name__ for c in type(hf_wrapper).__mro__[:6]]}")
print(f"Is nn.Module: {isinstance(hf_wrapper, torch.nn.Module)}")
try:
    print(f"attrs: {[a for a in dir(hf_wrapper) if not a.startswith('_')][:40]}")
except Exception as e:
    print(f"  dir() failed: {e}")

# ── item_uid / cache key structure ───────────────────────────────────────────

print("\n=== CACHE KEY STRUCTURE ===")
try:
    item_uid = infra.item_uid
    helper   = model.data.video_feature._event_types_helper
    df_test  = make_video_only_df(vp)
    events   = helper.extract(df_test)
    for ev in events:
        key = item_uid(ev)
        print(f"  event type: {type(ev).__name__}")
        print(f"  cache key:  {key!r}")
        in_cache = (cache_dict is not None) and (key in cache_dict)
        print(f"  in cache:   {in_cache}")
        break
except Exception as e:
    print(f"  item_uid inspection failed: {e}")

# ── Print the __call__ / extract source of infra ─────────────────────────────

print("\n=== INFRA __call__ SOURCE ===")
try:
    import inspect
    src = inspect.getsource(type(infra).__call__)
    print(src[:3000])
except Exception as e:
    print(f"  Could not get source: {e}")
    # Try the extract method if __call__ fails
    try:
        src = inspect.getsource(type(infra).extract)
        print(src[:3000])
    except Exception as e2:
        print(f"  Could not get extract source either: {e2}")

print("\nDone.")