"""
find_cache_intercept.py — Find where exca caches V-JEPA2 output and identify
the correct hook point for live abliteration.

Run this if tribe_validation.py reports hook_calls=0.
It traces all module forward calls during a single predict() to find which
modules actually execute vs which are bypassed by cache.
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

print("Loading model...")
model         = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
vjepa2_module = model.data.video_feature.image.model.model

# ── Hook every nn.Module in vjepa2_module ────────────────────────────────────

fired = {}   # module_name → call_count
hooks = []

def make_hook(name):
    def h(module, inp, out):
        fired[name] = fired.get(name, 0) + 1
    return h

for name, mod in vjepa2_module.named_modules():
    h = mod.register_forward_hook(make_hook(name or "ROOT"))
    hooks.append(h)

print(f"Registered hooks on {len(hooks)} modules inside vjepa2_module")

# ── Also hook the top-level video feature extractor ──────────────────────────
# to catch if exca intercepts before vjepa2_module is ever called

video_feature = model.data.video_feature
image_model   = video_feature.image

top_fired = {}
top_hooks = []

def make_top_hook(name):
    def h(module, inp, out):
        top_fired[name] = top_fired.get(name, 0) + 1
    return h

for name, mod in image_model.named_modules():
    h = mod.register_forward_hook(make_top_hook(f"image.{name}" or "image.ROOT"))
    top_hooks.append(h)

print(f"Registered {len(top_hooks)} hooks on image_model wrapper")

# ── Run one prediction ────────────────────────────────────────────────────────

val_videos = sorted(VAL_DIR.glob("*.mp4"))
if not val_videos:
    raise FileNotFoundError("No .mp4 files in val_data/")

vp = val_videos[0]
print(f"\nRunning predict() on {vp.name}...")
df = make_video_only_df(vp)
preds, _ = model.predict(events=df)
print(f"predict() returned shape: {preds.shape}")

# ── Remove hooks ─────────────────────────────────────────────────────────────

for h in hooks + top_hooks:
    h.remove()

# ── Report ────────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("MODULES THAT FIRED INSIDE vjepa2_module:")
print("="*60)
if not fired:
    print("  NONE — vjepa2_module forward was never called!")
    print("  exca is returning fully-cached features before V-JEPA2 runs.")
    print("  You need to intercept at the image_model or video_feature level.")
else:
    # Show only modules that fired, sorted by call count
    for name, count in sorted(fired.items(), key=lambda x: -x[1])[:30]:
        print(f"  {count:4d}x  {name}")

print("\n" + "="*60)
print("MODULES THAT FIRED IN image_model WRAPPER:")
print("="*60)
if not top_fired:
    print("  NONE")
else:
    for name, count in sorted(top_fired.items(), key=lambda x: -x[1])[:30]:
        print(f"  {count:4d}x  {name}")

# ── Check what exca actually caches ──────────────────────────────────────────

print("\n" + "="*60)
print("EXCA CACHE INSPECTION:")
print("="*60)

try:
    cache_dict = model.data.video_feature.infra.cache_dict
    keys       = list(cache_dict.keys())
    print(f"  cache_dict keys: {len(keys)}")
    if keys:
        k = keys[0]
        v = cache_dict[k]
        print(f"  Sample key:   {k}")
        print(f"  Sample value type: {type(v)}")
        if hasattr(v, 'shape'):
            print(f"  Sample value shape: {v.shape}")
        elif isinstance(v, dict):
            for kk, vv in v.items():
                shape = vv.shape if hasattr(vv, 'shape') else type(vv)
                print(f"    {kk}: {shape}")
except Exception as e:
    print(f"  Could not inspect cache_dict: {e}")

# ── Print the call stack for image_model ─────────────────────────────────────

print("\n" + "="*60)
print("RECOMMENDATION:")
print("="*60)
if not fired:
    print("""
  exca is caching the FULL V-JEPA2 output. The transformer never runs.

  To abliterate, you need to hook at the point where cached features
  are consumed downstream — i.e., in the regression/encoding head that
  maps V-JEPA2 features to cortical predictions.

  This means the abliteration direction needs to be projected out of
  the CACHED FEATURE VECTOR, not the internal attention activations.

  Next steps:
  1. Check what shape the cached values are (printed above)
  2. Hook the module that READS from cache and processes features
  3. Project out the abliteration direction there

  Look for something like:
    model.data.video_feature.image.model  (the _HuggingFace wrapper)
  or
    model.data  (the top-level data module)

  The abliteration direction was computed from encoder.layer[TARGET_IDX]
  hidden states — those are 1408-dim. If the cached value shape matches
  1408 or is derived from it, you can hook at the cache read point.
""")
else:
    print(f"""
  V-JEPA2 DID run ({len(fired)} modules fired).
  The hook at encoder.layer[TARGET_IDX] should work.
  
  If tribe_validation.py still shows no effect, check:
  1. That TARGET_IDX matches the block you hooked during collection
  2. That the direction signs are correct (run with alpha=-0.5 to test reversal)
  3. That the ortho vectors are on the correct device (DEVICE={DEVICE})
""")