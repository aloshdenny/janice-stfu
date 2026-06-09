"""
diagnose_surgery.py — Diagnostic script to inspect model structure,
state dict keys, and verify weight loading.
"""

import os, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import torch
from pathlib import Path
from tribev2.demo_utils import TribeModel

OUT_DIR    = Path("./abliterated")
CACHE_BASE = Path("./cache")
abl_path   = OUT_DIR / "vjepa2_abliterated.pt"

print("=== 1. Loading TribeModel ===")
try:
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load TribeModel: {e}")
    exit(1)

print("\n=== 2. Inspecting Class Structure ===")
try:
    video_feature = model.data.video_feature
    print(f"video_feature class: {type(video_feature).__module__}.{type(video_feature).__name__}")
    
    image_ext = video_feature.image
    print(f"video_feature.image class: {type(image_ext).__module__}.{type(image_ext).__name__}")
    
    hf_wrapper = image_ext.model
    print(f"video_feature.image.model class: {type(hf_wrapper).__module__}.{type(hf_wrapper).__name__}")
    
    vjepa2 = hf_wrapper.model
    print(f"vjepa2 class: {type(vjepa2).__module__}.{type(vjepa2).__name__}")
    print(f"vjepa2 is nn.Module: {isinstance(vjepa2, torch.nn.Module)}")
    
    N_LAYERS   = len(vjepa2.encoder.layer)
    TARGET_IDX = int(N_LAYERS * 0.75)
    print(f"Layers count: {N_LAYERS}, TARGET_IDX: {TARGET_IDX}")
except Exception as e:
    print(f"✗ Error during class inspection: {e}")

print("\n=== 3. Inspecting State Dict Keys ===")
if not abl_path.exists():
    print(f"✗ Saved abliterated file {abl_path} does not exist. Run surgery.py first.")
else:
    try:
        saved_state = torch.load(abl_path, map_location="cpu")
        active_state = vjepa2.state_dict()
        
        saved_keys = set(saved_state.keys())
        active_keys = set(active_state.keys())
        
        print(f"Saved state dict keys: {len(saved_keys)}")
        print(f"Active model state dict keys: {len(active_keys)}")
        
        common = saved_keys.intersection(active_keys)
        print(f"Common keys (intersection): {len(common)}")
        
        if len(saved_keys) != len(active_keys) or len(common) != len(saved_keys):
            print(f"  Only in saved state: {list(saved_keys - active_keys)[:5]}")
            print(f"  Only in active model: {list(active_keys - saved_keys)[:5]}")
        else:
            print("✓ Perfect key match between saved state and active model!")
    except Exception as e:
        print(f"✗ Error during state dict comparison: {e}")

print("\n=== 4. Testing load_state_dict ===")
if abl_path.exists():
    try:
        w_before = vjepa2.encoder.layer[TARGET_IDX].attention.value.weight.data.clone()
        
        missing, unexpected = vjepa2.load_state_dict(saved_state, strict=False)
        print(f"load_state_dict strict=False result:")
        print(f"  Missing keys: {len(missing)} (sample: {missing[:5]})")
        print(f"  Unexpected keys: {len(unexpected)} (sample: {unexpected[:5]})")
        
        w_after = vjepa2.encoder.layer[TARGET_IDX].attention.value.weight.data
        diff = (w_before - w_after).abs().max().item()
        print(f"Max weight difference before vs after loading: {diff:.8f}")
        if diff > 1e-8:
            print("✓ Weights verified as CHANGED after load_state_dict!")
        else:
            print("✗ WARNING: Weights did NOT change after load_state_dict!")
    except Exception as e:
        print(f"✗ Error during load_state_dict test: {e}")

print("\n=== 5. Checking Exca Cache Directory ===")
try:
    infra = model.data.video_feature.infra
    uid_folder = Path(str(infra.uid_folder))
    print(f"infra.uid_folder: {uid_folder}")
    print(f"Folder exists: {uid_folder.exists()}")
    if uid_folder.exists():
        files = list(uid_folder.glob("*"))
        print(f"Files inside uid_folder: {len(files)}")
        for f in files[:5]:
            print(f"  {f.name} ({f.stat().st_size / 1024:.1f} KB)")
except Exception as e:
    print(f"✗ Error checking cache directory: {e}")

print("\nDone.")
