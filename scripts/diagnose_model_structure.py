"""
diagnose_model_structure.py — Detailed diagnostics of TribeModel structure,
exca cache, and weight surgery simulation.
"""
import os
import sys
import time
import shutil
import warnings
import logging
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import torch
from tribev2.demo_utils import TribeModel

OUT_DIR = Path("./abliterated")
CACHE_BASE = Path("./cache")

print("=== 1. Checking Saved Abliterated Checkpoints ===")
checkpoint_files = list(OUT_DIR.glob("*.pt")) if OUT_DIR.exists() else []
if not checkpoint_files:
    print("No .pt files found in ./abliterated")
else:
    for f in sorted(checkpoint_files):
        mtime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(f.stat().st_mtime))
        size_mb = f.stat().st_size / 1024**2
        print(f"  - {f.name} (Size: {size_mb:.1f} MB, Modified: {mtime})")

print("\n=== 2. Loading Model & Inspecting Structure ===")
try:
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"✗ Failed to load TribeModel: {e}")
    sys.exit(1)

video_feature = model.data.video_feature
image_ext = video_feature.image
hf_wrapper = image_ext.model
vjepa2 = hf_wrapper.model

N_LAYERS = len(vjepa2.encoder.layer)
TARGET_IDX = int(N_LAYERS * 0.75)
print(f"Layers count: {N_LAYERS}")
print(f"Target Layer Index (75%): {TARGET_IDX}")

# Check shapes of the target layer weights
block = vjepa2.encoder.layer[TARGET_IDX]
print("\nTarget Layer Weights Shapes:")
print(f"  vjepa2.encoder.layer[{TARGET_IDX}].attention.value.weight: {block.attention.value.weight.shape}")
print(f"  vjepa2.encoder.layer[{TARGET_IDX}].attention.proj.weight:  {block.attention.proj.weight.shape}")

print("\n=== 3. Simulating In-Memory Weight Surgery ===")
# We load the directions if they exist
gore_dirs_path = OUT_DIR / "gore_directions.npy"
porn_dirs_path = OUT_DIR / "porn_directions.npy"

if not (gore_dirs_path.exists() and porn_dirs_path.exists()):
    print("✗ Directions files not found in ./abliterated. Cannot run surgery simulation.")
else:
    gore_dirs = np.load(gore_dirs_path)
    porn_dirs = np.load(porn_dirs_path)
    all_dirs = np.concatenate([gore_dirs, porn_dirs], axis=0)
    dirs_t = torch.tensor(all_dirs, dtype=torch.float32)

    # Gram-Schmidt
    ortho = []
    for d in dirs_t:
        for q in ortho:
            d = d - (d @ q) * q
        if d.norm() > 1e-6:
            ortho.append(d / d.norm())
    ortho = torch.stack(ortho).to(block.attention.value.weight.device)
    print(f"Number of orthogonal directions: {len(ortho)}")

    # Simulating surgery with different alphas
    for test_alpha in [0.1, 0.2, 0.5]:
        # clone weights
        W_val_orig = block.attention.value.weight.data.clone()
        W_proj_orig = block.attention.proj.weight.data.clone()

        # apply surgery
        W_val_surg = W_val_orig.clone()
        W_proj_surg = W_proj_orig.clone()

        for q in ortho:
            W_val_surg -= test_alpha * (W_val_surg @ q).unsqueeze(-1) * q
            W_proj_surg -= test_alpha * (W_proj_surg @ q).unsqueeze(-1) * q

        diff_val = (W_val_orig - W_val_surg).abs().max().item()
        diff_proj = (W_proj_orig - W_proj_surg).abs().max().item()
        max_diff = max(diff_val, diff_proj)

        print(f"  For Alpha = {test_alpha:.1f}:")
        print(f"    Max weight delta (value): {diff_val:.8f}")
        print(f"    Max weight delta (proj):  {diff_proj:.8f}")
        print(f"    Combined max delta:      {max_diff:.8f}")

print("\n=== 4. Checking Exca Cache Settings ===")
infra = model.data.video_feature.infra
print(f"infra.folder: {infra.folder}")
print(f"infra.mode:   {infra.mode}")
print(f"infra.keep_in_ram: {infra.keep_in_ram}")

try:
    uid_folder = infra.uid_folder()
    print(f"infra.uid_folder(): {uid_folder}")
    if uid_folder is not None:
        print(f"  exists: {uid_folder.exists()}")
        if uid_folder.exists():
            files = list(uid_folder.glob("*"))
            print(f"  files count: {len(files)}")
            # Check if any .data files or .jsonl files exist here
            data_files = list(uid_folder.glob("*.data"))
            print(f"  .data files count: {len(data_files)}")
except Exception as e:
    print(f"Error checking uid_folder: {e}")

print("\nDone.")
