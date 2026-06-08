"""
collect_acts.py — Phase 1 only: extract and cache activations per video.
No model lives in the parent process. Run this first, then run surgery.py.

Usage:
    python collect_acts.py [--batch N]   (default batch=1 video per subprocess)
"""

import os, sys, warnings, logging, argparse, subprocess, gc
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────

STUDY_ROOT    = Path("./tribe_study")
MASK_DIR      = STUDY_ROOT / "masks"
CACHE_DIR     = Path("./cache")
DATA_DIR      = Path("./data")
OUT_DIR       = Path("./abliterated")
OUT_DIR.mkdir(exist_ok=True)

# These must match surgery.py
GORE_MASK_FILE = MASK_DIR / "gore_strict_bicontrast_strict.npy"
PORN_MASK_FILE = MASK_DIR / "porn_no_food_strict.npy"

CLIP_FRAMES   = 16
CLIP_DURATION = 4
IMG_SIZE      = 256
# TARGET_IDX computed inside subprocess — not needed here at all

CATEGORIES = {
    "gore": [f"gore{i}.mp4" for i in range(1, 49)],
    "porn": [f"porn{i}.mp4" for i in range(1, 49)],
}

# ── Subprocess script template ────────────────────────────────────────────────
# Receives a batch of filenames. Loads model ONCE per subprocess, processes
# all videos in the batch, then exits — OS reclaims all memory.

WORKER_TEMPLATE = '''
import os, warnings, logging, gc, sys
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import torch
from pathlib import Path
import torchvision.io as tvio
from torchvision import transforms
from torchvision.transforms.functional import resize
from tribev2.demo_utils import TribeModel

OUT_DIR    = Path({out_dir!r})
STUDY_ROOT = Path({study_root!r})
DATA_DIR   = Path({data_dir!r})
CACHE_DIR  = Path({cache_dir!r})
DEVICE        = {device!r}
CLIP_FRAMES   = {clip_frames}
CLIP_DURATION = {clip_duration}
IMG_SIZE      = {img_size}
category      = {category!r}
mask_file     = {mask_file!r}
fnames        = {fnames!r}

acts_dir = OUT_DIR / f"acts_{{category}}"
acts_dir.mkdir(exist_ok=True)

# ── Check which videos actually need processing ───────────────────────────────
todo = []
for fname in fnames:
    stem     = Path(fname).stem
    act_path = acts_dir / f"{{stem}}_acts.npy"
    y_path   = acts_dir / f"{{stem}}_y.npy"
    if act_path.exists() and y_path.exists():
        print(f"  [CACHED] {{fname}}", flush=True)
    else:
        todo.append(fname)

if not todo:
    print("  All cached, exiting.", flush=True)
    sys.exit(0)

# ── Load model ONCE for this batch ───────────────────────────────────────────
print(f"  Loading model for batch of {{len(todo)}} videos...", flush=True)
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
vjepa2_module  = model.data.video_feature.image.model.model
encoder_blocks = vjepa2_module.encoder.layer
N_LAYERS       = len(encoder_blocks)
TARGET_IDX     = int(N_LAYERS * 0.75)

for m in vjepa2_module.modules():
    m._forward_hooks.clear()
    m._forward_pre_hooks.clear()

collected_acts = []
def hook_fn(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    collected_acts.clear()
    collected_acts.append(hidden.mean(dim=1).detach().cpu().float())

hook_handle = encoder_blocks[TARGET_IDX].register_forward_hook(hook_fn)

normalize_fn = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

mask = np.load(mask_file)

vjepa2_module.eval()

# ── Process each video ────────────────────────────────────────────────────────
for fname in todo:
    stem       = Path(fname).stem
    act_path   = acts_dir / f"{{stem}}_acts.npy"
    y_path     = acts_dir / f"{{stem}}_y.npy"
    preds_path = STUDY_ROOT / category / stem / "preds.npy"
    video_path = (DATA_DIR / fname).resolve()

    if not preds_path.exists() or not video_path.exists():
        print(f"  [SKIP] {{fname}}", flush=True)
        continue

    preds = np.load(preds_path)[:30]
    y_tr  = preds[:, mask].mean(axis=1)

    try:
        vframes, _, info = tvio.read_video(str(video_path), pts_unit="sec")
        try:
            vframes   = vframes.float() / 255.0
            vframes   = vframes.permute(0, 3, 1, 2)
            fps       = info.get("video_fps", 30.0)
            total_f   = vframes.shape[0]
            spf       = CLIP_DURATION * fps
            n_clips   = max(1, int(total_f // spf))

            clip_acts, clip_ys = [], []
            for c in range(n_clips):
                start = int(c * spf)
                end   = min(start + int(spf), total_f)
                chunk = vframes[start:end]
                idx   = torch.linspace(0, len(chunk) - 1, CLIP_FRAMES).long()
                clip  = chunk[idx]
                clip  = torch.stack([
                    normalize_fn(resize(clip[i], [IMG_SIZE, IMG_SIZE]))
                    for i in range(len(clip))
                ])
                inp = clip.unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    vjepa2_module(pixel_values_videos=inp)
                if collected_acts:
                    clip_acts.append(collected_acts[-1].squeeze(0).numpy().copy())
                    t_start = int(c * CLIP_DURATION)
                    t_end   = min(t_start + CLIP_DURATION, 30)
                    clip_ys.append(float(y_tr[t_start:t_end].mean()))
                del inp, clip
                torch.cuda.empty_cache()
        finally:
            del vframes
            gc.collect()

    except Exception as e:
        print(f"  [ERROR] {{fname}}: {{e}}", flush=True)
        continue

    if clip_acts:
        np.save(act_path, np.stack(clip_acts))
        np.save(y_path,   np.array(clip_ys))
        print(f"  {{fname}}: {{len(clip_acts)}} clips saved  "
              f"y=[{{min(clip_ys):.3f}}, {{max(clip_ys):.3f}}]", flush=True)

hook_handle.remove()
del model
torch.cuda.empty_cache()
gc.collect()
print("  Batch done.", flush=True)
'''

# ── Helpers ───────────────────────────────────────────────────────────────────

def run_batch(category, fnames, mask_file, batch_id):
    script = WORKER_TEMPLATE.format(
        out_dir      = str(OUT_DIR.resolve()),
        study_root   = str(STUDY_ROOT.resolve()),
        data_dir     = str(DATA_DIR.resolve()),
        cache_dir    = str(CACHE_DIR.resolve()),
        device       = "cuda" if __import__("torch").cuda.is_available() else "cpu",
        clip_frames  = CLIP_FRAMES,
        clip_duration= CLIP_DURATION,
        img_size     = IMG_SIZE,
        category     = category,
        mask_file    = str(mask_file.resolve()),
        fnames       = fnames,
    )
    print(f"\n[Batch {batch_id}] {category} × {len(fnames)} videos", flush=True)
    result = subprocess.run(
        [sys.executable, "-c", script],
        timeout=600,
    )
    if result.returncode != 0:
        print(f"  [BATCH FAILED] exit code {result.returncode}", flush=True)
    return result.returncode == 0


def collect_category(category, filenames, mask_file, batch_size):
    acts_dir = OUT_DIR / f"acts_{category}"
    acts_dir.mkdir(exist_ok=True)

    # Split into batches
    batches = [filenames[i:i+batch_size] for i in range(0, len(filenames), batch_size)]
    for bid, batch in enumerate(batches):
        run_batch(category, batch, mask_file, bid)
        gc.collect()

    # Report final coverage
    done  = sum(1 for f in filenames
                if (acts_dir / f"{Path(f).stem}_acts.npy").exists())
    print(f"\n{category}: {done}/{len(filenames)} videos collected")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1,
                        help="Videos per subprocess (default 1 = max isolation)")
    args = parser.parse_args()

    print(f"Batch size: {args.batch} video(s) per subprocess")

    print("\n=== Collecting GORE activations ===")
    collect_category("gore", CATEGORIES["gore"], GORE_MASK_FILE, args.batch)

    print("\n=== Collecting PORN activations ===")
    collect_category("porn", CATEGORIES["porn"], PORN_MASK_FILE, args.batch)

    print("\nCollection complete. Run surgery.py next.")
