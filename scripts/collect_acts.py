"""
collect_acts.py — Activation collection with aggressive memory management.

Strategy:
- Single persistent process (no subprocess overhead)
- Model loaded ONCE, stays loaded for all videos
- Hook output written directly to preallocated mmap arrays
- Explicit CUDA graph clearing between videos
- ulimit-aware: checks available memory before each video

Usage:
    python collect_acts.py [--category gore|porn|both]
"""

import os, warnings, logging, gc, argparse, time
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"
# Limit torch threads to reduce memory overhead
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import torchvision.io as tvio
from torchvision import transforms
from torchvision.transforms.functional import resize
from tribev2.demo_utils import TribeModel

# ── Config ────────────────────────────────────────────────────────────────────

STUDY_ROOT    = Path("./tribe_study")
MASK_DIR      = STUDY_ROOT / "masks"
CACHE_DIR     = Path("./cache")
DATA_DIR      = Path("./data")
OUT_DIR       = Path("./abliterated")
OUT_DIR.mkdir(exist_ok=True)

GORE_MASK_FILE = MASK_DIR / "gore_strict_bicontrast_strict.npy"
PORN_MASK_FILE = MASK_DIR / "porn_no_food_strict.npy"

CLIP_FRAMES   = 16
CLIP_DURATION = 4
IMG_SIZE      = 256

CATEGORIES = {
    "gore": [f"gore{i}.mp4" for i in range(1, 49)],
    "porn": [f"porn{i}.mp4" for i in range(1, 49)],
}

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--category", default="both", choices=["gore", "porn", "both"])
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Memory reporting ──────────────────────────────────────────────────────────

def report_mem(tag=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**2
        r = torch.cuda.memory_reserved() / 1024**2
        print(f"  [MEM{(' '+tag) if tag else ''}] VRAM alloc={a:.0f}MB reserved={r:.0f}MB")

def ram_available_mb():
    try:
        import subprocess
        r = subprocess.run(['free', '-m'], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith('Mem:'):
                return int(line.split()[6])  # available column
    except Exception:
        pass
    return 99999  # unknown, proceed

# ── Load model ONCE ───────────────────────────────────────────────────────────

print("Loading TribeModel (once)...")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)

vjepa2_module  = model.data.video_feature.image.model.model
encoder_blocks = vjepa2_module.encoder.layer
N_LAYERS       = len(encoder_blocks)
TARGET_IDX     = int(N_LAYERS * 0.75)
print(f"Encoder layers: {N_LAYERS}, target: {TARGET_IDX}")

for m in vjepa2_module.modules():
    m._forward_hooks.clear()
    m._forward_pre_hooks.clear()

report_mem("after model load")

# ── Hook — uses a single slot, never accumulates ──────────────────────────────

_hook_buffer = [None]  # single slot — replaced on every forward pass

def hook_fn(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    # mean over token dim immediately, move to CPU, detach
    _hook_buffer[0] = hidden.mean(dim=1).detach().cpu().float()

hook_handle = encoder_blocks[TARGET_IDX].register_forward_hook(hook_fn)

normalize_fn = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

vjepa2_module.eval()

# ── Per-video collection ──────────────────────────────────────────────────────

def process_video(category, fname, mask):
    stem       = Path(fname).stem
    acts_dir   = OUT_DIR / f"acts_{category}"
    acts_dir.mkdir(exist_ok=True)
    act_path   = acts_dir / f"{stem}_acts.npy"
    y_path     = acts_dir / f"{stem}_y.npy"

    if act_path.exists() and y_path.exists():
        print(f"  [CACHED] {fname}")
        return True

    preds_path = STUDY_ROOT / category / stem / "preds.npy"
    video_path = (DATA_DIR / fname).resolve()

    if not preds_path.exists():
        print(f"  [SKIP] no preds: {fname}")
        return False
    if not video_path.exists():
        print(f"  [SKIP] no video: {fname}")
        return False

    # Check RAM before reading video
    ram = ram_available_mb()
    if ram < 4000:
        print(f"  [WAIT] low RAM ({ram}MB), sleeping 5s...")
        time.sleep(5)
        gc.collect()
        torch.cuda.empty_cache()

    preds = np.load(preds_path)[:30]
    y_tr  = preds[:, mask].mean(axis=1)   # (30,)

    # Read video — this is the big RAM spike
    try:
        vframes, _, info = tvio.read_video(str(video_path), pts_unit="sec")
    except Exception as e:
        print(f"  [ERROR] read_video {fname}: {e}")
        del preds, y_tr
        return False

    try:
        vframes = vframes.float() / 255.0
        vframes = vframes.permute(0, 3, 1, 2)   # (T, C, H, W)
        fps     = info.get("video_fps", 30.0)
        total_f = vframes.shape[0]
        spf     = CLIP_DURATION * fps
        n_clips = max(1, int(total_f // spf))

        clip_acts = np.empty((n_clips, 1408), dtype=np.float32)  # preallocate
        clip_ys   = np.empty((n_clips,),      dtype=np.float32)
        valid     = 0

        for c in range(n_clips):
            start = int(c * spf)
            end   = min(start + int(spf), total_f)
            chunk = vframes[start:end]
            idx   = torch.linspace(0, len(chunk) - 1, CLIP_FRAMES).long()
            clip  = chunk[idx]
            clip  = torch.stack([
                normalize_fn(resize(clip[i], [IMG_SIZE, IMG_SIZE]))
                for i in range(len(clip))
            ])  # (T, C, H, W)

            inp = clip.unsqueeze(0).to(DEVICE)  # (1, T, C, H, W)
            _hook_buffer[0] = None

            with torch.no_grad():
                vjepa2_module(pixel_values_videos=inp)

            if _hook_buffer[0] is not None:
                clip_acts[valid] = _hook_buffer[0].squeeze(0).numpy()
                t_start = int(c * CLIP_DURATION)
                t_end   = min(t_start + CLIP_DURATION, 30)
                clip_ys[valid] = float(y_tr[t_start:t_end].mean())
                valid += 1

            # Immediately free clip tensors
            del inp, clip, chunk
            _hook_buffer[0] = None
            # Don't call empty_cache every clip — it's slow; do it per video

        if valid > 0:
            np.save(act_path, clip_acts[:valid])
            np.save(y_path,   clip_ys[:valid])
            print(f"  {fname}: {valid} clips  "
                  f"y=[{clip_ys[:valid].min():.3f}, {clip_ys[:valid].max():.3f}]")
        else:
            print(f"  [WARN] {fname}: no valid clips")

        return valid > 0

    except Exception as e:
        print(f"  [ERROR] {fname}: {e}")
        return False

    finally:
        del vframes, preds, y_tr
        torch.cuda.empty_cache()
        gc.collect()


# ── Main loop ─────────────────────────────────────────────────────────────────

def run_category(category):
    mask_file = GORE_MASK_FILE if category == "gore" else PORN_MASK_FILE
    mask      = np.load(mask_file)
    filenames = CATEGORIES[category]

    print(f"\n=== {category.upper()} ({len(filenames)} videos) ===")
    report_mem("start")

    done, failed = 0, 0
    for i, fname in enumerate(filenames):
        ok = process_video(category, fname, mask)
        if ok:
            done += 1
        else:
            failed += 1

        # Periodic cache flush
        if i % 8 == 7:
            torch.cuda.empty_cache()
            gc.collect()
            report_mem(f"after {i+1} videos")

    print(f"\n{category}: {done} collected, {failed} failed")
    report_mem("end")


cats = ["gore", "porn"] if args.category == "both" else [args.category]
for cat in cats:
    run_category(cat)

hook_handle.remove()
del model
torch.cuda.empty_cache()
gc.collect()

print("\nCollection complete. Run surgery.py next.")
print(f"Output: {OUT_DIR}/acts_gore/  and  {OUT_DIR}/acts_porn/")