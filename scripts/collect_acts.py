"""
collect_acts.py — Activation collection with aggressive memory management.

Strategy:
- Single persistent process (no subprocess overhead)
- Model loaded ONCE, stays loaded for all videos
- VideoReader: streams only CLIP_FRAMES*4 raw frames per clip window —
  the full video is never decoded into RAM at once
- Hook uses a single-slot buffer, no accumulation possible
- Hidden dim derived at runtime — no hardcoded 1408 assumption

Usage:
    python collect_acts.py [--category gore|porn|both]
"""

import os, warnings, logging, gc, argparse, time
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch
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
CLIP_DURATION = 4      # seconds per clip
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

# ── Hook — single-slot buffer, no accumulation ────────────────────────────────

_hook_buffer = [None]  # single slot — replaced on every forward pass

def hook_fn(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    # Mean over token dim immediately; move to CPU, detach from graph
    _hook_buffer[0] = hidden.mean(dim=1).detach().cpu().float()

hook_handle = encoder_blocks[TARGET_IDX].register_forward_hook(hook_fn)

normalize_fn = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

vjepa2_module.eval()

# ── Per-video collection (streaming — never loads full video into RAM) ────────

def process_video(category, fname, mask):
    stem     = Path(fname).stem
    acts_dir = OUT_DIR / f"acts_{category}"
    acts_dir.mkdir(exist_ok=True)
    act_path = acts_dir / f"{stem}_acts.npy"
    y_path   = acts_dir / f"{stem}_y.npy"

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

    ram = ram_available_mb()
    if ram < 2000:
        print(f"  [WAIT] low RAM ({ram}MB), sleeping 10s...")
        time.sleep(10)
        gc.collect()
        torch.cuda.empty_cache()

    preds = np.load(preds_path)[:30]
    y_tr  = preds[:, mask].mean(axis=1)   # (30,)

    # Open VideoReader — only a handful of frames in memory at once
    try:
        reader = tvio.VideoReader(str(video_path), "video")
        meta   = reader.get_metadata()
        fps    = meta["video"]["fps"][0]      if meta["video"]["fps"]      else 30.0
        dur    = meta["video"]["duration"][0] if meta["video"]["duration"] else 30.0
    except Exception as e:
        print(f"  [ERROR] open {fname}: {e}")
        del preds, y_tr
        return False

    n_clips          = max(1, int((dur * fps) // (CLIP_DURATION * fps)))
    clip_acts_list: list[np.ndarray] = []
    clip_ys_list:   list[float]      = []

    try:
        for c in range(n_clips):
            t_seek  = float(c * CLIP_DURATION)
            t_end_s = t_seek + CLIP_DURATION

            # Stream only the frames that fall in this clip window
            frames: list[torch.Tensor] = []
            try:
                reader.seek(t_seek)
                for frame_data in reader:
                    if frame_data["pts"] >= t_end_s:
                        break
                    frames.append(frame_data["data"])    # uint8 (C, H, W)
                    if len(frames) >= CLIP_FRAMES * 4:   # safety cap
                        break
            except Exception as e:
                print(f"  [WARN] clip {c}/{n_clips} of {fname}: {e}")
                del frames
                continue

            if len(frames) < 2:
                del frames
                continue

            # Subsample to exactly CLIP_FRAMES, normalise, resize — CPU only
            frames_t = torch.stack(frames).float() / 255.0  # (T, C, H, W)
            del frames
            idx  = torch.linspace(0, len(frames_t) - 1, CLIP_FRAMES).long()
            clip = frames_t[idx]                              # (CLIP_FRAMES, C, H, W)
            del frames_t
            clip = torch.stack([
                normalize_fn(resize(clip[i], [IMG_SIZE, IMG_SIZE]))
                for i in range(CLIP_FRAMES)
            ])                                                # (CLIP_FRAMES, C, H, W)

            inp = clip.unsqueeze(0).to(DEVICE)                # (1, T, C, H, W)
            del clip
            _hook_buffer[0] = None

            with torch.no_grad():
                vjepa2_module(pixel_values_videos=inp)
            del inp

            if _hook_buffer[0] is not None:
                # hidden_dim derived at runtime — no hardcode needed
                clip_acts_list.append(_hook_buffer[0].squeeze(0).numpy().copy())
                t_start_tr = int(c * CLIP_DURATION)
                t_end_tr   = min(t_start_tr + CLIP_DURATION, 30)
                clip_ys_list.append(float(y_tr[t_start_tr:t_end_tr].mean()))
            _hook_buffer[0] = None

    except Exception as e:
        print(f"  [ERROR] {fname}: {e}")
        return False

    finally:
        try:
            del reader
        except Exception:
            pass
        del preds, y_tr
        torch.cuda.empty_cache()
        gc.collect()

    valid = len(clip_acts_list)
    if valid > 0:
        arr_acts = np.stack(clip_acts_list)                   # (valid, hidden_dim)
        arr_ys   = np.array(clip_ys_list, dtype=np.float32)
        np.save(act_path, arr_acts)
        np.save(y_path,   arr_ys)
        print(f"  {fname}: {valid} clips  "
              f"y=[{arr_ys.min():.3f}, {arr_ys.max():.3f}]  "
              f"act_dim={arr_acts.shape[1]}")
        del arr_acts, arr_ys
    else:
        print(f"  [WARN] {fname}: no valid clips")

    del clip_acts_list, clip_ys_list
    return valid > 0


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

        # Periodic CUDA flush
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