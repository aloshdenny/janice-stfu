"""
abliteration.py — Unified abliteration pipeline.
Fully data-driven: target layers, masks, and surgery weights are all
derived from analysis outputs. No hardcoded category assumptions.

Usage:
    python scripts/abliteration.py --alpha 0.2 --n_components 3
    python scripts/abliteration.py --categories gore porn --alpha 0.15
"""

import os, gc, sys, time, json, warnings, logging, argparse
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.update({"PYTHONWARNINGS": "ignore", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})

import numpy as np
import torch
import torchvision.io as tvio
from torchvision import transforms
from torchvision.transforms.functional import resize
from tribev2.demo_utils import TribeModel

# ── Paths ─────────────────────────────────────────────────────────────────────

STUDY_ROOT = Path("./tribe_study")
MASK_DIR   = STUDY_ROOT / "masks"
CACHE_DIR  = Path("./cache")
DATA_DIR   = Path("./data")
OUT_DIR    = Path("./abliterated")
OUT_DIR.mkdir(exist_ok=True)

CONFIG_PATH        = MASK_DIR / "abliteration_config.json"
LAYER_PROFILE_PATH = MASK_DIR / "layer_profiles.npz"   # written by layer_analysis.py

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--alpha",        type=float, default=0.2)
parser.add_argument("--n_components", type=int,   default=1)
parser.add_argument("--categories",   nargs="+",  default=None,
                    help="Categories to abliterate. Defaults to all keys in config.")
parser.add_argument("--layer_mode",   choices=["auto", "fixed", "dual"], default="auto",
                    help="auto=peak from profile, fixed=75pct depth, dual=two peaks")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_FRAMES   = 16
CLIP_DURATION = 4
IMG_SIZE      = 256

# ── Load config (written by strict_analysis.py) ───────────────────────────────

if not CONFIG_PATH.exists():
    sys.exit(f"[FATAL] {CONFIG_PATH} not found. Run strict_analysis.py first.")

with open(CONFIG_PATH) as f:
    config = json.load(f)

target_categories = args.categories or list(config.keys())
print(f"Target categories: {target_categories}")
for cat in target_categories:
    if cat not in config:
        sys.exit(f"[FATAL] '{cat}' not in config. Run strict_analysis.py first.")

# ── Layer target selection ────────────────────────────────────────────────────

def pick_target_layers(cat, n_layers, mode, layer_profiles=None):
    """
    Returns a list of layer indices to run surgery on.
    
    auto:  load per-category layer correlation profile from analysis output,
           find the peak layer within each of the two TRIBE-sampled windows
           (shallow: 0–19, deep: 20–39), return the one with higher peak r.
           Falls back to fixed if profiles are unavailable.
    fixed: single layer at 75% depth (original behaviour).
    dual:  both window peaks — runs surgery on two layers.
    """
    if mode == "fixed" or layer_profiles is None:
        return [int(n_layers * 0.75)]

    if cat not in layer_profiles:
        print(f"  [WARN] No layer profile for {cat}, falling back to fixed.")
        return [int(n_layers * 0.75)]

    profile = layer_profiles[cat]   # shape (n_layers,) — mean |r| across ROIs

    shallow_peak = int(np.argmax(np.abs(profile[:20])))
    deep_peak    = int(np.argmax(np.abs(profile[20:]))) + 20

    shallow_val  = float(np.abs(profile[shallow_peak]))
    deep_val     = float(np.abs(profile[deep_peak]))

    print(f"  [{cat}] shallow peak: L{shallow_peak} (|r|={shallow_val:.4f})  "
          f"deep peak: L{deep_peak} (|r|={deep_val:.4f})")

    if mode == "dual":
        return [shallow_peak, deep_peak]
    else:   # auto — pick the stronger window
        return [shallow_peak if shallow_val > deep_val else deep_peak]


def load_layer_profiles():
    """
    Load per-category mean |Pearson r| profiles across ROIs.
    These are computed in layer_analysis.py and saved as a .npz.
    Returns None if file doesn't exist (graceful fallback).
    """
    if not LAYER_PROFILE_PATH.exists():
        print(f"  [WARN] {LAYER_PROFILE_PATH} not found — using fixed layer mode.")
        return None
    data = np.load(LAYER_PROFILE_PATH)
    return {k: data[k] for k in data.files}

# ── Memory helpers ────────────────────────────────────────────────────────────

def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def ram_available_mb():
    try:
        import subprocess
        r = subprocess.run(['free', '-m'], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if line.startswith('Mem:'):
                return int(line.split()[6])
    except Exception:
        pass
    return 99999

def report_mem(tag=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1024**2
        r = torch.cuda.memory_reserved() / 1024**2
        print(f"  [MEM{(' '+tag) if tag else ''}] alloc={a:.0f}MB reserved={r:.0f}MB")

# ── Video processing ──────────────────────────────────────────────────────────

normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

def iter_clips_from_video(video_path):
    """Yield (clip_tensor, clip_index) for every CLIP_DURATION-second window."""
    try:
        reader = tvio.VideoReader(str(video_path), "video")
        meta   = reader.get_metadata()
        fps    = meta["video"]["fps"][0]      if meta["video"]["fps"]      else 30.0
        dur    = meta["video"]["duration"][0] if meta["video"]["duration"] else 30.0
    except Exception as e:
        print(f"  [ERROR] open {video_path.name}: {e}")
        return

    n_clips = max(1, int((dur * fps) // (CLIP_DURATION * fps)))
    try:
        for c in range(n_clips):
            t_seek = float(c * CLIP_DURATION)
            t_end  = t_seek + CLIP_DURATION
            frames = []
            try:
                reader.seek(t_seek)
                for fd in reader:
                    if fd["pts"] >= t_end:
                        break
                    frames.append(fd["data"])
                    if len(frames) >= CLIP_FRAMES * 4:
                        break
            except Exception:
                continue
            if len(frames) < 2:
                continue
            ft   = torch.stack(frames).float() / 255.0
            idx  = torch.linspace(0, len(ft) - 1, CLIP_FRAMES).long()
            clip = torch.stack([normalize_fn(resize(ft[idx[i]], [IMG_SIZE, IMG_SIZE]))
                                for i in range(CLIP_FRAMES)])
            yield clip, c
            del frames, ft, clip
    finally:
        del reader

# ── Activation collection for one video ──────────────────────────────────────

def collect_video_activations(video_path, mask, preds_path,
                               vjepa2_module, hook_buffer,
                               acts_dir, cat):
    stem     = video_path.stem
    act_path = acts_dir / f"{stem}_acts.npy"
    y_path   = acts_dir / f"{stem}_y.npy"

    if act_path.exists() and y_path.exists():
        try:
            cached_y = np.load(y_path)
            if not np.isnan(cached_y).any():
                print(f"  [CACHED] {video_path.name}")
                return True
        except Exception:
            pass

    if not preds_path.exists():
        print(f"  [SKIP] no preds: {video_path.name}")
        return False

    if ram_available_mb() < 2000:
        print("  [WAIT] low RAM, sleeping 10s...")
        time.sleep(10); free()

    preds = np.load(preds_path)[:30]
    if mask.sum() == 0:
        raise ValueError(f"Mask is empty for {cat}!")
    y_tr = preds[:, mask].mean(axis=1)
    if np.isnan(y_tr).any():
        raise ValueError(f"NaNs in y_tr for {video_path.name}")

    clip_acts, clip_ys = [], []
    for clip, c in iter_clips_from_video(video_path):
        inp = clip.unsqueeze(0).to(DEVICE)
        hook_buffer[0] = None
        with torch.no_grad():
            vjepa2_module(pixel_values_videos=inp)
        del inp
        if hook_buffer[0] is not None:
            clip_acts.append(hook_buffer[0].squeeze(0).numpy().copy())
            t_s = int(c * CLIP_DURATION)
            t_e = min(t_s + CLIP_DURATION, 30)
            clip_ys.append(float(y_tr[t_s:t_e].mean()))
        hook_buffer[0] = None
        free()

    del preds, y_tr
    if clip_acts:
        np.save(act_path, np.stack(clip_acts))
        np.save(y_path,   np.array(clip_ys, dtype=np.float32))
        print(f"  {video_path.name}: {len(clip_acts)} clips")
        return True
    print(f"  [WARN] {video_path.name}: no valid clips")
    return False

# ── Phase 1: collect activations for all target categories ───────────────────

def run_activation_collection(model, vjepa2_module, encoder_blocks, n_layers):
    for cat in target_categories:
        mask_path = Path(config[cat]["mask_file"])
        mask      = np.load(mask_path)
        acts_dir  = OUT_DIR / f"acts_{cat}"
        acts_dir.mkdir(exist_ok=True)

        video_files = sorted(DATA_DIR.glob(f"{cat}*.mp4"))
        if not video_files:
            print(f"  [WARN] No videos found for {cat}")
            continue

        # Determine which layer to hook based on mode and profiles
        layer_profiles = load_layer_profiles()
        target_layers  = pick_target_layers(cat, n_layers, args.layer_mode, layer_profiles)
        # For collection we only need one layer; use the first (primary) target
        hook_layer = target_layers[0]

        # Clear all existing hooks
        for m in vjepa2_module.modules():
            m._forward_hooks.clear()
            m._forward_pre_hooks.clear()

        hook_buffer = [None]
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            hook_buffer[0] = hidden.mean(dim=1).detach().cpu().float()

        handle = encoder_blocks[hook_layer].register_forward_hook(hook_fn)

        print(f"\n=== {cat.upper()} — {len(video_files)} videos, hook on L{hook_layer} ===")
        report_mem("start")

        done = failed = 0
        for i, vp in enumerate(video_files):
            preds_path = STUDY_ROOT / cat / vp.stem / "preds.npy"
            ok = collect_video_activations(
                vp, mask, preds_path,
                vjepa2_module, hook_buffer,
                acts_dir, cat
            )
            done += ok; failed += (not ok)
            if i % 8 == 7:
                free(); report_mem(f"after {i+1} videos")

        handle.remove()
        print(f"  {cat}: {done} ok, {failed} failed")
        free()

# ── Direction finding (PCA on activation residuals) ───────────────────────────

def load_activations(cat):
    acts_dir = OUT_DIR / f"acts_{cat}"
    X_list, y_list, missing = [], [], []
    for p in sorted(acts_dir.glob("*_acts.npy")):
        y_p = acts_dir / p.name.replace("_acts", "_y")
        if not y_p.exists():
            missing.append(p.name); continue
        y_val = np.load(y_p)
        if np.isnan(y_val).any():
            raise ValueError(f"NaN in y for {p.name}")
        X_list.append(np.load(p))
        y_list.append(y_val)
    if missing:
        print(f"  [{cat}] missing y files: {len(missing)}")
    if not X_list:
        raise FileNotFoundError(f"No activations for {cat}")
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    print(f"  [{cat}] {X.shape[0]} clips, dim={X.shape[1]}, y∈[{y.min():.3f},{y.max():.3f}]")
    return X, y

def find_directions(X, y, n_components, label):
    y_range = y.max() - y.min()
    if y_range < 1e-9:
        weights = np.ones(len(y)) / len(y)
    else:
        weights = (y - y.min()) / (y_range + 1e-9)
        weights /= weights.sum()
    X_mean = (X * weights[:, None]).sum(axis=0, keepdims=True)
    X_c    = (X - X_mean) * np.sqrt(weights[:, None])
    _, S, Vt = np.linalg.svd(X_c, full_matrices=False)
    print(f"  [{label}] singular values: {S[:5].round(4)}")
    dirs = Vt[:n_components].copy()
    for i in range(n_components):
        proj = X @ dirs[i]
        if float(np.corrcoef(proj, y)[0, 1]) < 0:
            dirs[i] *= -1
            print(f"  [{label}] flipped direction {i}")
    return dirs

# ── Phase 2: surgery ──────────────────────────────────────────────────────────

def apply_surgery(vjepa2_module, encoder_blocks, n_layers, all_dirs_by_layer):
    """
    all_dirs_by_layer: dict mapping layer_idx -> np.array of directions (k, hidden_dim)
    Applies Gram-Schmidt orthogonalised projection-removal to value + proj weights.
    """
    for layer_idx, dirs in all_dirs_by_layer.items():
        dirs_t = torch.tensor(dirs, dtype=torch.float32).to(DEVICE)

        # Gram-Schmidt orthogonalisation across all directions for this layer
        ortho = []
        for d in dirs_t:
            for q in ortho:
                d = d - (d @ q) * q
            n = d.norm()
            if n > 1e-6:
                ortho.append(d / n)
        if not ortho:
            print(f"  [WARN] No valid directions for layer {layer_idx}, skipping")
            continue

        ortho = torch.stack(ortho)
        block = encoder_blocks[layer_idx]
        print(f"\n  Surgery on encoder.layer[{layer_idx}] — {len(ortho)} direction(s), alpha={args.alpha}")

        for attr_path in ["attention.value", "attention.proj"]:
            mod = block
            for part in attr_path.split("."):
                mod = getattr(mod, part)
            W = mod.weight.data.clone()
            for q in ortho:
                W -= args.alpha * (W @ q).unsqueeze(-1) * q
            mod.weight.data = W
            print(f"    {attr_path}: {tuple(W.shape)} updated")

def run_surgery_pipeline(vjepa2_module, encoder_blocks, n_layers):
    layer_profiles = load_layer_profiles()

    # Accumulate directions per layer across all target categories
    # so that if two categories share a target layer their directions
    # are orthogonalised together rather than applied in sequence
    dirs_by_layer = {}   # layer_idx -> list of direction arrays

    for cat in target_categories:
        target_layers = pick_target_layers(cat, n_layers, args.layer_mode, layer_profiles)
        X, y = load_activations(cat)
        dirs = find_directions(X, y, args.n_components, cat)
        del X, y; free()

        np.save(OUT_DIR / f"{cat}_directions.npy", dirs)
        np.save(OUT_DIR / f"{cat}_mask.npy", np.load(config[cat]["mask_file"]))

        # If dual mode, split directions evenly across the two target layers
        # (first n_components//2 to shallow, rest to deep)
        if len(target_layers) == 2:
            mid = max(1, args.n_components // 2)
            for tl, d in zip(target_layers, [dirs[:mid], dirs[mid:]]):
                dirs_by_layer.setdefault(tl, [])
                if len(d) > 0:
                    dirs_by_layer[tl].append(d)
        else:
            tl = target_layers[0]
            dirs_by_layer.setdefault(tl, [])
            dirs_by_layer[tl].append(dirs)

    # Merge per-layer and apply
    merged = {li: np.concatenate(ds, axis=0) for li, ds in dirs_by_layer.items()}

    # Print selectivity table before surgery
    print(f"\n{'Category':12s}", end="")
    for cat in target_categories:
        print(f"  {cat+'_mask':>14}", end="")
    print()
    print("-" * (14 + 16 * len(target_categories)))
    all_cats = list(config.keys())
    for eval_cat in all_cats:
        paths = sorted((STUDY_ROOT / eval_cat).glob("*/preds.npy"))
        if not paths:
            continue
        cat_mean = np.stack([np.load(p)[:30].mean(axis=0) for p in paths]).mean(axis=0)
        print(f"  {eval_cat:12s}", end="")
        for tcat in target_categories:
            m    = np.load(config[tcat]["mask_file"])
            val  = float(cat_mean[m].mean()) if m.sum() > 0 else 0.0
            print(f"  {val:14.4f}", end="")
        print()
        del cat_mean

    apply_surgery(vjepa2_module, encoder_blocks, n_layers, merged)

    # Save
    tag      = f"a{args.alpha}_c{args.n_components}_{'_'.join(target_categories)}"
    out_name = f"vjepa2_abliterated_{tag}.pt"
    torch.save(vjepa2_module.state_dict(), OUT_DIR / out_name)
    canonical = OUT_DIR / "vjepa2_abliterated.pt"
    torch.save(vjepa2_module.state_dict(), canonical)
    print(f"\n  Saved → {OUT_DIR / out_name}  ({(OUT_DIR / out_name).stat().st_size/1e6:.1f} MB)")
    print(f"  Canonical → {canonical}")

    # Write surgery log
    surgery_log = {
        "alpha":             args.alpha,
        "n_components":      args.n_components,
        "layer_mode":        args.layer_mode,
        "target_categories": target_categories,
        "layers_operated":   {str(li): int(len(ds)) for li, ds in merged.items()},
        "masks_used":        {cat: config[cat]["mask_name"] for cat in target_categories},
    }
    with open(OUT_DIR / "surgery_log.json", "w") as f:
        json.dump(surgery_log, f, indent=2)
    print(f"  Log → {OUT_DIR / 'surgery_log.json'}")

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Alpha={args.alpha}  n_components={args.n_components}  "
          f"layer_mode={args.layer_mode}  categories={target_categories}")

    model          = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    vjepa2_module  = model.data.video_feature.image.model.model
    encoder_blocks = vjepa2_module.encoder.layer
    n_layers       = len(encoder_blocks)
    vjepa2_module.eval()
    print(f"Encoder layers: {n_layers}")

    run_activation_collection(model, vjepa2_module, encoder_blocks, n_layers)
    run_surgery_pipeline(vjepa2_module, encoder_blocks, n_layers)
    print("\nDone.")