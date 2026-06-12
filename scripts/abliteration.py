"""
abliteration.py — Unified abliteration pipeline.
Includes strict mask generation (was strict_analysis.py), activation
collection, PCA direction finding, and weight surgery.

One target category is abliterated at a time against n=7 baselines.

Tolerance scale (continuous, replaces alpha):
  -1       -0.5       0       +0.5       +1
   |--------|---------|---------|--------|
   Repulsion Aversion  Neutral  Acceptance Attraction

  tolerance < 0 → suppress / repel the concept
  tolerance = 0 → neutral (no surgery effect)
  tolerance > 0 → accept / tolerate the concept

Usage:
    python scripts/abliteration.py --target porn --tolerance -0.8
    python scripts/abliteration.py --target porn --tolerance -1.0 --n_components 3
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
MASK_DIR.mkdir(parents=True, exist_ok=True)

LAYER_PROFILE_PATH = MASK_DIR / "layer_profiles.npz"   # written by layer_analysis.py

def discover_categories():
    """Auto-discover categories from subdirectory names in DATA_DIR."""
    return sorted([d.name for d in DATA_DIR.iterdir()
                   if d.is_dir() and any(d.glob("*.mp4"))])

ALL_CATEGORIES = discover_categories()

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--target",       type=str,   required=True,
                    help="Target category to abliterate (e.g. porn)")
parser.add_argument("--tolerance",    type=float, default=-0.8,
                    help="Tolerance scale: -1=repulsion, 0=neutral, +1=attraction")
parser.add_argument("--n_components", type=int,   default=1)
parser.add_argument("--layer_mode",   choices=["auto", "fixed", "dual"], default="auto",
                    help="auto=peak from profile, fixed=75pct depth, dual=two peaks")
args = parser.parse_args()

if args.target not in ALL_CATEGORIES:
    sys.exit(f"[FATAL] Unknown target '{args.target}'. "
             f"Available: {ALL_CATEGORIES}")

TARGET_CAT     = args.target
BASELINE_CATS  = [c for c in ALL_CATEGORIES if c != TARGET_CAT]
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_FRAMES   = 16
CLIP_DURATION = 4
IMG_SIZE      = 256
MAX_TRS       = 30

print(f"Discovered {len(ALL_CATEGORIES)} categories: {ALL_CATEGORIES}")
print(f"Target:    {TARGET_CAT}")
print(f"Baselines: {BASELINE_CATS}  (n={len(BASELINE_CATS)})")

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

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 0: Strict mask generation  (was strict_analysis.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_category_mean(category):
    paths = sorted((STUDY_ROOT / category).glob("*/preds.npy"))
    if not paths:
        sys.exit(f"[FATAL] No preds found for '{category}' in {STUDY_ROOT / category}. "
                 f"Run infer_bulk.py first.")
    arrays = [np.load(p)[:MAX_TRS].mean(axis=0) for p in paths]
    return np.stack(arrays).mean(axis=0)

def make_strict_mask(contrast, pct=90):
    thresh = np.percentile(contrast, pct)
    return (contrast > thresh) & (contrast > 0)

def score_mask(target_cat, mask, means, categories):
    """
    Composite score — fully data-driven, no hardcoded category assumptions.
      1. selectivity = target_mean - mean_of_others  (higher is better)
      2. max_leak = max activation of any non-target category in this mask
      3. leak_penalty = max(0, max_leak - target_mean)  (zero is ideal)
      4. n_verts >= 100 sanity gate
    Returns None if mask fails the gate.
    """
    if mask.sum() < 100:
        return None
    target_val = float(means[target_cat][mask].mean())
    other_cats = [c for c in categories if c != target_cat]
    other_val  = float(np.stack([means[c][mask] for c in other_cats]).mean())
    max_leak   = float(max(means[c][mask].mean() for c in other_cats))
    selectivity  = target_val - other_val
    leak_penalty = max(0.0, max_leak - target_val)
    return selectivity - 2.0 * leak_penalty

def run_strict_mask_generation():
    """Phase 0: generate contrast masks and auto-select the best one for TARGET_CAT."""
    print(f"\n{'='*60}")
    print(f"PHASE 0 — Strict mask generation for '{TARGET_CAT}'")
    print(f"{'='*60}")

    print("Loading category means...")
    means = {cat: load_category_mean(cat) for cat in ALL_CATEGORIES}

    # ── Build contrasts — fully data-driven LOSO ───────────────────────────
    # No hardcoded neutral_low / neutral_high / confound_map.

    # 1. LOSO: target vs mean of all baselines
    baseline_mean = np.stack([means[c] for c in BASELINE_CATS]).mean(axis=0)
    target_loso = means[TARGET_CAT] - baseline_mean

    CONTRASTS = {
        f"{TARGET_CAT}_loso": target_loso,
    }

    # 2. LOSO-k: leave one baseline out at a time, build contrast
    #    This discovers which baselines are confounds without hardcoding.
    for leave_out in BASELINE_CATS:
        remaining = [c for c in BASELINE_CATS if c != leave_out]
        remaining_mean = np.stack([means[c] for c in remaining]).mean(axis=0)
        CONTRASTS[f"{TARGET_CAT}_drop_{leave_out}"] = means[TARGET_CAT] - remaining_mean

    # 3. Pairwise: target vs each individual baseline
    for base in BASELINE_CATS:
        CONTRASTS[f"{TARGET_CAT}_vs_{base}"] = means[TARGET_CAT] - means[base]

    # 4. Multivariate strict: target > every single other category (logical AND)
    multivariate_mask = np.ones(baseline_mean.shape, dtype=bool)
    for c in ALL_CATEGORIES:
        if c != TARGET_CAT:
            multivariate_mask &= (means[TARGET_CAT] - means[c] > 0)
    CONTRASTS[f"{TARGET_CAT}_strict_multivariate"] = None  # mask-only

    # ── Generate masks ─────────────────────────────────────────────────────
    print(f"\n{'Name':35s}  {'n_verts':>8}  {'LH':>6}  {'RH':>6}  {'mean_val':>10}")
    print("-" * 75)

    new_masks = {}
    for name, data in CONTRASTS.items():
        if data is not None:
            mask = make_strict_mask(data, pct=90)
        elif "multivariate" in name:
            mask = multivariate_mask
        else:
            continue
        new_masks[name] = mask
        lh = mask[:10242].sum()
        rh = mask[10242:].sum()
        # For contrast-based masks, report mean contrast value
        if data is not None:
            mean_val = float(data[mask].mean()) if mask.sum() > 0 else 0
        else:
            mean_val = float(means[TARGET_CAT][mask].mean()) if mask.sum() > 0 else 0
        print(f"  {name:33s}  {mask.sum():8d}  {lh:6d}  {rh:6d}  {mean_val:10.4f}")

    # ── Selectivity check ──────────────────────────────────────────────────
    print("\nSelectivity check (mean activation per category in each mask):")
    print(f"{'Category':12s}", end="")
    for name in new_masks:
        short = name.replace(f"{TARGET_CAT}_", "")[:12]
        print(f"  {short:>12}", end="")
    print()
    print("-" * (14 + 14 * len(new_masks)))

    for cat in ALL_CATEGORIES:
        preds_paths = sorted((STUDY_ROOT / cat).glob("*/preds.npy"))
        if not preds_paths:
            continue
        cat_mean = np.stack([np.load(p)[:MAX_TRS].mean(axis=0)
                             for p in preds_paths]).mean(axis=0)
        print(f"  {cat:12s}", end="")
        for mask in new_masks.values():
            val = float(cat_mean[mask].mean()) if mask.sum() > 0 else 0
            print(f"  {val:12.4f}", end="")
        print()

    # ── Save masks ─────────────────────────────────────────────────────────
    for name, mask in new_masks.items():
        np.save(MASK_DIR / f"{name}_strict.npy", mask)
    print(f"\nMasks saved → {MASK_DIR}")

    # ── Auto-select best mask ──────────────────────────────────────────────
    candidates = list(new_masks.keys())
    best_name, best_score = None, -np.inf
    for name in candidates:
        mask = new_masks[name]
        s = score_mask(TARGET_CAT, mask, means, ALL_CATEGORIES)
        if s is not None and s > best_score:
            best_score, best_name = s, name

    if best_name is None:
        sys.exit(f"[FATAL] No valid mask found for '{TARGET_CAT}'")

    config = {
        TARGET_CAT: {
            "mask_file": str(MASK_DIR / f"{best_name}_strict.npy"),
            "mask_name": best_name,
            "score":     round(float(best_score), 6),
            "n_verts":   int(new_masks[best_name].sum()),
        }
    }
    print(f"\nAUTO-SELECTED: {best_name}  score={best_score:.4f}  "
          f"n={new_masks[best_name].sum()} vertices")

    config_path = MASK_DIR / "abliteration_config.json"

    # Merge with existing config (other targets from prior runs)
    if config_path.exists():
        with open(config_path) as f:
            existing = json.load(f)
        existing.update(config)
        config = existing

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written → {config_path}")

    return config

# ══════════════════════════════════════════════════════════════════════════════
# Layer target selection
# ══════════════════════════════════════════════════════════════════════════════

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
    if not LAYER_PROFILE_PATH.exists():
        print(f"  [WARN] {LAYER_PROFILE_PATH} not found — using fixed layer mode.")
        return None
    data = np.load(LAYER_PROFILE_PATH)
    return {k: data[k] for k in data.files}

# ══════════════════════════════════════════════════════════════════════════════
# Video processing helpers
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: Activation collection for target category
# ══════════════════════════════════════════════════════════════════════════════

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


def run_activation_collection(config, vjepa2_module, encoder_blocks, n_layers):
    print(f"\n{'='*60}")
    print(f"PHASE 1 — Activation collection for '{TARGET_CAT}'")
    print(f"{'='*60}")

    mask_path = Path(config[TARGET_CAT]["mask_file"])
    mask      = np.load(mask_path)
    acts_dir  = OUT_DIR / f"acts_{TARGET_CAT}"
    acts_dir.mkdir(exist_ok=True)

    # Find target videos in data/{TARGET_CAT}/
    video_files = sorted((DATA_DIR / TARGET_CAT).glob("*.mp4"))
    if not video_files:
        sys.exit(f"[FATAL] No videos found for {TARGET_CAT} in {DATA_DIR / TARGET_CAT}")

    # Determine which layer to hook
    layer_profiles = load_layer_profiles()
    target_layers  = pick_target_layers(TARGET_CAT, n_layers, args.layer_mode, layer_profiles)
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

    print(f"\n  {len(video_files)} videos, hook on L{hook_layer}")
    report_mem("start")

    done = failed = 0
    for i, vp in enumerate(video_files):
        preds_path = STUDY_ROOT / TARGET_CAT / vp.stem / "preds.npy"
        ok = collect_video_activations(
            vp, mask, preds_path,
            vjepa2_module, hook_buffer,
            acts_dir, TARGET_CAT
        )
        done += ok; failed += (not ok)
        if i % 8 == 7:
            free(); report_mem(f"after {i+1} videos")

    handle.remove()
    print(f"  {TARGET_CAT}: {done} ok, {failed} failed")
    free()

    return target_layers

# ══════════════════════════════════════════════════════════════════════════════
# Direction finding (PCA on activation residuals)
# ══════════════════════════════════════════════════════════════════════════════

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

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Surgery
# ══════════════════════════════════════════════════════════════════════════════

def apply_surgery(vjepa2_module, encoder_blocks, n_layers, all_dirs_by_layer):
    """
    all_dirs_by_layer: dict mapping layer_idx -> np.array of directions (k, hidden_dim)
    Applies Gram-Schmidt orthogonalised projection-removal to value + proj weights.
    """
    for layer_idx, dirs in all_dirs_by_layer.items():
        dirs_t = torch.tensor(dirs, dtype=torch.float32).to(DEVICE)

        # Gram-Schmidt orthogonalisation
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
        print(f"\n  Surgery on encoder.layer[{layer_idx}] — "
              f"{len(ortho)} direction(s), tolerance={args.tolerance}")

        for attr_path in ["attention.value", "attention.proj"]:
            mod = block
            for part in attr_path.split("."):
                mod = getattr(mod, part)
            W = mod.weight.data.clone()
            for q in ortho:
                W += args.tolerance * (W @ q).unsqueeze(-1) * q
            mod.weight.data = W
            print(f"    {attr_path}: {tuple(W.shape)} updated")


def run_surgery_pipeline(config, vjepa2_module, encoder_blocks, n_layers, target_layers):
    print(f"\n{'='*60}")
    print(f"PHASE 2 — Surgery for '{TARGET_CAT}'")
    print(f"{'='*60}")

    X, y = load_activations(TARGET_CAT)
    dirs = find_directions(X, y, args.n_components, TARGET_CAT)
    del X, y; free()

    np.save(OUT_DIR / f"{TARGET_CAT}_directions.npy", dirs)
    np.save(OUT_DIR / f"{TARGET_CAT}_mask.npy", np.load(config[TARGET_CAT]["mask_file"]))

    # Map directions to target layers
    dirs_by_layer = {}
    if len(target_layers) == 2:
        mid = max(1, args.n_components // 2)
        for tl, d in zip(target_layers, [dirs[:mid], dirs[mid:]]):
            if len(d) > 0:
                dirs_by_layer[tl] = d
    else:
        dirs_by_layer[target_layers[0]] = dirs

    # Print selectivity table before surgery
    mask = np.load(config[TARGET_CAT]["mask_file"])
    print(f"\nPre-surgery selectivity ({TARGET_CAT} mask, {int(mask.sum())} verts):")
    print(f"  {'Category':12s}  {'mean_act':>10}")
    print(f"  {'-'*26}")
    for cat in ALL_CATEGORIES:
        paths = sorted((STUDY_ROOT / cat).glob("*/preds.npy"))
        if not paths:
            continue
        cat_mean = np.stack([np.load(p)[:MAX_TRS].mean(axis=0) for p in paths]).mean(axis=0)
        val = float(cat_mean[mask].mean()) if mask.sum() > 0 else 0.0
        marker = " ◄ TARGET" if cat == TARGET_CAT else ""
        print(f"  {cat:12s}  {val:10.4f}{marker}")
        del cat_mean

    apply_surgery(vjepa2_module, encoder_blocks, n_layers, dirs_by_layer)

    # Save checkpoint
    tag      = f"t{args.tolerance}_c{args.n_components}_{TARGET_CAT}"
    out_name = f"vjepa2_abliterated_{tag}.pt"
    torch.save(vjepa2_module.state_dict(), OUT_DIR / out_name)
    canonical = OUT_DIR / "vjepa2_abliterated.pt"
    torch.save(vjepa2_module.state_dict(), canonical)
    print(f"\n  Saved → {OUT_DIR / out_name}  "
          f"({(OUT_DIR / out_name).stat().st_size/1e6:.1f} MB)")
    print(f"  Canonical → {canonical}")

    # Write surgery log
    surgery_log = {
        "target":            TARGET_CAT,
        "baselines":         BASELINE_CATS,
        "tolerance":         args.tolerance,
        "n_components":      args.n_components,
        "layer_mode":        args.layer_mode,
        "layers_operated":   {str(li): int(d.shape[0]) for li, d in dirs_by_layer.items()},
        "mask_used":         config[TARGET_CAT]["mask_name"],
        "mask_score":        config[TARGET_CAT]["score"],
    }
    with open(OUT_DIR / "surgery_log.json", "w") as f:
        json.dump(surgery_log, f, indent=2)
    print(f"  Log → {OUT_DIR / 'surgery_log.json'}")

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"\nTolerance={args.tolerance}  n_components={args.n_components}  "
          f"layer_mode={args.layer_mode}")

    # Phase 0: strict mask generation
    config = run_strict_mask_generation()

    # Load model (shared across phases 1 & 2)
    print("\nLoading TribeModel...")
    model          = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    vjepa2_module  = model.data.video_feature.image.model.model
    encoder_blocks = vjepa2_module.encoder.layer
    n_layers       = len(encoder_blocks)
    vjepa2_module.eval()
    print(f"Encoder layers: {n_layers}")

    # Phase 1: activation collection
    target_layers = run_activation_collection(config, vjepa2_module, encoder_blocks, n_layers)

    # Phase 2: surgery
    run_surgery_pipeline(config, vjepa2_module, encoder_blocks, n_layers, target_layers)

    print("\nDone.")