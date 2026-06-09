"""
abliteration_diagnosis.py — Diagnostic script to test which layers (or sets of layers)
are eligible for abliteration and which yield the maximum suppression with minimum side effects.
"""

import os
import gc
import sys
import time
import shutil
import warnings
import logging
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import torch
import pandas as pd
import torchvision.io as tvio
from torchvision import transforms
from torchvision.transforms.functional import resize
from tribev2.demo_utils import TribeModel

# ── Config ────────────────────────────────────────────────────────────────────

STUDY_ROOT    = Path("./tribe_study")
MASK_DIR      = STUDY_ROOT / "masks"
CACHE_DIR     = Path("./cache")
DATA_DIR      = Path("./data")
VAL_DIR       = Path("./val_data")
OUT_DIR       = Path("./abliterated")

GORE_MASK_FILE = MASK_DIR / "gore_strict_bicontrast_strict.npy"
PORN_MASK_FILE = MASK_DIR / "porn_no_food_strict.npy"

CLIP_FRAMES   = 16
CLIP_DURATION = 4
IMG_SIZE      = 256
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

CANDIDATE_LAYERS = [20, 24, 28, 30, 32, 36, 39]
DIAG_VIDEOS = {
    "gore": [f"gore{i}.mp4" for i in range(1, 49)],
    "porn": [f"porn{i}.mp4" for i in range(1, 49)],
}

# ── Helper: get duration ──────────────────────────────────────────────────────

def get_duration(video_path):
    import subprocess
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
                       capture_output=True, text=True)
    try:
        return round(float(r.stdout.strip()) - 0.1, 3)
    except:
        return 29.9

def make_video_only_df(video_path):
    dur = get_duration(video_path)
    return pd.DataFrame([{"type": "Video", "start": 0.0, "duration": dur,
                          "timeline": "default", "subject": "default", "session": "", "task": "", "run": "",
                          "filepath": str(video_path.resolve()), "frequency": 60.0, "offset": 0.0,
                          "stop": dur, "context": float("nan")}])

# ── Step 1: Collect Activations for All Candidate Layers ──────────────────────

def collect_diagnostic_activations():
    print("=== Step 1: Collecting Activations for All Candidate Layers ===")
    
    print("Loading TribeModel...")
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
    vjepa2_module  = model.data.video_feature.image.model.model
    encoder_blocks = vjepa2_module.encoder.layer
    vjepa2_module.eval()
    
    # Set up hooks for all candidate layers
    hook_buffers = {L: [] for L in CANDIDATE_LAYERS}
    hook_handles = []
    
    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            # Mean pool over tokens and move to CPU
            hook_buffers[layer_idx].append(hidden.mean(dim=1).detach().cpu().float())
        return hook_fn

    for L in CANDIDATE_LAYERS:
        handle = encoder_blocks[L].register_forward_hook(make_hook(L))
        hook_handles.append(handle)
        
    normalize_fn = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    X_data = {L: {"gore": [], "porn": []} for L in CANDIDATE_LAYERS}
    y_data = {"gore": [], "porn": []}
    
    # We collect from the subset of videos
    for cat in ["gore", "porn"]:
        mask_file = GORE_MASK_FILE if cat == "gore" else PORN_MASK_FILE
        mask = np.load(mask_file)
        
        for fname in DIAG_VIDEOS[cat]:
            stem = Path(fname).stem
            preds_path = STUDY_ROOT / cat / stem / "preds.npy"
            video_path = DATA_DIR / fname
            
            if not preds_path.exists() or not video_path.exists():
                continue
                
            print(f"  Streaming {fname}...")
            preds = np.load(preds_path)[:30]
            y_tr = preds[:, mask].mean(axis=1)
            
            # Read video
            try:
                reader = tvio.VideoReader(str(video_path), "video")
                meta = reader.get_metadata()
                fps = meta["video"]["fps"][0] if meta["video"]["fps"] else 30.0
                dur = meta["video"]["duration"][0] if meta["video"]["duration"] else 30.0
            except Exception as e:
                print(f"    Failed to open {fname}: {e}")
                continue
                
            n_clips = max(1, int((dur * fps) // (CLIP_DURATION * fps)))
            
            for c in range(n_clips):
                t_seek = float(c * CLIP_DURATION)
                t_end_s = t_seek + CLIP_DURATION
                
                frames = []
                try:
                    reader.seek(t_seek)
                    for frame_data in reader:
                        if frame_data["pts"] >= t_end_s:
                            break
                        frames.append(frame_data["data"])
                        if len(frames) >= CLIP_FRAMES * 4:
                            break
                except:
                    continue
                    
                if len(frames) < 2:
                    continue
                    
                frames_t = torch.stack(frames).float() / 255.0
                idx = torch.linspace(0, len(frames_t) - 1, CLIP_FRAMES).long()
                clip = frames_t[idx]
                clip = torch.stack([normalize_fn(resize(clip[i], [IMG_SIZE, IMG_SIZE])) for i in range(CLIP_FRAMES)])
                
                inp = clip.unsqueeze(0).to(DEVICE)
                
                # Clear buffers before pass
                for L in CANDIDATE_LAYERS:
                    hook_buffers[L].clear()
                    
                with torch.no_grad():
                    vjepa2_module(pixel_values_videos=inp)
                    
                # Store pooled activations
                valid_pass = True
                for L in CANDIDATE_LAYERS:
                    if not hook_buffers[L]:
                        valid_pass = False
                        break
                        
                if valid_pass:
                    for L in CANDIDATE_LAYERS:
                        X_data[L][cat].append(hook_buffers[L][-1].squeeze(0).numpy().copy())
                    t_start_tr = int(c * CLIP_DURATION)
                    t_end_tr = min(t_start_tr + CLIP_DURATION, 30)
                    if cat == "gore":
                        y_data["gore"].append(float(y_tr[t_start_tr:t_end_tr].mean()))
                    else:
                        y_data["porn"].append(float(y_tr[t_start_tr:t_end_tr].mean()))
                        
            del reader
            torch.cuda.empty_cache()
            gc.collect()

    # Remove hooks
    for handle in hook_handles:
        handle.remove()
    del model, vjepa2_module, encoder_blocks
    torch.cuda.empty_cache()
    gc.collect()
    
    # Package results
    results = {}
    for L in CANDIDATE_LAYERS:
        X_gore = np.stack(X_data[L]["gore"])
        y_gore = np.array(y_data["gore"])
        X_porn = np.stack(X_data[L]["porn"])
        y_porn = np.array(y_data["porn"])
        results[L] = {"X_gore": X_gore, "y_gore": y_gore, "X_porn": X_porn, "y_porn": y_porn}
        
    return results

# ── Step 2: Compute PCA & Explained Variance for Each Layer ──────────────────

def compute_directions_for_layers(activations):
    print("\n=== Step 2: Computing SVD Directions & Explained Variance ===")
    directions = {}
    
    for L in CANDIDATE_LAYERS:
        print(f"Layer {L}:")
        data = activations[L]
        
        # Gore SVD
        X_g, y_g = data["X_gore"], data["y_gore"]
        w_g = (y_g - y_g.min()) / (y_g.max() - y_g.min() + 1e-9)
        w_g /= w_g.sum()
        X_g_mean = (X_g * w_g[:, None]).sum(axis=0, keepdims=True)
        X_g_c = (X_g - X_g_mean) * np.sqrt(w_g[:, None])
        _, S_g, Vt_g = np.linalg.svd(X_g_c, full_matrices=False)
        var_g = (S_g[0]**2 / (S_g**2).sum())
        dir_g = Vt_g[0].copy()
        if np.corrcoef(X_g @ dir_g, y_g)[0, 1] < 0:
            dir_g *= -1
            
        # Porn SVD
        X_p, y_p = data["X_porn"], data["y_porn"]
        w_p = (y_p - y_p.min()) / (y_p.max() - y_p.min() + 1e-9)
        w_p /= w_p.sum()
        X_p_mean = (X_p * w_p[:, None]).sum(axis=0, keepdims=True)
        X_p_c = (X_p - X_p_mean) * np.sqrt(w_p[:, None])
        _, S_p, Vt_p = np.linalg.svd(X_p_c, full_matrices=False)
        var_p = (S_p[0]**2 / (S_p**2).sum())
        dir_p = Vt_p[0].copy()
        if np.corrcoef(X_p @ dir_p, y_p)[0, 1] < 0:
            dir_p *= -1
            
        print(f"  Gore Top Component Expl. Var: {var_g:.3f}")
        print(f"  Porn Top Component Expl. Var: {var_p:.3f}")
        
        directions[L] = {"gore": dir_g, "porn": dir_p, "var_gore": var_g, "var_porn": var_p}
        
    return directions

# ── Step 3: Run Validation & Evaluate Suppression Delta ───────────────────────

def _dump_cache_structure(label=""):
    """Print the cache directory structure and first lines of every info.jsonl found."""
    print(f"\n[CACHE DUMP{' ' + label if label else ''}] CACHE_DIR={CACHE_DIR.resolve()}")
    if not CACHE_DIR.exists():
        print("  (does not exist)")
        return
    for entry in sorted(CACHE_DIR.iterdir()):
        if not entry.is_dir():
            print(f"  FILE: {entry.name}")
            continue
        files = sorted(entry.iterdir())
        print(f"  DIR:  {entry.name}/  ({len(files)} items)")
        for f in files[:6]:
            if f.name == "info.jsonl":
                try:
                    lines = f.read_text().splitlines()
                    preview = lines[0][:200] if lines else "(empty)"
                    print(f"    info.jsonl[0]: {preview}")
                    if len(lines) > 1:
                        print(f"    info.jsonl[1]: {lines[1][:200]}")
                except OSError as e:
                    print(f"    info.jsonl: (read error: {e})")
            elif f.is_dir():
                sub_files = list(f.iterdir())
                print(f"    subdir: {f.name}/  ({len(sub_files)} items)")
                for sf in sub_files[:3]:
                    if sf.name == "info.jsonl":
                        try:
                            lines = sf.read_text().splitlines()
                            preview = lines[0][:200] if lines else "(empty)"
                            print(f"      info.jsonl[0]: {preview}")
                        except OSError:
                            pass
            else:
                print(f"    {f.name} ({f.stat().st_size} bytes)")
    print("[END CACHE DUMP]\n")


_debug_first_call = [True]

def _nuke_cache_for_video(video_path: Path):
    """
    Nuke every exca cache entry for this video.

    exca may not store the filename in info.jsonl at all — it may key on a
    hash of the serialised input dict.  So the safest strategy is:
      1. Do a filename-string scan (catches cases where path IS in jsonl).
      2. If that finds nothing on the first call, dump the structure so we
         can see what exca actually stores.
      3. Fallback: delete every subdir under CACHE_DIR that contains a .data
         memmap file (any exca activation cache).  Safe because the only
         preds we preserve are in tribe_study/, not CACHE_DIR.
    """
    video_name = video_path.name
    video_stem = video_path.stem
    deleted_by_scan = 0

    if not CACHE_DIR.exists():
        return 0

    # Pass 1: filename-string scan (two levels deep)
    dirs_to_delete = []
    for entry in CACHE_DIR.iterdir():
        if not entry.is_dir():
            continue
        matched = False
        info_file = entry / "info.jsonl"
        if info_file.exists():
            try:
                text = info_file.read_text()
                if video_name in text or video_stem in text:
                    matched = True
            except OSError:
                pass
        if not matched:
            for sub in entry.iterdir():
                if not sub.is_dir():
                    continue
                sub_info = sub / "info.jsonl"
                if sub_info.exists():
                    try:
                        text = sub_info.read_text()
                        if video_name in text or video_stem in text:
                            matched = True
                            break
                    except OSError:
                        pass
        if matched:
            dirs_to_delete.append(entry)

    for d in dirs_to_delete:
        shutil.rmtree(d, ignore_errors=True)
        deleted_by_scan += 1

    if deleted_by_scan == 0 and _debug_first_call[0]:
        _debug_first_call[0] = False
        _dump_cache_structure("before first nuke")

    # Pass 2: fallback — delete every subdir that contains a .data memmap.
    # exca stores activations as <uid>/<key>.data; if the filename scan
    # found nothing the key doesn't embed the path, so we must delete
    # everything that looks like an activation cache.
    deleted_by_fallback = 0
    if CACHE_DIR.exists():
        for entry in list(CACHE_DIR.iterdir()):
            if not entry.is_dir():
                continue
            has_data = any(entry.rglob("*.data"))
            if has_data:
                shutil.rmtree(entry, ignore_errors=True)
                deleted_by_fallback += 1

    total = deleted_by_scan + deleted_by_fallback
    print(f"  [CACHE] scan={deleted_by_scan} fallback={deleted_by_fallback} dirs nuked")
    return total


def _fingerprint_weights(encoder_blocks, layer_indices):
    """Return a small float that changes if any target weight matrix changes."""
    total = 0.0
    for L in layer_indices:
        block = encoder_blocks[L]
        for layer_name in ["attention.value", "attention.proj"]:
            mod = block
            for p in layer_name.split("."):
                mod = getattr(mod, p)
            total += float(mod.weight.data.sum())
    return total


def evaluate_layer_suppression(directions, validation_video):
    print(f"\n=== Step 3: Running Validation Inference on {validation_video.name} ===")

    gore_mask = np.load(GORE_MASK_FILE)
    porn_mask = np.load(PORN_MASK_FILE)

    # Load baseline predictions for the video
    stem = validation_video.stem
    base_preds_path = STUDY_ROOT / "gore" / stem / "preds.npy"
    if not base_preds_path.exists():
        for cat_dir in STUDY_ROOT.iterdir():
            if not cat_dir.is_dir():
                continue
            cand = cat_dir / stem / "preds.npy"
            if cand.exists():
                base_preds_path = cand
                break

    if not base_preds_path.exists():
        raise FileNotFoundError(f"Baseline predictions for {validation_video.name} not found in tribe_study.")

    preds_base = np.load(base_preds_path)[:30]
    base_gore_mean = float(preds_base[:, gore_mask].mean())
    base_porn_mean = float(preds_base[:, porn_mask].mean())
    base_wb_mean   = float(preds_base.mean())

    print(f"Baseline Predictions on {validation_video.name}:")
    print(f"  Gore Mask Mean: {base_gore_mean:.4f}")
    print(f"  Porn Mask Mean: {base_porn_mean:.4f}")
    print(f"  Whole Brain Mean: {base_wb_mean:.4f}")

    comparison_results = []

    # Test each layer individually and a few multi-layer combinations
    test_cases = [(L,) for L in CANDIDATE_LAYERS] + [
        (28, 30, 32),
        (30, 32),
    ]

    for case in test_cases:
        case_name = "+".join(map(str, case))
        print(f"\nEvaluating abliteration on Layer(s): {case_name}")

        # ── 1. Nuke the on-disk cache for this video BEFORE loading the model.
        #    This prevents the freshly-loaded model from immediately reading a
        #    stale cache entry that was written by the previous iteration.
        n_deleted = _nuke_cache_for_video(validation_video)
        print(f"  [CACHE] Deleted {n_deleted} cache dirs for {validation_video.name}")

        # ── 2. Load a fresh model instance.
        model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
        vjepa2_module  = model.data.video_feature.image.model.model
        encoder_blocks = vjepa2_module.encoder.layer

        # ── 3. Also clear the in-memory cache_dict on every infra we can reach,
        #    just in case a previous iteration populated it.
        #    cache_dict is a property that raises ValueError when the infra has
        #    neither folder nor keep_in_ram configured, so catch both.
        for attr_path in [
            "data.video_feature.infra",
            "data.video_feature.image.infra",
        ]:
            try:
                infra = model
                for part in attr_path.split("."):
                    infra = getattr(infra, part)
                for k in list(infra.cache_dict.keys()):
                    del infra.cache_dict[k]
            except (AttributeError, ValueError, Exception):
                pass

        # ── 4. Snapshot weight fingerprint before surgery (sanity check).
        fp_before = _fingerprint_weights(encoder_blocks, case)

        # ── 5. Apply surgery.
        alpha = 0.2
        for L in case:
            dir_g = directions[L]["gore"]
            dir_p = directions[L]["porn"]
            dirs_t = torch.tensor(np.stack([dir_g, dir_p]), dtype=torch.float32).to(DEVICE)

            ortho = []
            for d in dirs_t:
                for q in ortho:
                    d = d - (d @ q) * q
                if d.norm() > 1e-6:
                    ortho.append(d / d.norm())
            ortho = torch.stack(ortho)

            block = encoder_blocks[L]
            for layer_name in ["attention.value", "attention.proj"]:
                mod = block
                for p in layer_name.split("."):
                    mod = getattr(mod, p)
                W = mod.weight.data.clone()
                for q in ortho:
                    W -= alpha * (W @ q).unsqueeze(-1) * q
                mod.weight.data = W

        # ── 6. Confirm weights actually changed.
        fp_after = _fingerprint_weights(encoder_blocks, case)
        weight_delta = abs(fp_after - fp_before)
        if weight_delta < 1e-6:
            print(f"  [WARN] Weight fingerprint unchanged (Δ={weight_delta:.2e}) — surgery may not have applied!")
        else:
            print(f"  [OK] Weights modified (fingerprint Δ={weight_delta:.4f})")

        # ── 7. Run inference.  Because the cache was nuked in step 1 and the
        #    in-memory dict was cleared in step 3, exca must re-run the forward
        #    pass through the now-modified weights.
        df = make_video_only_df(validation_video)
        preds_abl, _ = model.predict(events=df)
        preds_abl = preds_abl[:30]

        abl_gore_mean = float(preds_abl[:, gore_mask].mean())
        abl_porn_mean = float(preds_abl[:, porn_mask].mean())
        abl_wb_mean   = float(preds_abl.mean())

        gore_diff = abl_gore_mean - base_gore_mean
        gore_pct  = 100 * gore_diff / (abs(base_gore_mean) + 1e-9)

        porn_diff = abl_porn_mean - base_porn_mean
        porn_pct  = 100 * porn_diff / (abs(base_porn_mean) + 1e-9)

        wb_diff = abl_wb_mean - base_wb_mean
        wb_pct  = 100 * wb_diff / (abs(base_wb_mean) + 1e-9)

        print(f"  Gore Mask: {base_gore_mean:.4f} -> {abl_gore_mean:.4f} (Δ={gore_diff:+.4f}, {gore_pct:+.1f}%)")
        print(f"  Porn Mask: {base_porn_mean:.4f} -> {abl_porn_mean:.4f} (Δ={porn_diff:+.4f}, {porn_pct:+.1f}%)")
        print(f"  Whole Brain: {base_wb_mean:.4f} -> {abl_wb_mean:.4f} (Δ={wb_diff:+.4f}, {wb_pct:+.1f}%)")

        comparison_results.append({
            "layers": case_name,
            "gore_delta": gore_diff,
            "gore_pct": gore_pct,
            "porn_delta": porn_diff,
            "porn_pct": porn_pct,
            "wb_delta": wb_diff,
            "wb_pct": wb_pct,
        })

        del model, vjepa2_module
        torch.cuda.empty_cache()
        gc.collect()

    return comparison_results

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Choose validation video (gore1.mp4)
    val_videos = sorted(VAL_DIR.glob("*.mp4"))
    if not val_videos:
        print("✗ No validation videos found in val_data/!")
        sys.exit(1)
        
    val_video = val_videos[0]
    
    # Run pipeline
    activations = collect_diagnostic_activations()
    directions = compute_directions_for_layers(activations)
    results = evaluate_layer_suppression(directions, val_video)
    
    # Print final comparison table
    print("\n" + "="*80)
    print("FINAL ABLITERATION DIAGNOSIS REPORT")
    print("="*80)
    print(f"{'Layer(s)':15s} | {'Gore Delta %':14s} | {'Porn Delta %':14s} | {'Whole Brain Delta %':18s}")
    print("-" * 80)
    for r in results:
        print(f"{r['layers']:15s} | {r['gore_pct']:+13.1f}% | {r['porn_pct']:+13.1f}% | {r['wb_pct']:+17.1f}%")
    print("="*80)
    
    # Identify the optimal configuration
    # We want max negative gore delta (max suppression) with min absolute whole brain delta (min distortion)
    best_single_layer = None
    best_score = -999999
    
    for r in results:
        # Simple heuristic: suppression ratio minus absolute side-effect
        score = -r['gore_pct'] - abs(r['wb_pct'])
        if score > best_score:
            best_score = score
            best_config = r
            
    print(f"\nRecommended Optimal Configuration:")
    print(f"  Layer(s): {best_config['layers']}")
    print(f"  Gore Suppression: {best_config['gore_pct']:.1f}%")
    print(f"  Whole Brain Side-effect: {best_config['wb_pct']:.1f}%")
    print("="*80)