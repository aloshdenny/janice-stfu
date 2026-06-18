"""
validation.py — Validates abliterated model against baseline preds.

Approach: Load the abliterated checkpoint directly into the vjepa2 encoder,
clear all caches, and run inference. Compare against baseline preds from
tribe_study/. No monkeypatching — works for surgery on ANY layer.

Also generates side-by-side brain scan PNGs (pre vs post) for each video,
saved to val_results/{video_stem}.png.
"""

import os, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse
import json
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from tribev2.demo_utils import TribeModel
import gc, subprocess, shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize as MplNorm
from matplotlib.cm import ScalarMappable

# ── Parse Command Line Arguments ──────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Validate abliterated model.")
parser.add_argument("--target", type=str, required=True,
                    help="Target category (e.g., 'porn')")
args = parser.parse_args()

VAL_DIR     = Path("./val_data")
OUT_DIR     = Path("./abliterated")
CACHE_BASE  = Path("./cache")
STUDY_ROOT  = Path("./tribe_study")
VAL_RESULTS = Path("./val_results")
VAL_RESULTS.mkdir(exist_ok=True)

# ── Load surgery log ──────────────────────────────────────────────────────────

surgery_log_path = OUT_DIR / "surgery_log.json"
if not surgery_log_path.exists():
    raise FileNotFoundError(f"surgery_log.json not found in {OUT_DIR}. Run abliteration.py first.")

with open(surgery_log_path) as f:
    surgery_log = json.load(f)

OPERATED_LAYERS = sorted([int(k) for k in surgery_log["layers_operated"].keys()])
TOLERANCE = surgery_log["tolerance"]

print(f"Target: {args.target}")
print(f"Tolerance: {TOLERANCE}")
print(f"Operated layers: {OPERATED_LAYERS}")
print(f"Mask: {surgery_log['mask_used']} (score={surgery_log['mask_score']:.6f})")
print(f"Components per layer: {surgery_log['n_components']}")

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_duration(video_path):
    r = subprocess.run(["ffprobe","-v","error","-show_entries","format=duration",
                        "-of","default=noprint_wrappers=1:nokey=1",str(video_path)],
                       capture_output=True, text=True)
    try: return round(float(r.stdout.strip())-0.1, 3)
    except: return 29.9

def make_video_only_df(video_path):
    dur = get_duration(video_path)
    return pd.DataFrame([{"type":"Video","start":0.0,"duration":dur,
        "timeline":"default","subject":"default","session":"","task":"","run":"",
        "filepath":str(video_path.resolve()),"frequency":60.0,"offset":0.0,
        "stop":dur,"context":float("nan")}])

# ── Load Target Mask + baseline preds ─────────────────────────────────────────

target_mask_path = OUT_DIR / f"{args.target}_mask.npy"
if not target_mask_path.exists():
    raise FileNotFoundError(f"Mask file not found: {target_mask_path}")

target_mask = np.load(target_mask_path)
print(f"\nTarget ({args.target}) mask: {target_mask.sum()} verts")

val_videos = sorted(VAL_DIR.glob("*.mp4"))
print(f"Found {len(val_videos)} validation videos")

baseline_preds = {}
for vp in val_videos:
    for cat_dir in STUDY_ROOT.iterdir():
        if not cat_dir.is_dir(): continue
        cand = cat_dir / vp.stem / "preds.npy"
        if cand.exists():
            baseline_preds[vp.stem] = np.load(cand)[:30]
            print(f"  {vp.name}: baseline from {cand.relative_to(STUDY_ROOT)}")
            break
    else:
        print(f"  [WARN] {vp.name}: no baseline")

# ── Brain rendering setup ─────────────────────────────────────────────────────

print("\nLoading fsaverage5 mesh for brain rendering...")
try:
    from nilearn import datasets, surface as nisurf
    fsaverage  = datasets.fetch_surf_fsaverage(mesh="fsaverage5")
    lh_coords, lh_faces = nisurf.load_surf_mesh(fsaverage["pial_left"])
    rh_coords, rh_faces = nisurf.load_surf_mesh(fsaverage["pial_right"])
    lh_sulc = nisurf.load_surf_data(fsaverage["sulc_left"])
    rh_sulc = nisurf.load_surf_data(fsaverage["sulc_right"])
    def _norm01(x): return (x - x.min()) / (x.max() - x.min() + 1e-9)
    lh_sulc = _norm01(lh_sulc)
    rh_sulc = _norm01(rh_sulc)
    BRAIN_RENDER_AVAILABLE = True
    print("  fsaverage5 loaded OK")
except Exception as e:
    print(f"  [WARN] nilearn not available, brain rendering disabled: {e}")
    BRAIN_RENDER_AVAILABLE = False

_hot = plt.get_cmap("hot")

def _vertex_colors(sulc, activation, threshold_pct=80):
    """Return (N,3) uint8 RGB array blending hot colormap over sulcal gray."""
    thresh = float(np.nanpercentile(np.abs(activation), threshold_pct))
    vmax   = float(np.nanpercentile(np.abs(activation), 99)) + 1e-9
    base   = (120 + sulc * 100).astype(np.float32)
    rgb    = np.stack([base, base, base], axis=1)       # (N,3) gray
    above  = np.abs(activation) >= thresh
    t      = np.clip((np.abs(activation[above]) - thresh) / (vmax - thresh), 0, 1)
    hot_rgb = (_hot(t)[:, :3] * 255).astype(np.uint8)
    rgb[above] = hot_rgb
    return rgb.astype(np.uint8)


def _render_lateral_view(coords, faces, sulc, activation,
                          ax, title, vmax_shared=None, threshold_pct=80):
    """
    Software-render a lateral brain view using matplotlib trisurf projection.
    Projects 3-D mesh vertices onto 2-D using a simple lateral (side) view.
    """
    # Lateral view: project onto (y, z) plane (right hemisphere is mirrored)
    vx = coords[:, 1]   # anterior-posterior
    vy = coords[:, 2]   # superior-inferior

    threshold_pct = threshold_pct
    thresh = float(np.nanpercentile(np.abs(activation), threshold_pct))
    vmax   = vmax_shared if vmax_shared is not None else \
             float(np.nanpercentile(np.abs(activation), 99)) + 1e-9

    # Base sulcal gray: 0.35–0.65 range
    face_sulc = sulc[faces].mean(axis=1)
    gray_val  = 0.35 + face_sulc * 0.30

    # Per-face activation (mean of vertices)
    face_act  = activation[faces].mean(axis=1)
    face_above = np.abs(face_act) >= thresh
    t          = np.clip((np.abs(face_act) - thresh) / (vmax - thresh + 1e-9), 0, 1)

    # Build RGBA for each face
    n_faces = len(faces)
    face_colors = np.zeros((n_faces, 4), dtype=np.float32)
    # gray for below-threshold
    face_colors[~face_above, :3] = gray_val[~face_above, None]
    face_colors[~face_above, 3]  = 1.0
    # hot colormap for above-threshold
    if face_above.any():
        hot_rgba = _hot(t[face_above])
        face_colors[face_above] = hot_rgba

    # Depth sort (painter's algorithm) — faces furthest in x drawn first
    depth = coords[faces, 0].mean(axis=1)   # lateral depth
    order = np.argsort(depth)

    ax.set_facecolor("#0d0d0d")
    ax.set_aspect("equal")
    ax.axis("off")

    from matplotlib.collections import PolyCollection
    verts2d = np.stack([vx[faces], vy[faces]], axis=2)  # (F,3,2)
    verts2d = verts2d[order]
    fc      = face_colors[order]

    pc = PolyCollection(verts2d, facecolors=fc, edgecolors="none", linewidths=0)
    ax.add_collection(pc)
    ax.set_xlim(vx.min()-5, vx.max()+5)
    ax.set_ylim(vy.min()-5, vy.max()+5)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=8)


def render_brain_comparison(stem, preds_base, preds_abl, target_mask,
                             base_val, abl_val, diff, pct):
    """
    Generate a side-by-side brain scan comparison PNG for one video.
    Layout: Left = Pre-abliteration (LH + RH lateral), Right = Post-abliteration.
    """
    if not BRAIN_RENDER_AVAILABLE:
        return

    # Mean prediction across TRs: shape (n_verts,)
    act_pre  = preds_base.mean(axis=0).astype(np.float32)
    act_post = preds_abl.mean(axis=0).astype(np.float32)

    # Shared colour scale across both panels so comparison is fair
    vmax_shared = float(np.nanpercentile(np.abs(act_pre), 99)) + 1e-9

    lh_pre  = act_pre[:10242];   rh_pre  = act_pre[10242:]
    lh_post = act_post[:10242];  rh_post = act_post[10242:]

    # ── Figure layout ────────────────────────────────────────────────────────
    # 2 rows × 4 cols: [LH_pre | RH_pre | LH_post | RH_post]
    # with a title bar on top and a stats bar on bottom

    fig = plt.figure(figsize=(20, 9), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(3, 4, figure=fig,
                            height_ratios=[0.06, 1.0, 0.06],
                            hspace=0.08, wspace=0.04)

    # ── Title row ────────────────────────────────────────────────────────────
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.set_facecolor("#0d0d0d")
    tag = args.target.upper()
    fig.text(0.5, 0.97,
             f"TRIBE v2 — Abliteration Comparison  |  {stem}  |  target={tag}  |  "
             f"tolerance={TOLERANCE}  |  layers={OPERATED_LAYERS}",
             ha="center", va="top", color="white", fontsize=12, fontweight="bold")

    # ── Section labels ───────────────────────────────────────────────────────
    fig.text(0.25, 0.91, "PRE-ABLITERATION",
             ha="center", color="#aaaaaa", fontsize=10)
    fig.text(0.75, 0.91, "POST-ABLITERATION",
             ha="center", color="#aaaaaa", fontsize=10)

    # Divider line between pre and post
    fig.add_artist(plt.matplotlib.lines.Line2D([0.5, 0.5], [0.08, 0.93],
                   transform=fig.transFigure, color="#444", linewidth=1.2))

    # ── Brain panels ─────────────────────────────────────────────────────────
    ax_lh_pre  = fig.add_subplot(gs[1, 0])
    ax_rh_pre  = fig.add_subplot(gs[1, 1])
    ax_lh_post = fig.add_subplot(gs[1, 2])
    ax_rh_post = fig.add_subplot(gs[1, 3])

    _render_lateral_view(lh_coords, lh_faces, lh_sulc, lh_pre,
                         ax_lh_pre,  "Left Hemisphere", vmax_shared)
    _render_lateral_view(rh_coords, rh_faces, rh_sulc, rh_pre,
                         ax_rh_pre,  "Right Hemisphere", vmax_shared)
    _render_lateral_view(lh_coords, lh_faces, lh_sulc, lh_post,
                         ax_lh_post, "Left Hemisphere", vmax_shared)
    _render_lateral_view(rh_coords, rh_faces, rh_sulc, rh_post,
                         ax_rh_post, "Right Hemisphere", vmax_shared)

    # ── Colorbar ─────────────────────────────────────────────────────────────
    cbar_ax = fig.add_axes([0.92, 0.18, 0.012, 0.60])
    sm = ScalarMappable(cmap="hot",
                        norm=MplNorm(vmin=0, vmax=vmax_shared))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Activity", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    cbar_ax.set_facecolor("#0d0d0d")
    fig.text(0.945, 0.80, "High", ha="center", va="bottom", color="white", fontsize=8)
    fig.text(0.945, 0.17, "Low",  ha="center", va="top",    color="white", fontsize=8)

    # ── Stats bar ────────────────────────────────────────────────────────────
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis("off")
    ax_stats.set_facecolor("#0d0d0d")

    tag_sym  = "✓" if diff < -0.005 else ("✗" if abs(diff) < 0.005 else "↑")
    tag_word = "suppressed" if diff < -0.005 else ("no change" if abs(diff) < 0.005 else "increased")
    tag_col  = "#22dd66" if diff < -0.005 else ("#ff5555" if diff > 0.005 else "#aaaaaa")

    stats_txt = (
        f"mask region ({int(target_mask.sum())} verts)   "
        f"base = {base_val:.4f}   →   post = {abl_val:.4f}   "
        f"Δ = {diff:+.4f} ({pct:+.1f}%)"
    )
    fig.text(0.5, 0.035, stats_txt,
             ha="center", color="white", fontsize=10)
    fig.text(0.5, 0.012, f"{tag_sym}  {tag_word}",
             ha="center", color=tag_col, fontsize=11, fontweight="bold")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = VAL_RESULTS / f"{stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#0d0d0d", edgecolor="none")
    plt.close(fig)
    print(f"  Brain scan saved → {out_path}")


# ── Load model + abliterated weights ──────────────────────────────────────────

abliterated_ckpt = OUT_DIR / "vjepa2_abliterated.pt"
if not abliterated_ckpt.exists():
    raise FileNotFoundError(f"Abliterated checkpoint not found: {abliterated_ckpt}")

print(f"\nLoading model...")
model          = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_BASE)
vjepa2_module  = model.data.video_feature.image.model.model
encoder_blocks = vjepa2_module.encoder.layer
N_LAYERS       = len(encoder_blocks)

print(f"Loading abliterated checkpoint: {abliterated_ckpt.name}")
state_dict = torch.load(abliterated_ckpt, map_location="cpu")
vjepa2_module.load_state_dict(state_dict)
del state_dict
vjepa2_module.eval()
print(f"  Loaded — surgery on layers {OPERATED_LAYERS} is baked into the weights")

# ── Clear ALL caches ──────────────────────────────────────────────────────────

cache_dict = model.data.video_feature.infra.cache_dict
item_uid   = model.data.video_feature.infra.item_uid
helper     = model.data.video_feature._event_types_helper

print("\nClearing caches for validation videos...")
for vp in val_videos:
    events = helper.extract(make_video_only_df(vp))
    for ev in events:
        key = item_uid(ev)
        if key in cache_dict:
            del cache_dict[key]

val_names = {vp.name for vp in val_videos}
disk_cleared = 0
info_files = list(CACHE_BASE.rglob("*info.jsonl"))
for info_file in info_files:
    try:
        if info_file.exists() and any(n in info_file.read_text() for n in val_names):
            shutil.rmtree(info_file.parent)
            disk_cleared += 1
    except Exception:
        pass
print(f"  In-memory: cleared entries for {len(val_videos)} videos")
print(f"  Disk: cleared {disk_cleared} cache dirs")

# ── Inference loop ────────────────────────────────────────────────────────────

results = []
for vp in val_videos:
    stem = vp.stem
    if stem not in baseline_preds:
        print(f"\n[SKIP] {vp.name}")
        continue

    print(f"\n{'='*55}\nVideo: {vp.name}")

    preds_base = baseline_preds[stem]
    df = make_video_only_df(vp)
    preds_abl, _ = model.predict(events=df)
    preds_abl = preds_abl[:30]

    # Target mask evaluation
    base_val = float(preds_base[:, target_mask].mean())
    abl_val  = float(preds_abl[:, target_mask].mean())
    diff = abl_val - base_val
    pct  = 100 * diff / (abs(base_val) + 1e-9)
    tag  = "✓ suppressed" if diff < -0.005 else \
           ("✗ no change" if abs(diff) < 0.005 else "↑ increased")
    print(f"  {args.target + '_mask':12s}  base={base_val:.4f}  abl={abl_val:.4f}  "
          f"Δ={diff:+.4f} ({pct:+.1f}%)  {tag}")

    # Whole brain
    bg = float(preds_base.mean()); ag = float(preds_abl.mean())
    print(f"  {'whole_brain':12s}  base={bg:.4f}  abl={ag:.4f}  "
          f"Δ={ag-bg:+.4f} ({100*(ag-bg)/(abs(bg)+1e-9):+.1f}%)")

    # Brain scan comparison PNG
    render_brain_comparison(stem, preds_base, preds_abl, target_mask,
                            base_val, abl_val, diff, pct)

    torch.cuda.empty_cache(); gc.collect()
    results.append({"video": vp.name, "preds_base": preds_base,
                    "preds_abl": preds_abl, "diff": diff, "pct": pct})

# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*55 + "\nSUMMARY")
print(f"  Checkpoint: {abliterated_ckpt.name}")
print(f"  Layers operated: {OPERATED_LAYERS}")
print(f"  Mask: {surgery_log['mask_used']} ({int(target_mask.sum())} verts)")
print(f"  Tolerance: {TOLERANCE}")
print(f"  Brain scans → {VAL_RESULTS}/")
print()

all_diffs = [r["diff"] for r in results]
for r in results:
    pct = r["pct"]; diff = r["diff"]
    tag = "✓" if diff < -0.005 else ("✗" if abs(diff) < 0.005 else "↑")
    print(f"  {r['video']:20s}  Δ={diff:+.4f} ({pct:+.1f}%)  {tag}")

if all_diffs:
    print(f"\n  Mean Δ across all videos: {np.mean(all_diffs):+.4f}")
    target_diffs = [r["diff"] for r in results if r["video"].startswith(args.target)]
    other_diffs  = [r["diff"] for r in results if not r["video"].startswith(args.target)]
    if target_diffs:
        print(f"  Mean Δ on {args.target} videos: {np.mean(target_diffs):+.4f}")
    if other_diffs:
        print(f"  Mean Δ on other videos:  {np.mean(other_diffs):+.4f}")

# Save numpy results
for r in results:
    s = Path(r["video"]).stem
    np.save(OUT_DIR / f"val_{s}_base.npy", r["preds_base"])
    np.save(OUT_DIR / f"val_{s}_abl.npy",  r["preds_abl"])

print("\nDone.")