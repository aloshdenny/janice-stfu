import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tribev2.demo_utils import TribeModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gc, subprocess, json

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("./data")
STUDY_ROOT = Path("./tribe_study")
CACHE_DIR  = Path("./cache")
OUT_DIR    = Path("./diagnosis")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORIES = ["porn", "gore", "cute", "nature", "food", "kissing", "chase", "fight"]
N_VIDEOS_PER_CAT = 16   # all videos

# ── Video / model constants ───────────────────────────────────────────────────
# Resolution: never touched — ffmpeg outputs native H×W pixels.
# Duration:   every second of every video is covered.  No cap on clips.
CLIP_FRAMES   = 16     # frames sampled per 4-second clip
CLIP_DURATION = 4.0    # seconds (unchanged from original study)

from torchvision import transforms

normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

# ── Load model ────────────────────────────────────────────────────────────────

model           = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
lightning_model = model._model
vjepa2_module   = model.data.video_feature.image.model.model
encoder_blocks  = vjepa2_module.encoder.layer      # ModuleList, 40 blocks
N_LAYERS        = len(encoder_blocks)
print(f"V-JEPA2 encoder blocks: {N_LAYERS}")

vjepa2_module.eval()
vjepa2_module.to(DEVICE)

# ── Inspect brain encoder ─────────────────────────────────────────────────────

print("\n=== Lightning model children ===")
for name, child in lightning_model.named_children():
    print(f"  .{name} → {type(child).__name__}")
    for subname, sub in child.named_children():
        print(f"    .{subname} → {type(sub).__name__}")

print("\n=== Linear layers in brain encoder ===")
for name, mod in lightning_model.named_modules():
    if isinstance(mod, nn.Linear):
        print(f"  .{name}  in={mod.in_features}  out={mod.out_features}")

# ── Memory helpers ────────────────────────────────────────────────────────────

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def gpu_mem_gb():
    return torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

# ── Hook infrastructure ───────────────────────────────────────────────────────

layer_acts = {}   # layer_idx → np.ndarray (hidden_dim,)

def make_hook(idx):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # Mean over all spatial/temporal tokens → scalar vector per sample.
        # hidden shape: (batch, n_tokens, hidden_dim)
        layer_acts[idx] = hidden.mean(dim=1)[0].detach().float().cpu().numpy()
    return hook

def register_all_hooks():
    return [encoder_blocks[i].register_forward_hook(make_hook(i))
            for i in range(N_LAYERS)]

def remove_hooks(handles):
    for h in handles:
        h.remove()

# ── Video probing (zero RAM) ──────────────────────────────────────────────────

def probe_video(path: Path):
    """Return (fps, total_duration_sec, width, height) via ffprobe. No RAM cost."""
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", str(path)]
    out = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(out.stdout)
    vs = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if vs is None:
        raise RuntimeError(f"No video stream: {path}")

    num, den = vs.get("avg_frame_rate", "30/1").split("/")
    fps = float(num) / max(float(den), 1e-9)

    # Prefer explicit duration field; fall back to nb_frames/fps
    dur = vs.get("duration") or info.get("format", {}).get("duration")
    if dur:
        duration = float(dur)
    else:
        nb = int(vs.get("nb_frames", 0))
        duration = nb / fps if nb else 30.0

    return fps, duration, int(vs["width"]), int(vs["height"])


# ── Streaming clip decoder ────────────────────────────────────────────────────
# Peak RAM per clip: CLIP_FRAMES × 3 × H × W × 4 B  (≈ 168 MB at 1280×720)
# The full video is NEVER loaded into memory.

def decode_clip(path: Path, start_sec: float, dur_sec: float,
                n_frames: int, w: int, h: int) -> np.ndarray:
    """
    Decode exactly n_frames evenly-spaced frames from [start_sec, start_sec+dur_sec].
    Returns float32 (n_frames, 3, h, w) in [0, 1]. Native resolution preserved.
    """
    target_fps = n_frames / dur_sec
    cmd = [
        "ffmpeg", "-v", "quiet",
        "-ss", f"{start_sec:.6f}",      # seek before -i = fast keyframe seek
        "-t",  f"{dur_sec:.6f}",
        "-i",  str(path),
        "-vf", f"fps={target_fps:.6f}",
        "-frames:v", str(n_frames),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    raw    = result.stdout
    need   = n_frames * h * w * 3

    if len(raw) < need:
        if len(raw) == 0:
            raise RuntimeError("ffmpeg returned 0 bytes")
        # Pad by repeating the last complete frame
        frame_bytes = h * w * 3
        last_frame  = raw[-(len(raw) // frame_bytes) * frame_bytes
                          or -frame_bytes:][-frame_bytes:]
        raw += last_frame * ((need - len(raw) + frame_bytes - 1) // frame_bytes)

    frames = np.frombuffer(raw[:need], dtype=np.uint8) \
               .reshape(n_frames, h, w, 3) \
               .astype(np.float32) / 255.0
    return frames.transpose(0, 3, 1, 2)   # (T, 3, H, W)


def iter_clips(path: Path):
    """
    Yield all non-overlapping 4-second clips from the video, one at a time.
    Includes the final partial clip if ≥ 1 s of footage remains.
    No clip is skipped, no second of footage is discarded.
    """
    fps, total_dur, w, h = probe_video(path)
    n_clips = max(1, int(total_dur // CLIP_DURATION))
    # Include tail: if there's ≥ 1 s left after the last full clip, add one more
    if total_dur - n_clips * CLIP_DURATION >= 1.0:
        n_clips += 1

    for c in range(n_clips):
        start  = c * CLIP_DURATION
        actual = min(CLIP_DURATION, total_dur - start)
        if actual < 0.5:          # < 0.5 s is too short to sample meaningfully
            break
        # For a partial tail clip, sample proportionally fewer frames but still
        # return CLIP_FRAMES by sampling from whatever is available
        frames_np = decode_clip(path, start, actual, CLIP_FRAMES, w, h)
        clip = torch.from_numpy(frames_np)                 # (T, 3, H, W)
        clip = torch.stack([normalize_fn(clip[t]) for t in range(CLIP_FRAMES)])
        yield clip    # (CLIP_FRAMES, 3, H, W)
        del frames_np, clip


# ── Forward pass with bf16 autocast (RTX 4090) ───────────────────────────────
# bf16 halves VRAM and ~doubles throughput with no meaningful accuracy loss
# for the purpose of reading internal activations.

USE_AMP = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def run_forward(clip_tensor):
    """Run one clip through V-JEPA2 encoder; fills layer_acts in-place."""
    inp = clip_tensor.unsqueeze(0).to(DEVICE)   # (1, T, C, H, W)
    with torch.no_grad():
        if USE_AMP:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vjepa2_module(pixel_values_videos=inp)
        else:
            vjepa2_module(pixel_values_videos=inp)
    del inp


# ── Pass 1: collect per-layer activations for all categories ─────────────────
# Stored shape per category: (total_clips, N_LAYERS, hidden_dim)
# At 16 videos × 7 clips × 40 layers × 1408 floats ≈ 25 MB — trivial.

cat_layer_acts = {}

for cat in CATEGORIES:
    video_paths = sorted(DATA_DIR.glob(f"{cat}*.mp4"))[:N_VIDEOS_PER_CAT]
    if not video_paths:
        print(f"  [SKIP] no videos for {cat}")
        continue

    all_clips = []
    n_errors  = 0

    for vp in video_paths:
        handles = register_all_hooks()
        try:
            for clip in iter_clips(vp):
                layer_acts.clear()
                run_forward(clip)
                if len(layer_acts) == N_LAYERS:
                    all_clips.append(
                        np.stack([layer_acts[i] for i in range(N_LAYERS)])
                    )   # (N_LAYERS, hidden_dim)
                free_memory()
        except Exception as e:
            n_errors += 1
            print(f"  [ERROR] {vp.name}: {e}")
        finally:
            remove_hooks(handles)

    if all_clips:
        cat_layer_acts[cat] = np.stack(all_clips)   # (n_clips, N_LAYERS, hidden_dim)
        print(f"  {cat}: {len(all_clips)} clips from {len(video_paths)} videos  "
              f"shape={cat_layer_acts[cat].shape}  errors={n_errors}  "
              f"GPU={gpu_mem_gb():.1f} GB")

    free_memory()


# ── Plotting helpers ──────────────────────────────────────────────────────────

DARK_BG   = "#0d0d0d"
DARK_AXES = "#1a1a1a"
GRID_COL  = "#333333"
SPINE_COL = "#555555"
layer_indices = np.arange(N_LAYERS)

TRIBE_FRACS = [(0.5, "L×0.5"), (0.75, "L×0.75"), (1.0, "L×1.0")]

def style_ax(ax):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(SPINE_COL)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    for frac, lbl in TRIBE_FRACS:
        li = int(N_LAYERS * frac) - 1
        ax.axvline(li, color="#666", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.text(li + 0.3, ax.get_ylim()[1] * 0.97, lbl,
                color="#888", fontsize=7, va="top")


# ── Plot 1: BAR GRAPH — avg activation magnitude per layer per category ───────
# Each category gets its own subplot so bars don't overlap and are readable.

print("\nPlot 1: bar graph of mean activation magnitude per layer …")

n_cats  = len(cat_layer_acts)
fig, axes = plt.subplots(n_cats, 1, figsize=(18, 3.5 * n_cats), sharex=True)
fig.patch.set_facecolor(DARK_BG)
if n_cats == 1:
    axes = [axes]

cat_colors = {
    cat: c for cat, c in zip(
        CATEGORIES,
        plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES)))
    )
}

for ax, (cat, acts) in zip(axes, cat_layer_acts.items()):
    # acts: (n_clips, N_LAYERS, hidden_dim)
    norms = np.linalg.norm(acts, axis=-1).mean(axis=0)   # (N_LAYERS,)
    color = cat_colors[cat]
    ax.bar(layer_indices, norms, color=color, alpha=0.85, width=0.85)
    ax.set_ylabel("Mean L2 norm", color="white")
    ax.set_title(f"{cat}  ({acts.shape[0]} clips)", color="white")
    style_ax(ax)

axes[-1].set_xlabel("Layer index (0 → 39)", color="white")
fig.suptitle("V-JEPA2 per-layer activation magnitude — one bar = one layer",
             color="white", fontsize=13, y=1.002)
plt.tight_layout()
plt.savefig(OUT_DIR / "plot1_activation_magnitude_bars.png", dpi=150,
            bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved → {OUT_DIR}/plot1_activation_magnitude_bars.png")


# ── Plot 2: Overlay line — differential activation vs neutral baseline ─────────

print("Plot 2: differential activation (target − neutral baseline) …")

neutral_cats = [c for c in ["cute", "nature", "food"] if c in cat_layer_acts]
if neutral_cats:
    neutral_mean_norms = np.stack([
        np.linalg.norm(cat_layer_acts[c], axis=-1).mean(axis=0)
        for c in neutral_cats
    ]).mean(axis=0)   # (N_LAYERS,)

    target_cats = [c for c in ["porn", "gore", "kissing", "chase", "fight"]
                   if c in cat_layer_acts]

    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    for cat in target_cats:
        acts      = cat_layer_acts[cat]
        cat_norms = np.linalg.norm(acts, axis=-1).mean(axis=0)
        diff      = cat_norms - neutral_mean_norms
        ax.plot(layer_indices, diff, label=f"{cat} − neutral",
                color=cat_colors.get(cat, "white"), linewidth=2, alpha=0.9)

    ax.axhline(0, color="#555", linewidth=1)
    ax.fill_between(layer_indices,
                    [0] * N_LAYERS, [0] * N_LAYERS,
                    alpha=0)   # dummy for consistent legend spacing
    ax.set_xlabel("Layer index (0 → 39)", color="white")
    ax.set_ylabel("Δ L2 norm vs neutral mean", color="white")
    ax.set_title("Per-layer differential activation  (target content − neutral baseline)",
                 color="white")
    ax.legend(facecolor=DARK_AXES, labelcolor="white", framealpha=0.9)
    style_ax(ax)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "plot2_differential_activation.png", dpi=150,
                bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"  Saved → {OUT_DIR}/plot2_differential_activation.png")
else:
    print("  [SKIP] no neutral-category data available")


# ── Plot 3: Layer → fMRI vertex correlation ───────────────────────────────────
# For each (porn|gore) video that has a matching preds.npy, we:
#   1. load the fMRI predictions (30 TRs × 20484 vertices)
#   2. average across the relevant mask → (30,) vertex signal
#   3. bucket the vertex signal into n_clips bins aligned to the clips
#   4. Pearson-correlate per-layer activation norm vs vertex signal
# Final bar shows mean Pearson r across all contributing videos.

print("Plot 3: layer → fMRI vertex correlation …")

gore_mask = np.load("./abliterated/gore_mask.npy")
porn_mask = np.load("./abliterated/porn_mask.npy")

layer_corrs     = {"porn": np.zeros(N_LAYERS), "gore": np.zeros(N_LAYERS)}
video_count     = {"porn": 0,                  "gore": 0}

for cat in ["porn", "gore"]:
    mask        = porn_mask if cat == "porn" else gore_mask
    video_paths = sorted(DATA_DIR.glob(f"{cat}*.mp4"))[:N_VIDEOS_PER_CAT]

    for vp in video_paths:
        preds_path = STUDY_ROOT / cat / vp.stem / "preds.npy"
        if not preds_path.exists():
            continue

        preds         = np.load(preds_path)[:30]            # (30, 20484)
        vertex_signal = preds[:, mask].mean(axis=1)         # (30,)

        clip_acts = []
        handles   = register_all_hooks()
        try:
            for clip in iter_clips(vp):
                layer_acts.clear()
                run_forward(clip)
                if len(layer_acts) == N_LAYERS:
                    clip_acts.append(
                        np.stack([layer_acts[i] for i in range(N_LAYERS)])
                    )
                free_memory()
        except Exception as e:
            print(f"  [ERROR] {vp.name}: {e}")
        finally:
            remove_hooks(handles)

        if not clip_acts:
            continue

        clip_acts = np.stack(clip_acts)   # (n_clips, N_LAYERS, hidden_dim)
        n_clips   = len(clip_acts)

        # Align vertex signal to clip timeline by bucketing the 30 TRs
        bucket_s   = 30.0 / n_clips
        y_bucketed = np.array([
            vertex_signal[int(c * bucket_s): max(int(c * bucket_s) + 1,
                                                  int((c + 1) * bucket_s))].mean()
            for c in range(n_clips)
        ])

        if y_bucketed.std() < 1e-9:
            continue   # flat signal — no information

        # Per-layer Pearson r
        per_layer_r = np.zeros(N_LAYERS)
        for li in range(N_LAYERS):
            x = np.linalg.norm(clip_acts[:, li, :], axis=-1)   # (n_clips,)
            if x.std() > 1e-9:
                per_layer_r[li] = float(np.corrcoef(x, y_bucketed)[0, 1])

        layer_corrs[cat] += per_layer_r
        video_count[cat] += 1

        del clip_acts
        free_memory()

    # Average over contributing videos
    if video_count[cat]:
        layer_corrs[cat] /= video_count[cat]
        print(f"  {cat}: averaged over {video_count[cat]} videos with preds")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
fig.patch.set_facecolor(DARK_BG)

for ax, cat, color in [
    (axes[0], "porn", "#ff6b6b"),
    (axes[1], "gore", "#ffa94d"),
]:
    corrs = layer_corrs[cat]
    ax.set_facecolor(DARK_BG)
    bars = ax.bar(layer_indices, corrs, color=color, alpha=0.75, width=0.85)

    # Highlight top-5 by absolute r
    top5 = np.argsort(np.abs(corrs))[-5:]
    for li in top5:
        bars[li].set_alpha(1.0)
        bars[li].set_edgecolor("white")
        bars[li].set_linewidth(0.8)
        ax.text(li, corrs[li] + 0.004 * np.sign(corrs[li]) + 1e-9,
                str(li), color="white", fontsize=7, ha="center", va="bottom")

    ax.axhline(0, color="#555", linewidth=1)
    ax.set_ylabel("Mean Pearson r", color="white")
    n = video_count[cat]
    ax.set_title(
        f"Layer activation → fMRI vertex signal  [{cat} mask, n={n} videos]",
        color="white"
    )
    style_ax(ax)

axes[1].set_xlabel("Layer index (0 → 39)", color="white")
plt.tight_layout()
plt.savefig(OUT_DIR / "plot3_layer_fmri_correlation.png", dpi=150,
            bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved → {OUT_DIR}/plot3_layer_fmri_correlation.png")


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n=== Best layers for abliteration (by fMRI correlation) ===")
for cat in ["porn", "gore"]:
    print(f"\n  Top 10 layers correlated with {cat}_mask vertices:")
    for li in np.argsort(layer_corrs[cat])[-10:][::-1]:
        r = layer_corrs[cat][li]
        print(f"    layer {li:2d}  r={r:+.4f}  depth={li/N_LAYERS:.2f}")

print(f"\nAll outputs saved to {OUT_DIR.resolve()}")