from tribev2.demo_utils import TribeModel
from pathlib import Path
import numpy as np
import os
import warnings
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from multiprocessing import Process, cpu_count
from concurrent.futures import ThreadPoolExecutor
import time

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

CHUNK_TIMEOUT = 120
CHUNK = 10

# ── Video categories ──────────────────────────────────────────────────────────

DATA_DIR = Path("./data")

VIDEOS = {
    "porn":    ["porn5.mp4",    "porn6.mp4",    "porn7.mp4",    "porn8.mp4"],
    "gore":    ["gore5.mp4",    "gore6.mp4",    "gore7.mp4",    "gore8.mp4"],
    "cute":    ["cute5.mp4",    "cute6.mp4",    "cute7.mp4",    "cute8.mp4"],
    "nature":  ["nature5.mp4",  "nature6.mp4",  "nature7.mp4",  "nature8.mp4"],
    "food":    ["food5.mp4",    "food6.mp4",    "food7.mp4",    "food8.mp4"],
    "kissing": ["kissing5.mp4", "kissing6.mp4", "kissing7.mp4", "kissing8.mp4"],
}

# ── Worker ────────────────────────────────────────────────────────────────────

def render_worker(i, end, preds_chunk, segments_chunk, out_path):
    import warnings, logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)
    os.environ["DISPLAY"] = ":99"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import numpy as np
    from tribev2.plotting import PlotBrain

    try:
        plotter = PlotBrain(mesh="fsaverage5")
        fig = plotter.plot_timesteps(
            preds_chunk,
            segments=segments_chunk,
            cmap="fire",
            norm_percentile=99,
            alpha_cmap=(0, 0.2),
            show_stimuli=False,
        )
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        np.save(out_path, img)
    except Exception as e:
        print(f"  [ERROR] t={i}–{end}: {e}")

# ── Display ───────────────────────────────────────────────────────────────────

os.environ["DISPLAY"] = ":99"
if not os.path.exists("/tmp/.X11-unix/X99"):
    os.system("Xvfb :99 -screen 0 1024x768x24 &")
    time.sleep(1)

# ── Model (load once) ─────────────────────────────────────────────────────────

CACHE_FOLDER = Path("./cache")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_FOLDER)
n_cores = cpu_count()

# ── Per-video pipeline ────────────────────────────────────────────────────────

def process_video(video_path: Path, out_dir: Path):
    """Run inference + chunked render for one video. Saves preds/segments + brain PNG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(exist_ok=True)

    preds_path    = out_dir / "preds.npy"
    segments_path = out_dir / "segments.npy"
    final_path    = out_dir / "brain_full.png"

    if final_path.exists():
        print(f"  [SKIP] Already done: {final_path}")
        return

    # Inference (skip if cached)
    if preds_path.exists() and segments_path.exists():
        print(f"  [CACHED] Loading preds for {video_path.name}")
        preds    = np.load(preds_path)
        segments = np.load(segments_path, allow_pickle=True)
    else:
        print(f"  [INFER] {video_path.name}")
        df = model.get_events_dataframe(video_path=video_path)
        preds, segments = model.predict(events=df)
        print(f"  Predictions shape: {preds.shape}")
        np.save(preds_path, preds)
        np.save(segments_path, segments)

    # Chunked parallel rendering
    total = len(preds)
    chunk_ranges = [(i, min(i + CHUNK, total)) for i in range(0, total, CHUNK)]

    def run_chunk(i, end):
        out_path = chunk_dir / f"chunk_{i:05d}.npy"
        if out_path.exists():
            return
        p = Process(
            target=render_worker,
            args=(i, end, preds[i:end], segments[i:end], str(out_path)),
            daemon=True,
        )
        p.start()
        p.join(timeout=CHUNK_TIMEOUT)
        if p.is_alive():
            print(f"    TIMEOUT t={i}–{end}")
            p.kill(); p.join()
        elif p.exitcode != 0:
            print(f"    CRASHED (exit {p.exitcode}) t={i}–{end}")

    print(f"  Rendering {total} timesteps across {len(chunk_ranges)} chunks...")
    with ThreadPoolExecutor(max_workers=n_cores) as ex:
        for f in [ex.submit(run_chunk, i, end) for i, end in chunk_ranges]:
            f.result()

    # Composite
    row_images = []
    for i, end in chunk_ranges:
        p = chunk_dir / f"chunk_{i:05d}.npy"
        if p.exists():
            row_images.append(np.load(str(p)))
        else:
            row_images.append(np.ones((200, CHUNK * 150, 3), dtype=np.uint8) * 30)

    max_width = max(img.shape[1] for img in row_images)
    padded = []
    for img in row_images:
        if img.shape[1] < max_width:
            pad = np.ones((img.shape[0], max_width - img.shape[1], 3), dtype=np.uint8) * 255
            img = np.concatenate([img, pad], axis=1)
        padded.append(img)

    full_image = np.concatenate(padded, axis=0)
    plt.figure(figsize=(max_width / 150, full_image.shape[0] / 150))
    plt.imshow(full_image)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(final_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"  Saved → {final_path}")


# ── Main loop ─────────────────────────────────────────────────────────────────

ROOT_OUTPUT = Path("./tribe_study")

for category, filenames in VIDEOS.items():
    print(f"\n{'='*50}")
    print(f"Category: {category}")
    print(f"{'='*50}")
    for fname in filenames:
        video_path = (DATA_DIR / fname).resolve()
        if not video_path.exists():
            print(f"  [MISSING] {fname} — skipping")
            continue
        # e.g. tribe_study/porn/porn_1/
        stem = Path(fname).stem
        out_dir = ROOT_OUTPUT / category / stem
        print(f"\n  Video: {fname}")
        process_video(video_path, out_dir)

print("\n\nAll videos processed.")
print(f"Outputs in: {ROOT_OUTPUT.resolve()}")