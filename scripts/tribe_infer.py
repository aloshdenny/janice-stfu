import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

from tribev2.demo_utils import TribeModel
from tribev2.plotting import PlotBrain
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from multiprocessing import Process, cpu_count
import multiprocessing as mp
import time

CHUNK_TIMEOUT = 120
CHUNK = 10

# ── Worker: renders one chunk, saves as .npy to disk ─────────────────────────

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
        print(f"  [ERROR] t={i}s–{end}s: {e}")


# ── Setup display (safe) ─────────────────────────────────────────────────────

os.environ["DISPLAY"] = ":99"
if not os.path.exists("/tmp/.X11-unix/X99"):
    os.system("Xvfb :99 -screen 0 1024x768x24 &")
    time.sleep(1)

# ── Model + inference ─────────────────────────────────────────────────────────

CACHE_FOLDER = Path("./cache")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_FOLDER)

video_path = Path("funkytown.mp4").resolve()

# ✅ sanity check
if not video_path.exists():
    raise FileNotFoundError(f"Video not found: {video_path}")

print(f"Using video: {video_path}")

df = model.get_events_dataframe(video_path=video_path)
preds, segments = model.predict(events=df)
print(f"Predictions shape: {preds.shape}")

# ── Save outputs ─────────────────────────────────────────────────────────────

output_dir = Path("./tribe_outputs")
output_dir.mkdir(exist_ok=True)
chunk_dir = output_dir / "chunks"
chunk_dir.mkdir(exist_ok=True)

np.save(output_dir / "preds.npy", preds)
np.save(output_dir / "segments.npy", segments)

# ── Parallel rendering ───────────────────────────────────────────────────────

total = len(preds)
n_cores = cpu_count()
print(f"Rendering {total} timesteps, chunk={CHUNK}, cores={n_cores}...")

chunk_ranges = [(i, min(i + CHUNK, total)) for i in range(0, total, CHUNK)]

def run_chunk(i, end):
    out_path = chunk_dir / f"chunk_{i:05d}.npy"
    if out_path.exists():
        print(f"  Cached: t={i}s–{end}s")
        return

    p = Process(
        target=render_worker,
        args=(i, end, preds[i:end], segments[i:end], str(out_path)),
        daemon=True,
    )
    p.start()
    p.join(timeout=CHUNK_TIMEOUT)

    if p.is_alive():
        print(f"  TIMEOUT — killing t={i}s–{end}s")
        p.kill()
        p.join()
    elif p.exitcode != 0:
        print(f"  CRASHED (exit {p.exitcode}) — skipping t={i}s–{end}s")
    else:
        print(f"  Done: t={i}s–{end}s")


from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=n_cores) as ex:
    futures = [ex.submit(run_chunk, i, end) for i, end in chunk_ranges]
    for f in futures:
        f.result()

# ── Composite ────────────────────────────────────────────────────────────────

print("Compositing...")

row_images = []
for i, end in chunk_ranges:
    out_path = chunk_dir / f"chunk_{i:05d}.npy"
    if out_path.exists():
        row_images.append(np.load(str(out_path)))
    else:
        h, w = 200, CHUNK * 150
        blank = np.ones((h, w, 3), dtype=np.uint8) * 30
        row_images.append(blank)
        print(f"  Blank placeholder: t={i}s–{end}s")

max_width = max(img.shape[1] for img in row_images)

padded = []
for img in row_images:
    if img.shape[1] < max_width:
        pad = np.ones((img.shape[0], max_width - img.shape[1], 3), dtype=np.uint8) * 255
        img = np.concatenate([img, pad], axis=1)
    padded.append(img)

full_image = np.concatenate(padded, axis=0)
heights = [img.shape[0] for img in row_images]

final_path = output_dir / "brain_full.png"

plt.figure(figsize=(max_width / 150, sum(heights) / 150))
plt.imshow(full_image)
plt.axis("off")
plt.tight_layout(pad=0)
plt.savefig(final_path, dpi=150, bbox_inches="tight", pad_inches=0)
plt.close()

print(f"Saved → {final_path.resolve()}")
print("All done.")