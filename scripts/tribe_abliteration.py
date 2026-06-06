import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tribev2.demo_utils import TribeModel
import torchvision.io as tvio
from torchvision import transforms

# ── Config ────────────────────────────────────────────────────────────────────

STUDY_ROOT = Path("./tribe_study")
MASK_DIR   = STUDY_ROOT / "masks"
CACHE_DIR  = Path("./cache")
DATA_DIR   = Path("./data")
OUT_DIR    = Path("./abliterated")
OUT_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PORN_MASK_FILE = MASK_DIR / "porn_no_romance.npy"
GORE_MASK_FILE = MASK_DIR / "gore_motion_corrected.npy"

CATEGORIES = {
    "porn": [f"porn{i}.mp4" for i in range(1, 9)],
    "gore": [f"gore{i}.mp4" for i in range(1, 9)],
}

# ── Load model ────────────────────────────────────────────────────────────────

print("Loading TRIBEv2...")
model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)

# Drill to the actual VJEPA2 encoder blocks
# image_extractor → _HuggingFace wrapper → VJEPA2Model → encoder.layer
vjepa2_module = model.data.video_feature.image.model.model  # VJEPA2Model
encoder_blocks = vjepa2_module.encoder.layer                # ModuleList, len=40

N_LAYERS   = len(encoder_blocks)   # 40
TARGET_IDX = int(N_LAYERS * 0.75)  # block 30
print(f"Hooking into encoder.layer[{TARGET_IDX}] of {N_LAYERS} (depth=0.75)")

# ── Inspect block structure to find output projection ────────────────────────

target_block = encoder_blocks[TARGET_IDX]
print("\nTarget block children:")
for name, mod in target_block.named_children():
    print(f"  .{name} → {type(mod).__name__}")
    for subname, submod in mod.named_children():
        print(f"    .{name}.{subname} → {type(submod).__name__}")

# ── Hook: collect CLS token activations ──────────────────────────────────────

collected_acts = []

def hook_fn(module, input, output):
    # VJEPA2 blocks output a tuple or tensor — handle both
    if isinstance(output, tuple):
        hidden = output[0]
    else:
        hidden = output
    # hidden: (batch, seq_len, hidden_dim) — CLS is token 0
    collected_acts.append(hidden[:, 0, :].detach().cpu())

hook_handle = encoder_blocks[TARGET_IDX].register_forward_hook(hook_fn)

# ── Load masks ────────────────────────────────────────────────────────────────

porn_mask = np.load(PORN_MASK_FILE)
gore_mask = np.load(GORE_MASK_FILE)
print(f"\nPorn mask: {porn_mask.sum()} vertices")
print(f"Gore mask: {gore_mask.sum()} vertices")

# ── Frame extractor ───────────────────────────────────────────────────────────

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def extract_frames(video_path, n_frames=30):
    """Extract ~1 frame per second up to n_frames."""
    try:
        vframes, _, info = tvio.read_video(str(video_path), pts_unit="sec")
        fps = info.get("video_fps", 30)
        step = max(1, int(fps))
        frames = vframes[::step][:n_frames]          # (T, H, W, C)
        frames = frames.float() / 255.0
        frames = frames.permute(0, 3, 1, 2)          # (T, C, H, W)
        return frames
    except Exception as e:
        print(f"    [ERROR] reading {video_path.name}: {e}")
        return None

def frames_to_acts(frames):
    """Run frames through V-JEPA2 with hook, return activations (T, hidden)."""
    acts = []
    vjepa2_module.eval()
    with torch.no_grad():
        for frame in frames:
            frame = preprocess(frame).unsqueeze(0).to(DEVICE)
            vjepa2_module(frame)
            if collected_acts:
                acts.append(collected_acts[-1].squeeze(0).numpy())
                collected_acts.clear()
    return np.stack(acts) if acts else None

# ── Collect activations + paired vertex targets ───────────────────────────────

def collect_for_category(category, filenames, mask):
    X_all, y_all = [], []
    for fname in filenames:
        stem       = Path(fname).stem
        preds_path = STUDY_ROOT / category / stem / "preds.npy"
        video_path = (DATA_DIR / fname).resolve()

        if not preds_path.exists():
            print(f"  [SKIP] no preds: {fname}")
            continue
        if not video_path.exists():
            print(f"  [SKIP] no video: {fname}")
            continue

        preds = np.load(preds_path)[:30]         # (30, 20484)
        y     = preds[:, mask].mean(axis=1)      # (30,) vertex mean per TR

        frames = extract_frames(video_path, n_frames=30)
        if frames is None:
            continue

        acts = frames_to_acts(frames)            # (T, hidden)
        if acts is None:
            continue

        min_len = min(len(acts), len(y))
        X_all.append(acts[:min_len])
        y_all.append(y[:min_len])
        print(f"  {fname}: {min_len} pairs, y∈[{y[:min_len].min():.3f}, {y[:min_len].max():.3f}]")

    if not X_all:
        return None, None
    return np.concatenate(X_all, axis=0), np.concatenate(y_all, axis=0)


print("\nCollecting porn activations...")
X_porn, y_porn = collect_for_category("porn", CATEGORIES["porn"], porn_mask)

print("\nCollecting gore activations...")
X_gore, y_gore = collect_for_category("gore", CATEGORIES["gore"], gore_mask)

hook_handle.remove()

# ── Weighted PCA → abliteration directions ────────────────────────────────────

def find_directions(X, y, n_components=3, label=""):
    weights = (y - y.min()) / (y.max() - y.min() + 1e-9)
    weights /= weights.sum()

    X_mean    = (X * weights[:, None]).sum(axis=0, keepdims=True)
    X_centered = X - X_mean
    X_weighted = X_centered * np.sqrt(weights[:, None])

    _, S, Vt = np.linalg.svd(X_weighted, full_matrices=False)
    print(f"  [{label}] top singular values: {S[:5].round(4)}")
    return Vt[:n_components]   # (n_components, hidden_dim)


print("\nComputing porn direction...")
porn_dirs = find_directions(X_porn, y_porn, n_components=3, label="porn")

print("Computing gore direction...")
gore_dirs = find_directions(X_gore, y_gore, n_components=3, label="gore")

np.save(OUT_DIR / "porn_directions.npy", porn_dirs)
np.save(OUT_DIR / "gore_directions.npy", gore_dirs)
print(f"Saved directions → {OUT_DIR}")

# ── Build Gram-Schmidt projection hook ───────────────────────────────────────

def make_gs_hook(directions_np):
    """Project out abliteration directions from every token at inference time."""
    dirs = torch.tensor(directions_np, dtype=torch.float32).to(DEVICE)
    ortho = []
    for d in dirs:
        for q in ortho:
            d = d - (d @ q) * q
        norm = d.norm()
        if norm > 1e-6:
            ortho.append(d / norm)
    ortho = torch.stack(ortho)   # (n, hidden)

    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        for q in ortho:
            proj   = (hidden @ q).unsqueeze(-1) * q
            hidden = hidden - proj
        if isinstance(output, tuple):
            return (hidden,) + output[1:]
        return hidden

    return hook


print("\nRegistering abliteration hooks...")
porn_hook_handle = encoder_blocks[TARGET_IDX].register_forward_hook(make_gs_hook(porn_dirs))
gore_hook_handle = encoder_blocks[TARGET_IDX].register_forward_hook(make_gs_hook(gore_dirs))

# ── Validate: compare masked activation before/after ─────────────────────────

def masked_activation(video_path, mask):
    df    = model.get_events_dataframe(video_path=video_path)
    preds, _ = model.predict(events=df)
    return float(preds[:30, mask].mean())


val_pairs = [
    ("porn1.mp4",   porn_mask, "porn→porn_mask"),
    ("nature1.mp4", porn_mask, "nature→porn_mask"),
    ("gore1.mp4",   gore_mask, "gore→gore_mask"),
    ("cute1.mp4",   gore_mask, "cute→gore_mask"),
]

print("\nValidation (hooks active):")
for fname, mask, label in val_pairs:
    vp = (DATA_DIR / fname).resolve()
    if vp.exists():
        act = masked_activation(vp, mask)
        print(f"  {label:30s}  activation = {act:.4f}")

# ── Optional: permanent weight surgery ───────────────────────────────────────

def apply_weight_surgery():
    """
    Bake the projection into the attention output weight matrix permanently.
    Removes the need for runtime hooks.
    """
    porn_hook_handle.remove()
    gore_hook_handle.remove()

    all_dirs = np.concatenate([porn_dirs, gore_dirs], axis=0)
    dirs_t   = torch.tensor(all_dirs, dtype=torch.float32).to(DEVICE)

    # Gram-Schmidt orthonormalize all directions together
    ortho = []
    for d in dirs_t:
        for q in ortho:
            d = d - (d @ q) * q
        norm = d.norm()
        if norm > 1e-6:
            ortho.append(d / norm)
    ortho = torch.stack(ortho)

    # Find the attention output projection weight in the target block
    # Try common VJEPA2 attention attribute names
    block = encoder_blocks[TARGET_IDX]
    W = None
    for path in ["attention.output.dense", "attention.out_proj",
                 "attn.proj", "attn.out_proj", "self_attn.out_proj"]:
        parts = path.split(".")
        try:
            mod = block
            for p in parts:
                mod = getattr(mod, p)
            W = mod.weight.data
            print(f"  Weight surgery target: block[{TARGET_IDX}].{path}  shape={W.shape}")
            break
        except AttributeError:
            continue

    if W is None:
        # Fallback: print all linear layers in block and pick first
        print("  Could not find attn output proj — listing linear layers:")
        for name, mod in block.named_modules():
            if isinstance(mod, nn.Linear):
                print(f"    .{name}  weight={mod.weight.shape}")
        print("  Set path manually and re-run apply_weight_surgery()")
        return

    for q in ortho:
        W -= (W @ q).unsqueeze(-1) * q

    torch.save(vjepa2_module.state_dict(),
               OUT_DIR / "vjepa2_abliterated.pt")
    print(f"Saved abliterated weights → {OUT_DIR / 'vjepa2_abliterated.pt'}")


# Uncomment when ready to make permanent:
# apply_weight_surgery()

print("\nDone. Runtime hooks active for this session.")
print(f"Directions saved to {OUT_DIR}")