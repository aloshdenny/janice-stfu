"""
diagnose_layers_by_region.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SCIENTIFIC SCOPE — what we CAN and CANNOT map
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRIBE v2 predicts activity on the fsaverage5 CORTICAL SURFACE MESH only:
  20,484 vertices, 10,242 per hemisphere, covering neocortex + allocortex.

This means the following regions from the Wikipedia "List of regions in the
human brain" are COMPLETELY OUTSIDE the output space and are NOT mapped here:
  • Medulla oblongata and all its nuclei
  • Pons and pontine nuclei
  • Cerebellum
  • Midbrain (tectum, tegmentum, substantia nigra, VTA, red nucleus, PAG)
  • Thalamus and all thalamic nuclei
  • Hypothalamus and all hypothalamic nuclei / pituitary
  • Hippocampus (subcortical in FreeSurfer; not on the pial surface)
  • Amygdala (subcortical)
  • Basal ganglia (striatum, globus pallidus, subthalamic nucleus)
  • Basal forebrain
  • Claustrum
  • All white-matter tracts (corpus callosum, fasciculi, etc.)
  • Ventricular system / CSF spaces
  • Spinal cord

MAPPABLE regions (all have vertices on the fsaverage5 pial surface):
  Via the Destrieux (2010) atlas — 74 cortical gyri/sulci per hemisphere —
  distributed with FreeSurfer and available in nilearn with NO extra install:

  Occipital lobe
    V1   Primary visual cortex          (BA 17 = pericalcarine)
    V2   Secondary visual cortex        (cuneus + lingual)
    V3/V4 Higher visual areas           (inferior occipital gyrus/sulcus)
    MT   Middle temporal / V5           (middle temporal gyrus, posterior)

  Temporal lobe
    A1   Primary auditory cortex        (transverse temporal / Heschl)
    STS  Superior temporal sulcus       (superior temporal sulcus)
    STG  Superior temporal gyrus        (BA 22)
    FFA  Fusiform face area             (fusiform gyrus)
    PPA  Parahippocampal place area     (parahippocampal gyrus)
    MTG  Middle temporal gyrus          (BA 21)
    ITG  Inferior temporal gyrus        (BA 20)

  Parietal lobe
    S1   Primary somatosensory cortex   (post-central gyrus; BA 1/2/3)
    IPL  Inferior parietal lobule       (supramarginal + angular; BA 39/40)
    SPL  Superior parietal lobule       (BA 5/7)
    PCC  Posterior cingulate cortex     (posterior cingulate)

  Frontal lobe
    M1   Primary motor cortex           (pre-central gyrus; BA 4)
    PMC  Premotor cortex / SMA          (superior frontal gyrus, posterior)
    DLPFC Dorsolateral PFC              (middle frontal gyrus; BA 9/46)
    IFG  Inferior frontal gyrus/Broca   (inferior frontal gyrus; BA 44/45)
    OFC  Orbitofrontal cortex           (orbital gyri; BA 11)
    ACC  Anterior cingulate cortex      (anterior cingulate; BA 24/32)
    MPFC Medial PFC / vmPFC             (medial frontal gyrus; BA 10/25)

  Insular cortex
    INS  Insula                         (short + long insular gyri)

All mappings use exact Destrieux region name substrings — deterministic,
atlas-grounded, no assumptions.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

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

CATEGORIES       = ["porn", "gore", "cute", "nature", "food", "kissing", "chase", "fight"]
N_VIDEOS_PER_CAT = 16
CLIP_FRAMES      = 16
CLIP_DURATION    = 4.0

from torchvision import transforms
normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

# ─────────────────────────────────────────────────────────────────────────────
# ROI TABLE
# ─────────────────────────────────────────────────────────────────────────────
# Each entry:
#   key          : short identifier
#   label        : display name
#   lobe         : anatomical lobe (for colour grouping)
#   wiki_region  : exact name as it appears on the Wikipedia list
#   ba           : Brodmann area(s), if applicable
#   function     : canonical function from peer-reviewed literature
#   destrieux    : list of Destrieux atlas region name substrings
#                  (matched case-insensitively against the atlas labels)
#   source       : primary citation for the functional assignment

ROIS = [
    # ── Occipital lobe ────────────────────────────────────────────────────
    dict(
        key="V1", label="V1", lobe="occipital",
        wiki_region="Primary visual cortex (V1)",
        ba="BA 17",
        function="Oriented edges, retinotopic map, low-level luminance/contrast",
        destrieux=["Pericalcarine"],
        source="Hubel & Wiesel 1962; Wandell et al. 2007",
    ),
    dict(
        key="V2", label="V2/V3", lobe="occipital",
        wiki_region="V2, V3",
        ba="BA 18/19",
        function="Contours, illusory edges, slightly enlarged receptive fields",
        destrieux=["Cuneus", "Lingual"],
        source="Hegdé & Van Essen 2000",
    ),
    dict(
        key="V4", label="V4", lobe="occipital",
        wiki_region="V4",
        ba="BA 19",
        function="Curvature, shape fragments, colour selectivity",
        destrieux=["Inferior occipital", "Occipital"],
        source="Pasupathy & Connor 2002",
    ),
    dict(
        key="MT", label="MT/V5", lobe="occipital",
        wiki_region="V5 / MT",
        ba="BA 19/37",
        function="Motion direction, speed, optic flow; dorsal-stream gateway",
        destrieux=["Middle temporal", "Inferior temporal"],
        source="Maunsell & Van Essen 1983; Born & Bradley 2005",
    ),
    # ── Temporal lobe ─────────────────────────────────────────────────────
    dict(
        key="A1", label="A1 (Heschl)", lobe="temporal",
        wiki_region="Primary auditory cortex (BA 41/42)",
        ba="BA 41",
        function="Primary auditory cortex — frequency tonotopy, onset responses",
        destrieux=["Transverse temporal"],
        source="Formisano et al. 2003",
    ),
    dict(
        key="STG", label="STG", lobe="temporal",
        wiki_region="Superior temporal gyrus (BA 22)",
        ba="BA 22",
        function="Auditory association, speech perception, Wernicke area (L)",
        destrieux=["Superior temporal", "Planum temporale", "Planum polare"],
        source="Scott & Johnsrude 2003",
    ),
    dict(
        key="STS", label="STS", lobe="temporal",
        wiki_region="Superior temporal sulcus",
        ba="BA 21/22",
        function="Biological motion, faces-in-motion, social perception, speech",
        destrieux=["Superior temporal sulcus", "Posterior ramus"],
        source="Allison et al. 2000; Pelphrey et al. 2005",
    ),
    dict(
        key="FFA", label="FFA", lobe="temporal",
        wiki_region="Fusiform gyrus (BA 37) — Fusiform Face Area",
        ba="BA 37",
        function="Face identity, face detection; also words, expertise objects",
        destrieux=["Fusiform"],
        source="Kanwisher et al. 1997; Haxby et al. 2001",
    ),
    dict(
        key="PPA", label="PPA", lobe="temporal",
        wiki_region="Parahippocampal gyrus — Parahippocampal Place Area",
        ba="BA 27/35/36",
        function="Scenes, buildings, spatial layouts, navigational context",
        destrieux=["Parahippocampal"],
        source="Epstein & Kanwisher 1998",
    ),
    dict(
        key="MTG", label="MTG", lobe="temporal",
        wiki_region="Middle temporal gyrus (BA 21)",
        ba="BA 21",
        function="Semantic memory, lexical retrieval, social cognition",
        destrieux=["Middle temporal gyrus"],
        source="Binder et al. 2009",
    ),
    dict(
        key="ITG", label="ITG", lobe="temporal",
        wiki_region="Inferior temporal gyrus (BA 20)",
        ba="BA 20",
        function="Object recognition, high-level visual categorisation",
        destrieux=["Inferior temporal gyrus"],
        source="Tanaka 1996; DiCarlo et al. 2012",
    ),
    # ── Parietal lobe ─────────────────────────────────────────────────────
    dict(
        key="S1", label="S1", lobe="parietal",
        wiki_region="Primary somatosensory cortex (BA 1/2/3)",
        ba="BA 1/2/3",
        function="Touch, proprioception, pain; somatotopic body map",
        destrieux=["Postcentral"],
        source="Kaas 1983; Purves et al. 2012",
    ),
    dict(
        key="SPL", label="SPL", lobe="parietal",
        wiki_region="Superior parietal lobule (BA 5/7)",
        ba="BA 5/7",
        function="Visuospatial attention, tool use, dorsal visual stream",
        destrieux=["Superior parietal"],
        source="Culham & Valyear 2006",
    ),
    dict(
        key="IPL", label="IPL (SMG/AG)", lobe="parietal",
        wiki_region="Inferior parietal lobule (BA 39/40) — Supramarginal & Angular gyri",
        ba="BA 39/40",
        function="Number processing, language, tool semantics, mirror system",
        destrieux=["Supramarginal", "Angular"],
        source="Corbetta & Shulman 2002; Rizzolatti & Craighero 2004",
    ),
    dict(
        key="PCC", label="PCC / RSC", lobe="parietal",
        wiki_region="Posterior cingulate cortex (BA 23/31)",
        ba="BA 23/31",
        function="Default mode network hub; episodic memory retrieval, self-referential",
        destrieux=["Posterior cingulate", "Isthmus cingulate"],
        source="Buckner et al. 2008; Leech & Sharp 2014",
    ),
    # ── Frontal lobe ──────────────────────────────────────────────────────
    dict(
        key="M1", label="M1", lobe="frontal",
        wiki_region="Primary motor cortex (BA 4)",
        ba="BA 4",
        function="Voluntary movement execution; somatotopic motor map",
        destrieux=["Precentral"],
        source="Penfield & Boldrey 1937",
    ),
    dict(
        key="PMC", label="PMC / SMA", lobe="frontal",
        wiki_region="Premotor cortex / Supplementary motor area (BA 6)",
        ba="BA 6",
        function="Motor planning, sequence learning, action preparation",
        destrieux=["Superior frontal gyrus", "Paracentral"],
        source="Passingham 1993; Nachev et al. 2008",
    ),
    dict(
        key="DLPFC", label="DLPFC", lobe="frontal",
        wiki_region="Dorsolateral prefrontal cortex (BA 9/46)",
        ba="BA 9/46",
        function="Working memory, cognitive control, top-down attention",
        destrieux=["Middle frontal", "Middle frontal sulcus"],
        source="Goldman-Rakic 1995; Miller & Cohen 2001",
    ),
    dict(
        key="IFG", label="IFG (Broca)", lobe="frontal",
        wiki_region="Inferior frontal gyrus (BA 44/45) — Broca area",
        ba="BA 44/45",
        function="Language production (L), syntactic processing, action observation",
        destrieux=["Inferior frontal", "Triangular", "Opercular"],
        source="Broca 1861; Friederici 2011",
    ),
    dict(
        key="OFC", label="OFC", lobe="frontal",
        wiki_region="Orbitofrontal cortex (BA 11/47)",
        ba="BA 11/47",
        function="Reward valuation, emotion regulation, social value / disgust",
        destrieux=["Orbital", "Medial orbital"],
        source="Wallis 2007; Rolls 2019",
    ),
    dict(
        key="MPFC", label="mPFC / vmPFC", lobe="frontal",
        wiki_region="Medial prefrontal cortex (BA 10/25)",
        ba="BA 10/25",
        function="Default mode, self-referential processing, social cognition",
        destrieux=["Medial frontal", "Paraolfactory"],
        source="Amodio & Frith 2006",
    ),
    dict(
        key="ACC", label="ACC", lobe="frontal",
        wiki_region="Anterior cingulate cortex (BA 24/32)",
        ba="BA 24/32",
        function="Conflict monitoring, pain affect, error detection, salience",
        destrieux=["Anterior cingulate", "Middle cingulate"],
        source="Bush et al. 2000; Shackman et al. 2011",
    ),
    # ── Insula ────────────────────────────────────────────────────────────
    dict(
        key="INS", label="Insula", lobe="insula",
        wiki_region="Insular cortex",
        ba="BA 13/14",
        function="Interoception, disgust, pain, empathy, salience network hub",
        destrieux=["Long insular", "Short insular", "Circular insular"],
        source="Craig 2002; Uddin 2015",
    ),
]

LOBE_COLORS = {
    "occipital": "#4fc3f7",   # light blue
    "temporal":  "#81c784",   # green
    "parietal":  "#ffb74d",   # amber
    "frontal":   "#f48fb1",   # pink
    "insula":    "#ce93d8",   # lavender
}

# ─────────────────────────────────────────────────────────────────────────────
# Build ROI vertex masks via Destrieux atlas on fsaverage5 (nilearn)
# ─────────────────────────────────────────────────────────────────────────────

print("Building ROI masks from Destrieux atlas (fsaverage5) …")
try:
    from nilearn import datasets as nl_datasets
    destrieux   = nl_datasets.fetch_atlas_surf_destrieux()
    lh_labels   = np.array(destrieux["map_left"])    # (10242,) int
    rh_labels   = np.array(destrieux["map_right"])   # (10242,) int
    atlas_names = [
        (n.decode() if isinstance(n, bytes) else n)
        for n in destrieux["labels"]
    ]
    print(f"  Destrieux atlas loaded: {len(atlas_names)} labels")
    print(f"  Label names sample: {atlas_names[1:6]}")

    def destrieux_mask(substrings):
        """Boolean (20484,) mask for vertices whose label name contains any substring."""
        idxs = [i for i, n in enumerate(atlas_names)
                if any(s.lower() in n.lower() for s in substrings)]
        if not idxs:
            return np.zeros(20484, dtype=bool)
        lh = np.isin(lh_labels, idxs)
        rh = np.isin(rh_labels, idxs)
        return np.concatenate([lh, rh])

    for roi in ROIS:
        roi["mask"] = destrieux_mask(roi["destrieux"])
        n = roi["mask"].sum()
        status = "OK" if n > 0 else "EMPTY — ROI dropped"
        print(f"  {roi['key']:8s}  {n:5d} vertices  ({status})")

    ROIS = [r for r in ROIS if r["mask"].sum() > 0]
    print(f"  {len(ROIS)} ROIs retained after atlas lookup")

except Exception as e:
    print(f"  [FATAL] nilearn unavailable: {e}")
    print("  Cannot build atlas-grounded masks — exiting.")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Load V-JEPA2 encoder via TribeModel
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading TribeModel …")
model           = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
lightning_model = model._model
vjepa2_module   = model.data.video_feature.image.model.model
encoder_blocks  = vjepa2_module.encoder.layer
N_LAYERS        = len(encoder_blocks)
print(f"V-JEPA2 encoder blocks: {N_LAYERS}")

vjepa2_module.eval()
vjepa2_module.to(DEVICE)

# TRIBE v2 reads these two layer indices into its brain encoder
# (per training config, confirmed in arxiv:2605.13904)
TRIBE_LAYERS = {19: "TRIBE L×0.5", 39: "TRIBE L×1.0"}

# ─────────────────────────────────────────────────────────────────────────────
# Hooks + memory helpers
# ─────────────────────────────────────────────────────────────────────────────

layer_acts = {}

def make_hook(idx):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        layer_acts[idx] = hidden.mean(dim=1)[0].detach().float().cpu().numpy()
    return hook

def register_all_hooks():
    return [encoder_blocks[i].register_forward_hook(make_hook(i))
            for i in range(N_LAYERS)]

def remove_hooks(handles):
    for h in handles:
        h.remove()

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

USE_AMP = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

def run_forward(clip_tensor):
    inp = clip_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if USE_AMP:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vjepa2_module(pixel_values_videos=inp)
        else:
            vjepa2_module(pixel_values_videos=inp)
    del inp

# ─────────────────────────────────────────────────────────────────────────────
# Streaming video decoder (full resolution, full duration, no RAM spike)
# ─────────────────────────────────────────────────────────────────────────────

def probe_video(path):
    cmd = ["ffprobe", "-v", "quiet", "-print_format", "json",
           "-show_streams", str(path)]
    out  = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(out.stdout)
    vs   = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if vs is None:
        raise RuntimeError(f"No video stream: {path}")
    num, den = vs.get("avg_frame_rate", "30/1").split("/")
    fps      = float(num) / max(float(den), 1e-9)
    dur      = vs.get("duration") or info.get("format", {}).get("duration")
    duration = float(dur) if dur else int(vs.get("nb_frames", 900)) / fps
    return fps, duration, int(vs["width"]), int(vs["height"])

def decode_clip(path, start_sec, dur_sec, n_frames, w, h):
    target_fps = n_frames / dur_sec
    cmd = ["ffmpeg", "-v", "quiet",
           "-ss", f"{start_sec:.6f}", "-t", f"{dur_sec:.6f}",
           "-i", str(path),
           "-vf", f"fps={target_fps:.6f}",
           "-frames:v", str(n_frames),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw  = subprocess.run(cmd, capture_output=True).stdout
    need = n_frames * h * w * 3
    if len(raw) < need:
        if len(raw) == 0:
            raise RuntimeError("ffmpeg returned 0 bytes")
        fb   = h * w * 3
        last = raw[-(len(raw) // fb) * fb or -fb:][-fb:]
        raw += last * ((need - len(raw) + fb - 1) // fb)
    frames = np.frombuffer(raw[:need], dtype=np.uint8).reshape(n_frames, h, w, 3)
    return (frames.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)

def iter_clips(path):
    fps, total_dur, w, h = probe_video(path)
    n_clips = max(1, int(total_dur // CLIP_DURATION))
    if total_dur - n_clips * CLIP_DURATION >= 1.0:
        n_clips += 1
    for c in range(n_clips):
        start  = c * CLIP_DURATION
        actual = min(CLIP_DURATION, total_dur - start)
        if actual < 0.5:
            break
        frames_np = decode_clip(path, start, actual, CLIP_FRAMES, w, h)
        clip = torch.from_numpy(frames_np)
        clip = torch.stack([normalize_fn(clip[t]) for t in range(CLIP_FRAMES)])
        yield clip
        del frames_np, clip

# ─────────────────────────────────────────────────────────────────────────────
# Pass 1: extract V-JEPA2 layer activations for all categories
# ─────────────────────────────────────────────────────────────────────────────

print("\nExtracting V-JEPA2 layer activations …")
cat_layer_acts = {}

for cat in CATEGORIES:
    vpaths = sorted(DATA_DIR.glob(f"{cat}*.mp4"))[:N_VIDEOS_PER_CAT]
    if not vpaths:
        print(f"  [SKIP] {cat}: no videos")
        continue
    all_clips = []
    for vp in vpaths:
        handles = register_all_hooks()
        try:
            for clip in iter_clips(vp):
                layer_acts.clear()
                run_forward(clip)
                if len(layer_acts) == N_LAYERS:
                    all_clips.append(
                        np.stack([layer_acts[i] for i in range(N_LAYERS)])
                    )
                free_memory()
        except Exception as e:
            print(f"  [ERROR] {vp.name}: {e}")
        finally:
            remove_hooks(handles)
    if all_clips:
        cat_layer_acts[cat] = np.stack(all_clips)
        print(f"  {cat}: {len(all_clips)} clips  shape={cat_layer_acts[cat].shape}")
    free_memory()

cats_with_data = [c for c in CATEGORIES if c in cat_layer_acts]

# ─────────────────────────────────────────────────────────────────────────────
# Pass 2: layer → ROI Pearson r via preds.npy
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing layer → ROI Pearson r from preds.npy …")

# layer_roi_r[cat][roi_key] = (N_LAYERS,) mean Pearson r across videos
layer_roi_r   = {cat: {roi["key"]: np.zeros(N_LAYERS) for roi in ROIS}
                 for cat in CATEGORIES}
video_n       = {cat: {roi["key"]: 0 for roi in ROIS}
                 for cat in CATEGORIES}

for cat in CATEGORIES:
    vpaths = sorted(DATA_DIR.glob(f"{cat}*.mp4"))[:N_VIDEOS_PER_CAT]
    if not vpaths:
        continue
    for vp in vpaths:
        preds_path = STUDY_ROOT / cat / vp.stem / "preds.npy"
        if not preds_path.exists():
            continue
        preds = np.load(preds_path)[:30]   # (30, 20484)

        # Pre-compute vertex signal for every ROI (cheap)
        roi_sigs = {}
        for roi in ROIS:
            m = roi["mask"]
            if m.sum() > 0:
                roi_sigs[roi["key"]] = preds[:, m].mean(axis=1)  # (30,)

        if not roi_sigs:
            continue

        # Extract clip-level activations for this video
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

        for roi_key, vsig in roi_sigs.items():
            # Align 30 TRs → n_clips bins
            bsz = 30.0 / n_clips
            y = np.array([
                vsig[int(c * bsz): max(int(c * bsz) + 1, int((c+1) * bsz))].mean()
                for c in range(n_clips)
            ])
            if y.std() < 1e-9:
                continue
            r_vec = np.zeros(N_LAYERS)
            for li in range(N_LAYERS):
                x = np.linalg.norm(clip_acts[:, li, :], axis=-1)
                if x.std() > 1e-9:
                    r_vec[li] = float(np.corrcoef(x, y)[0, 1])
            layer_roi_r[cat][roi_key] += r_vec
            video_n[cat][roi_key]     += 1

        del clip_acts
        free_memory()

# Average
for cat in CATEGORIES:
    for roi in ROIS:
        k = roi["key"]
        if video_n[cat][k] > 0:
            layer_roi_r[cat][k] /= video_n[cat][k]

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG    = "#0d0d0d"
SPINE_COL  = "#444444"
layer_idx  = np.arange(N_LAYERS)

CAT_COLORS = {c: col for c, col in zip(
    CATEGORIES, plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES)))
)}

def _style(ax, title="", ylabel="", xlabel=""):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.set_title(title, color="white", fontsize=10, pad=4)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(SPINE_COL)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    for li, lbl in TRIBE_LAYERS.items():
        ax.axvline(li, color="#666", linestyle="--", linewidth=0.9, alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(li + 0.2, ylim[0] + (ylim[1] - ylim[0]) * 0.97,
                lbl, color="#777", fontsize=6, va="top")

# ─────────────────────────────────────────────────────────────────────────────
# Plot A: one figure per ROI
#   X = layer index, one line per category, shaded expected depth band
#   = "which layers drive this brain region, and does it vary by content?"
# ─────────────────────────────────────────────────────────────────────────────

print("\nPlot A: per-ROI layer–fMRI correlation (one file per ROI) …")

for roi in ROIS:
    key   = roi["key"]
    color = LOBE_COLORS[roi["lobe"]]

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)

    for cat in cats_with_data:
        n = video_n[cat][key]
        if n == 0:
            continue
        r = layer_roi_r[cat][key]
        ax.plot(layer_idx, r,
                label=f"{cat} (n={n})",
                color=CAT_COLORS.get(cat, "white"),
                linewidth=1.8, alpha=0.9)

    ax.axhline(0, color="#555", linewidth=0.8)
    _style(ax,
           title=f"{roi['label']}  [{roi['wiki_region']}]  {roi['ba']}\n"
                 f"{roi['function']}",
           ylabel="Mean Pearson r  (layer activation norm → ROI BOLD)",
           xlabel="V-JEPA2 layer index  (0 = shallowest, 39 = deepest)")

    ax.legend(facecolor="#111", labelcolor="white", framealpha=0.85,
              fontsize=8, ncol=4, loc="upper left")

    # Annotation: cite source
    ax.text(0.99, 0.02, f"Source: {roi['source']}",
            color="#666", fontsize=6, ha="right", va="bottom",
            transform=ax.transAxes)

    plt.tight_layout()
    fname = OUT_DIR / f"roiA_{key.lower()}_layer_corr.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

print(f"  Saved {len(ROIS)} per-ROI plots to {OUT_DIR}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot B: per-category full ROI grid
#   Rows of subplots, one per ROI, bar chart of layer–ROI r
#   = "for this content type, which layers drive which cortical regions?"
# ─────────────────────────────────────────────────────────────────────────────

print("Plot B: per-category full ROI grid …")

N_COLS = 4
N_ROWS = (len(ROIS) + N_COLS - 1) // N_COLS

for cat in cats_with_data:
    fig, axes = plt.subplots(N_ROWS, N_COLS,
                              figsize=(22, 4 * N_ROWS),
                              sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    axes_flat = axes.flatten()

    for ri, roi in enumerate(ROIS):
        ax    = axes_flat[ri]
        key   = roi["key"]
        r     = layer_roi_r[cat][key]
        col   = LOBE_COLORS[roi["lobe"]]
        n     = video_n[cat][key]

        ax.set_facecolor(DARK_BG)
        ax.bar(layer_idx, r, color=col, alpha=0.8, width=0.85)
        ax.axhline(0, color="#555", linewidth=0.7)

        # Mark TRIBE's sampled layers
        for li in TRIBE_LAYERS:
            ax.axvline(li, color="#666", linestyle="--", linewidth=0.7)

        # Label top-3 layers by |r|
        top3 = np.argsort(np.abs(r))[-3:]
        for li in top3:
            ax.text(li, r[li] + 0.003 * np.sign(r[li]),
                    str(li), color="white", fontsize=6, ha="center")

        ax.set_title(f"{roi['label']}  (n={n})", color=col, fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color(SPINE_COL)
        ax.set_xlim(-0.5, N_LAYERS - 0.5)
        if ri % N_COLS == 0:
            ax.set_ylabel("Pearson r", color="white", fontsize=8)

    # Hide unused subplots
    for ri in range(len(ROIS), len(axes_flat)):
        axes_flat[ri].set_visible(False)

    axes_flat[min(len(ROIS) - 1, len(axes_flat) - N_COLS)].set_xlabel(
        "V-JEPA2 layer index  (0→39)", color="white", fontsize=9)

    fig.suptitle(
        f"{cat.upper()}  ·  Layer→ROI Pearson r for all {len(ROIS)} cortical regions\n"
        f"bar label = best layer  |  --- = TRIBE v2 sampled layers (19, 39)",
        color="white", fontsize=12, y=1.005)
    plt.tight_layout()
    fname = OUT_DIR / f"roiB_cat_{cat}_all_rois.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()

print(f"  Saved {len(cats_with_data)} per-category grid plots")

# ─────────────────────────────────────────────────────────────────────────────
# Plot C: grand summary heatmap  (categories × ROIs)
#   Cell = peak |r| across all layers, annotated with the best layer index
# ─────────────────────────────────────────────────────────────────────────────

print("Plot C: grand category × ROI heatmap …")

roi_keys = [r["key"]   for r in ROIS]
roi_labs = [r["label"] for r in ROIS]

peak_r   = np.zeros((len(cats_with_data), len(ROIS)))
best_li  = np.zeros((len(cats_with_data), len(ROIS)), dtype=int)

for ci, cat in enumerate(cats_with_data):
    for ri, roi in enumerate(ROIS):
        r          = layer_roi_r[cat][roi["key"]]
        best       = int(np.argmax(np.abs(r)))
        best_li[ci, ri]  = best
        peak_r[ci, ri]   = r[best]

fig, ax = plt.subplots(figsize=(max(14, len(ROIS) * 0.9),
                                max(6, len(cats_with_data) * 0.85)))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

vmax = max(0.01, np.abs(peak_r).max())
im   = ax.imshow(peak_r, aspect="auto", cmap="RdBu_r",
                 vmin=-vmax, vmax=vmax, interpolation="nearest")

ax.set_xticks(range(len(ROIS)))
ax.set_xticklabels(roi_labs, color="white", fontsize=9, rotation=45, ha="right")
ax.set_yticks(range(len(cats_with_data)))
ax.set_yticklabels(cats_with_data, color="white", fontsize=10)

for ci in range(len(cats_with_data)):
    for ri in range(len(ROIS)):
        val = peak_r[ci, ri]
        li  = best_li[ci, ri]
        txt = f"{val:+.2f}\nL{li}"
        fc  = "white" if abs(val) > vmax * 0.45 else "#bbb"
        ax.text(ri, ci, txt, ha="center", va="center", color=fc, fontsize=6.5)

cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cbar.ax.tick_params(colors="white")
cbar.set_label("Peak Pearson r  (best layer)", color="white", fontsize=9)

ax.set_title(
    "Grand summary: Content category × Cortical region\n"
    "colour = peak Pearson r,  cell label = best V-JEPA2 layer index",
    color="white", fontsize=12)
ax.tick_params(colors="white")
for sp in ax.spines.values():
    sp.set_edgecolor(SPINE_COL)

# Vertical lines separating lobes
lobe_order = ["occipital", "temporal", "parietal", "frontal", "insula"]
prev_lobe, boundary_x = None, []
for ri, roi in enumerate(ROIS):
    if roi["lobe"] != prev_lobe and prev_lobe is not None:
        boundary_x.append(ri - 0.5)
    prev_lobe = roi["lobe"]
for bx in boundary_x:
    ax.axvline(bx, color="#888", linewidth=1.2, linestyle=":")

plt.tight_layout()
fname = OUT_DIR / "roiC_summary_heatmap.png"
plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved → {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# Plot D: lobe-level summary — mean |r| per lobe per category
# ─────────────────────────────────────────────────────────────────────────────

print("Plot D: lobe-level summary …")

lobes = ["occipital", "temporal", "parietal", "frontal", "insula"]
lobe_r = np.zeros((len(cats_with_data), len(lobes)))

for ci, cat in enumerate(cats_with_data):
    for li, lobe in enumerate(lobes):
        vals = [
            np.abs(layer_roi_r[cat][roi["key"]]).max()
            for roi in ROIS if roi["lobe"] == lobe
        ]
        lobe_r[ci, li] = np.mean(vals) if vals else 0.0

x       = np.arange(len(lobes))
width   = 0.8 / max(len(cats_with_data), 1)
fig, ax = plt.subplots(figsize=(14, 5))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)

for ci, cat in enumerate(cats_with_data):
    offset = (ci - len(cats_with_data) / 2) * width + width / 2
    ax.bar(x + offset, lobe_r[ci], width * 0.92,
           color=CAT_COLORS.get(cat, "white"), label=cat, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels([l.capitalize() for l in lobes], color="white", fontsize=11)
ax.set_ylabel("Mean peak |Pearson r| across ROIs in lobe", color="white")
ax.set_title("Lobe-level layer→fMRI strength by content category", color="white")
ax.tick_params(colors="white")
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["bottom", "left"]:
    ax.spines[sp].set_color(SPINE_COL)
ax.legend(facecolor="#111", labelcolor="white", framealpha=0.85,
          fontsize=9, ncol=4)

plt.tight_layout()
fname = OUT_DIR / "roiD_lobe_summary.png"
plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved → {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# Printed mapping table
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*80)
print("DEFINITIVE LAYER → BRAIN REGION MAPPING  (empirical, atlas-grounded)")
print("="*80)
print(f"{'ROI':<9} {'Wiki region':<42} {'BA':<10} "
      f"{'Best cat':<10} {'peak r':>7}  {'best layer':>10}  {'depth':>6}")
print("-"*80)

for roi in ROIS:
    key     = roi["key"]
    best_cat, best_r, best_layer = None, -999.0, -1
    for cat in cats_with_data:
        r  = layer_roi_r[cat][key]
        li = int(np.argmax(np.abs(r)))
        if abs(r[li]) > abs(best_r):
            best_r, best_cat, best_layer = r[li], cat, li
    print(f"{roi['label']:<9} {roi['wiki_region'][:40]:<42} "
          f"{roi['ba']:<10} {str(best_cat):<10} "
          f"{best_r:>+7.4f}  layer {best_layer:2d}     "
          f"{best_layer/N_LAYERS:>5.2f}")

print("\nNOT MAPPED (outside fsaverage5 cortical surface):")
excluded = [
    "Medulla oblongata & all nuclei",
    "Pons & pontine nuclei",
    "Cerebellum",
    "Midbrain (tectum, tegmentum, SN, VTA, PAG, red nucleus)",
    "Thalamus & all thalamic nuclei",
    "Hypothalamus & all nuclei / pituitary",
    "Hippocampus  (subcortical in FreeSurfer)",
    "Amygdala  (subcortical)",
    "Basal ganglia (striatum, globus pallidus, STN)",
    "Basal forebrain / claustrum",
    "All white-matter tracts & commissures",
    "Ventricular system / CSF",
]
for e in excluded:
    print(f"  ✗  {e}")

print(f"\nAll outputs saved to {OUT_DIR.resolve()}")
print("Files:")
for f in sorted(OUT_DIR.glob("roi*.png")):
    print(f"  {f.name}")