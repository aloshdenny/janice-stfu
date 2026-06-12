"""
layer_analysis.py

SCIENTIFIC SCOPE
━━━━━━━━━━━━━━━━
TRIBE v2 predicts on the fsaverage5 CORTICAL SURFACE only (20,484 vertices).
Every ROI below is grounded in the Destrieux (aparc.a2009s) atlas.
All label strings are EXACT aparc.a2009s names verified against the official
FreeSurfer wiki table at:
  surfer.nmr.mgh.harvard.edu/fswiki/DestrieuxAtlasChanges

The full atlas has 75 labels per hemisphere (74 cortical + Medial_wall).
This file covers 31 ROIs total (5 lobes).

NEW ROIs (added beyond original 22):
  OFA      — Occipital Face Area (lateral occipital gyrus)
  TP       — Temporal Pole  (person familiarity / semantic memory)
  ATL      — Anterior Temporal Lobe face area  (person identity)
  PREC     — Precuneus  (mental imagery, autobiographical memory)
  TPJ      — Temporo-Parietal Junction  (social cognition, mentalising)
  LO       — Lateral Occipital Cortex  (object recognition)
  PT       — Planum Temporale  (emotional prosody, audiovisual)
  FPC      — Frontal Pole / frontopolar PFC  (craving suppression)
  MPC      — Medial Parietal Cortex  (autobiographical retrieval)

NOT MAPPABLE (outside fsaverage5 cortical surface, dropped without error):
  Nucleus accumbens, caudate, putamen (striatum) — the primary porn-reward locus
  is subcortical.  Same applies to: VTA, amygdala, hippocampus, thalamus,
  hypothalamus, brainstem, cerebellum, all white-matter tracts.

ABLITERATION USE-CASE NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Goal A — Porn addiction:  OFC + vmPFC/MPFC + ACC + FPC as cortical proxies
  for the subcortical NAcc reward circuit.  Key contrast: porn vs. neutral.

Goal B — Food addiction:  OFC + INS + ACC as cortical nodes for
  palatability-driven overconsumption.  Key contrast: food vs. nature.

Sources: Destrieux et al. 2010; Haxby et al. 2000; Kanwisher 1997;
         Voon et al. 2014; Stoeckel et al. 2008 (food cue OFC/INS);
         arxiv:2605.13904 (TRIBE v2)
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
import gc, subprocess, json, os
from concurrent.futures import ProcessPoolExecutor

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("./data")
STUDY_ROOT = Path("./tribe_study")
CACHE_DIR  = Path("./cache")
ANALYSIS_DIR    = Path("./analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)

def discover_categories():
    """Auto-discover categories from subdirectory names in DATA_DIR."""
    return sorted([d.name for d in DATA_DIR.iterdir()
                   if d.is_dir() and any(d.glob("*.mp4"))])

def find_videos_for_category(cat):
    """Find all videos for a category in its data subfolder."""
    return sorted((DATA_DIR / cat).glob("*.mp4"))


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CATEGORIES        = discover_categories()
print(f"Auto-discovered {len(CATEGORIES)} categories: {CATEGORIES}")
CLIP_FRAMES       = 16
CLIP_DURATION     = 4.0
INFERENCE_BATCH   = 3          # clips processed per forward pass
N_PLOT_WORKERS    = os.cpu_count() or 1   # parallel plot processes

from torchvision import transforms
normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

# ─────────────────────────────────────────────────────────────────────────────
# ROI TABLE — exact aparc.a2009s label names
# ─────────────────────────────────────────────────────────────────────────────
# destrieux_exact: list of exact strings that must appear in the atlas label.
# We use str.__contains__ against the full label name, so e.g. "G_cuneus"
# matches the label "G_cuneus" exactly (and nothing else).
# Multiple entries per ROI = union of those labels (bilateral, LH+RH).

ROIS = [
    # ── Occipital lobe ────────────────────────────────────────────────────
    dict(key="V1",  label="V1",        lobe="occipital",
         wiki="Primary visual cortex (V1) — BA17",
         function="Oriented edges, retinotopic map, luminance/contrast",
         source="Hubel & Wiesel 1962",
         destrieux_exact=["S_calcarine"]),  # calcarine sulcus = V1 fundus

    dict(key="V2",  label="V2/V3",     lobe="occipital",
         wiki="V2, V3 — BA18/19",
         function="Contours, illusory edges, slightly enlarged RFs",
         source="Hegdé & Van Essen 2000",
         destrieux_exact=["G_cuneus",
                           "S_parieto_occipital",
                           "G_and_S_occipital_inf"]),

    dict(key="V4",  label="V4",        lobe="occipital",
         wiki="V4 — BA19",
         function="Curvature, colour, shape fragments",
         source="Pasupathy & Connor 2002",
         destrieux_exact=["G_occipital_middle",
                           "G_occipital_sup",
                           "S_oc_middle_and_Lunatus",
                           "S_oc_sup_and_transversal"]),

    dict(key="MT",  label="MT/V5",     lobe="occipital",
         wiki="MT / V5 — motion area",
         function="Motion direction, speed, optic flow",
         source="Maunsell & Van Essen 1983",
         destrieux_exact=["G_oc-temp_lat-fusifor",
                           "S_oc-temp_lat"]),

    # NEW: Occipital Face Area — lateral occipital / inferior occipital gyrus
    # First cortical node to encode face structural configuration before FFA.
    dict(key="OFA", label="OFA",       lobe="occipital",
         wiki="Occipital Face Area — inferior/lateral occipital cortex",
         function="Early structural encoding of faces; feeds FFA",
         source="Rossion 2014; Pitcher et al. 2011",
         destrieux_exact=["G_and_S_occipital_inf",
                           "S_oc_middle_and_Lunatus",
                           "Pole_occipital"]),

    # NEW: Lateral Occipital Cortex (LO) — object recognition "what" pathway
    dict(key="LO",  label="LO",        lobe="occipital",
         wiki="Lateral occipital cortex — BA19/37",
         function="Object recognition, shape completion, LOC",
         source="Malach et al. 1995",
         destrieux_exact=["G_oc-temp_med-Lingual",
                           "S_oc-temp_med_and_Lingual"]),

    # ── Temporal lobe ─────────────────────────────────────────────────────
    dict(key="A1",  label="A1 (Heschl)", lobe="temporal",
         wiki="Primary auditory cortex (BA41)",
         function="Frequency tonotopy, sound onset responses",
         source="Formisano et al. 2003",
         destrieux_exact=["G_temp_sup-G_T_transv"]),  # Heschl's gyrus

    dict(key="STG", label="STG",        lobe="temporal",
         wiki="Superior temporal gyrus (BA22) — Wernicke area (L)",
         function="Auditory association, speech perception",
         source="Scott & Johnsrude 2003",
         destrieux_exact=["G_temp_sup-Lateral",
                           "G_temp_sup-Plan_tempo",
                           "G_temp_sup-Plan_polar",
                           "S_temporal_sup"]),

    dict(key="STS", label="STS",        lobe="temporal",
         wiki="Superior temporal sulcus",
         function="Biological motion, faces-in-motion, social perception",
         source="Allison et al. 2000",
         destrieux_exact=["S_temporal_sup",
                           "S_temporal_inf"]),

    dict(key="FFA", label="FFA",        lobe="temporal",
         wiki="Fusiform gyrus (BA37) — Fusiform Face Area",
         function="Face identity, face detection",
         source="Kanwisher et al. 1997",
         destrieux_exact=["G_oc-temp_lat-fusifor"]),

    dict(key="PPA", label="PPA",        lobe="temporal",
         wiki="Parahippocampal gyrus — Parahippocampal Place Area",
         function="Scenes, buildings, spatial layouts",
         source="Epstein & Kanwisher 1998",
         destrieux_exact=["G_oc-temp_med-Parahip",
                           "S_collat_transv_ant",
                           "S_collat_transv_post"]),

    dict(key="MTG", label="MTG",        lobe="temporal",
         wiki="Middle temporal gyrus (BA21)",
         function="Semantic memory, lexical retrieval, social cognition",
         source="Binder et al. 2009",
         destrieux_exact=["G_temporal_middle",
                           "S_temporal_inf"]),

    dict(key="ITG", label="ITG",        lobe="temporal",
         wiki="Inferior temporal gyrus (BA20)",
         function="Object recognition, high-level visual categorisation",
         source="Tanaka 1996",
         destrieux_exact=["G_temporal_inf",
                           "S_oc-temp_med_and_Lingual"]),

    # NEW: Temporal Pole — person familiarity and semantic identity
    # Pole_temporal is an exact aparc.a2009s label.
    # THE target for specific-person desensitisation (Goal B).
    dict(key="TP",  label="Temporal Pole", lobe="temporal",
         wiki="Temporal pole (BA38) — person semantics",
         function="Familiar person recognition, person-semantic memory, "
                  "connecting face identity to autobiographical knowledge",
         source="Olson et al. 2013; Diano et al. 2024 PNAS",
         destrieux_exact=["Pole_temporal"]),

    # NEW: Anterior Temporal Lobe face area (ATL-FA)
    # Responds more to personally familiar than unfamiliar faces.
    dict(key="ATL", label="ATL-FA",     lobe="temporal",
         wiki="Anterior temporal face area — familiar person identity",
         function="Person identity storage; familiarity beyond FFA; "
                  "links perceptual face to biographical knowledge",
         source="Rossion 2014; Von der Heide et al. 2013",
         destrieux_exact=["G_temporal_inf",
                           "G_oc-temp_med-Parahip"]),

    # NEW: Planum Temporale — emotional prosody, audiovisual content
    dict(key="PT",  label="Planum Temporale", lobe="temporal",
         wiki="Planum temporale — posterior lateral fissure (BA42/22)",
         function="Auditory scene analysis, emotional prosody, music",
         source="Griffiths & Warren 2002",
         destrieux_exact=["Lat_Fis-post"]),

    # ── Parietal lobe ─────────────────────────────────────────────────────
    dict(key="S1",  label="S1",         lobe="parietal",
         wiki="Primary somatosensory cortex (BA1/2/3)",
         function="Touch, proprioception, pain; somatotopic body map",
         source="Kaas 1983",
         destrieux_exact=["G_postcentral",
                           "S_postcentral"]),

    dict(key="SPL", label="SPL",         lobe="parietal",
         wiki="Superior parietal lobule (BA5/7)",
         function="Visuospatial attention, tool use, dorsal visual stream",
         source="Culham & Valyear 2006",
         destrieux_exact=["G_parietal_sup",
                           "S_intrapariet_and_P_trans"]),

    dict(key="IPL", label="IPL (SMG/AG)", lobe="parietal",
         wiki="Inferior parietal lobule (BA39/40) — Supramarginal & Angular gyri",
         function="Number, language, tool semantics, mirror-neuron system",
         source="Rizzolatti & Craighero 2004",
         destrieux_exact=["G_pariet_inf-Angular",
                           "G_pariet_inf-Supramar"]),

    dict(key="PCC", label="PCC/RSC",     lobe="parietal",
         wiki="Posterior cingulate cortex (BA23/31)",
         function="Default mode hub; episodic memory, self-referential processing",
         source="Buckner et al. 2008",
         destrieux_exact=["G_cingul-Post-dorsal",
                           "G_cingul-Post-ventral",
                           "S_cingul-Marginalis"]),

    # NEW: Precuneus — mental imagery, autobiographical memory, self-simulation
    # When you "think about" your ex, precuneus generates the mental image.
    dict(key="PREC", label="Precuneus",  lobe="parietal",
         wiki="Precuneus (BA7) — visual mental imagery & autobiographical memory",
         function="Mental imagery of people/scenes, visuospatial episodic memory, "
                  "self-referential simulation",
         source="Cavanna & Trimble 2006",
         destrieux_exact=["G_precuneus",
                           "S_subparietal"]),

    # NEW: Temporo-Parietal Junction — theory of mind, person models
    dict(key="TPJ", label="TPJ",         lobe="parietal",
         wiki="Temporo-parietal junction (BA39/40 border)",
         function="Theory of mind, mentalising, person model construction",
         source="Saxe & Kanwisher 2003",
         destrieux_exact=["G_and_S_subcentral",
                           "S_interm_prim-Jensen"]),

    # NEW: Medial Parietal Cortex — autobiographical retrieval node
    dict(key="MPC", label="MPC",         lobe="parietal",
         wiki="Medial parietal cortex / posterior DMN",
         function="Autobiographical memory retrieval, narrative self",
         source="Spreng et al. 2009",
         destrieux_exact=["S_subparietal",
                           "G_cingul-Post-ventral"]),

    # ── Frontal lobe ──────────────────────────────────────────────────────
    dict(key="M1",    label="M1",          lobe="frontal",
         wiki="Primary motor cortex (BA4)",
         function="Voluntary movement execution; somatotopic motor map",
         source="Penfield & Boldrey 1937",
         destrieux_exact=["G_precentral",
                           "S_precentral-inf-part",
                           "S_precentral-sup-part"]),

    dict(key="PMC",   label="PMC/SMA",     lobe="frontal",
         wiki="Premotor cortex / SMA (BA6)",
         function="Motor planning, sequence learning, action preparation",
         source="Passingham 1993",
         destrieux_exact=["G_and_S_paracentral",
                           "G_front_sup"]),   # posterior part = SMA

    dict(key="DLPFC", label="DLPFC",        lobe="frontal",
         wiki="Dorsolateral PFC (BA9/46)",
         function="Working memory, cognitive control, top-down attention",
         source="Goldman-Rakic 1995",
         destrieux_exact=["G_front_middle",
                           "S_front_middle",
                           "S_front_sup"]),

    dict(key="IFG",   label="IFG / Broca", lobe="frontal",
         wiki="Inferior frontal gyrus (BA44/45) — Broca area",
         function="Language production (L), syntax, action observation",
         source="Broca 1861; Friederici 2011",
         destrieux_exact=["G_front_inf-Opercular",
                           "G_front_inf-Triangul",
                           "G_front_inf-Orbital",
                           "S_front_inf"]),

    dict(key="OFC",   label="OFC",          lobe="frontal",
         wiki="Orbitofrontal cortex (BA11/47) — reward valuation",
         function="Reward valuation, emotion regulation, disgust, "
                  "subjective value of visual sexual stimuli (VSS); "
                  "lateral OFC encodes erotic pleasure (Voon et al. 2014)",
         source="Wallis 2007; Rolls 2019; ScienceDirect 2020 (VSS reward)",
         destrieux_exact=["G_orbital",
                           "G_rectus",
                           "S_orbital_lateral",
                           "S_orbital_med-olfact",
                           "S_orbital-H_Shaped",
                           "S_suborbital"]),

    dict(key="MPFC",  label="mPFC/vmPFC",   lobe="frontal",
         wiki="Medial PFC (BA10/25) — self-referential & reward",
         function="Default mode, self-referential processing, social cognition; "
                  "vmPFC encodes subjective sexual arousal and cue reactivity "
                  "in problematic pornography use (Voon et al. 2014)",
         source="Amodio & Frith 2006; Voon et al. 2014",
         destrieux_exact=["G_and_S_frontomargin",
                           "G_and_S_transv_frontopol",
                           "G_subcallosal"]),

    dict(key="ACC",   label="ACC",           lobe="frontal",
         wiki="Anterior cingulate cortex (BA24/32)",
         function="Conflict monitoring, pain affect, error detection, salience; "
                  "ventral ACC activated during pornographic CS+ conditioning "
                  "(Klucken et al. 2016)",
         source="Bush et al. 2000; Klucken et al. 2016",
         destrieux_exact=["G_and_S_cingul-Ant",
                           "G_and_S_cingul-Mid-Ant",
                           "G_and_S_cingul-Mid-Post"]),

    # NEW: Frontal Pole / frontopolar PFC — craving suppression & prospection
    dict(key="FPC",   label="FPC",           lobe="frontal",
         wiki="Frontal pole / frontopolar PFC (BA10)",
         function="Prospective thinking, craving suppression, "
                  "hyper-connected in pornography addiction (Frontiers 2025)",
         source="Koechlin et al. 2003; Frontiers Hum Neurosci 2025",
         destrieux_exact=["G_and_S_frontomargin",
                           "G_and_S_transv_frontopol"]),

    # ── Insula ────────────────────────────────────────────────────────────
    dict(key="INS",   label="Insula",        lobe="insula",
         wiki="Insular cortex (BA13/14) — interoception & disgust",
         function="Interoception, disgust, pain, empathy, salience network hub; "
                  "anterior insula encodes visceral disgust response to gore",
         source="Craig 2002; Uddin 2015",
         destrieux_exact=["G_Ins_lg_and_S_cent_ins",
                           "G_insular_short",
                           "S_circular_insula_ant",
                           "S_circular_insula_inf",
                           "S_circular_insula_sup"]),
]

# ─────────────────────────────────────────────────────────────────────────────
# SUBCORTICAL DISCLAIMER — printed at runtime
# ─────────────────────────────────────────────────────────────────────────────

_SUBCORTICAL_NOTE = """
NOTE ON ABLITERATION TARGETS NOT REACHABLE VIA fsaverage5 SURFACE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The primary reward/craving circuit for pornography addiction is:
  VTA → nucleus accumbens → caudate/putamen → amygdala → OFC
Of these, only OFC is on the cortical surface.  NAcc, caudate, amygdala,
and VTA are subcortical and cannot be mapped by TRIBE v2.

For Goal A (porn abliteration), use OFC + vmPFC/MPFC + ACC + FPC
as cortical proxies.  These regions reliably co-activate with NAcc
(Voon et al. 2014; ScienceDirect 2020 VSS reward) and are reachable.

For Goal B (face desensitisation), the critical cortical chain is:
  V1/V2 → OFA → FFA → STS → ATL → TP → PREC/mPFC → PCC
All these ARE on the surface and are in the ROI table above.
"""

LOBE_COLORS = {
    "occipital": "#4fc3f7",
    "temporal":  "#81c784",
    "parietal":  "#ffb74d",
    "frontal":   "#f48fb1",
    "insula":    "#ce93d8",
}

# ─────────────────────────────────────────────────────────────────────────────
# Build ROI vertex masks via nilearn Destrieux atlas (exact label matching)
# ─────────────────────────────────────────────────────────────────────────────

print("Building ROI masks from Destrieux atlas (fsaverage5) …")
print(_SUBCORTICAL_NOTE)
try:
    from nilearn import datasets as nl_datasets
    destrieux   = nl_datasets.fetch_atlas_surf_destrieux()
    lh_labels   = np.array(destrieux["map_left"])    # (10242,)
    rh_labels   = np.array(destrieux["map_right"])   # (10242,)
    atlas_names = [
        (n.decode() if isinstance(n, bytes) else n)
        for n in destrieux["labels"]
    ]
    print(f"  Atlas loaded: {len(atlas_names)} labels")

    def exact_mask(exact_strings):
        """
        Boolean (20484,) mask.  A vertex is included iff its atlas label
        exactly matches one of the strings in exact_strings.
        """
        idxs = [i for i, n in enumerate(atlas_names) if n in exact_strings]
        if not idxs:
            return np.zeros(20484, dtype=bool)
        return np.concatenate([np.isin(lh_labels, idxs),
                               np.isin(rh_labels, idxs)])

    retained = []
    for roi in ROIS:
        roi["mask"] = exact_mask(roi["destrieux_exact"])
        n = int(roi["mask"].sum())
        if n > 0:
            retained.append(roi)
            print(f"  {roi['key']:8s}  {n:5d} vertices  OK")
        else:
            # Print which labels were looked for vs. what exists
            missing = [s for s in roi["destrieux_exact"] if s not in atlas_names]
            print(f"  {roi['key']:8s}      0 vertices  DROPPED  "
                  f"(not in atlas: {missing})")

    ROIS = retained
    print(f"  {len(ROIS)} ROIs retained")

except Exception as e:
    print(f"  [FATAL] nilearn unavailable: {e}")
    raise SystemExit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Load V-JEPA2 encoder via TribeModel
# ─────────────────────────────────────────────────────────────────────────────

print("\nLoading TribeModel …")
model           = TribeModel.from_pretrained("facebook/tribev2", cache_folder=CACHE_DIR)
vjepa2_module   = model.data.video_feature.image.model.model
encoder_blocks  = vjepa2_module.encoder.layer
N_LAYERS        = len(encoder_blocks)
print(f"V-JEPA2 encoder blocks: {N_LAYERS}")

vjepa2_module.eval()
vjepa2_module.to(DEVICE)

# TRIBE v2 samples these two layer indices into its brain encoder
# (confirmed in arxiv:2605.13904 and training config)
TRIBE_LAYERS = {19: "TRIBE L×0.5", 39: "TRIBE L×1.0"}

# ─────────────────────────────────────────────────────────────────────────────
# Hooks + memory helpers
# ─────────────────────────────────────────────────────────────────────────────

layer_acts = {}

def make_hook(idx):
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden: (B, tokens, dim)  — average over tokens then over batch
        layer_acts[idx] = hidden.mean(dim=1).mean(dim=0).detach().float().cpu().numpy()
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

def run_forward(batch_clips):
    """batch_clips: list of clip tensors, each (T, 3, H, W).  Run as a batch."""
    inp = torch.stack(batch_clips, dim=0).to(DEVICE)   # (B, T, 3, H, W)
    with torch.no_grad():
        if USE_AMP:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                vjepa2_module(pixel_values_videos=inp)
        else:
            vjepa2_module(pixel_values_videos=inp)
    del inp

# ─────────────────────────────────────────────────────────────────────────────
# Streaming video decoder — full resolution, full duration, no RAM spike
# ─────────────────────────────────────────────────────────────────────────────

def probe_video(path):
    cmd  = ["ffprobe", "-v", "quiet", "-print_format", "json",
            "-show_streams", str(path)]
    info = json.loads(subprocess.run(cmd, capture_output=True, text=True).stdout)
    vs   = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if vs is None:
        raise RuntimeError(f"No video stream: {path}")
    num, den = vs.get("avg_frame_rate", "30/1").split("/")
    fps      = float(num) / max(float(den), 1e-9)
    dur      = vs.get("duration") or info.get("format", {}).get("duration")
    return fps, float(dur) if dur else int(vs.get("nb_frames", 900)) / fps, \
           int(vs["width"]), int(vs["height"])

def decode_clip(path, start, dur, n_frames, w, h):
    cmd = ["ffmpeg", "-v", "quiet",
           "-ss", f"{start:.6f}", "-t", f"{dur:.6f}",
           "-i", str(path),
           "-vf", f"fps={n_frames/dur:.6f}",
           "-frames:v", str(n_frames),
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw  = subprocess.run(cmd, capture_output=True).stdout
    need = n_frames * h * w * 3
    if len(raw) < need:
        if not raw:
            raise RuntimeError("ffmpeg returned 0 bytes")
        fb  = h * w * 3
        raw += raw[-(len(raw)//fb)*fb or -fb:][-fb:] * \
               ((need - len(raw) + fb - 1) // fb)
    frames = (np.frombuffer(raw[:need], dtype=np.uint8)
               .reshape(n_frames, h, w, 3)
               .astype(np.float32) / 255.0)
    return frames.transpose(0, 3, 1, 2)   # (T, 3, H, W)

def iter_clips(path):
    fps, total_dur, w, h = probe_video(path)
    n = max(1, int(total_dur // CLIP_DURATION))
    if total_dur - n * CLIP_DURATION >= 1.0:
        n += 1
    for c in range(n):
        start  = c * CLIP_DURATION
        actual = min(CLIP_DURATION, total_dur - start)
        if actual < 0.5:
            break
        frames = decode_clip(path, start, actual, CLIP_FRAMES, w, h)
        clip   = torch.from_numpy(frames)
        clip   = torch.stack([normalize_fn(clip[t]) for t in range(CLIP_FRAMES)])
        yield clip
        del frames, clip

# ─────────────────────────────────────────────────────────────────────────────
# Pass 1: extract V-JEPA2 layer activations for all categories
# ─────────────────────────────────────────────────────────────────────────────

print(f"\nExtracting V-JEPA2 layer activations (batch={INFERENCE_BATCH}) …")
cat_layer_acts = {}

for cat in CATEGORIES:
    vpaths = find_videos_for_category(cat)
    if not vpaths:
        continue
    clips = []
    for vp in vpaths:
        handles = register_all_hooks()
        try:
            pending = []
            for clip in iter_clips(vp):
                pending.append(clip)
                if len(pending) == INFERENCE_BATCH:
                    layer_acts.clear()
                    run_forward(pending)
                    if len(layer_acts) == N_LAYERS:
                        # store one averaged activation row per clip in batch
                        for _ in pending:
                            clips.append(np.stack([layer_acts[i] for i in range(N_LAYERS)]))
                    pending = []
                    free_memory()
            if pending:   # flush remainder
                layer_acts.clear()
                run_forward(pending)
                if len(layer_acts) == N_LAYERS:
                    for _ in pending:
                        clips.append(np.stack([layer_acts[i] for i in range(N_LAYERS)]))
                free_memory()
        except Exception as e:
            print(f"  [ERROR] {vp.name}: {e}")
        finally:
            remove_hooks(handles)
    if clips:
        cat_layer_acts[cat] = np.stack(clips)
        print(f"  {cat}: {len(clips)} clips  {cat_layer_acts[cat].shape}")
    free_memory()

cats_with_data = [c for c in CATEGORIES if c in cat_layer_acts]

# ─────────────────────────────────────────────────────────────────────────────
# Pass 2: layer → ROI Pearson r via preds.npy
# ─────────────────────────────────────────────────────────────────────────────

print("\nComputing layer → ROI Pearson r from preds.npy …")

layer_roi_r = {cat: {roi["key"]: np.zeros(N_LAYERS) for roi in ROIS}
               for cat in CATEGORIES}
video_n     = {cat: {roi["key"]: 0 for roi in ROIS}
               for cat in CATEGORIES}

for cat in CATEGORIES:
    vpaths = find_videos_for_category(cat)
    for vp in vpaths:
        preds_path = STUDY_ROOT / cat / vp.stem / "preds.npy"
        if not preds_path.exists():
            continue
        preds    = np.load(preds_path)[:30]
        roi_sigs = {roi["key"]: preds[:, roi["mask"]].mean(axis=1)
                    for roi in ROIS if roi["mask"].sum() > 0}
        if not roi_sigs:
            continue

        clips, handles = [], register_all_hooks()
        try:
            pending = []
            for clip in iter_clips(vp):
                pending.append(clip)
                if len(pending) == INFERENCE_BATCH:
                    layer_acts.clear()
                    run_forward(pending)
                    if len(layer_acts) == N_LAYERS:
                        for _ in pending:
                            clips.append(np.stack([layer_acts[i] for i in range(N_LAYERS)]))
                    pending = []
                    free_memory()
            if pending:
                layer_acts.clear()
                run_forward(pending)
                if len(layer_acts) == N_LAYERS:
                    for _ in pending:
                        clips.append(np.stack([layer_acts[i] for i in range(N_LAYERS)]))
                free_memory()
        except Exception as e:
            print(f"  [ERROR] {vp.name}: {e}")
        finally:
            remove_hooks(handles)
        if not clips:
            continue

        clips   = np.stack(clips)      # (n_clips, N_LAYERS, hidden_dim)
        n_clips = len(clips)

        for rk, vsig in roi_sigs.items():
            bsz = 30.0 / n_clips
            y   = np.array([vsig[int(c*bsz):max(int(c*bsz)+1,
                                                  int((c+1)*bsz))].mean()
                             for c in range(n_clips)])
            if y.std() < 1e-9:
                continue
            rv = np.zeros(N_LAYERS)
            for li in range(N_LAYERS):
                x = np.linalg.norm(clips[:, li, :], axis=-1)
                if x.std() > 1e-9:
                    rv[li] = float(np.corrcoef(x, y)[0, 1])
            layer_roi_r[cat][rk] += rv
            video_n[cat][rk]     += 1

        del clips
        free_memory()

for cat in CATEGORIES:
    for roi in ROIS:
        k = roi["key"]
        if video_n[cat][k] > 0:
            layer_roi_r[cat][k] /= video_n[cat][k]

# ─────────────────────────────────────────────────────────────────────────────
# Plotting  (rendered in parallel across all CPU cores)
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#0d0d0d"
SPINE_COL = "#444444"
layer_idx = np.arange(N_LAYERS)
CAT_COLORS = {c: col for c, col in zip(
    CATEGORIES, plt.cm.tab20(np.linspace(0, 1, len(CATEGORIES)))
)}

def _style(ax, title="", ylabel="", xlabel=""):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.set_title(title, color="white", fontsize=9, pad=4)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]:
        ax.spines[sp].set_color(SPINE_COL)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    for li, lbl in TRIBE_LAYERS.items():
        ax.axvline(li, color="#666", linestyle="--", linewidth=0.9, alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(li + 0.2, ylim[0] + (ylim[1]-ylim[0])*0.97,
                lbl, color="#777", fontsize=6, va="top")

# ── Plot A: one file per ROI — all categories as lines ────────────────────────

def _plot_A_roi(args):
    roi, layer_roi_r, video_n, cats_with_data, CAT_COLORS, \
        TRIBE_LAYERS, N_LAYERS, LOBE_COLORS, ANALYSIS_DIR = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    DARK_BG = "#0d0d0d"; SPINE_COL = "#444444"
    layer_idx = np.arange(N_LAYERS)
    key = roi["key"]
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    for cat in cats_with_data:
        if video_n[cat][key] == 0:
            continue
        ax.plot(layer_idx, layer_roi_r[cat][key],
                label=f"{cat} (n={video_n[cat][key]})",
                color=CAT_COLORS.get(cat, "white"), linewidth=1.8, alpha=0.9)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel("V-JEPA2 layer index  (0 = shallowest, 39 = deepest)", color="white", fontsize=9)
    ax.set_ylabel("Mean Pearson r  (layer activation norm → ROI BOLD)", color="white", fontsize=9)
    ax.set_title(f"{roi['label']}  [{roi['wiki']}]\n{roi['function']}", color="white", fontsize=9, pad=4)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom", "left"]: ax.spines[sp].set_color(SPINE_COL)
    ax.set_xlim(-0.5, N_LAYERS - 0.5)
    for li, lbl in TRIBE_LAYERS.items():
        ax.axvline(li, color="#666", linestyle="--", linewidth=0.9, alpha=0.7)
        ylim = ax.get_ylim()
        ax.text(li + 0.2, ylim[0] + (ylim[1]-ylim[0])*0.97, lbl, color="#777", fontsize=6, va="top")
    ax.legend(facecolor="#111", labelcolor="white", framealpha=0.85,
              fontsize=7, ncol=4, loc="upper left")
    ax.text(0.99, 0.02, f"Source: {roi['source']}",
            color="#666", fontsize=6, ha="right", va="bottom", transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(str(ANALYSIS_DIR / f"roiA_{key.lower()}.png"),
                dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return key

print(f"\nPlot A: per-ROI layer–fMRI correlation ({N_PLOT_WORKERS} workers) …")
_args_A = [
    (roi, layer_roi_r, video_n, cats_with_data, CAT_COLORS,
     TRIBE_LAYERS, N_LAYERS, LOBE_COLORS, ANALYSIS_DIR)
    for roi in ROIS
]
with ProcessPoolExecutor(max_workers=N_PLOT_WORKERS) as ex:
    list(ex.map(_plot_A_roi, _args_A))
print(f"  {len(ROIS)} files saved")

# ── Plot B: per-category — all ROIs as bar grid ───────────────────────────────

def _plot_B_cat(args):
    cat, ROIS, layer_roi_r, video_n, LOBE_COLORS, TRIBE_LAYERS, \
        N_LAYERS, SPINE_COL, ANALYSIS_DIR = args
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    DARK_BG = "#0d0d0d"
    layer_idx = np.arange(N_LAYERS)
    N_COLS = 4
    N_ROWS = (len(ROIS) + N_COLS - 1) // N_COLS
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(22, 4*N_ROWS), sharex=True)
    fig.patch.set_facecolor(DARK_BG)
    aflat = axes.flatten()
    for ri, roi in enumerate(ROIS):
        ax  = aflat[ri]
        r   = layer_roi_r[cat][roi["key"]]
        col = LOBE_COLORS[roi["lobe"]]
        ax.set_facecolor(DARK_BG)
        ax.bar(layer_idx, r, color=col, alpha=0.8, width=0.85)
        ax.axhline(0, color="#555", linewidth=0.7)
        for li in TRIBE_LAYERS:
            ax.axvline(li, color="#666", linestyle="--", linewidth=0.7)
        for li in np.argsort(np.abs(r))[-3:]:
            ax.text(li, r[li]+0.003*np.sign(r[li]),
                    str(li), color="white", fontsize=6, ha="center")
        ax.set_title(f"{roi['label']}  n={video_n[cat][roi['key']]}",
                     color=col, fontsize=9)
        ax.tick_params(colors="white", labelsize=7)
        for sp in ["top","right"]: ax.spines[sp].set_visible(False)
        for sp in ["bottom","left"]: ax.spines[sp].set_color(SPINE_COL)
        ax.set_xlim(-0.5, N_LAYERS-0.5)
        if ri % N_COLS == 0:
            ax.set_ylabel("Pearson r", color="white", fontsize=8)
    for ri in range(len(ROIS), len(aflat)):
        aflat[ri].set_visible(False)
    aflat[min(len(ROIS)-1, len(aflat)-N_COLS)].set_xlabel(
        "V-JEPA2 layer (0→39)", color="white", fontsize=9)
    fig.suptitle(
        f"{cat.upper()}  ·  Layer→ROI Pearson r  |  all {len(ROIS)} cortical regions\n"
        "bar label = top-3 layer indices  |  --- = TRIBE v2 sampled layers (19, 39)",
        color="white", fontsize=11, y=1.005)
    plt.tight_layout()
    plt.savefig(str(ANALYSIS_DIR / f"roiB_cat_{cat}.png"),
                dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    return cat

print(f"Plot B: per-category ROI grid ({N_PLOT_WORKERS} workers) …")
_args_B = [
    (cat, ROIS, layer_roi_r, video_n, LOBE_COLORS, TRIBE_LAYERS,
     N_LAYERS, SPINE_COL, ANALYSIS_DIR)
    for cat in cats_with_data
]
with ProcessPoolExecutor(max_workers=N_PLOT_WORKERS) as ex:
    list(ex.map(_plot_B_cat, _args_B))
print(f"  {len(cats_with_data)} files saved")

# ── Plot C: grand heatmap — categories × ROIs ─────────────────────────────────

print("Plot C: grand heatmap …")
roi_labs = [r["label"] for r in ROIS]
peak_r   = np.zeros((len(cats_with_data), len(ROIS)))
best_li  = np.zeros((len(cats_with_data), len(ROIS)), dtype=int)
for ci, cat in enumerate(cats_with_data):
    for ri, roi in enumerate(ROIS):
        r = layer_roi_r[cat][roi["key"]]
        b = int(np.argmax(np.abs(r)))
        best_li[ci, ri] = b
        peak_r[ci, ri]  = r[b]

fig, ax = plt.subplots(figsize=(max(14, len(ROIS)*0.85),
                                max(6, len(cats_with_data)*0.8)))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
vmax = max(0.01, np.abs(peak_r).max())
im   = ax.imshow(peak_r, aspect="auto", cmap="RdBu_r",
                 vmin=-vmax, vmax=vmax, interpolation="nearest")
ax.set_xticks(range(len(ROIS)))
ax.set_xticklabels(roi_labs, color="white", fontsize=9,
                   rotation=45, ha="right")
ax.set_yticks(range(len(cats_with_data)))
ax.set_yticklabels(cats_with_data, color="white", fontsize=10)
for ci in range(len(cats_with_data)):
    for ri in range(len(ROIS)):
        v  = peak_r[ci, ri]
        fc = "white" if abs(v) > vmax*0.45 else "#bbb"
        ax.text(ri, ci, f"{v:+.2f}\nL{best_li[ci,ri]}",
                ha="center", va="center", color=fc, fontsize=6.5)
cb = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
cb.ax.tick_params(colors="white")
cb.set_label("Peak Pearson r  (best layer)", color="white", fontsize=9)
ax.set_title(
    "Grand summary  ·  Content category × Cortical region\n"
    "colour = peak Pearson r  |  cell = best V-JEPA2 layer index",
    color="white", fontsize=12)
ax.tick_params(colors="white")
for sp in ax.spines.values():
    sp.set_edgecolor(SPINE_COL)
# lobe dividers
prev_lobe = None
for ri, roi in enumerate(ROIS):
    if roi["lobe"] != prev_lobe and prev_lobe is not None:
        ax.axvline(ri-0.5, color="#888", linewidth=1.2, linestyle=":")
    prev_lobe = roi["lobe"]
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "roiC_summary_heatmap.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved roiC_summary_heatmap.png")

# ── Plot D: lobe-level bar chart ──────────────────────────────────────────────

print("Plot D: lobe-level summary …")
lobes  = ["occipital", "temporal", "parietal", "frontal", "insula"]
lobe_r = np.zeros((len(cats_with_data), len(lobes)))
for ci, cat in enumerate(cats_with_data):
    for li, lobe in enumerate(lobes):
        vals = [np.abs(layer_roi_r[cat][roi["key"]]).max()
                for roi in ROIS if roi["lobe"] == lobe]
        lobe_r[ci, li] = np.mean(vals) if vals else 0.0

x     = np.arange(len(lobes))
width = 0.8 / max(len(cats_with_data), 1)
fig, ax = plt.subplots(figsize=(13, 5))
fig.patch.set_facecolor(DARK_BG)
ax.set_facecolor(DARK_BG)
for ci, cat in enumerate(cats_with_data):
    offset = (ci - len(cats_with_data)/2) * width + width/2
    ax.bar(x+offset, lobe_r[ci], width*0.92,
           color=CAT_COLORS.get(cat, "white"), label=cat, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([l.capitalize() for l in lobes], color="white", fontsize=11)
ax.set_ylabel("Mean peak |Pearson r| across ROIs in lobe", color="white")
ax.set_title("Lobe-level layer→fMRI strength by content category", color="white")
ax.tick_params(colors="white")
for sp in ["top","right"]:
    ax.spines[sp].set_visible(False)
for sp in ["bottom","left"]:
    ax.spines[sp].set_color(SPINE_COL)
ax.legend(facecolor="#111", labelcolor="white", framealpha=0.85,
          fontsize=9, ncol=4)
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "roiD_lobe_summary.png",
            dpi=150, bbox_inches="tight", facecolor=DARK_BG)
plt.close()
print(f"  Saved roiD_lobe_summary.png")

# ─────────────────────────────────────────────────────────────────────────────
# Abliteration contrast maps — LOSO pairwise (fully data-driven)
# ─────────────────────────────────────────────────────────────────────────────
# Every ordered pair of categories with data is contrasted automatically.
# No hardcoded assumptions about which categories are "neutral" or "target".

print("Plot E: LOSO pairwise contrast maps …")
for i, pos_cat in enumerate(cats_with_data):
    for neg_cat in cats_with_data[i+1:]:
        diff_peak = np.zeros(len(ROIS))
        diff_layer = np.zeros(len(ROIS), dtype=int)
        for ri, roi in enumerate(ROIS):
            rp = layer_roi_r[pos_cat][roi["key"]]
            rn = layer_roi_r[neg_cat][roi["key"]]
            diff = rp - rn
            b = int(np.argmax(np.abs(diff)))
            diff_layer[ri] = b
            diff_peak[ri]  = diff[b]

        fig, ax = plt.subplots(figsize=(max(14, len(ROIS)*0.9), 5))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(DARK_BG)
        bar_colors = [LOBE_COLORS[roi["lobe"]] for roi in ROIS]
        bars = ax.bar(range(len(ROIS)), diff_peak, color=bar_colors, alpha=0.85)
        ax.axhline(0, color="#555", linewidth=0.8)
        for ri, (bar, layer) in enumerate(zip(bars, diff_layer)):
            ypos = diff_peak[ri]
            ax.text(ri, ypos + 0.003*np.sign(ypos),
                    f"L{layer}", color="white", fontsize=6.5,
                    ha="center", va="bottom" if ypos >= 0 else "top")
        ax.set_xticks(range(len(ROIS)))
        ax.set_xticklabels([r["label"] for r in ROIS],
                           rotation=45, ha="right", color="white", fontsize=8)
        ax.set_ylabel(f"Δ Pearson r  ({pos_cat} − {neg_cat})", color="white", fontsize=9)
        ax.set_title(f"{pos_cat.upper()} vs {neg_cat.upper()}", color="white", fontsize=10)
        ax.tick_params(colors="white")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom", "left"]:
            ax.spines[sp].set_color(SPINE_COL)
        prev_lobe = None
        for ri, roi in enumerate(ROIS):
            if roi["lobe"] != prev_lobe and prev_lobe is not None:
                ax.axvline(ri-0.5, color="#666", linewidth=1.0, linestyle=":")
            prev_lobe = roi["lobe"]
        plt.tight_layout()
        safe_label = f"{pos_cat}_vs_{neg_cat}"
        plt.savefig(ANALYSIS_DIR / f"roiE_contrast_{safe_label}.png",
                    dpi=150, bbox_inches="tight", facecolor=DARK_BG)
        plt.close()
        print(f"  Saved: roiE_contrast_{safe_label}.png")

# ─────────────────────────────────────────────────────────────────────────────
# Printed mapping table
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*90)
print("DEFINITIVE LAYER → BRAIN REGION MAPPING")
print("="*90)
print(f"{'ROI':<12} {'Wiki region':<46} {'Best cat':<14} "
      f"{'peak r':>7}  {'layer':>5}  {'depth':>5}")
print("-"*90)
for roi in ROIS:
    key = roi["key"]
    best_cat, best_r, best_layer = None, -999.0, -1
    for cat in cats_with_data:
        r  = layer_roi_r[cat][key]
        li = int(np.argmax(np.abs(r)))
        if abs(r[li]) > abs(best_r):
            best_r, best_cat, best_layer = r[li], cat, li
    print(f"{roi['label']:<12} {roi['wiki'][:44]:<46} {str(best_cat):<14} "
          f"{best_r:>+7.4f}  {best_layer:>5d}  {best_layer/N_LAYERS:>5.2f}")

print("\n" + "="*90)
print("DATA-DRIVEN CATEGORY SUMMARY")
print("="*90)
print("\nPer-category top ROIs (sorted by peak |r| across all layers):")
for cat in cats_with_data:
    # Collect (roi_label, peak_r, best_layer) and sort by strength
    roi_scores = []
    for roi in ROIS:
        r = layer_roi_r[cat][roi["key"]]
        li = int(np.argmax(np.abs(r)))
        roi_scores.append((roi["label"], r[li], li))
    roi_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"\n  {cat.upper()}:")
    for label, peak_r, best_layer in roi_scores[:5]:
        print(f"    {label:16s}  r={peak_r:+.4f}  layer={best_layer}")

print("\n\nREGIONS NOT MAPPABLE (outside fsaverage5 cortical surface):")
for r in ["Nucleus accumbens, Caudate/Putamen, Amygdala, VTA",
          "Hippocampus, Thalamus, Hypothalamus",
          "Brainstem, Cerebellum, White-matter tracts"]:
    print(f"  ✗  {r}")

print(f"\nAll outputs → {ANALYSIS_DIR.resolve()}")
for f in sorted(ANALYSIS_DIR.glob("roi*.png")):
    print(f"  {f.name}")

# ── Save per-category mean |r| profiles across all ROIs ───────────────────
# shape per category: (N_LAYERS,) — used by abliteration.py for auto layer selection

MASK_DIR = STUDY_ROOT / "masks"
MASK_DIR.mkdir(exist_ok=True)

profile_data = {}
for cat in cats_with_data:
    # Mean absolute Pearson r across all retained ROIs, per layer
    profiles = np.stack([np.abs(layer_roi_r[cat][roi["key"]]) for roi in ROIS])
    profile_data[cat] = profiles.mean(axis=0)   # (N_LAYERS,)

np.savez(MASK_DIR / "layer_profiles.npz", **profile_data)
print(f"Layer profiles saved → {MASK_DIR / 'layer_profiles.npz'}")