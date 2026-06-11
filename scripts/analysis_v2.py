"""
analysis.py  (v2 — expanded ROIs + abliteration-purpose categories)

SCIENTIFIC SCOPE
━━━━━━━━━━━━━━━━
TRIBE v2 predicts on the fsaverage5 CORTICAL SURFACE only (20,484 vertices).
Every ROI below is grounded in the Destrieux (aparc.a2009s) atlas.
All label strings are EXACT aparc.a2009s names verified against the official
FreeSurfer wiki table at:
  surfer.nmr.mgh.harvard.edu/fswiki/DestrieuxAtlasChanges

The full atlas has 75 labels per hemisphere (74 cortical + Medial_wall).
v1 covered 22 ROIs.  v2 adds 9 more for 31 ROIs total, filling genuine gaps:

NEW ROIs IN v2:
  OFA      — Occipital Face Area (lateral occipital gyrus; G_occipital_middle
             overlaps OFA in standard fMRI localizers — same region used here)
  TP       — Temporal Pole  (Pole_temporal; person familiarity / semantic memory)
  ATL      — Anterior Temporal Lobe face area  (G_temporal_inf anterior section;
             person identity storage beyond FFA, critical for "knowing who")
  PREC     — Precuneus  (G_precuneus; self-referential imagery, mental simulation,
             autobiographical memory — key for "getting over" someone)
  TPJ      — Temporo-Parietal Junction  (G_and_S_subcentral; social cognition,
             mentalising, person models)
  LO       — Lateral Occipital Cortex  (G_oc-temp_med-Lingual; object recognition
             complement to V4/MT)
  EC       — Entorhinal Cortex  (S_oc-temp_med_and_Lingual posterior section maps
             onto perirhinal/entorhinal belt; face-familiarity gate to memory)
  FPC      — Frontal Pole / frontopolar PFC  (G_and_S_frontomargin +
             G_and_S_transv_frontopol; prospective thinking, craving suppression)
  MPC      — Medial Parietal Cortex / posterior cingulate junction
             (S_subparietal + G_precuneus posterior; autobiographical retrieval)
  LAT_FIS  — Lateral Fissure / planum temporale  (Lat_Fis-post; language, music,
             emotion prosody — relevant for audiovisual sexual content)

NOT MAPPABLE (outside fsaverage5 cortical surface, dropped without error):
  Nucleus accumbens, caudate, putamen (striatum) — the primary porn-reward locus
  is subcortical.  TRIBE v2 cannot model it.  For abliteration of porn-craving
  circuits you would need subcortical brain encoders (e.g. deep-brain fMRI models)
  or indirect OFC/vmPFC/ACC proxies which ARE mappable here.
  Same applies to: VTA, amygdala, hippocampus, thalamus, hypothalamus,
  brainstem, cerebellum, all white-matter tracts.

ABLITERATION USE-CASE NOTES
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Goal A — Porn addiction:  The cortical nodes most relevant are OFC (reward
  valuation, already in v1), vmPFC/MPFC (cue reactivity, in v1), ACC (craving
  monitoring, in v1), and the newly added FPC (frontopolar — cue suppression).
  The subcortical nucleus accumbens is outside the surface model; use OFC/MPFC
  as its cortical proxy.  Key contrast: porn vs. neutral.

Goal B — Specific-face desensitisation ("getting over an ex"):
  The relevant circuit is FFA (in v1) → OFA (new) → TP/ATL (new, person
  semantics) → PREC (new, mental simulation / "thinking about them") → mPFC
  (in v1, self-relevance).  Key contrast: target-person vs. unknown-face.

Sources: Destrieux et al. 2010; Haxby et al. 2000; Kanwisher 1997;
         Voon et al. 2014 (compulsive sexual behaviour); Rossion 2014 (OFA);
         Olson et al. 2013 (ATL face patches); PNAS 2024 doi:10.1073/pnas.2321346121
         (temporal pole); arxiv:2605.13904 (TRIBE v2 feature viz)
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

DATA_DIR      = Path("./data")
STUDY_ROOT    = Path("./tribe_study")
CACHE_DIR     = Path("./cache")
ANALYSIS_DIR  = Path("./analysis")
ANALYSIS_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Categories (v2 — expanded for abliteration research) ─────────────────────
# Original 8 kept; 7 new ones added.
# Rationale for additions:
#
#   face_familiar   — Videos/photos of ONE specific known person (e.g. the ex).
#                     Activates TP, ATL, PREC, mPFC more than generic faces.
#                     Critical for Goal B.
#   face_unknown    — Control: strangers.  Isolates generic FFA/OFA from
#                     person-semantic TP/ATL signal.
#   erotic_static   — Single-frame erotic images (no motion).  Separates OFC/
#                     vmPFC reward response from MT/STS motion response in "porn".
#   disgust         — High-arousal negative content without violence.  Needed
#                     to dissociate valence from arousal in gore/porn contrasts.
#                     Activates INS + OFC in a distinct direction.
#   craving_neutral — Matched neutral clips for baseline subtraction.
#   sport           — High-motion, high-arousal, no sexuality/gore.  Controls for
#                     arousal and motion in MT/STS when contrasting porn/gore.
#   relax           — Nature-like but specifically calming (meditation scenes).
#                     Separates "nature" arousal from genuine relaxation baseline.

CATEGORIES = [
    # original
    "porn", "gore", "cute", "nature", "food", "kissing", "chase", "fight",
    # new
    "face_familiar", "face_unknown", "erotic_static",
    "disgust", "craving_neutral", "sport", "relax",
]

CLIP_FRAMES   = 16
CLIP_DURATION = 4.0

from torchvision import transforms
normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

# ─────────────────────────────────────────────────────────────────────────────
# ROI TABLE — exact aparc.a2009s label names (verified against FreeSurfer wiki)
# ─────────────────────────────────────────────────────────────────────────────

ROIS = [
    # ── Occipital lobe ────────────────────────────────────────────────────
    dict(key="V1",  label="V1",        lobe="occipital",
         wiki="Primary visual cortex (V1) — BA17",
         function="Oriented edges, retinotopic map, luminance/contrast",
         source="Hubel & Wiesel 1962",
         destrieux_exact=["S_calcarine"]),

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
    # In standard fMRI localizers OFA sits at the junction of lateral occipital
    # and inferior occipital gyrus, well captured by S_oc_middle_and_Lunatus +
    # G_and_S_occipital_inf.  It is the first cortical node to encode face
    # structural configuration before FFA.  Rossion 2014; Pitcher et al. 2011.
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
         destrieux_exact=["G_temp_sup-G_T_transv"]),

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
    # Critical for "knowing who this is" — the TP stores person-level semantic
    # memories.  Damage → associative prosopagnosia (Olson et al. 2013; PNAS 2024).
    # This is THE target for Goal B (desensitisation to a specific person's face).
    dict(key="TP",  label="Temporal Pole", lobe="temporal",
         wiki="Temporal pole (BA38) — person semantics",
         function="Familiar person recognition, person-semantic memory, "
                  "connecting face identity to autobiographical knowledge",
         source="Olson et al. 2013; Diano et al. 2024 PNAS",
         destrieux_exact=["Pole_temporal"]),

    # NEW: Anterior Temporal Lobe face area (ATL-FA)
    # Sits on the ventral surface of anterior temporal cortex, anterior MTG/ITG.
    # Responds more to personally familiar than unfamiliar faces.
    # In Destrieux it overlaps with G_temporal_inf (anterior portion) and
    # G_oc-temp_med-Parahip anterior extent.
    dict(key="ATL", label="ATL-FA",     lobe="temporal",
         wiki="Anterior temporal face area — familiar person identity",
         function="Person identity storage; familiarity beyond FFA; "
                  "links perceptual face to biographical knowledge",
         source="Rossion 2014; Von der Heide et al. 2013",
         destrieux_exact=["G_temporal_inf",
                           "G_oc-temp_med-Parahip"]),

    # ── Lateral Fissure / planum temporale (new) ──────────────────────────
    # Lat_Fis-post = posterior ramus of lateral (Sylvian) fissure = planum temporale.
    # Processes prosody, emotional tone in audiovisual content.
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
    # G_precuneus is an exact aparc.a2009s label.
    # When you "think about" your ex, the precuneus is active generating the
    # mental image.  A key node for the "getting over someone" circuit alongside
    # PCC/RSC and mPFC.
    dict(key="PREC", label="Precuneus",  lobe="parietal",
         wiki="Precuneus (BA7) — visual mental imagery & autobiographical memory",
         function="Mental imagery of people/scenes, visuospatial episodic memory, "
                  "self-referential simulation ('imagining the ex')",
         source="Cavanna & Trimble 2006",
         destrieux_exact=["G_precuneus",
                           "S_subparietal"]),

    # NEW: Temporo-Parietal Junction — theory of mind, person models
    # G_and_S_subcentral is the subcentral gyrus/sulcus complex at the lower
    # end of the postcentral region, maps onto the TPJ in functional atlases.
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
                           "G_front_sup"]),

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
    # G_and_S_frontomargin + G_and_S_transv_frontopol are the canonical
    # frontopolar labels.  This region is activated during cue-exposure +
    # suppression tasks and in craving regulation.  fNIRS porn-addiction study
    # (Frontiers 2025) found FPC hyper-connectivity in addicted group.
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
    for h in handles: h.remove()

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
# Streaming video decoder
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
    return frames.transpose(0, 3, 1, 2)

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

print("\nExtracting V-JEPA2 layer activations …")
cat_layer_acts = {}

for cat in CATEGORIES:
    vpaths = sorted(DATA_DIR.glob(f"{cat}*.mp4"))
    if not vpaths:
        continue
    clips = []
    for vp in vpaths:
        handles = register_all_hooks()
        try:
            for clip in iter_clips(vp):
                layer_acts.clear()
                run_forward(clip)
                if len(layer_acts) == N_LAYERS:
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
    vpaths = sorted(DATA_DIR.glob(f"{cat}*.mp4"))
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
            for clip in iter_clips(vp):
                layer_acts.clear()
                run_forward(clip)
                if len(layer_acts) == N_LAYERS:
                    clips.append(np.stack([layer_acts[i] for i in range(N_LAYERS)]))
                free_memory()
        except Exception as e:
            print(f"  [ERROR] {vp.name}: {e}")
        finally:
            remove_hooks(handles)
        if not clips:
            continue

        clips   = np.stack(clips)
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
# Abliteration contrast maps  (NEW in v2)
# ─────────────────────────────────────────────────────────────────────────────
# These contrasts directly inform which ROIs and which V-JEPA2 layers to target
# for weight surgery.

ABLITERATION_CONTRASTS = [
    dict(label="Porn vs Neutral",
         pos="porn",   neg="craving_neutral",
         title="PORN vs NEUTRAL  —  reward/craving cortical signature\n"
               "Target nodes for porn-addiction abliteration"),
    dict(label="Porn vs Sport",
         pos="porn",   neg="sport",
         title="PORN vs SPORT  —  sexual specificity (controls arousal+motion)"),
    dict(label="Gore vs Disgust",
         pos="gore",   neg="disgust",
         title="GORE vs DISGUST  —  violence specificity\n"
               "(controls for disgust/insula activation)"),
    dict(label="Face_familiar vs Face_unknown",
         pos="face_familiar", neg="face_unknown",
         title="FAMILIAR vs UNKNOWN FACE  —  person-semantic circuit\n"
               "Target nodes for specific-person desensitisation (Goal B)"),
    dict(label="Face_familiar vs Nature",
         pos="face_familiar", neg="nature",
         title="FAMILIAR FACE vs NATURE  —  identity memory vs visual baseline"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Plotting
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

# ── Plot A: per-ROI all categories ───────────────────────────────────────────
print("\nPlot A: per-ROI layer–fMRI correlation …")
for roi in ROIS:
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
    _style(ax,
           title=(f"{roi['label']}  [{roi['wiki']}]\n{roi['function']}"),
           ylabel="Mean Pearson r  (layer activation norm → ROI BOLD)",
           xlabel="V-JEPA2 layer index  (0 = shallowest, 39 = deepest)")
    ax.legend(facecolor="#111", labelcolor="white", framealpha=0.85,
              fontsize=7, ncol=4, loc="upper left")
    ax.text(0.99, 0.02, f"Source: {roi['source']}",
            color="#666", fontsize=6, ha="right", va="bottom",
            transform=ax.transAxes)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f"roiA_{key.lower()}.png",
                dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
print(f"  {len(ROIS)} files saved")

# ── Plot B: per-category bar grid ────────────────────────────────────────────
print("Plot B: per-category ROI grid …")
N_COLS = 4
N_ROWS = (len(ROIS) + N_COLS - 1) // N_COLS

for cat in cats_with_data:
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
        for sp in ["top","right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["bottom","left"]:
            ax.spines[sp].set_color(SPINE_COL)
        ax.set_xlim(-0.5, N_LAYERS-0.5)
        if ri % N_COLS == 0:
            ax.set_ylabel("Pearson r", color="white", fontsize=8)
    for ri in range(len(ROIS), len(aflat)):
        aflat[ri].set_visible(False)
    fig.suptitle(
        f"{cat.upper()}  ·  Layer→ROI Pearson r  |  all {len(ROIS)} cortical regions\n"
        "bar label = top-3 layer indices  |  --- = TRIBE v2 sampled layers (19, 39)",
        color="white", fontsize=11, y=1.005)
    plt.tight_layout()
    plt.savefig(ANALYSIS_DIR / f"roiB_cat_{cat}.png",
                dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
print(f"  {len(cats_with_data)} files saved")

# ── Plot C: grand heatmap ────────────────────────────────────────────────────
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

# ── Plot D: lobe-level bar chart ─────────────────────────────────────────────
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

# ── Plot E: abliteration contrast maps (NEW) ──────────────────────────────────
print("Plot E: abliteration contrast maps …")
for contrast in ABLITERATION_CONTRASTS:
    pos_cat = contrast["pos"]
    neg_cat = contrast["neg"]
    if pos_cat not in cats_with_data or neg_cat not in cats_with_data:
        print(f"  Skipping '{contrast['label']}' — missing data")
        continue

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
    ax.set_title(contrast["title"], color="white", fontsize=10)
    ax.tick_params(colors="white")
    for sp in ["top","right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]:
        ax.spines[sp].set_color(SPINE_COL)
    # lobe separators
    prev_lobe = None
    for ri, roi in enumerate(ROIS):
        if roi["lobe"] != prev_lobe and prev_lobe is not None:
            ax.axvline(ri-0.5, color="#666", linewidth=1.0, linestyle=":")
        prev_lobe = roi["lobe"]
    plt.tight_layout()
    safe_label = contrast["label"].replace(" ", "_").replace("/", "-")
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
print("ABLITERATION TARGET SUMMARY")
print("="*90)
print("""
Goal A — Porn addiction (cortical proxies for NAcc reward circuit):
  Primary:   OFC  (reward valuation of VSS, lateral OFC = erotic pleasure)
             MPFC (cue reactivity, vmPFC subjective arousal)
             ACC  (ventral ACC activated in porn CS+ conditioning)
  Secondary: FPC  (frontopolar — hyper-connected in addicted group)
             INS  (anterior insula — craving interoception)
  Contrast to use: 'Porn vs Neutral' and 'Porn vs Sport'

Goal B — Specific-person desensitisation (ex-partner / target face):
  Primary:   TP   (person-semantic memory — "knowing who this is")
             ATL  (anterior temporal face area — familiar person identity)
             FFA  (face detection gateway)
             OFA  (structural face encoding, feeds FFA)
  Secondary: PREC (mental imagery/simulation of the person)
             MPFC (self-relevance of the person)
             PCC  (autobiographical episodic hub)
  Contrast to use: 'Face_familiar vs Face_unknown'
  Note: for single-person abliteration, you need videos/photos of THAT
  specific person as one category and unknown faces as the other.
  Weight surgery should target the layer + ROI with peak Δr in that contrast.
""")

print("\nREGIONS NOT MAPPABLE (outside fsaverage5 cortical surface):")
for r in ["Nucleus accumbens  ← PRIMARY porn-reward target (subcortical)",
          "Caudate / Putamen  ← habit-formation in addiction",
          "Amygdala           ← emotional salience / fear conditioning",
          "VTA / Substantia nigra  ← dopamine source",
          "Hippocampus  (subcortical in FreeSurfer)",
          "Thalamus & all thalamic nuclei",
          "Hypothalamus & nuclei",
          "Brainstem, cerebellum",
          "All white-matter tracts & ventricular system"]:
    print(f"  ✗  {r}")

print(f"\nAll outputs → {ANALYSIS_DIR.resolve()}")
for f in sorted(ANALYSIS_DIR.glob("roi*.png")):
    print(f"  {f.name}")