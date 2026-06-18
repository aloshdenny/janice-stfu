The dependency chain then becomes fully linear with no human decision points:

Preparing the dataset:
Videos in each category are set to 30s, 48 each and resolution 256x256

Data layout:
  data/{category}/  — one subfolder per category, auto-discovered
  Categories are NOT hardcoded. Add a new folder → it's picked up automatically.

  Current categories (13):
    argument, butcher, chase, cute, dance, fight, food, gore,
    kissing, nature, porn, sports, surgery

  Target is specified at runtime: --target porn
  All other categories become baselines automatically.

Pipeline:
  infer_bulk.py       → tribe_study/*/preds.npy          (skips already-processed)
  layer_analysis.py   → masks/layer_profiles.npz         + analysis/roi*.png
  abliteration.py     → masks + config + per-layer acts  + abliterated checkpoint
  validation.py       → validates abliterated model      against baseline preds

Contrast design — LOSO (Leave-One-Subject-Out):
  No hardcoded neutral_low / neutral_high. Every category is contrasted
  against every other in a fully data-driven design.

  For each category:
    LOSO contrast    = category − mean(all others)
    LOSO-k contrast  = category − mean(all others except k)  (discovers confounds)
    Pairwise         = category − each other category individually
    Multivariate     = vertices where category > every other category

  Mask scoring uses margin-based selection:
    score = min_margin × selectivity
    where min_margin = target_activation − max_competitor_activation
    A mask where porn barely exceeds food (margin ≈ 0) scores near zero.
    A mask where porn clearly dominates all others scores high.

Multi-layer surgery:
  --n_layers N selects the top-N layers by correlation strength from
  layer_analysis.py profiles. Activations are collected at EACH layer
  independently, PCA directions are found per-layer, and surgery is
  applied per-layer with its own directions.

  Per-layer direction files: abliterated/{target}_directions_L{k}.npy
  Surgery log: abliterated/surgery_log.json (lists all operated layers)
  Validation reads surgery_log.json to patch the correct layers.

Tolerance scale:
  -1       -0.5       0       +0.5       +1
   |--------|---------|---------|--------|
   Repulsion Aversion  Neutral  Acceptance Attraction

  tolerance < 0  → suppress / repel the concept
  tolerance = 0  → neutral — no surgery effect
  tolerance > 0  → accept / tolerate the concept

  Positive tolerance ≠ preference.
  A tolerant brain doesn't seek the concept — it simply doesn't experience a push away.

  Default: --tolerance -0.8

  Examples:
    python scripts/abliteration.py --target porn --tolerance -1.0 --n_components 3 --n_layers 5
    python scripts/abliteration.py --target porn --tolerance -0.5 --n_components 2 --n_layers 3
    python scripts/validation.py --target porn

Notes:
  If you run into deadlocks, run:
    find ./cache -name "inflight.db" -delete