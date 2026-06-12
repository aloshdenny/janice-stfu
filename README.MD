The dependency chain then becomes fully linear with no human decision points:

Preparing the dataset:
Videos in each category are set to 30s, 48 each and resolution 256x256

Data layout:
  data/{category}/  — one subfolder per category, auto-discovered
  Categories are NOT hardcoded. Add a new folder → it's picked up automatically.

  Current categories (10):
    argument, chase, cute, fight, food, gore, kissing, nature, porn, sports

  Target is specified at runtime: --target porn
  All other categories become baselines automatically.

Pipeline:
  infer_bulk.py       → tribe_study/*/preds.npy          (skips already-processed)
  layer_analysis.py   → masks/layer_profiles.npz         + analysis/roi*.png
  abliteration.py     → masks + config + activations     + abliterated checkpoint
  validation.py       → validates abliterated model      against baseline preds

Contrast design — LOSO (Leave-One-Subject-Out):
  No hardcoded neutral_low / neutral_high. Every category is contrasted
  against every other in a fully data-driven design.

  For each category:
    LOSO contrast    = category − mean(all others)
    LOSO-k contrast  = category − mean(all others except k)  (discovers confounds)
    Pairwise         = category − each other category individually
    Multivariate     = vertices where category > every other category

  Like a radar chart: all categories form the vertices, and we can see
  which component overlaps with which other — derived from readings,
  not from assumptions.

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
    python scripts/abliteration.py --target porn --tolerance -1.0
    python scripts/abliteration.py --target porn --tolerance -0.5
    python scripts/abliteration.py --target porn --tolerance  0.0