The dependency chain then becomes fully linear with no human decision points:

Preparing the dataset:
Videos in each category are set to 30s, 48 each and resolution 256x256

There are 7 baselines categories: nature, food, chase, fight, cute, kissing, gore

And one target ablation category: porn

Data layout:
  data/baselines/   — cute, nature, food, kissing, chase, fight, gore (7×48 = 336 videos)
  data/targets/     — porn (1×48 = 48 videos)

Pipeline:
  infer_bulk.py       → tribe_study/*/preds.npy          (skips already-processed)
  layer_analysis.py   → masks/layer_profiles.npz         + analysis/roi*.png
  abliteration.py     → masks + config + activations     + abliterated checkpoint
  validation.py       → validates abliterated model      against baseline preds

Tolerance scale:
  -1       -0.5       0       +0.5       +1
   |--------|---------|---------|--------|
   Repulsion Aversion  Neutral  Acceptance Attraction

  tolerance < 0  → suppress / repel the concept
                   the model steers away from the target activation pattern
  tolerance = 0  → neutral — no surgery effect
  tolerance > 0  → accept / tolerate the concept
                   the model no longer pushes away from that semantic region

  Positive tolerance ≠ preference.
  A tolerant brain doesn't seek the concept — it simply doesn't experience a push away.

  Default: --tolerance -0.8

  Examples:
    python scripts/abliteration.py --target porn --tolerance -1.0    # immense repulsion
    python scripts/abliteration.py --target porn --tolerance -0.5    # moderate aversion
    python scripts/abliteration.py --target porn --tolerance  0.0    # neutral (no-op)
    python scripts/abliteration.py --target porn --tolerance  0.5    # acceptance