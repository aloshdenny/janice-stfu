The dependency chain then becomes fully linear with no human decision points:

infer_bulk.py          → tribe_study/*/preds.npy
strict_analysis.py     → masks/abliteration_config.json  +  masks/*_strict.npy
layer_analysis.py            → masks/layer_profiles.npz  +  analysis/roi*.png
abliteration.py        → reads config + profiles, runs end-to-end