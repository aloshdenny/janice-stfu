"""
strict_analysis.py
"""

import os
import warnings
import logging

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
from pathlib import Path

STUDY_ROOT = Path("./tribe_study")
MASK_DIR   = STUDY_ROOT / "masks"
MAX_TRS    = 30

CATEGORIES = ["porn", "gore", "cute", "nature", "food",
              "kissing", "chase", "fight"]

def load_category_mean(category):
    paths = sorted((STUDY_ROOT / category).glob("*/preds.npy"))
    arrays = [np.load(p)[:MAX_TRS].mean(axis=0) for p in paths]
    return np.stack(arrays).mean(axis=0)

print("Loading category means...")
means = {cat: load_category_mean(cat) for cat in CATEGORIES}

# ── Stricter contrast design ───────────────────────────────────────────────
# Problem: low-arousal neutral (cute/nature/food) doesn't control for
# visual complexity or skin tones. Use ALL non-target categories as neutral.

# Porn target: subtract everything except porn
porn_neutral  = np.stack([means[c] for c in
                          ["gore","cute","nature","food","kissing","chase","fight"]]
                         ).mean(axis=0)

# Gore target: subtract everything except gore
gore_neutral  = np.stack([means[c] for c in
                          ["porn","cute","nature","food","kissing","chase","fight"]]
                         ).mean(axis=0)

porn_contrast = means["porn"] - porn_neutral
gore_contrast = means["gore"] - gore_neutral

# Cross-subtract to remove shared variance
porn_specific = porn_contrast - gore_contrast
gore_specific = gore_contrast - porn_contrast

# Also try: porn vs its closest confound (kissing + chase — intimate + high motion)
porn_tight = means["porn"] - np.stack([means["kissing"], means["chase"]]).mean(axis=0)
gore_tight  = means["gore"] - np.stack([means["fight"],  means["chase"]]).mean(axis=0)

porn_no_food = means["porn"] - np.stack([
    means["kissing"], means["chase"], means["food"]
]).mean(axis=0)

gore_no_food = means["gore"] - np.stack([
    means["cute"], means["nature"], means["kissing"], means["chase"], means["fight"]
]).mean(axis=0)

CONTRASTS = {
    "porn_allneutral":  porn_contrast,
    "gore_allneutral":  gore_contrast,
    "porn_specific":    porn_specific,
    "gore_specific":    gore_specific,
    "porn_tight":       porn_tight,
    "gore_tight":       gore_tight,
    "porn_no_food":     porn_no_food,
    "gore_no_food":     gore_no_food,
}

# ── Stricter masking: require top 10% AND positive ─────────────────────────
def make_strict_mask(contrast, pct=90):
    thresh = np.percentile(contrast, pct)
    return (contrast > thresh) & (contrast > 0)

print("\nNew masks:")
print(f"{'Name':25s}  {'n_verts':>8}  {'LH':>6}  {'RH':>6}  {'mean_val':>10}")
print("-" * 65)

new_masks = {}
for name, data in CONTRASTS.items():
    mask = make_strict_mask(data, pct=90)
    new_masks[name] = mask
    lh = mask[:10242].sum()
    rh = mask[10242:].sum()
    mean_val = float(data[mask].mean()) if mask.sum() > 0 else 0
    print(f"  {name:23s}  {mask.sum():8d}  {lh:6d}  {rh:6d}  {mean_val:10.4f}")

# Option 3: Logical AND (no percentile thresholding)
name = "gore_strict_bicontrast"
mask = (means["gore"] - means["food"] > 0) & \
       (means["gore"] - means["porn"] > 0) & \
       (means["gore"] - means["cute"] > 0)
new_masks[name] = mask
lh = mask[:10242].sum()
rh = mask[10242:].sum()
bicontrast_data = np.minimum(np.minimum(means["gore"] - means["food"], means["gore"] - means["porn"]), means["gore"] - means["cute"])
mean_val = float(bicontrast_data[mask].mean()) if mask.sum() > 0 else 0
print(f"  {name:23s}  {mask.sum():8d}  {lh:6d}  {rh:6d}  {mean_val:10.4f}")

# Option 3 (fully strict): gore is greater than all other categories
name_all = "gore_strict_multivariate_all"
mask_all = (means["gore"] - means["food"] > 0) & \
           (means["gore"] - means["porn"] > 0) & \
           (means["gore"] - means["cute"] > 0) & \
           (means["gore"] - means["nature"] > 0) & \
           (means["gore"] - means["kissing"] > 0) & \
           (means["gore"] - means["chase"] > 0) & \
           (means["gore"] - means["fight"] > 0)
new_masks[name_all] = mask_all
lh_all = mask_all[:10242].sum()
rh_all = mask_all[10242:].sum()
multivariate_data = means["gore"]
for c in CATEGORIES:
    if c != "gore":
        multivariate_data = np.minimum(multivariate_data, means["gore"] - means[c])
mean_val_all = float(multivariate_data[mask_all].mean()) if mask_all.sum() > 0 else 0
print(f"  {name_all:23s}  {mask_all.sum():8d}  {lh_all:6d}  {rh_all:6d}  {mean_val_all:10.4f}")

# ── Selectivity check ──────────────────────────────────────────────────────
print("\nSelectivity check (mean activation per category in each mask):")
print(f"{'Category':12s}", end="")
for name in new_masks:
    print(f"  {name[:12]:>12}", end="")
print()
print("-" * 100)

for cat in CATEGORIES:
    preds_paths = sorted((STUDY_ROOT / cat).glob("*/preds.npy"))
    if not preds_paths:
        continue
    cat_mean = np.stack([np.load(p)[:MAX_TRS].mean(axis=0)
                         for p in preds_paths]).mean(axis=0)
    print(f"  {cat:12s}", end="")
    for name, mask in new_masks.items():
        val = float(cat_mean[mask].mean()) if mask.sum() > 0 else 0
        print(f"  {val:12.4f}", end="")
    print()

# ── Selectivity score (target activation - mean of all other categories):
print("\nSelectivity score (target activation - mean of all other categories):")
for target_cat, mask_name in [("porn", "porn_specific"), ("gore", "gore_specific"),
                               ("porn", "porn_tight"),    ("gore", "gore_tight"),
                               ("porn", "porn_allneutral"),("gore","gore_allneutral"),
                               ("porn", "porn_no_food"),   ("gore", "gore_no_food"),
                               ("gore", "gore_strict_bicontrast"),
                               ("gore", "gore_strict_multivariate_all")]:
    mask = new_masks[mask_name]
    if mask.sum() == 0:
        continue
    target_val  = float(means[target_cat][mask].mean())
    other_cats  = [c for c in CATEGORIES if c != target_cat]
    other_val   = float(np.stack([means[c][mask] for c in other_cats]).mean())
    score       = target_val - other_val
    print(f"  {target_cat:6s} in {mask_name:20s}: target={target_val:.4f}  others={other_val:.4f}  score={score:+.4f}")

# ── Food selectivity sanity check ─────────────────────────────────────────
for name in ["gore_no_food", "gore_strict_bicontrast", "gore_strict_multivariate_all"]:
    if name in new_masks:
        gore_mask_to_check = new_masks[name]
        gore_act = float(means["gore"][gore_mask_to_check].mean())
        food_act = float(means["food"][gore_mask_to_check].mean())
        print(f"\nFood selectivity sanity check for {name}:")
        print(f"  Gore activation in Gore mask: {gore_act:.4f}")
        print(f"  Food activation in Gore mask: {food_act:.4f}")
        if food_act >= gore_act:
            print(f"  WARNING: Food activation is HIGHER than or equal to Gore activation in the Gore mask!")
        else:
            print(f"  ✓ Sanity check passed: Food activation is lower than Gore activation in the Gore mask.")

# ── Save best masks ────────────────────────────────────────────────────────
MASK_DIR.mkdir(exist_ok=True)
for name, mask in new_masks.items():
    np.save(MASK_DIR / f"{name}_strict.npy", mask)
print(f"\nStrict masks saved → {MASK_DIR}")

# ── Auto-select best masks and emit config ─────────────────────────────────

import json

def score_mask(target_cat, mask, means, categories):
    """
    Composite score:
      1. selectivity = target_mean - mean_of_others  (higher is better)
      2. food_penalty = max(0, food_mean - target_mean)  (zero is ideal)
      3. n_verts >= 100 sanity gate
    Returns None if mask fails the gate.
    """
    if mask.sum() < 100:
        return None
    target_val = float(means[target_cat][mask].mean())
    other_cats = [c for c in categories if c != target_cat]
    other_val  = float(np.stack([means[c][mask] for c in other_cats]).mean())
    food_val   = float(means["food"][mask].mean())
    selectivity   = target_val - other_val
    food_penalty  = max(0.0, food_val - target_val)
    return selectivity - 2.0 * food_penalty   # food leak weighted 2x

MASK_CANDIDATES = {
    "porn": ["porn_specific", "porn_tight", "porn_allneutral", "porn_no_food"],
    "gore": ["gore_specific", "gore_tight", "gore_allneutral", "gore_no_food",
             "gore_strict_bicontrast", "gore_strict_multivariate_all"],
}

auto_selected = {}
for target_cat, candidates in MASK_CANDIDATES.items():
    best_name, best_score = None, -np.inf
    for name in candidates:
        mask = new_masks.get(name)
        if mask is None:
            continue
        s = score_mask(target_cat, mask, means, CATEGORIES)
        if s is not None and s > best_score:
            best_score, best_name = s, name
    if best_name:
        auto_selected[target_cat] = {
            "mask_file": str(MASK_DIR / f"{best_name}_strict.npy"),
            "mask_name": best_name,
            "score":     round(float(best_score), 6),
            "n_verts":   int(new_masks[best_name].sum()),
        }
        print(f"\nAUTO-SELECTED [{target_cat}]: {best_name}  score={best_score:.4f}  n={new_masks[best_name].sum()}")

config_path = MASK_DIR / "abliteration_config.json"
with open(config_path, "w") as f:
    json.dump(auto_selected, f, indent=2)
print(f"\nConfig written → {config_path}")