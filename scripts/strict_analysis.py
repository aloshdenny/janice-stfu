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

CONTRASTS = {
    "porn_allneutral":  porn_contrast,
    "gore_allneutral":  gore_contrast,
    "porn_specific":    porn_specific,
    "gore_specific":    gore_specific,
    "porn_tight":       porn_tight,
    "gore_tight":       gore_tight,
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

# ── Selectivity check ──────────────────────────────────────────────────────
print("\nSelectivity check (mean activation per category in each mask):")
print(f"{'Category':12s}", end="")
for name in CONTRASTS:
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

# ── Pick best masks based on selectivity ──────────────────────────────────
# Best mask = highest (target - mean_of_others) in that mask
print("\nSelectivity score (target activation - mean of all other categories):")
for target_cat, mask_name in [("porn", "porn_specific"), ("gore", "gore_specific"),
                               ("porn", "porn_tight"),    ("gore", "gore_tight"),
                               ("porn", "porn_allneutral"),("gore","gore_allneutral")]:
    mask = new_masks[mask_name]
    if mask.sum() == 0:
        continue
    target_val  = float(means[target_cat][mask].mean())
    other_cats  = [c for c in CATEGORIES if c != target_cat]
    other_val   = float(np.stack([means[c][mask] for c in other_cats]).mean())
    score       = target_val - other_val
    print(f"  {target_cat:6s} in {mask_name:20s}: target={target_val:.4f}  others={other_val:.4f}  score={score:+.4f}")

# ── Save best masks ────────────────────────────────────────────────────────
MASK_DIR.mkdir(exist_ok=True)
for name, mask in new_masks.items():
    np.save(MASK_DIR / f"{name}_strict.npy", mask)
print(f"\nStrict masks saved → {MASK_DIR}")