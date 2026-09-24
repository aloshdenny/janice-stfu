# TRIBE v2 Concept Abliteration Pipeline

A fully data-driven pipeline for surgeon-like concept suppression or tolerance inside the TRIBE v2 brain model, featuring Leave-One-Subject-Out (LOSO) contrast designs, multi-layer weight surgery, and automated checkpoint chunking.

---

## 🚀 Setup & Installation

Follow these steps to set up the project locally using your existing `tribev2` Conda environment:

```bash
# 1. Clone the repository
git clone https://github.com/aloshdenny/janice-stfu.git
cd janice-stfu

# 2. Activate your tribev2 conda environment
conda activate tribev2

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 📂 Data Layout

The pipeline automatically discovers categories from subfolders under `data/`. Categories are not hardcoded. Adding a new directory automatically integrates it into the pipeline.

```
data/
  ├── argument/     # ~48 videos, 30s each, 256x256
  ├── butcher/
  ├── chase/
  ├── cute/
  ├── dance/
  ├── fight/
  ├── food/
  ├── gore/
  ├── kissing/
  ├── nature/
  ├── porn/         # Target or baseline
  ├── sports/
  └── surgery/
```

- **Current categories (13)**: `argument`, `butcher`, `chase`, `cute`, `dance`, `fight`, `food`, `gore`, `kissing`, `nature`, `porn`, `sports`, `surgery`.
- **Target Specification**: The target concept is specified at runtime (e.g. `--target porn`). All other discovered categories automatically serve as baseline controls.

---

## ⚙️ Abliteration Pipeline

The dependency chain is fully linear and runs via four sequential scripts:

```mermaid
flowchart TD
    A["1. Bulk Inference - infer_bulk.py"] --> B["2. Layer Analysis - layer_analysis.py"]
    B --> C["3. Abliteration - abliteration.py"]
    C --> D["4. Validation - validation.py"]
    D --> E["5. Generate Brain Scans - val_results"]
```

### 1. Bulk Inference
Precomputes and saves model predictions (`preds.npy`) for all videos. Already-processed videos are skipped.
```bash
python scripts/infer_bulk.py
```

### 2. Layer Analysis
Analyzes representations layer-by-layer to generate masks and identify which transformer layers are most selective for the target concept. Output is saved to `tribe_study/masks/layer_profiles.npz` along with diagnostic plots in `analysis/`.
```bash
python scripts/layer_analysis.py
```

### 3. Concept Abliteration (Surgery)
Extracts PCA concept directions and applies surgical suppression/tolerance to the selected encoder layers. 
```bash
# Run surgery with 5 operated layers and 3 components per layer
python scripts/abliteration.py --target porn --tolerance -1.0 --n_components 3 --n_layers 5
```

### 4. Validation
Loads the abliterated model, clears inference caches, compares predictions against baseline outputs, and saves side-by-side brain scan visualization PNGs to `val_results/`.
```bash
python scripts/validation.py --target porn
```

---

## 💾 Checkpoint Chunking & Versioning

Because the abliterated PyTorch checkpoints (`.pt`) are extremely large (~4.14 GB each) and GitHub enforces a strict **100MB file limit** and a **2GB push size limit**:
- **Automatic Chunking**: Checkpoints are automatically sliced into binary chunks of `<25MB` (default is `23 * 1024 * 1024` bytes / 24.12 MB) when saving in `abliteration.py`. They are saved as `vjepa2_abliterated_chunk_000.pt`, `vjepa2_abliterated_chunk_001.pt`, etc.
- **On-Disk Re-fusing**: When loaded in `validation.py`, these chunks are merged into a temporary file on disk, read via `torch.load()`, and immediately deleted to ensure memory safety on 16GB RAM machines.
- **Batch Pushed**: A helper utility [batch_push.py](file:///Users/aoxo/vscode/janice-stfu/scripts/batch_push.py) is used to commit and push changes to GitHub in safe `<1.2GB` batches.

---

## 🧠 Methodology & Concepts

### 1. Contrast Design — Leave-One-Subject-Out (LOSO)
Rather than comparing against static arbitrary reference frames, every category is contrasted against all other categories in a fully data-driven configuration:
- **LOSO contrast** = `category − mean(all other categories)`
- **LOSO-k contrast** = `category − mean(all others except category k)` (isolates confounding signals)
- **Multivariate Mask** = Vertices where the target category activation is strictly greater than every other category.

### 2. Selectivity Mask Scoring
Selectivity masks are isolated using margin-based selection:
$$\text{Score} = \text{Min Margin} \times \text{Selectivity}$$
$$\text{where } \text{Min Margin} = \text{Target Activation} - \text{Max Competitor Activation}$$
This guarantees that vertices where the target activation only marginally exceeds competitor activation score near zero, while vertices with clear target dominance score high.

### 3. Multi-Layer Surgery
`--n_layers N` selects the top-N layers from `layer_profiles.npz` based on concept correlation. Activations are collected per-layer, PCA directions are calculated independently per-layer, and surgery is applied independently using layer-specific directions.
- Directory of directions: `abliterated/{target}_directions_L{layer_idx}.npy`
- Surgery log: `abliterated/surgery_log.json` (lists operated layers; parsed automatically by the validation loop)

### 4. Tolerance Scale
Continuous scale mapping surgeon intervention outcomes:
```
  -1       -0.5       0       +0.5       +1
   |--------|---------|---------|--------|
   Repulsion Aversion  Neutral  Acceptance Attraction
```
- **Tolerance < 0**: Suppress / repel the concept.
- **Tolerance = 0**: Neutral (no effect).
- **Tolerance > 0**: Accept / tolerate the concept.
> **Note**: Positive tolerance represents the absence of aversion (no pushback), not a preference for the concept.

---

## 🛠️ Troubleshooting

### Deadlock on Cache Access
If multi-process file locks deadlock, remove the inflight database in your cache directory:
```bash
find ./cache -name "inflight.db" -delete
```