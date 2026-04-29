# SEM Grain Segmentation

Automated grain boundary detection and size analysis for SEM images, using a trained Random Forest pixel classifier. Includes a Streamlit web app for interactive use and Jupyter notebooks for training and exploratory analysis.

---

## Quick Start — Streamlit App

The app lets you upload any SEM `.tif`, tune all pipeline parameters interactively, inspect diagnostic intermediate images, and download a PDF report, CSV data, and a JSON parameter file for reproducibility.

```bash
# 1. Create and activate the virtual environment
python -m venv .sem_seg
source .sem_seg/bin/activate        # macOS / Linux
# .sem_seg\Scripts\activate         # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

Then open the URL printed in the terminal (usually `http://localhost:8501`).

> **Note:** `best_pixel_classifier.joblib` must be present in the project directory (it is excluded from git due to file size — obtain it separately or train your own; see below).

---

## Setup

### Option A — pip + venv (recommended for the app)

```bash
python -m venv .sem_seg
source .sem_seg/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

### Option B — Conda (recommended for the notebooks)

```bash
conda env create -f environment.yml
conda activate mini-proj
jupyter notebook
```

Key dependencies: `scikit-image`, `scikit-learn`, `scipy`, `opencv-python`, `joblib`, `pandas`, `numpy`, `matplotlib`, `streamlit`, `Pillow`.

---

## Streamlit App (`app.py`)

### Features

- **Upload** any SEM `.tif` image (RGB or grayscale)
- **Tune parameters** in the sidebar with tooltips explaining each knob
- **Cached RF prediction** — changing only postprocessing params reruns the fast steps without re-running the classifier
- **Diagnostic mode** — toggle to reveal 8 intermediate images showing the effect of each parameter group at every pipeline stage
- **Grain statistics** — grain count, mean/std/median diameter, summary table, diameter and area histograms
- **Downloads:**
  - `*_grain_report.pdf` — 4-page report: overlay image, size distributions, statistics table, full parameter list
  - `*_grain_analysis.csv` — per-grain measurements (area, diameter, perimeter, aspect ratio, eccentricity)
  - `*_params.json` — all parameters + results summary for exact reproducibility

### Sidebar parameters

| Section | Parameter | Effect of increasing |
|---------|-----------|----------------------|
| — | Scale factor (px/nm) | Adjusts all reported sizes; calibrate from ImageJ |
| Preprocessing | Top-hat radius | Removes broader background gradients; must exceed grain diameter |
| Preprocessing | Contrast clip % | Clips intensity outliers; helps cross-session normalization (try 1–2) |
| Preprocessing | Gaussian sigma | More smoothing, fewer spurious boundaries; may blur fine boundaries |
| Cleaning | Opening radius | Removes more speckle noise; may break thin boundaries |
| Cleaning | Line length | Bridges longer boundary gaps |
| Watershed | Min seed distance | Fewer splits; merges touching grains |
| Watershed | Erosion passes | Wider boundary lines, cleaner grain separation |
| Filtering | Min object size | Discards smaller grain fragments |
| Filtering | Max hole size | Fills larger internal voids |

### Command-line usage

```bash
python inference_pipeline.py image.tif outputs/ \
    --scale-factor 0.603 \
    --tophat-radius 50 \
    --gaussian-sigma 2 \
    --watershed-min-distance 40 \
    --save-intermediates
```

Run `python inference_pipeline.py --help` for all options.

---

## Notebook Pipeline

Three notebooks form the original training-and-analysis workflow.

### Stage 1 — Train (`preprocess_train.ipynb`)

1. Load `train_image.tif` → convert to grayscale
2. White top-hat (disk radius 50) to remove background brightness gradient
3. CLAHE contrast enhancement + Gaussian denoise
4. Extract multiscale features (`sigma_min=1`, `sigma_max=10`, intensity + texture + edges)
5. Load `train_labels.tif` (white=255 → boundary label 1, black=0 → grain label 0)
6. Train `RandomForestClassifier` with `GridSearchCV` (balanced class weights, 3-fold CV)
7. Save best model → `best_pixel_classifier.joblib`

**Creating training labels in ImageJ/Fiji:**
Draw grain boundaries with white (value 255). Adjust contrast and threshold so the grain interior becomes fully black (value 0). This binary image is your `train_labels.tif`. The training script converts it to 0/1 labels automatically.

### Stage 2 — Predict (`predict.ipynb`)

1. Set `test_image_path` at the top of the notebook
2. Apply identical preprocessing as training
3. Load `best_pixel_classifier.joblib` and predict pixel labels
4. Save binary prediction → `predict_GBs.tif` (0 = grain interior, 1 = boundary; appears nearly black when viewed directly — this is expected)

### Stage 3 — Postprocess & Analyse (`postprocess_analysis.ipynb`)

1. Load `train_image.tif` and `predict_GBs.tif` (update paths at the top)
2. Morphological opening to remove salt noise
3. Optional: connect dotted boundaries
4. Watershed grain separation
5. Fill internal holes, remove border-touching grains, label
6. Run `analyze_grains(scale_factor=0.603)`
7. Save: `segmentation_result_with_scalebar.tif`, `grain_analysis.csv`

---

## Repository Structure

```
├── app.py                        # Streamlit web app
├── inference_pipeline.py         # Reusable pipeline (importable + CLI)
├── BSSEM_utils.py                # Shared utility functions
├── preprocess_train.ipynb        # Stage 1: training
├── predict.ipynb                 # Stage 2: prediction
├── postprocess_analysis.ipynb    # Stage 3: postprocessing & analysis
├── requirements.txt              # pip dependencies
├── environment.yml               # Conda environment
└── CLAUDE.md                     # AI assistant guidance
```

> `best_pixel_classifier.joblib` and all `.tif` images are excluded from git (file size). Store them separately or use Git LFS.

---

## Key Constants

- `SCALE_FACTOR = 0.603` px/nm — calibrated from ImageJ; update for different SEM sessions
- Grain boundary convention: 0 = grain interior, 1 = boundary
- `sigma_min=1`, `sigma_max=10` for multiscale features — fixed; changing these requires retraining
