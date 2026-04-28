# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

Conda (recommended):
```bash
conda env create -f environment.yml
conda activate mini-proj
jupyter notebook
```

Pip alternative:
```bash
python -m venv sem_seg && source sem_seg/bin/activate
pip install -r requirements.txt
```

Key dependencies: `scikit-image`, `scikit-learn`, `scipy`, `opencv-python` (`cv2`), `joblib`, `pandas`, `numpy`, `matplotlib`.

## Pipeline Architecture

The project is a three-stage Jupyter notebook pipeline for SEM grain boundary segmentation using a trained Random Forest pixel classifier.

### Stage 1 — Train (`preprocess_train.ipynb`)
1. Load `train_image.tif` → convert to grayscale
2. Preprocess: white top-hat (disk radius 50) to remove background brightness gradient, then CLAHE contrast enhancement + Gaussian denoise
3. Extract multiscale features via `skimage.feature.multiscale_basic_features` (intensity + texture + edges, `sigma_min=1`, `sigma_max=10`)
4. Load `train_labels.tif` (white=255 → grain boundary label 1, black=0 → grain label 0)
5. Train `RandomForestClassifier` with `GridSearchCV` (balanced class weights, 3-fold stratified CV)
6. Save best model to `best_pixel_classifier.joblib`

### Stage 2 — Predict (`predict.ipynb`)
1. Set `test_image_path` at the top of the notebook (the only required edit per new image)
2. Apply identical preprocessing as training (white top-hat → contrast rescale → Gaussian)
3. Extract the same multiscale features
4. Load `best_pixel_classifier.joblib` and predict pixel labels
5. Save binary prediction as `predict_GBs.tif` (0 = grain, 1 = boundary; appears nearly black)

### Stage 3 — Postprocess & Analyze (`postprocess_analysis.ipynb`)
1. Load `train_image.tif` and `predict_GBs.tif` (update paths at the top per new image)
2. Morphological cleanup: `apply_opening(radius=3)` to remove salt noise
3. Optional: `connect_boundary_dots()` to close gaps in dotted boundaries
4. Invert to get grain mask, then `separate_touching(min_distance=40, num_erosions=2)` via watershed
5. Fill holes: `fill_grain_holes(max_hole_size=2000)`
6. Label grains, clear border artifacts, run `analyze_grains(scale_factor=0.603)`
7. Outputs: overlay TIF with scale bar (`segmentation_result_with_scalebar.tif`), grain stats CSV

## Shared Utility Module

`BSSEM_utils.py` contains all reusable functions used by the notebooks:
- **Visualization**: `visualize_image`, `visualize_labeled`, `visualize_overlay`, `visualize_grain_statistics`, `add_simple_scale_bar`
- **Morphology**: `apply_erosion/dilation/opening/closing`, `connect_boundary_dots`, `fill_grain_holes`, `remove_small_objects`
- **Segmentation**: `separate_watershed`, `separate_touching` (watershed-based grain splitting)
- **Analysis**: `analyze_grains` (returns areas, diameters, perimeters, aspect ratios in nm), `save_grain_analysis` (CSV export)

## Key Constants

- `SCALE_FACTOR = 0.603` pixel/nm — used to convert pixel measurements to nanometers in `analyze_grains` and `add_simple_scale_bar`
- Grain boundary label convention: 0 = grain interior, 1 = boundary (in predictions and training labels)
- The `predict_GBs.tif` output is uint8 and will appear almost entirely black when viewed directly

## Known Image Quality Issues

- SEM stage drift can cause non-uniform background brightness across an image; the white top-hat transform (disk 50) corrects for this. If a new image has severe brightness gradients, tune the disk radius.
- If the classifier was trained on one SEM session and used on another with different brightness characteristics, predictions degrade — retrain or use `connect_boundary_dots` more aggressively in postprocessing.
