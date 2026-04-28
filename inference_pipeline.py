"""
SEM grain boundary segmentation inference pipeline.

Run from the command line:
    python inference_pipeline.py <image_path> <output_dir> [options]

Or import and call programmatically:
    from inference_pipeline import run_pipeline
    results = run_pipeline("image.tif", "outputs/", scale_factor=0.603)
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, exposure, filters, segmentation
from skimage.feature import multiscale_basic_features
from skimage.io import imread, imsave
from skimage.measure import label
from skimage.morphology import disk, white_tophat

import BSSEM_utils

# --------------- DEFAULT PARAMETERS ---------------
DEFAULTS = dict(
    model_path="best_pixel_classifier.joblib",
    # preprocessing
    tophat_radius=50,           # disk radius for background removal (px)
    gaussian_sigma=2,           # Gaussian denoise strength
    # feature extraction — must match training settings
    sigma_min=1,
    sigma_max=10,
    # postprocessing
    opening_radius=3,           # morphological opening to remove salt noise
    connect_dots=True,          # bridge gaps in dotted grain boundaries
    line_length=5,              # line element length for gap bridging (px)
    min_object_size=50,         # minimum object size to keep after cleaning (px²)
    watershed_min_distance=40,  # minimum distance between watershed seed points (px)
    watershed_erosions=2,       # erosion passes after watershed separation
    max_hole_size=2000,         # maximum hole to fill inside a grain (px²)
    # calibration and output
    scale_factor=0.603,         # px/nm, calibrated from ImageJ
    scale_bar_length_nm=500,    # physical length shown on the scale bar (nm)
    # diagnostics
    save_intermediates=False,   # save preprocessed/raw-boundary/postprocessed TIFs to output_dir
)


def load_image(image_path):
    """Load a .tif SEM image and convert to grayscale if the file is RGB."""
    image = imread(str(image_path))
    if len(image.shape) > 2:
        image = color.rgb2gray(image)
    return image


def preprocess(image, tophat_radius=50, gaussian_sigma=2):
    """Remove background brightness gradient, rescale contrast, and denoise."""
    image = white_tophat(image, disk(tophat_radius))
    image = exposure.rescale_intensity(image)
    image = filters.gaussian(image, sigma=gaussian_sigma)
    return image


def predict_boundaries(image_processed, model_path, sigma_min=1, sigma_max=10):
    """Extract multiscale features and classify each pixel as grain (0) or boundary (1)."""
    feats = multiscale_basic_features(
        image=image_processed,
        intensity=True,
        texture=True,
        edges=True,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        channel_axis=None,
    )
    classifier = joblib.load(str(model_path))
    predicted = classifier.predict(feats.reshape(-1, feats.shape[2]))
    return predicted.reshape(image_processed.shape).astype(np.uint8)


def postprocess(boundary_mask, opening_radius=3, connect_dots=True, line_length=5,
                min_object_size=50, watershed_min_distance=40, watershed_erosions=2,
                max_hole_size=2000):
    """
    Convert a raw boundary prediction into a labeled grain image.

    Steps: salt removal → optional gap bridging → invert → watershed separation
    → hole fill → border-touching grain removal → integer labeling.
    """
    cleaned = BSSEM_utils.apply_opening(boundary_mask, opening_radius)
    if connect_dots:
        cleaned = BSSEM_utils.connect_boundary_dots(
            cleaned, line_length=line_length, min_size=min_object_size
        )
    grain_mask = 1 - cleaned
    separated, _, _ = BSSEM_utils.separate_touching(
        grain_mask, min_distance=watershed_min_distance, num_erosions=watershed_erosions
    )
    filled = BSSEM_utils.fill_grain_holes(separated, max_hole_size=max_hole_size)
    return segmentation.clear_border(label(filled))


def analyze_and_save(image_raw, final_labels, output_dir, scale_factor=0.603,
                     scale_bar_length_nm=500):
    """
    Compute grain statistics and write all outputs to output_dir.

    Saves: grain_analysis.csv, segmentation_result_with_scalebar.tif,
    grain_size_histogram.png.

    Returns:
        (grain_stats_df, overlay_image)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grain_data = BSSEM_utils.analyze_grains(final_labels, scale_factor=scale_factor)

    # Guard against divide-by-zero on degenerate grains with zero minor axis
    minor = grain_data['minor_axis_lengths']
    major = grain_data['major_axis_lengths']
    grain_data['aspect_ratios'] = np.where(
        minor > 0, major / np.maximum(minor, 1e-10), np.nan
    )

    stats_df = pd.DataFrame({
        'Area_nm2': grain_data['areas'],
        'Diameter_nm': grain_data['diameters'],
        'Perimeter_nm': grain_data['perimeters'],
        'MajorAxis_nm': major,
        'MinorAxis_nm': minor,
        'AspectRatio': grain_data['aspect_ratios'],
        'Eccentricity': grain_data['eccentricities'],
    })
    stats_df.to_csv(output_dir / 'grain_analysis.csv', index=False)

    overlay = _make_overlay(image_raw, final_labels)
    overlay_with_bar = BSSEM_utils.add_simple_scale_bar(
        overlay, scale_factor=scale_factor, bar_length_nm=scale_bar_length_nm
    )
    imsave(str(output_dir / 'segmentation_result_with_scalebar.tif'), overlay_with_bar)

    _save_grain_histogram(grain_data, output_dir / 'grain_size_histogram.png')

    return stats_df, overlay_with_bar


def _make_overlay(image_raw, labeled_image, alpha=0.3):
    """Render labeled grain regions as a colored overlay on the raw image, returning uint8 RGB."""
    rgb = np.stack([image_raw] * 3, axis=-1).astype(float)
    rgb /= rgb.max()
    overlay = rgb.copy()

    rng = np.random.RandomState(0)
    n_labels = int(labeled_image.max()) + 1
    colors = np.zeros((n_labels, 3))
    if n_labels > 1:
        colors[1:] = rng.rand(n_labels - 1, 3)

    for i in range(1, n_labels):
        mask = labeled_image == i
        overlay[mask] = overlay[mask] * (1 - alpha) + colors[i] * alpha

    return (overlay * 255).astype(np.uint8)


def _save_grain_histogram(grain_data, save_path):
    """Save a 4-panel grain size and area distribution figure to save_path."""
    fig, axs = plt.subplots(1, 4, figsize=(16, 5))

    axs[0].hist(grain_data['diameters'], bins=30, edgecolor='black', color='skyblue')
    axs[0].set_xlabel('Grain Diameter (nm)')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title('Grain Size Distribution')
    axs[0].grid(True, linestyle='--', alpha=0.7)

    bp0 = axs[1].boxplot(grain_data['diameters'], vert=True, patch_artist=True)
    for patch in bp0['boxes']:
        patch.set_facecolor('skyblue')
    axs[1].set_ylabel('Grain Diameter (nm)')
    axs[1].set_title('Grain Size Quartiles')
    axs[1].set_xticks([])
    axs[1].grid(True, linestyle='--', alpha=0.7)

    axs[2].hist(grain_data['areas'], bins=30, edgecolor='black', color='lightgreen')
    axs[2].set_xlabel('Grain Area (nm²)')
    axs[2].set_ylabel('Frequency')
    axs[2].set_title('Grain Area Distribution')
    axs[2].grid(True, linestyle='--', alpha=0.7)

    bp1 = axs[3].boxplot(grain_data['areas'], vert=True, patch_artist=True)
    for patch in bp1['boxes']:
        patch.set_facecolor('lightgreen')
    axs[3].set_ylabel('Grain Area (nm²)')
    axs[3].set_title('Grain Area Quartiles')
    axs[3].set_xticks([])
    axs[3].grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    fig.savefig(str(save_path), dpi=150)
    plt.close(fig)


def run_pipeline(image_path, output_dir, **params):
    """
    Run the full SEM grain segmentation pipeline end-to-end.

    Args:
        image_path: Path to the input SEM image (.tif).
        output_dir: Directory where all outputs are written.
        **params: Override any value from DEFAULTS (e.g. scale_factor=0.75,
                  model_path='other.joblib', connect_dots=False).

    Returns:
        dict with keys:
            grain_stats_df  -- DataFrame of per-grain measurements
            overlay_image   -- uint8 RGB ndarray with colored overlay and scale bar
            boundary_mask   -- uint8 ndarray of raw classifier output (0=grain, 1=boundary)
    """
    p = {**DEFAULTS, **params}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_raw = load_image(image_path)
    image_processed = preprocess(
        image_raw, tophat_radius=p['tophat_radius'], gaussian_sigma=p['gaussian_sigma']
    )
    if p['save_intermediates']:
        imsave(str(output_dir / 'intermediate_preprocessed.tif'), image_processed.astype(np.float32))

    boundary_mask = predict_boundaries(
        image_processed, p['model_path'],
        sigma_min=p['sigma_min'], sigma_max=p['sigma_max'],
    )
    if p['save_intermediates']:
        imsave(str(output_dir / 'intermediate_boundary_raw.tif'), (boundary_mask * 255).astype(np.uint8))

    final_labels = postprocess(
        boundary_mask,
        opening_radius=p['opening_radius'],
        connect_dots=p['connect_dots'],
        line_length=p['line_length'],
        min_object_size=p['min_object_size'],
        watershed_min_distance=p['watershed_min_distance'],
        watershed_erosions=p['watershed_erosions'],
        max_hole_size=p['max_hole_size'],
    )
    if p['save_intermediates']:
        n = max(int(final_labels.max()), 1)
        imsave(str(output_dir / 'intermediate_postprocessed.tif'),
               ((final_labels / n) * 255).astype(np.uint8))

    grain_stats_df, overlay_image = analyze_and_save(
        image_raw, final_labels, output_dir,
        scale_factor=p['scale_factor'],
        scale_bar_length_nm=p['scale_bar_length_nm'],
    )

    imsave(str(output_dir / 'predict_GBs.tif'), boundary_mask)

    return dict(
        grain_stats_df=grain_stats_df,
        overlay_image=overlay_image,
        boundary_mask=boundary_mask,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SEM grain segmentation inference pipeline')
    parser.add_argument('image_path', help='Path to input SEM image (.tif)')
    parser.add_argument('output_dir', help='Directory for output files')
    parser.add_argument('--model', dest='model_path', default=DEFAULTS['model_path'],
                        help='Path to trained .joblib classifier')
    parser.add_argument('--scale-factor', type=float, default=DEFAULTS['scale_factor'],
                        help='Pixel/nm calibration factor (default: 0.603)')
    parser.add_argument('--no-connect-dots', dest='connect_dots', action='store_false',
                        help='Skip the boundary gap-bridging step')
    args = parser.parse_args()

    results = run_pipeline(
        args.image_path, args.output_dir,
        model_path=args.model_path,
        scale_factor=args.scale_factor,
        connect_dots=args.connect_dots,
    )
    df = results['grain_stats_df']
    print(f"Done. {len(df)} grains detected.")
    print(f"Mean diameter:   {df['Diameter_nm'].mean():.1f} nm")
    print(f"Median diameter: {df['Diameter_nm'].median():.1f} nm")
