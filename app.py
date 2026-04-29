"""
Streamlit app for SEM grain segmentation.

Launch with:
    streamlit run app.py
"""

import hashlib
import io
import json
import shutil
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from skimage.morphology import disk, white_tophat

from inference_pipeline import (
    DEFAULTS,
    analyze_and_save,
    load_image,
    load_model,
    postprocess,
    predict_boundaries,
    preprocess,
)


# --------------- HELPERS ---------------

def _to_display(arr):
    """Normalize any array to uint8 for st.image (handles float, uint16, label images)."""
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    return (arr * 255).astype(np.uint8)


def _slow_cache_key(image_bytes, model_path, tophat_radius, contrast_clip_pct, gaussian_sigma):
    return hashlib.md5(
        image_bytes
        + model_path.encode()
        + f"{tophat_radius},{contrast_clip_pct},{gaussian_sigma}".encode()
    ).hexdigest()


def _cleanup_prev_temps():
    for key in ("_prev_tmp_img", "_prev_output_dir"):
        path = st.session_state.pop(key, None)
        if not path:
            continue
        try:
            p = Path(path)
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            elif p.exists():
                p.unlink(missing_ok=True)
        except Exception:
            pass


@st.cache_resource
def _load_model_cached(model_path):
    return load_model(model_path)


def _generate_params_json(params, source_name, grain_stats_df):
    """Serialise run parameters + results summary to JSON bytes."""
    data = {
        "source_image": source_name or "",
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": params.get("model_path", ""),
        "preprocessing": {
            "tophat_radius": params.get("tophat_radius"),
            "contrast_clip_pct": params.get("contrast_clip_pct"),
            "gaussian_sigma": params.get("gaussian_sigma"),
        },
        "cleaning": {
            "opening_radius": params.get("opening_radius"),
            "connect_dots": params.get("connect_dots"),
            "line_length": params.get("line_length"),
        },
        "watershed": {
            "watershed_min_distance": params.get("watershed_min_distance"),
            "watershed_erosions": params.get("watershed_erosions"),
        },
        "filtering": {
            "min_object_size": params.get("min_object_size"),
            "max_hole_size": params.get("max_hole_size"),
        },
        "calibration": {
            "scale_factor": params.get("scale_factor"),
            "scale_bar_length_nm": params.get("scale_bar_length_nm"),
        },
        "results_summary": {
            "grain_count": int(len(grain_stats_df)),
            "mean_diameter_nm": round(float(grain_stats_df["Diameter_nm"].mean()), 2),
            "std_diameter_nm": round(float(grain_stats_df["Diameter_nm"].std()), 2),
            "median_diameter_nm": round(float(grain_stats_df["Diameter_nm"].median()), 2),
            "mean_area_nm2": round(float(grain_stats_df["Area_nm2"].mean()), 2),
        },
    }
    return json.dumps(data, indent=2).encode()


def _generate_pdf(overlay_image, grain_stats_df, source_name="", params=None):
    """Build a 4-page PDF (overlay, distributions, stats table, parameters) and return bytes."""
    label = f" — {source_name}" if source_name else ""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: segmentation overlay
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(overlay_image)
        ax.set_title(f"Grain Segmentation Result{label}")
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: diameter + area distributions
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"Grain Size Distributions{label}", fontsize=13)
        axs[0, 0].hist(grain_stats_df["Diameter_nm"], bins=30, edgecolor="black", color="skyblue")
        axs[0, 0].set_xlabel("Grain Diameter (nm)")
        axs[0, 0].set_ylabel("Frequency")
        axs[0, 0].set_title("Diameter Distribution")
        axs[0, 0].grid(True, linestyle="--", alpha=0.7)

        bp = axs[0, 1].boxplot(grain_stats_df["Diameter_nm"], vert=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("skyblue")
        axs[0, 1].set_ylabel("Grain Diameter (nm)")
        axs[0, 1].set_title("Diameter Quartiles")
        axs[0, 1].set_xticks([])
        axs[0, 1].grid(True, linestyle="--", alpha=0.7)

        axs[1, 0].hist(grain_stats_df["Area_nm2"], bins=30, edgecolor="black", color="lightgreen")
        axs[1, 0].set_xlabel("Grain Area (nm²)")
        axs[1, 0].set_ylabel("Frequency")
        axs[1, 0].set_title("Area Distribution")
        axs[1, 0].grid(True, linestyle="--", alpha=0.7)

        bp2 = axs[1, 1].boxplot(grain_stats_df["Area_nm2"], vert=True, patch_artist=True)
        for patch in bp2["boxes"]:
            patch.set_facecolor("lightgreen")
        axs[1, 1].set_ylabel("Grain Area (nm²)")
        axs[1, 1].set_title("Area Quartiles")
        axs[1, 1].set_xticks([])
        axs[1, 1].grid(True, linestyle="--", alpha=0.7)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 3: summary statistics table
        summary = grain_stats_df.describe().round(2)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis("off")
        tbl = ax.table(
            cellText=summary.values,
            rowLabels=summary.index,
            colLabels=summary.columns,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(col=list(range(len(summary.columns))))
        ax.set_title(f"Grain Statistics Summary{label}", pad=20)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 4: run parameters (only when params dict is provided)
        if params:
            param_groups = [
                ("Metadata", [
                    ("Source image", source_name or "—"),
                    ("Analysis date", datetime.now().strftime("%Y-%m-%d %H:%M")),
                    ("Model", params.get("model_path", "—")),
                ]),
                ("Preprocessing", [
                    ("Top-hat radius (px)", params.get("tophat_radius")),
                    ("Contrast clip percentile (%)", params.get("contrast_clip_pct")),
                    ("Gaussian sigma", params.get("gaussian_sigma")),
                ]),
                ("Cleaning", [
                    ("Opening radius (px)", params.get("opening_radius")),
                    ("Connect boundary dots", params.get("connect_dots")),
                    ("Line length (px)", params.get("line_length")),
                ]),
                ("Watershed", [
                    ("Min seed distance (px)", params.get("watershed_min_distance")),
                    ("Erosion passes", params.get("watershed_erosions")),
                ]),
                ("Filtering", [
                    ("Min object size (px²)", params.get("min_object_size")),
                    ("Max hole size (px²)", params.get("max_hole_size")),
                ]),
                ("Calibration", [
                    ("Scale factor (px/nm)", params.get("scale_factor")),
                    ("Scale bar length (nm)", params.get("scale_bar_length_nm")),
                ]),
            ]

            cell_text, cell_colors = [], []
            HEADER_BG, ROW_BG = "#cce5ff", "#ffffff"
            for section, rows in param_groups:
                cell_text.append([f"  {section}", ""])
                cell_colors.append([HEADER_BG, HEADER_BG])
                for name, val in rows:
                    cell_text.append([f"    {name}", str(val)])
                    cell_colors.append([ROW_BG, ROW_BG])

            fig, ax = plt.subplots(figsize=(8, len(cell_text) * 0.35 + 1.5))
            ax.axis("off")
            ptbl = ax.table(
                cellText=cell_text,
                colLabels=["Parameter", "Value"],
                cellColours=cell_colors,
                loc="center",
                cellLoc="left",
            )
            ptbl.auto_set_font_size(False)
            ptbl.set_fontsize(9)
            ptbl.auto_set_column_width([0, 1])
            ax.set_title(f"Analysis Parameters{label}", pad=12, fontsize=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    buf.seek(0)
    return buf.read()


# --------------- PAGE CONFIG ---------------

st.set_page_config(page_title="SEM Grain Segmentation", layout="wide")
st.title("SEM Grain Segmentation")

# --------------- SIDEBAR ---------------

with st.sidebar:
    st.header("Input & Settings")

    uploaded_file = st.file_uploader("Upload SEM image (.tif)", type=["tif", "tiff"])
    diagnostic_mode = st.toggle("Diagnostic Mode", value=False)

    st.divider()

    model_path = st.text_input(
        "Model path", value=DEFAULTS["model_path"],
        help="Path to the trained Random Forest classifier (.joblib). Changing this triggers a full re-run."
    )
    scale_factor = st.number_input(
        "Scale factor (px/nm)", value=DEFAULTS["scale_factor"], step=0.001, format="%.3f",
        help="Pixels per nanometer — calibrate from a known scale bar in ImageJ. All reported grain sizes scale with this value."
    )
    scale_bar_length_nm = st.number_input(
        "Scale bar length (nm)", value=DEFAULTS["scale_bar_length_nm"], step=10, min_value=1,
        help="Physical length of the white scale bar drawn on the overlay image."
    )

    with st.expander("Preprocessing  ← slow to change"):
        tophat_radius = st.number_input(
            "Top-hat radius (px)", value=DEFAULTS["tophat_radius"], step=1, min_value=1,
            help=(
                "Disk radius for the white top-hat transform that removes broad background "
                "brightness gradients (e.g. SEM stage drift). Should be larger than the typical "
                "grain diameter. Larger → removes more slowly-varying background; too large may "
                "suppress genuine large grains."
            ),
        )
        contrast_clip_pct = st.number_input(
            "Contrast clip percentile (%)", value=DEFAULTS["contrast_clip_pct"],
            step=0.5, min_value=0.0, max_value=10.0, format="%.1f",
            help=(
                "Clips this percentage of pixels from each end of the intensity histogram before "
                "rescaling to [0, 1]. Corrects images where a few very bright/dark outlier pixels "
                "(beam artefacts, contamination) compress the useful contrast range. "
                "0 = rescale by absolute min/max (default, matches training). "
                "Try 1–2 when using images from a different SEM session than the training data."
            ),
        )
        gaussian_sigma = st.number_input(
            "Gaussian sigma", value=float(DEFAULTS["gaussian_sigma"]), step=0.5, min_value=0.0,
            help=(
                "Standard deviation of the Gaussian blur applied to reduce high-frequency noise "
                "before feature extraction. Larger → smoother image, fewer spurious boundaries; "
                "too large → fine grain boundaries become blurred and may be missed."
            ),
        )

    with st.expander("Cleaning"):
        opening_radius = st.number_input(
            "Opening radius (px)", value=DEFAULTS["opening_radius"], step=1, min_value=1,
            help=(
                "Radius of the morphological opening (erosion → dilation) that removes isolated "
                "bright speckles from the raw boundary prediction. Larger → eliminates more noise "
                "spots; too large may erode and break thin continuous boundary lines."
            ),
        )
        connect_dots = st.checkbox(
            "Connect boundary dots", value=DEFAULTS["connect_dots"],
            help=(
                "Apply a multi-angle closing operation to bridge small gaps in broken or dotted "
                "grain boundaries. Enable when boundaries appear as dashes rather than solid lines."
            ),
        )
        line_length = st.number_input(
            "Line length (px)", value=DEFAULTS["line_length"], step=1, min_value=1,
            help=(
                "Length of the line structuring element used to bridge boundary gaps (only active "
                "when Connect boundary dots is on). Larger → bridges longer gaps; too large may "
                "incorrectly connect boundaries across adjacent grains."
            ),
        )

    with st.expander("Watershed"):
        watershed_min_distance = st.number_input(
            "Min seed distance (px)", value=DEFAULTS["watershed_min_distance"], step=1, min_value=1,
            help=(
                "Minimum pixel distance between seed points for the watershed grain-splitting "
                "algorithm. Larger → fewer seeds, so touching grains tend to merge into one "
                "region; smaller → more seeds, risking over-segmentation of large grains."
            ),
        )
        watershed_erosions = st.number_input(
            "Erosion passes", value=DEFAULTS["watershed_erosions"], step=1, min_value=0,
            help=(
                "Number of erosion passes applied after watershed to widen the dividing line "
                "between adjacent grains. Larger → cleaner separation but grains shrink slightly; "
                "set to 0 if grains appear too small after segmentation."
            ),
        )

    with st.expander("Filtering"):
        min_object_size = st.number_input(
            "Min object size (px²)", value=DEFAULTS["min_object_size"], step=10, min_value=0,
            help=(
                "Grain regions smaller than this area are discarded as noise or boundary "
                "fragments. Larger → removes more small artefacts; too large may discard "
                "genuine small grains near image edges."
            ),
        )
        max_hole_size = st.number_input(
            "Max hole size (px²)", value=DEFAULTS["max_hole_size"], step=100, min_value=0,
            help=(
                "Holes (dark voids) inside grains smaller than this area are filled. Larger → "
                "fills more internal voids for cleaner grain regions; too large may incorrectly "
                "fill the space between two nearly-touching grains."
            ),
        )

    st.caption(
        "sigma_min (1) and sigma_max (10) are fixed — changing them requires retraining.\n\n"
        "Preprocessing params (top-hat radius, gaussian sigma) and model path trigger "
        "a full re-run. All other params reuse the cached RF prediction."
    )

    run_button = st.button(
        "Run Segmentation", type="primary", disabled=(uploaded_file is None)
    )

# --------------- RUN PIPELINE ---------------

if run_button and uploaded_file is not None:
    _cleanup_prev_temps()

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_img:
        tmp_img.write(uploaded_file.getbuffer())
        tmp_img_path = tmp_img.name

    output_dir = tempfile.mkdtemp()
    st.session_state["_prev_tmp_img"] = tmp_img_path
    st.session_state["_prev_output_dir"] = output_dir

    slow_key = _slow_cache_key(
        uploaded_file.getvalue(), model_path,
        int(tophat_radius), float(contrast_clip_pct), float(gaussian_sigma)
    )

    progress = st.progress(0, text="Starting...")

    try:
        # ---- SLOW PART: load → preprocess → predict (cached by slow_key) ----
        if st.session_state.get("_slow_key") != slow_key:
            progress.progress(5, text="Loading image...")
            image_raw = load_image(tmp_img_path)

            progress.progress(15, text="Applying top-hat transform...")
            image_tophat = white_tophat(image_raw, disk(int(tophat_radius)))

            progress.progress(25, text="Preprocessing (contrast normalisation + Gaussian denoise)...")
            image_processed = preprocess(
                image_raw,
                tophat_radius=int(tophat_radius),
                contrast_clip_pct=float(contrast_clip_pct),
                gaussian_sigma=float(gaussian_sigma),
            )

            progress.progress(30, text="Loading model...")
            model = _load_model_cached(model_path)

            progress.progress(35, text="Running RF classifier — this may take 1–2 min...")
            boundary_mask = predict_boundaries(image_processed, model)

            st.session_state.update({
                "_slow_key": slow_key,
                "_image_raw": image_raw,
                "_image_tophat": image_tophat,
                "_image_processed": image_processed,
                "_boundary_mask": boundary_mask,
            })
        else:
            progress.progress(35, text="Using cached prediction (preprocessing unchanged)...")
            image_raw = st.session_state["_image_raw"]
            image_tophat = st.session_state["_image_tophat"]
            image_processed = st.session_state["_image_processed"]
            boundary_mask = st.session_state["_boundary_mask"]

        # ---- FAST PART: postprocess → analyze (always re-runs) ----
        progress.progress(60, text="Postprocessing boundaries...")
        intermediates = {}
        final_labels = postprocess(
            boundary_mask,
            opening_radius=int(opening_radius),
            connect_dots=connect_dots,
            line_length=int(line_length),
            min_object_size=int(min_object_size),
            watershed_min_distance=int(watershed_min_distance),
            watershed_erosions=int(watershed_erosions),
            max_hole_size=int(max_hole_size),
            _intermediates=intermediates,
        )

        progress.progress(85, text="Analyzing grains and building overlay...")
        grain_stats_df, overlay_image = analyze_and_save(
            image_raw, final_labels, output_dir,
            scale_factor=float(scale_factor),
            scale_bar_length_nm=int(scale_bar_length_nm),
        )

        progress.progress(100, text="Done!")

        run_params = dict(
            model_path=model_path,
            tophat_radius=int(tophat_radius),
            contrast_clip_pct=float(contrast_clip_pct),
            gaussian_sigma=float(gaussian_sigma),
            opening_radius=int(opening_radius),
            connect_dots=bool(connect_dots),
            line_length=int(line_length),
            watershed_min_distance=int(watershed_min_distance),
            watershed_erosions=int(watershed_erosions),
            min_object_size=int(min_object_size),
            max_hole_size=int(max_hole_size),
            scale_factor=float(scale_factor),
            scale_bar_length_nm=int(scale_bar_length_nm),
        )
        st.session_state["results"] = dict(
            grain_stats_df=grain_stats_df,
            overlay_image=overlay_image,
            boundary_mask=boundary_mask,
            image_raw=image_raw,
            source_name=uploaded_file.name,
            run_params=run_params,
        )
        st.session_state["diag_data"] = dict(
            original=image_raw,
            tophat=image_tophat,
            preprocessed=image_processed,
            boundary_raw=boundary_mask,
            after_opening=intermediates.get("after_opening"),
            after_connect_dots=intermediates.get("after_connect_dots"),
            after_watershed=intermediates.get("after_watershed"),
            final_grain_mask=(final_labels > 0).astype(np.uint8),
        )
        st.session_state["diagnostic_mode"] = diagnostic_mode

    except Exception:
        st.error(traceback.format_exc())
    finally:
        progress.empty()

# --------------- DISPLAY RESULTS ---------------

if "results" in st.session_state:
    results = st.session_state["results"]
    grain_stats_df = results["grain_stats_df"]
    overlay_image = results["overlay_image"]
    boundary_mask = results["boundary_mask"]
    image_raw = results["image_raw"]
    source_name = results.get("source_name", "")
    stem = Path(source_name).stem if source_name else "result"
    run_params = results.get("run_params", {})
    diag_mode = st.session_state.get("diagnostic_mode", False)

    # ---- Diagnostic: 2 rows × 4 columns ----
    if diag_mode and "diag_data" in st.session_state:
        d = st.session_state["diag_data"]

        st.subheader("Diagnostic — Preprocessing")
        p_cols = st.columns(4)
        with p_cols[0]:
            st.image(_to_display(d["original"]),
                     caption="1. Raw input — unprocessed SEM image as loaded.",
                     use_container_width=True)
        with p_cols[1]:
            st.image(_to_display(d["tophat"]),
                     caption=(
                         "2. After top-hat (tophat_radius) — slow-varying background gradient "
                         "removed. Grain interiors should appear uniformly grey. "
                         "Increase radius if bright/dark patches remain."
                     ),
                     use_container_width=True)
        with p_cols[2]:
            st.image(_to_display(d["preprocessed"]),
                     caption=(
                         "3. After contrast normalisation + denoise (contrast_clip_pct, "
                         "gaussian_sigma) — intensity rescaled and noise smoothed. Grain "
                         "boundaries should appear as dark lines on a bright background. "
                         "Increase clip percentile if overall contrast looks poor."
                     ),
                     use_container_width=True)
        with p_cols[3]:
            st.image(_to_display(d["boundary_raw"]),
                     caption=(
                         "4. RF boundary prediction (model) — white pixels are predicted grain "
                         "boundaries. Look for continuous outlines around each grain. Broken or "
                         "dotted outlines are handled in postprocessing."
                     ),
                     use_container_width=True)

        st.subheader("Diagnostic — Postprocessing")
        pp_cols = st.columns(4)
        with pp_cols[0]:
            if d["after_opening"] is not None:
                st.image(_to_display(d["after_opening"]),
                         caption=(
                             "5. After opening (opening_radius) — isolated speckles and "
                             "single-pixel noise removed. Only boundary segments wider than "
                             "the opening radius survive. Increase if noise spots remain."
                         ),
                         use_container_width=True)
        with pp_cols[1]:
            if d["after_connect_dots"] is not None:
                st.image(_to_display(d["after_connect_dots"]),
                         caption=(
                             "6. After gap bridging (connect_dots, line_length) — short gaps in "
                             "broken boundaries are closed. Grain outlines should now be fully "
                             "enclosed. Increase line_length if gaps are still visible."
                         ),
                         use_container_width=True)
        with pp_cols[2]:
            if d["after_watershed"] is not None:
                st.image(_to_display(d["after_watershed"]),
                         caption=(
                             "7. After watershed (min_distance, erosions) — touching grains "
                             "are split into separate regions. Decrease min_distance if large "
                             "grains are unsplit; increase if small grains are over-split."
                         ),
                         use_container_width=True)
        with pp_cols[3]:
            if d.get("final_grain_mask") is not None:
                st.image(_to_display(d["final_grain_mask"]),
                         caption=(
                             "8. Final grain regions (min_object_size, max_hole_size) — white = "
                             "accepted grain interior after hole filling and size filtering; "
                             "black = boundaries + discarded regions. Compare to panel 7 to see "
                             "holes filled and border grains removed."
                         ),
                         use_container_width=True)

        st.divider()

    # ---- Final results: 3-column image display ----
    st.subheader("Segmentation Results")
    r_cols = st.columns(3)
    with r_cols[0]:
        st.image(_to_display(image_raw), caption="Original Image", use_container_width=True)
    with r_cols[1]:
        st.image(_to_display(boundary_mask), caption="Boundary Mask", use_container_width=True)
    with r_cols[2]:
        st.image(overlay_image, caption="Segmentation Overlay", use_container_width=True)

    st.divider()

    # ---- Grain statistics ----
    st.subheader("Grain Statistics")

    m_cols = st.columns(5)
    with m_cols[0]:
        st.metric("Grains detected", len(grain_stats_df))
    with m_cols[1]:
        st.metric("Mean diameter (nm)", f"{grain_stats_df['Diameter_nm'].mean():.1f}")
    with m_cols[2]:
        st.metric("Std diameter (nm)", f"{grain_stats_df['Diameter_nm'].std():.1f}")
    with m_cols[3]:
        st.metric("Median diameter (nm)", f"{grain_stats_df['Diameter_nm'].median():.1f}")
    with m_cols[4]:
        st.metric("Mean area (nm²)", f"{grain_stats_df['Area_nm2'].mean():.0f}")

    # Summary statistics table (describe() for key columns)
    summary_cols = ["Diameter_nm", "Area_nm2", "AspectRatio", "Eccentricity"]
    st.table(grain_stats_df[summary_cols].describe().round(2))

    # Distribution charts: diameter (left) and area (right)
    c_left, c_right = st.columns(2)
    with c_left:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(grain_stats_df["Diameter_nm"], bins=30, edgecolor="black", color="skyblue")
        ax.set_xlabel("Grain Diameter (nm)")
        ax.set_ylabel("Frequency")
        ax.set_title("Diameter Distribution")
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with c_right:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(grain_stats_df["Area_nm2"], bins=30, edgecolor="black", color="lightgreen")
        ax.set_xlabel("Grain Area (nm²)")
        ax.set_ylabel("Frequency")
        ax.set_title("Area Distribution")
        ax.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Per-grain data hidden in expander
    with st.expander("Per-grain data table"):
        st.dataframe(grain_stats_df, use_container_width=True)

    st.divider()

    # ---- Downloads ----
    dl_cols = st.columns(3)
    with dl_cols[0]:
        pdf_bytes = _generate_pdf(
            overlay_image, grain_stats_df, source_name=source_name, params=run_params
        )
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"{stem}_grain_report.pdf",
            mime="application/pdf",
        )
    with dl_cols[1]:
        csv_bytes = grain_stats_df.to_csv(index=False).encode()
        st.download_button(
            label="Download CSV Data",
            data=csv_bytes,
            file_name=f"{stem}_grain_analysis.csv",
            mime="text/csv",
        )
    with dl_cols[2]:
        json_bytes = _generate_params_json(run_params, source_name, grain_stats_df)
        st.download_button(
            label="Download Parameters (JSON)",
            data=json_bytes,
            file_name=f"{stem}_params.json",
            mime="application/json",
        )
