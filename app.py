"""
Streamlit app for SEM grain segmentation.

Launch with:
    streamlit run app.py
"""

import io
import tempfile
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from skimage.io import imread

from inference_pipeline import DEFAULTS, run_pipeline


# --------------- HELPERS ---------------

def _to_display(arr):
    """Normalize any array to uint8 for st.image (handles float, uint16, label images)."""
    arr = np.asarray(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    return (arr * 255).astype(np.uint8)


def _generate_pdf(overlay_image, grain_stats_df):
    """Build a 3-page PDF (overlay, diameter histogram, summary table) and return bytes."""
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: segmentation overlay
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(overlay_image)
        ax.set_title("Grain Segmentation Result with Scale Bar")
        ax.axis("off")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # Page 2: diameter distribution
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        axs[0].hist(grain_stats_df["Diameter_nm"], bins=30, edgecolor="black", color="skyblue")
        axs[0].set_xlabel("Grain Diameter (nm)")
        axs[0].set_ylabel("Frequency")
        axs[0].set_title("Grain Size Distribution")
        axs[0].grid(True, linestyle="--", alpha=0.7)
        bp = axs[1].boxplot(grain_stats_df["Diameter_nm"], vert=True, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("skyblue")
        axs[1].set_ylabel("Grain Diameter (nm)")
        axs[1].set_title("Grain Size Quartiles")
        axs[1].set_xticks([])
        axs[1].grid(True, linestyle="--", alpha=0.7)
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
        ax.set_title("Grain Statistics Summary", pad=20)
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

    model_path = st.text_input("Model path", value=DEFAULTS["model_path"])
    scale_factor = st.number_input(
        "Scale factor (px/nm)", value=DEFAULTS["scale_factor"], step=0.001, format="%.3f"
    )
    scale_bar_length_nm = st.number_input(
        "Scale bar length (nm)", value=DEFAULTS["scale_bar_length_nm"], step=10, min_value=1
    )

    with st.expander("Preprocessing"):
        tophat_radius = st.number_input(
            "Top-hat radius (px)", value=DEFAULTS["tophat_radius"], step=1, min_value=1
        )
        gaussian_sigma = st.number_input(
            "Gaussian sigma", value=float(DEFAULTS["gaussian_sigma"]), step=0.5, min_value=0.0
        )

    with st.expander("Cleaning"):
        opening_radius = st.number_input(
            "Opening radius (px)", value=DEFAULTS["opening_radius"], step=1, min_value=1
        )
        connect_dots = st.checkbox("Connect boundary dots", value=DEFAULTS["connect_dots"])
        line_length = st.number_input(
            "Line length (px)", value=DEFAULTS["line_length"], step=1, min_value=1
        )

    with st.expander("Watershed"):
        watershed_min_distance = st.number_input(
            "Min seed distance (px)", value=DEFAULTS["watershed_min_distance"], step=1, min_value=1
        )
        watershed_erosions = st.number_input(
            "Erosion passes", value=DEFAULTS["watershed_erosions"], step=1, min_value=0
        )

    with st.expander("Filtering"):
        min_object_size = st.number_input(
            "Min object size (px²)", value=DEFAULTS["min_object_size"], step=10, min_value=0
        )
        max_hole_size = st.number_input(
            "Max hole size (px²)", value=DEFAULTS["max_hole_size"], step=100, min_value=0
        )

    st.caption(
        "sigma_min (1) and sigma_max (10) are fixed — "
        "changing them requires retraining the classifier."
    )

    run_button = st.button(
        "Run Segmentation", type="primary", disabled=(uploaded_file is None)
    )

# --------------- RUN PIPELINE ---------------

if run_button and uploaded_file is not None:
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_img:
        tmp_img.write(uploaded_file.getbuffer())
        tmp_img_path = tmp_img.name

    output_dir = tempfile.mkdtemp()

    params = dict(
        model_path=model_path,
        scale_factor=float(scale_factor),
        scale_bar_length_nm=int(scale_bar_length_nm),
        tophat_radius=int(tophat_radius),
        gaussian_sigma=float(gaussian_sigma),
        opening_radius=int(opening_radius),
        connect_dots=connect_dots,
        line_length=int(line_length),
        watershed_min_distance=int(watershed_min_distance),
        watershed_erosions=int(watershed_erosions),
        min_object_size=int(min_object_size),
        max_hole_size=int(max_hole_size),
        save_intermediates=diagnostic_mode,
    )

    with st.spinner("Running pipeline..."):
        try:
            results = run_pipeline(tmp_img_path, output_dir, **params)
            st.session_state["results"] = results
            st.session_state["output_dir"] = output_dir
            st.session_state["image_raw"] = imread(tmp_img_path)
            st.session_state["diagnostic_mode"] = diagnostic_mode
        except Exception:
            st.error(traceback.format_exc())

# --------------- DISPLAY RESULTS ---------------

if "results" in st.session_state:
    results = st.session_state["results"]
    output_dir = st.session_state["output_dir"]
    image_raw = st.session_state["image_raw"]
    diag_mode = st.session_state.get("diagnostic_mode", False)

    grain_stats_df = results["grain_stats_df"]
    overlay_image = results["overlay_image"]
    boundary_mask = results["boundary_mask"]

    # ---- Diagnostic intermediates (shown ABOVE final results) ----
    if diag_mode:
        st.subheader("Diagnostic: Intermediate Steps")
        d_cols = st.columns(4)

        preprocessed = imread(str(Path(output_dir) / "intermediate_preprocessed.tif"))
        boundary_raw = imread(str(Path(output_dir) / "intermediate_boundary_raw.tif"))
        postprocessed = imread(str(Path(output_dir) / "intermediate_postprocessed.tif"))

        with d_cols[0]:
            st.image(
                _to_display(preprocessed),
                caption="Preprocessed — tophat_radius, gaussian_sigma",
                use_container_width=True,
            )
        with d_cols[1]:
            st.image(
                _to_display(boundary_raw),
                caption="Raw boundary prediction — model, sigma_min/max",
                use_container_width=True,
            )
        with d_cols[2]:
            st.image(
                _to_display(postprocessed),
                caption="Postprocessed grains — opening_radius, connect_dots, watershed params",
                use_container_width=True,
            )
        with d_cols[3]:
            st.empty()

        st.divider()

    # ---- Final results ----
    st.subheader("Segmentation Results")
    r_cols = st.columns(3)
    with r_cols[0]:
        st.image(_to_display(image_raw), caption="Original Image", use_container_width=True)
    with r_cols[1]:
        st.image(_to_display(boundary_mask), caption="Boundary Mask", use_container_width=True)
    with r_cols[2]:
        st.image(overlay_image, caption="Segmentation Overlay", use_container_width=True)

    st.divider()

    # ---- Summary stats ----
    st.subheader("Grain Statistics")
    m_cols = st.columns(4)
    with m_cols[0]:
        st.metric("Grains detected", len(grain_stats_df))
    with m_cols[1]:
        st.metric("Mean diameter (nm)", f"{grain_stats_df['Diameter_nm'].mean():.1f}")
    with m_cols[2]:
        st.metric("Std diameter (nm)", f"{grain_stats_df['Diameter_nm'].std():.1f}")
    with m_cols[3]:
        st.metric("Median diameter (nm)", f"{grain_stats_df['Diameter_nm'].median():.1f}")

    st.dataframe(grain_stats_df, use_container_width=True)

    # ---- Grain area histogram ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(grain_stats_df["Area_nm2"], bins=30, edgecolor="black", color="skyblue")
    ax.set_xlabel("Grain Area (nm²)")
    ax.set_ylabel("Frequency")
    ax.set_title("Grain Area Distribution")
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ---- PDF download ----
    pdf_bytes = _generate_pdf(overlay_image, grain_stats_df)
    st.download_button(
        label="Download PDF Report",
        data=pdf_bytes,
        file_name="grain_segmentation_report.pdf",
        mime="application/pdf",
    )
