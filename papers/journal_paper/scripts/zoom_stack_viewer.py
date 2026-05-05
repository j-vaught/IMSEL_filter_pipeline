#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[3]
FIGURES_ROOT = ROOT / "papers" / "journal_paper" / "figures"
SUMMARY_JSON = (
    ROOT
    / "papers"
    / "journal_paper"
    / "figures"
    / "data"
    / "sec09_real_image_zoom_stack"
    / "sec09_real_image_zoom_stack_summary.json"
)
GARNET = "#73000A"
ATLANTIC = "#466A9F"
BLACK90 = "#363636"
BLACK30 = "#C7C7C7"
SNR_SLUG = "inf"
DEGREES = (1, 3, 5, 7, 9, 11)


@st.cache_data(show_spinner=False)
def load_summary(path: str) -> dict[str, object]:
    return json.loads(Path(path).read_text())


@st.cache_data(show_spinner=False)
def load_image(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")


def metric_color(wvf_value: float, baseline_value: float) -> str:
    return GARNET if float(wvf_value) < float(baseline_value) else ATLANTIC


def placeholder_image(width: int = 400, height: int = 400, message: str = "not available") -> Image.Image:
    image = Image.new("RGB", (width, height), color="#ECECEC")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width - 1, height - 1), outline=BLACK30, width=2)
    draw.multiline_text((width / 2, height / 2), message, fill=BLACK90, anchor="mm", align="center", spacing=6)
    return image


def resolve_asset(path: str | None) -> Path | None:
    if path in (None, ""):
        return None
    return FIGURES_ROOT / str(path)


def zoom_label(zoom_payload: dict[str, object]) -> str:
    return f"{zoom_payload['label']} ({zoom_payload['effective_vessel_diameter_px']})"


def build_cell_map(zoom_payload: dict[str, object]) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (int(cell["radius"]), int(cell["degree"])): cell
        for cell in zoom_payload["methods"]["wvf"]["cells"]
    }


def render_metric_line(label: str, value: float, baseline_value: float, suffix: str = "") -> str:
    color = metric_color(value, baseline_value)
    return (
        f"<span style='font-weight:700;color:{BLACK90}'>{label}</span> "
        f"<span style='font-weight:700;color:{color}'>{value:.4f}{suffix}</span>"
    )


def main() -> None:
    st.set_page_config(page_title="Zoom Stack Viewer", layout="wide")
    st.title("Section 9 Zoom-Stack Viewer")
    st.caption("Manual WVF tuning on the HRF zoom-stack headline. Displayed metrics are the clean-scenario values used by the headline figure.")

    summary = load_summary(str(SUMMARY_JSON))
    zoom_order = summary["zoom_order"]
    zooms = summary["zooms"]
    zoom_options = {zoom_label(zooms[key]): key for key in zoom_order}

    st.sidebar.header("Controls")
    zoom_display = st.sidebar.radio("Zoom level", list(zoom_options.keys()), index=0)
    zoom_key = zoom_options[zoom_display]
    zoom_payload = zooms[zoom_key]
    best_clean = zoom_payload["methods"]["wvf"]["best_by_snr"][SNR_SLUG]
    default_r = int(best_clean["radius"])
    default_d = int(best_clean["degree"])
    radius = st.sidebar.slider("Radius r", min_value=2, max_value=30, value=default_r, step=1)
    degree = st.sidebar.selectbox("Degree d", options=list(DEGREES), index=list(DEGREES).index(default_d))

    cell_map = build_cell_map(zoom_payload)
    selected_cell = cell_map.get((int(radius), int(degree)))

    static_assets = dict(zoom_payload.get("static_assets", {}))
    input_path = resolve_asset(static_assets.get("input_path") or zoom_payload.get("input_asset_path"))
    gt_path = resolve_asset(static_assets.get("ground_truth_path") or zoom_payload.get("ground_truth_asset_path") or zoom_payload.get("vessel_mask_asset_path"))
    baseline_assets = dict(zoom_payload.get("baseline_reference_assets", {}))
    baseline_mag_path = resolve_asset(
        baseline_assets.get("magnitude_path")
        or zoom_payload["methods"]["farid_simoncelli"]["clean_assets"]["magnitude_path"]
    )

    if selected_cell is not None and "clean_assets" in selected_cell:
        wvf_mag_path = resolve_asset(selected_cell["clean_assets"]["magnitude_path"])
        wvf_image = load_image(str(wvf_mag_path)) if wvf_mag_path is not None and wvf_mag_path.exists() else placeholder_image(message="asset missing")
        availability_note = None
    else:
        wvf_image = placeholder_image(message="not available\nkappa(A) above gate")
        availability_note = zoom_payload["conditioning_gate"]

    input_image = load_image(str(input_path)) if input_path is not None and input_path.exists() else placeholder_image(message="input missing")
    gt_image = load_image(str(gt_path)) if gt_path is not None and gt_path.exists() else placeholder_image(message="ground truth missing")
    baseline_image = load_image(str(baseline_mag_path)) if baseline_mag_path is not None and baseline_mag_path.exists() else placeholder_image(message="baseline missing")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Input")
        st.image(input_image, use_container_width=True)
    with col2:
        st.subheader("Ground truth")
        st.image(gt_image, use_container_width=True)
    with col3:
        st.subheader(f"WVF magnitude r={radius}, d={degree}")
        st.image(wvf_image, use_container_width=True)

    st.subheader("Best fixed baseline magnitude")
    st.image(baseline_image, width=400, caption="Farid–Simoncelli")

    baseline_metrics = zoom_payload["methods"]["farid_simoncelli"]["snr_metrics"][SNR_SLUG]
    if selected_cell is not None:
        wvf_metrics = selected_cell["snr_metrics"][SNR_SLUG]
        wvf_rmse = float(wvf_metrics["gradient_vector_rmse_mean"])
        wvf_mae = float(wvf_metrics["orientation_mae_deg_mean"])
        base_rmse = float(baseline_metrics["gradient_vector_rmse_mean"])
        base_mae = float(baseline_metrics["orientation_mae_deg_mean"])
        metrics_html = (
            render_metric_line("WVF RMSE", wvf_rmse, base_rmse)
            + f" &nbsp;&nbsp; <span style='color:{BLACK90}'>F-S RMSE {base_rmse:.4f}</span>"
            + "<br/>"
            + render_metric_line("WVF orientation MAE", wvf_mae, base_mae, "°")
            + f" &nbsp;&nbsp; <span style='color:{BLACK90}'>F-S orientation MAE {base_mae:.4f}°</span>"
        )
        st.markdown(metrics_html, unsafe_allow_html=True)
    else:
        st.markdown(
            f"<span style='color:{BLACK90};font-weight:700'>WVF cell unavailable.</span> "
            f"<span style='color:{ATLANTIC}'>The selected (r, d) pair was conditioning-gated out of the feasible grid.</span>",
            unsafe_allow_html=True,
        )

    if availability_note is not None:
        st.caption(availability_note)
    else:
        st.caption("Unavailable cells are omitted by the same conditioning gate used in the degree-radius interaction ablation.")


if __name__ == "__main__":
    main()
