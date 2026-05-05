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
    / "sec09_real_image_hd_viewer"
    / "sec09_real_image_hd_viewer_summary.json"
)
GARNET = "#73000A"
ATLANTIC = "#466A9F"
BLACK90 = "#363636"
BLACK30 = "#C7C7C7"
DEGREES = (1, 3, 5, 7, 9, 11)
ZOOM_OPTIONS = (1, 2, 4, 8)


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


def build_cell_map(summary: dict[str, object]) -> dict[tuple[int, int], dict[str, object]]:
    return {
        (int(cell["radius"]), int(cell["degree"])): cell
        for cell in summary["wvf_grid"]["feasible_cells"]
    }


def render_metric_line(label: str, value: float, baseline_value: float, suffix: str = "") -> str:
    color = metric_color(value, baseline_value)
    return (
        f"<span style='font-weight:700;color:{BLACK90}'>{label}</span> "
        f"<span style='font-weight:700;color:{color}'>{value:.4f}{suffix}</span>"
    )


def crop_for_view(
    image: Image.Image,
    rect_xywh: list[int],
    zoom_factor: int,
    pan_x_frac: float,
    pan_y_frac: float,
) -> Image.Image:
    x0, y0, width, height = (int(rect_xywh[0]), int(rect_xywh[1]), int(rect_xywh[2]), int(rect_xywh[3]))
    if int(zoom_factor) <= 1:
        return image.crop((x0, y0, x0 + width, y0 + height))
    sub_w = max(1, int(round(width / float(zoom_factor))))
    sub_h = max(1, int(round(height / float(zoom_factor))))
    max_dx = max(0, width - sub_w)
    max_dy = max(0, height - sub_h)
    dx = int(round(float(pan_x_frac) * float(max_dx)))
    dy = int(round(float(pan_y_frac) * float(max_dy)))
    left = x0 + dx
    top = y0 + dy
    return image.crop((left, top, left + sub_w, top + sub_h))


def main() -> None:
    st.set_page_config(page_title="HD WVF Tuning Viewer", layout="wide")
    st.title("Section 9 HD WVF Tuning Viewer")
    st.caption("Manual WVF tuning on one full-resolution HRF image. Metrics are computed over the full image and remain fixed while the viewer pans and zooms.")

    if not SUMMARY_JSON.exists():
        st.error(f"Summary JSON not found at {SUMMARY_JSON}")
        st.stop()

    summary = load_summary(str(SUMMARY_JSON))
    cell_map = build_cell_map(summary)
    baseline = dict(summary["baseline_reference"])
    crop_presets = dict(summary["crop_presets"])
    crop_order = list(summary["crop_preset_order"])
    default_cell = dict(summary["wvf_grid"]["default_cell"])

    st.sidebar.header("Controls")
    radius = st.sidebar.slider("Radius r", min_value=2, max_value=30, value=int(default_cell["radius"]), step=1)
    degree = st.sidebar.selectbox("Degree d", options=list(DEGREES), index=list(DEGREES).index(int(default_cell["degree"])))
    crop_options = {str(crop_presets[key]["label"]): key for key in crop_order}
    crop_label = st.sidebar.radio("Crop region", list(crop_options.keys()), index=0)
    crop_key = crop_options[crop_label]
    zoom_factor = st.sidebar.radio("Zoom level", options=list(ZOOM_OPTIONS), format_func=lambda value: "No zoom" if value == 1 else f"{int(value)}x")
    if int(zoom_factor) > 1:
        pan_x_pct = st.sidebar.slider("Pan X", min_value=0, max_value=100, value=50, step=1)
        pan_y_pct = st.sidebar.slider("Pan Y", min_value=0, max_value=100, value=50, step=1)
    else:
        pan_x_pct = 0
        pan_y_pct = 0

    selected_cell = cell_map.get((int(radius), int(degree)))
    static_assets = dict(summary["static_assets"])
    input_path = resolve_asset(static_assets.get("input_path"))
    gt_path = resolve_asset(static_assets.get("ground_truth_path"))
    baseline_assets = dict(baseline["assets"])
    baseline_mag_path = resolve_asset(baseline_assets.get("magnitude_path"))
    baseline_ori_path = resolve_asset(baseline_assets.get("orientation_path"))

    input_image = load_image(str(input_path)) if input_path is not None and input_path.exists() else placeholder_image(message="input missing")
    gt_image = load_image(str(gt_path)) if gt_path is not None and gt_path.exists() else placeholder_image(message="ground truth missing")
    baseline_mag = load_image(str(baseline_mag_path)) if baseline_mag_path is not None and baseline_mag_path.exists() else placeholder_image(message="baseline missing")
    baseline_ori = load_image(str(baseline_ori_path)) if baseline_ori_path is not None and baseline_ori_path.exists() else placeholder_image(message="baseline missing")

    if selected_cell is not None:
        wvf_assets = dict(selected_cell["assets"])
        wvf_mag_path = resolve_asset(wvf_assets.get("magnitude_path"))
        wvf_ori_path = resolve_asset(wvf_assets.get("orientation_path"))
        wvf_mag = load_image(str(wvf_mag_path)) if wvf_mag_path is not None and wvf_mag_path.exists() else placeholder_image(message="asset missing")
        wvf_ori = load_image(str(wvf_ori_path)) if wvf_ori_path is not None and wvf_ori_path.exists() else placeholder_image(message="asset missing")
        availability_note = None
    else:
        wvf_mag = placeholder_image(message="not available\nkappa(A) above gate")
        wvf_ori = placeholder_image(message="not available\nkappa(A) above gate")
        availability_note = str(summary["wvf_grid"]["conditioning_gate"])

    rect_xywh = list(crop_presets[crop_key]["rect_xywh"])
    pan_x_frac = float(pan_x_pct) / 100.0
    pan_y_frac = float(pan_y_pct) / 100.0

    tabs = st.tabs(["Magnitude", "Orientation"])
    for tab_name, tab in zip(("magnitude", "orientation"), tabs, strict=True):
        with tab:
            current_wvf = wvf_mag if tab_name == "magnitude" else wvf_ori
            current_baseline = baseline_mag if tab_name == "magnitude" else baseline_ori
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.subheader("Input")
                st.image(
                    crop_for_view(input_image, rect_xywh, int(zoom_factor), pan_x_frac, pan_y_frac),
                    use_container_width=True,
                )
            with col2:
                st.subheader("Vessel mask")
                st.image(
                    crop_for_view(gt_image, rect_xywh, int(zoom_factor), pan_x_frac, pan_y_frac),
                    use_container_width=True,
                )
            with col3:
                st.subheader(f"WVF {tab_name} r={radius}, d={degree}")
                st.image(
                    crop_for_view(current_wvf, rect_xywh, int(zoom_factor), pan_x_frac, pan_y_frac),
                    use_container_width=True,
                )
            with col4:
                st.subheader(f"Farid–Simoncelli {tab_name}")
                st.image(
                    crop_for_view(current_baseline, rect_xywh, int(zoom_factor), pan_x_frac, pan_y_frac),
                    use_container_width=True,
                )

    baseline_metrics = dict(baseline["metrics"])
    if selected_cell is not None:
        wvf_metrics = dict(selected_cell["metrics"])
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
        st.caption(
            f"Full-image metrics. support_cardinality={int(selected_cell['support_cardinality'])}, "
            f"kappa(A)={float(selected_cell['kappa_design_matrix']):.3e}, sigma_min={float(selected_cell['sigma_min']):.3e}"
        )
    else:
        st.markdown(
            f"<span style='color:{BLACK90};font-weight:700'>WVF cell unavailable.</span> "
            f"<span style='color:{ATLANTIC}'>The selected (r, d) pair was conditioning-gated out of the feasible grid.</span>",
            unsafe_allow_html=True,
        )

    preview_shape = list(summary["asset_rendering"]["preview_shape_px"])
    st.caption(
        f"Crop preset: {crop_presets[crop_key]['label']}. Preview canvas: {preview_shape[0]}x{preview_shape[1]} px. "
        f"Zoomed views pan inside the chosen preset rectangle without recomputing the filter."
    )
    if availability_note is not None:
        st.caption(availability_note)
    else:
        st.caption(
            "Unavailable cells are omitted by the same conditioning gate used in the degree-radius interaction ablation."
        )


if __name__ == "__main__":
    main()
