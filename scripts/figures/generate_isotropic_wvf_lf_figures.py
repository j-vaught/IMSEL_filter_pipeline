"""Generate figures for the isotropic-WVF LF decomposition note."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, hsv_to_rgb
from PIL import Image
from scipy import ndimage

from edgecritic.core.taylor import build_taylor_matrix, get_circular_neighbors, get_square_neighbors, rotate_coordinates
from edgecritic.nms_gmm import build_line_filter_kernels


GARNET = "#73000A"
ROSE = "#CC2E40"
ATLANTIC = "#466A9F"
BLACK90 = "#363636"
BLACK70 = "#5C5C5C"
BLACK10 = "#ECECEC"
WHITE = "#FFFFFF"

WEIGHT_CMAP = LinearSegmentedColormap.from_list("lf_weights", [ATLANTIC, WHITE, GARNET])


def _neighbors(np_count: int, neighbor_type: str) -> np.ndarray:
    if neighbor_type == "circular":
        return get_circular_neighbors(np_count)
    if neighbor_type == "square":
        return get_square_neighbors(np_count)
    raise ValueError("neighbor_type must be 'circular' or 'square'")


def _canonical_rows(np_count: int, order: int, neighbor_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    neighbors = _neighbors(np_count, neighbor_type)
    design = build_taylor_matrix(neighbors, order=order)
    pinv = np.linalg.pinv(design)
    return neighbors, pinv[1, :], pinv[2, :]


def _line_weights(half_width: int) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-half_width, half_width + 1, dtype=np.float64)
    if half_width == 0:
        return offsets, np.ones(1, dtype=np.float64)
    sigma = max(float(half_width) / 2.0, np.finfo(np.float64).eps)
    weights = np.exp(-0.5 * (offsets / sigma) ** 2)
    weights /= weights.sum()
    return offsets, weights


def _kernel_from_sparse(
    offsets: list[tuple[int, int]],
    weights: list[float],
    shape: tuple[int, int],
) -> np.ndarray:
    kernel = np.zeros(shape, dtype=np.float64)
    cy = shape[0] // 2
    cx = shape[1] // 2
    for (dy, dx), weight in zip(offsets, weights):
        kernel[cy + dy, cx + dx] += weight
    return kernel


def _derivative_kernels(
    np_count: int,
    order: int,
    neighbor_type: str,
    shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    neighbors, wx, wy = _canonical_rows(np_count, order, neighbor_type)
    if shape is None:
        max_dx = int(np.max(np.abs(neighbors[:, 0])))
        max_dy = int(np.max(np.abs(neighbors[:, 1])))
        shape = (2 * max_dy + 1, 2 * max_dx + 1)
    offsets = [(int(dy), int(dx)) for dx, dy in neighbors.astype(np.int64)]
    return _kernel_from_sparse(offsets, wx.tolist(), shape), _kernel_from_sparse(offsets, wy.tolist(), shape)


def _line_normal_lf_shape(
    half_width: int,
    np_count: int,
    order: int,
    n_orientations: int,
    neighbor_type: str,
) -> tuple[tuple[int, int], np.ndarray, int]:
    reference = build_line_filter_kernels(
        half_width=half_width,
        np_count=np_count,
        order=order,
        n_orientations=n_orientations,
        neighbor_type=neighbor_type,
    )
    return reference.kernels.shape[1:], reference.angles, reference.border


def build_standard_line_normal_lf_kernels(
    half_width: int = 7,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    neighbor_type: str = "circular",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build standard LF kernels with theta as line direction and WVF normal perpendicular to it."""
    kernel_shape, angles, border = _line_normal_lf_shape(
        half_width=half_width,
        np_count=np_count,
        order=order,
        n_orientations=n_orientations,
        neighbor_type=neighbor_type,
    )
    neighbors = _neighbors(np_count, neighbor_type)
    line_offsets, line_weights = _line_weights(half_width)

    kernels = np.zeros((len(angles), *kernel_shape), dtype=np.float64)
    for angle_index, theta in enumerate(angles):
        normal_theta = theta + np.pi / 2.0
        local = rotate_coordinates(neighbors, normal_theta)
        directional = np.linalg.pinv(build_taylor_matrix(local, order=order))[1, :]
        offsets: list[tuple[int, int]] = []
        weights: list[float] = []
        for line_offset, line_weight in zip(line_offsets, line_weights):
            vdx = int(round(line_offset * np.cos(theta)))
            vdy = int(round(line_offset * np.sin(theta)))
            for neighbor_index, (dx, dy) in enumerate(neighbors.astype(np.int64)):
                offsets.append((vdy + int(dy), vdx + int(dx)))
                weights.append(float(line_weight * directional[neighbor_index]))
        kernels[angle_index] = _kernel_from_sparse(offsets, weights, kernel_shape)

    return kernels, angles, border


def build_isotropic_decomposed_lf_kernels(
    half_width: int = 7,
    np_count: int = 15,
    order: int = 4,
    n_orientations: int = 36,
    neighbor_type: str = "circular",
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build line-normal LF kernels from one isotropic WVF derivative pair and line smoothing."""
    kernel_shape, angles, border = _line_normal_lf_shape(
        half_width=half_width,
        np_count=np_count,
        order=order,
        n_orientations=n_orientations,
        neighbor_type=neighbor_type,
    )
    neighbors, wx, wy = _canonical_rows(np_count, order, neighbor_type)
    line_offsets, line_weights = _line_weights(half_width)

    kernels = np.zeros((len(angles), *kernel_shape), dtype=np.float64)
    for angle_index, theta in enumerate(angles):
        directional = -np.sin(theta) * wx + np.cos(theta) * wy
        offsets: list[tuple[int, int]] = []
        weights: list[float] = []
        for line_offset, line_weight in zip(line_offsets, line_weights):
            vdx = int(round(line_offset * np.cos(theta)))
            vdy = int(round(line_offset * np.sin(theta)))
            for neighbor_index, (dx, dy) in enumerate(neighbors.astype(np.int64)):
                offsets.append((vdy + int(dy), vdx + int(dx)))
                weights.append(float(line_weight * directional[neighbor_index]))
        kernels[angle_index] = _kernel_from_sparse(offsets, weights, kernel_shape)

    return kernels, angles, border


def _standard_normal_rows(np_count: int, order: int, n_orientations: int, neighbor_type: str) -> tuple[np.ndarray, np.ndarray]:
    neighbors = _neighbors(np_count, neighbor_type)
    angles = np.linspace(0.0, np.pi, n_orientations, endpoint=False)
    rows = []
    for theta in angles:
        local = rotate_coordinates(neighbors, theta + np.pi / 2.0)
        rows.append(np.linalg.pinv(build_taylor_matrix(local, order=order))[1, :])
    return np.stack(rows, axis=0), angles


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    max_value = float(np.max(arr)) if arr.size else 0.0
    if max_value <= 0.0:
        return np.zeros_like(arr)
    return np.clip(arr / max_value, 0.0, 1.0)


def _angle_rgb(angle: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    hue = (np.asarray(angle) % np.pi) / np.pi
    value = np.maximum(_normalize(magnitude), 0.18)
    hsv = np.stack([hue, np.ones_like(hue), value], axis=-1)
    return hsv_to_rgb(hsv)


def _axial_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = (np.asarray(a) - np.asarray(b) + np.pi / 2.0) % np.pi - np.pi / 2.0
    return np.abs(np.rad2deg(diff))


def _read_gray(path: Path, max_side: int) -> np.ndarray:
    with Image.open(path) as im:
        gray = im.convert("L")
        gray.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        return np.asarray(gray, dtype=np.float64) / 255.0


def _style_axes(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(BLACK10)
        spine.set_linewidth(0.8)


def _plot_kernel(ax, kernel: np.ndarray, title: str, vmax: float | None = None) -> None:
    max_abs = float(np.max(np.abs(kernel))) if vmax is None else float(vmax)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs) if max_abs > 0 else None
    ax.imshow(kernel, cmap=WEIGHT_CMAP, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=8.5, color=BLACK90)
    _style_axes(ax)


def _figure_derivative_identity(output_dir: Path, np_count: int, order: int, neighbor_type: str) -> dict[str, float]:
    neighbors, wx, wy = _canonical_rows(np_count, order, neighbor_type)
    standard_rows, angles = _standard_normal_rows(np_count, order, 36, neighbor_type)
    combo_rows = np.array([-np.sin(theta) * wx + np.cos(theta) * wy for theta in angles])
    row_diff = standard_rows - combo_rows

    shape = (9, 9)
    kx, ky = _derivative_kernels(np_count, order, neighbor_type, shape=shape)
    selected_degrees = [0, 30, 60, 90]
    fig, axes = plt.subplots(3, 4, figsize=(9.3, 6.6), dpi=180)
    fig.patch.set_facecolor("white")

    for col, degree in enumerate(selected_degrees):
        theta = np.deg2rad(degree)
        normal_theta = theta + np.pi / 2.0
        local = rotate_coordinates(neighbors, normal_theta)
        standard_row = np.linalg.pinv(build_taylor_matrix(local, order=order))[1, :]
        standard_kernel = _kernel_from_sparse(
            [(int(dy), int(dx)) for dx, dy in neighbors.astype(np.int64)],
            standard_row.tolist(),
            shape,
        )
        combo_kernel = -np.sin(theta) * kx + np.cos(theta) * ky
        vmax = max(float(np.max(np.abs(standard_kernel))), float(np.max(np.abs(combo_kernel))))
        _plot_kernel(axes[0, col], standard_kernel, f"WVF normal to {degree} deg line", vmax=vmax)
        _plot_kernel(axes[1, col], combo_kernel, f"-sin/cos pair {degree} deg", vmax=vmax)
        _plot_kernel(axes[2, col], standard_kernel - combo_kernel, "difference", vmax=max(vmax, 1e-15))

    fig.suptitle("The LF normal derivative equals one isotropic Gx/Gy pair", color=GARNET, fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "fig_derivative_identity.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    return {
        "directional_row_max_abs_diff": float(np.max(np.abs(row_diff))),
        "directional_row_rmse": float(np.sqrt(np.mean(row_diff * row_diff))),
        "figure": str(path),
    }


def _figure_kernel_identity(
    output_dir: Path,
    standard_kernels: np.ndarray,
    decomposed_kernels: np.ndarray,
    angles: np.ndarray,
) -> dict[str, float]:
    selected_degrees = [0, 30, 60, 90, 120, 150]
    fig, axes = plt.subplots(len(selected_degrees), 3, figsize=(7.2, 10.8), dpi=180)
    fig.patch.set_facecolor("white")

    for row, degree in enumerate(selected_degrees):
        target = np.deg2rad(degree)
        index = int(np.argmin(np.abs(((angles - target + np.pi / 2.0) % np.pi) - np.pi / 2.0)))
        standard = standard_kernels[index]
        decomposed = decomposed_kernels[index]
        diff = standard - decomposed
        vmax = max(float(np.max(np.abs(standard))), float(np.max(np.abs(decomposed))))
        _plot_kernel(axes[row, 0], standard, f"standard LF {degree} deg", vmax=vmax)
        _plot_kernel(axes[row, 1], decomposed, "isotropic form", vmax=vmax)
        _plot_kernel(axes[row, 2], diff, "difference", vmax=max(vmax, 1e-15))

    fig.suptitle("Line-normal LF kernels match the isotropic-WVF line-smoothed kernels", color=GARNET, fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "fig_lf_kernel_identity.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    diff = standard_kernels - decomposed_kernels
    return {
        "kernel_max_abs_diff": float(np.max(np.abs(diff))),
        "kernel_rmse": float(np.sqrt(np.mean(diff * diff))),
        "figure": str(path),
    }


def _responses_from_kernels(image: np.ndarray, kernels: np.ndarray, border: int) -> np.ndarray:
    responses = np.empty((*image.shape, kernels.shape[0]), dtype=np.float64)
    for index, kernel in enumerate(kernels):
        responses[..., index] = ndimage.correlate(image, kernel, mode="reflect")
    if border > 0:
        b = int(border)
        responses[:b, :, :] = 0.0
        responses[-b:, :, :] = 0.0
        responses[:, :b, :] = 0.0
        responses[:, -b:, :] = 0.0
    return responses


def _magnitude_angle(responses: np.ndarray, angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    abs_responses = np.abs(responses)
    index = np.argmax(abs_responses, axis=-1)
    magnitude = np.take_along_axis(abs_responses, index[..., None], axis=-1)[..., 0]
    angle = angles[index]
    return magnitude, angle


def _figure_response_identity(
    output_dir: Path,
    image: np.ndarray,
    standard_responses: np.ndarray,
    decomposed_responses: np.ndarray,
    angles: np.ndarray,
) -> dict[str, float]:
    standard_mag, standard_angle = _magnitude_angle(standard_responses, angles)
    decomposed_mag, decomposed_angle = _magnitude_angle(decomposed_responses, angles)
    response_diff = standard_responses - decomposed_responses
    mag_diff = np.abs(standard_mag - decomposed_mag)
    angle_diff = _axial_diff_deg(standard_angle, decomposed_angle)
    gate = standard_mag > 0.1 * float(np.max(standard_mag))

    fig, axes = plt.subplots(2, 4, figsize=(11.4, 5.8), dpi=180)
    fig.patch.set_facecolor("white")
    panels = [
        ("Input", image, "gray"),
        ("Standard LF magnitude", _normalize(standard_mag), "gray"),
        ("Isotropic-form magnitude", _normalize(decomposed_mag), "gray"),
        ("Magnitude difference", mag_diff, "magma"),
        ("Standard LF angle", _angle_rgb(standard_angle, standard_mag), None),
        ("Isotropic-form angle", _angle_rgb(decomposed_angle, decomposed_mag), None),
        ("Angle difference deg", angle_diff, "magma"),
        ("Response max abs diff", np.max(np.abs(response_diff), axis=-1), "magma"),
    ]
    for ax, (title, data, cmap) in zip(axes.ravel(), panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=8.5, color=BLACK90)
        _style_axes(ax)

    fig.suptitle("Line-normal LF responses are unchanged by the isotropic-WVF decomposition", color=GARNET, fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = output_dir / "fig_response_identity.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight")
    plt.close(fig)

    return {
        "response_max_abs_diff": float(np.max(np.abs(response_diff))),
        "response_rmse": float(np.sqrt(np.mean(response_diff * response_diff))),
        "relative_response_rmse": float(np.sqrt(np.mean(response_diff * response_diff)) / (np.sqrt(np.mean(standard_responses * standard_responses)) + 1e-12)),
        "gated_angle_max_diff_deg": float(np.max(angle_diff[gate])) if np.any(gate) else 0.0,
        "gated_angle_mean_diff_deg": float(np.mean(angle_diff[gate])) if np.any(gate) else 0.0,
        "same_argmax_fraction_gated": float(np.mean(standard_angle[gate] == decomposed_angle[gate])) if np.any(gate) else 1.0,
        "figure": str(path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("papers/anisotropic-lf-proposal/figures/isotropic_wvf_lf"))
    parser.add_argument("--image", type=Path, default=Path("papers/filter-critique/presentation/figures/datasets/biped_RGB008.png"))
    parser.add_argument("--max-side", type=int, default=128)
    parser.add_argument("--half-width", type=int, default=7)
    parser.add_argument("--np-count", type=int, default=15)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--orientations", type=int, default=36)
    parser.add_argument("--neighbor-type", choices=("circular", "square"), default="circular")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    standard_kernels, angles, border = build_standard_line_normal_lf_kernels(
        half_width=args.half_width,
        np_count=args.np_count,
        order=args.order,
        n_orientations=args.orientations,
        neighbor_type=args.neighbor_type,
    )
    decomposed_kernels, _, decomposed_border = build_isotropic_decomposed_lf_kernels(
        half_width=args.half_width,
        np_count=args.np_count,
        order=args.order,
        n_orientations=args.orientations,
        neighbor_type=args.neighbor_type,
    )
    if decomposed_border != border:
        raise RuntimeError("standard and decomposed border sizes do not match")

    image = _read_gray(args.image, max_side=args.max_side)
    standard_responses = _responses_from_kernels(image, standard_kernels, border=border)
    decomposed_responses = _responses_from_kernels(image, decomposed_kernels, border=border)

    summary = {
        "config": {
            "half_width": args.half_width,
            "np_count": args.np_count,
            "order": args.order,
            "orientations": args.orientations,
            "neighbor_type": args.neighbor_type,
            "image": str(args.image),
            "image_shape": list(image.shape),
        },
        "derivative_identity": _figure_derivative_identity(args.output_dir, args.np_count, args.order, args.neighbor_type),
        "kernel_identity": _figure_kernel_identity(args.output_dir, standard_kernels, decomposed_kernels, angles),
        "response_identity": _figure_response_identity(args.output_dir, image, standard_responses, decomposed_responses, angles),
    }

    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
