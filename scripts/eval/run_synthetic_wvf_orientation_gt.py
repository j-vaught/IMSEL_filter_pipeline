"""Compare WVF gradient-orientation estimates against synthetic ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from edgecritic.wvf import wvf_component_gradients
from edgecritic.wvf._metal import metal_backend_available

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BRAND = {
    "garnet": (115, 0, 10),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "black90": (54, 54, 54),
    "black70": (92, 92, 92),
    "black30": (199, 199, 199),
    "black10": (235, 235, 235),
    "rose": (204, 46, 64),
    "atlantic": (70, 106, 159),
    "honeycomb": (164, 145, 55),
}

PLOT_COLORS = {
    "steerable_atan2": "#73000A",
    "brute": "#466A9F",
    "grid": "#ECECEC",
    "text": "#000000",
}


@dataclass(frozen=True)
class OrientationMetric:
    image: str
    palette: str
    size: int
    order: int
    radius: int
    estimator: str
    sample_count: int
    mask: str
    n_pixels: int
    mean_deg: float
    median_deg: float
    p90_deg: float
    p95_deg: float
    p99_deg: float
    max_deg: float
    mean_strength: float


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def star_points(
    center: tuple[float, float],
    outer_radius: float,
    inner_radius: float,
    rotation_deg: float,
) -> np.ndarray:
    cx, cy = center
    start = math.radians(rotation_deg - 90.0)
    points = []
    for index in range(10):
        radius = outer_radius if index % 2 == 0 else inner_radius
        angle = start + index * math.pi / 5.0
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return np.asarray(points, dtype=np.float64)


def draw_layer_mask(layer: dict[str, float | str], size: int) -> np.ndarray:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    cx = float(layer["center_x"])
    cy = float(layer["center_y"])
    width = float(layer["width"])
    height = float(layer["height"])
    if layer["shape"] == "star":
        points = star_points(
            (cx, cy),
            outer_radius=width / 2.0,
            inner_radius=width * 0.34,
            rotation_deg=float(layer["rotation_deg"]),
        )
        draw.polygon([tuple(point) for point in points], fill=255)
    elif layer["shape"] == "square":
        half = width / 2.0
        draw.rectangle((cx - half, cy - half, cx + half, cy + half), fill=255)
    elif layer["shape"] == "oval":
        draw.ellipse((cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0), fill=255)
    else:
        raise ValueError(f"unsupported shape: {layer['shape']}")
    return np.asarray(mask, dtype=bool)


def polygon_normal_angles(
    x: np.ndarray,
    y: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    best_dist = np.full(x.shape, np.inf, dtype=np.float64)
    best_angle = np.zeros(x.shape, dtype=np.float64)
    for index in range(len(points)):
        p0 = points[index]
        p1 = points[(index + 1) % len(points)]
        vx = p1[0] - p0[0]
        vy = p1[1] - p0[1]
        denom = vx * vx + vy * vy
        t = np.clip(((x - p0[0]) * vx + (y - p0[1]) * vy) / denom, 0.0, 1.0)
        proj_x = p0[0] + t * vx
        proj_y = p0[1] + t * vy
        dist = (x - proj_x) ** 2 + (y - proj_y) ** 2
        update = dist < best_dist
        normal_angle = math.atan2(vx, -vy)
        best_angle = np.where(update, normal_angle, best_angle)
        best_dist = np.where(update, dist, best_dist)
    return best_angle


def square_normal_angles(x: np.ndarray, y: np.ndarray, layer: dict[str, float | str]) -> np.ndarray:
    cx = float(layer["center_x"])
    cy = float(layer["center_y"])
    half = float(layer["width"]) / 2.0
    left = abs(x - (cx - half))
    right = abs(x - (cx + half))
    top = abs(y - (cy - half))
    bottom = abs(y - (cy + half))
    distances = np.stack([left, right, top, bottom], axis=0)
    side = np.argmin(distances, axis=0)
    angle = np.zeros(x.shape, dtype=np.float64)
    angle = np.where(side == 2, math.pi / 2.0, angle)
    angle = np.where(side == 3, math.pi / 2.0, angle)
    return angle


def oval_normal_angles(x: np.ndarray, y: np.ndarray, layer: dict[str, float | str]) -> np.ndarray:
    cx = float(layer["center_x"])
    cy = float(layer["center_y"])
    a = float(layer["width"]) / 2.0
    b = float(layer["height"]) / 2.0
    nx = (x - cx) / max(a * a, 1e-12)
    ny = (y - cy) / max(b * b, 1e-12)
    return np.arctan2(ny, nx)


def vertex_mask_for_layer(layer: dict[str, float | str], size: int, exclude_px: int) -> np.ndarray:
    if exclude_px <= 0:
        return np.zeros((size, size), dtype=bool)

    cx = float(layer["center_x"])
    cy = float(layer["center_y"])
    width = float(layer["width"])
    if layer["shape"] == "star":
        points = star_points((cx, cy), width / 2.0, width * 0.34, float(layer["rotation_deg"]))
    elif layer["shape"] == "square":
        half = width / 2.0
        points = np.asarray(
            [
                (cx - half, cy - half),
                (cx + half, cy - half),
                (cx + half, cy + half),
                (cx - half, cy + half),
            ],
            dtype=np.float64,
        )
    else:
        return np.zeros((size, size), dtype=bool)

    yy, xx = np.mgrid[0:size, 0:size]
    mask = np.zeros((size, size), dtype=bool)
    r2 = float(exclude_px * exclude_px)
    for px, py in points:
        mask |= (xx - px) ** 2 + (yy - py) ** 2 <= r2
    return mask


def layer_angles(layer: dict[str, float | str], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if layer["shape"] == "star":
        points = star_points(
            (float(layer["center_x"]), float(layer["center_y"])),
            outer_radius=float(layer["width"]) / 2.0,
            inner_radius=float(layer["width"]) * 0.34,
            rotation_deg=float(layer["rotation_deg"]),
        )
        return polygon_normal_angles(x, y, points)
    if layer["shape"] == "square":
        return square_normal_angles(x, y, layer)
    if layer["shape"] == "oval":
        return oval_normal_angles(x, y, layer)
    raise ValueError(f"unsupported shape: {layer['shape']}")


def build_gt_orientation(
    image_spec: dict[str, object],
    edge_band_px: int,
    vertex_exclude_px: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    size = int(image_spec["size"])
    orientation = np.zeros((size, size), dtype=np.float32)
    all_mask = np.zeros((size, size), dtype=bool)
    smooth_mask = np.zeros((size, size), dtype=bool)
    yy, xx = np.mgrid[0:size, 0:size]

    structure = np.ones((3, 3), dtype=bool)
    for layer in image_spec["layers"]:
        mask = draw_layer_mask(layer, size)
        inner = mask & ~ndimage.binary_erosion(mask, structure=structure, border_value=0)
        outer = ndimage.binary_dilation(mask, structure=structure) & ~mask
        boundary = inner | outer
        if edge_band_px > 0:
            boundary = ndimage.binary_dilation(boundary, iterations=edge_band_px)

        angles = layer_angles(layer, xx[boundary].astype(np.float64), yy[boundary].astype(np.float64))
        orientation[boundary] = wrap_pi(angles).astype(np.float32)
        all_mask |= boundary

        vertices = vertex_mask_for_layer(layer, size, vertex_exclude_px)
        if edge_band_px > 0:
            vertices = ndimage.binary_dilation(vertices, iterations=edge_band_px)
        smooth_mask |= boundary & ~vertices

    return orientation, all_mask, smooth_mask


def load_luminance(path: Path) -> tuple[np.ndarray, Image.Image]:
    rgb_image = Image.open(path).convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return np.ascontiguousarray(luminance, dtype=np.float32), rgb_image


def wrap_pi(angle: np.ndarray | float) -> np.ndarray:
    return np.mod(angle, math.pi)


def unsigned_angle_error_deg(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    diff = np.mod(est - gt + math.pi / 2.0, math.pi) - math.pi / 2.0
    return np.abs(diff) * (180.0 / math.pi)


def brute_grid_angle(steer_angle: np.ndarray, sample_count: int) -> np.ndarray:
    step = math.pi / float(sample_count)
    return np.mod(np.rint(wrap_pi(steer_angle) / step) * step, math.pi).astype(np.float32)


def metric_from_error(
    image_spec: dict[str, object],
    order: int,
    radius: int,
    estimator: str,
    sample_count: int,
    mask_name: str,
    mask: np.ndarray,
    error_deg: np.ndarray,
    strength: np.ndarray,
) -> OrientationMetric:
    values = error_deg[mask]
    strengths = strength[mask]
    return OrientationMetric(
        image=str(image_spec["path"]),
        palette=str(image_spec["palette"]),
        size=int(image_spec["size"]),
        order=order,
        radius=radius,
        estimator=estimator,
        sample_count=sample_count,
        mask=mask_name,
        n_pixels=int(values.size),
        mean_deg=float(np.mean(values)),
        median_deg=float(np.median(values)),
        p90_deg=float(np.percentile(values, 90)),
        p95_deg=float(np.percentile(values, 95)),
        p99_deg=float(np.percentile(values, 99)),
        max_deg=float(np.max(values)),
        mean_strength=float(np.mean(strengths)),
    )


def hue_orientation_image(angle: np.ndarray, mask: np.ndarray) -> Image.Image:
    hue = wrap_pi(angle) / math.pi
    h = (hue * 6.0).astype(np.float32)
    c = np.ones_like(h)
    x = 1.0 - np.abs(np.mod(h, 2.0) - 1.0)
    z = np.zeros_like(h)
    rgb = np.zeros((*h.shape, 3), dtype=np.float32)

    regions = [
        ((0 <= h) & (h < 1), (c, x, z)),
        ((1 <= h) & (h < 2), (x, c, z)),
        ((2 <= h) & (h < 3), (z, c, x)),
        ((3 <= h) & (h < 4), (z, x, c)),
        ((4 <= h) & (h < 5), (x, z, c)),
        ((5 <= h) & (h < 6), (c, z, x)),
    ]
    for region, channels in regions:
        rgb[..., 0] = np.where(region, channels[0], rgb[..., 0])
        rgb[..., 1] = np.where(region, channels[1], rgb[..., 1])
        rgb[..., 2] = np.where(region, channels[2], rgb[..., 2])
    rgb = np.where(mask[..., None], rgb * 255.0, np.asarray(BRAND["black"], dtype=np.float32))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def error_image(error_deg: np.ndarray, mask: np.ndarray, limit_deg: float = 30.0) -> Image.Image:
    scaled = np.clip(error_deg / limit_deg, 0.0, 1.0)
    lo = np.asarray(BRAND["black"], dtype=np.float32)
    hi = np.asarray(BRAND["rose"], dtype=np.float32)
    rgb = lo + (hi - lo) * scaled[..., None]
    rgb = np.where(mask[..., None], rgb, np.asarray(BRAND["black"], dtype=np.float32))
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def resize_tile(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def add_label(tile: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    label_h = 24
    output = Image.new("RGB", (tile.width, tile.height + label_h), BRAND["black"])
    output.paste(tile, (0, label_h))
    draw = ImageDraw.Draw(output)
    draw.text((6, 7), label, fill=BRAND["white"], font=font)
    return output


def make_contact_sheet(
    output_path: Path,
    rows: list[tuple[str, list[Image.Image]]],
    headers: list[str],
    title: str,
    tile_size: int,
) -> None:
    font = ImageFont.load_default()
    gutter = 10
    title_h = 40
    row_label_w = 250
    label_h = 24
    tile_h = tile_size + label_h
    width = row_label_w + len(headers) * tile_size + (len(headers) + 1) * gutter
    height = title_h + len(rows) * tile_h + (len(rows) + 1) * gutter
    sheet = Image.new("RGB", (width, height), BRAND["black"])
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, 12), title, fill=BRAND["white"], font=font)
    for row_index, (row_label, tiles) in enumerate(rows):
        y = title_h + gutter + row_index * (tile_h + gutter)
        draw.text((gutter, y + tile_size // 2), row_label, fill=BRAND["white"], font=font)
        for col, tile in enumerate(tiles):
            x = row_label_w + gutter + col * (tile_size + gutter)
            sheet.paste(add_label(tile, headers[col]), (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def write_metrics(output_dir: Path, metrics: list[OrientationMetric]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_dicts = [asdict(metric) for metric in metrics]
    (output_dir / "metrics.json").write_text(json.dumps(metric_dicts, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metric_dicts[0]))
        writer.writeheader()
        writer.writerows(metric_dicts)


def update_histogram(
    histograms: dict[tuple[int, int, str, str], np.ndarray],
    bins: np.ndarray,
    order: int,
    size: int,
    mask_name: str,
    estimator: str,
    values: np.ndarray,
) -> None:
    counts, _ = np.histogram(values, bins=bins)
    key = (order, size, mask_name, estimator)
    if key not in histograms:
        histograms[key] = np.zeros(len(bins) - 1, dtype=np.int64)
    histograms[key] += counts.astype(np.int64)


def normalized_counts(histograms: dict[tuple[int, int, str, str], np.ndarray], key: tuple[int, int, str, str]) -> np.ndarray:
    counts = histograms.get(key)
    if counts is None:
        return np.zeros(0, dtype=np.float64)
    total = int(np.sum(counts))
    if total == 0:
        return counts.astype(np.float64)
    return counts.astype(np.float64) / float(total)


def make_histogram_grid(
    output_path: Path,
    histograms: dict[tuple[int, int, str, str], np.ndarray],
    bins: np.ndarray,
    orders: list[int],
    sizes: list[int],
    mask_name: str,
    brute_estimator: str,
) -> None:
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(
        len(orders),
        len(sizes),
        figsize=(4.6 * len(sizes), 3.1 * len(orders)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for row, order in enumerate(orders):
        for col, size in enumerate(sizes):
            ax = axes[row][col]
            steer = normalized_counts(histograms, (order, size, mask_name, "steerable_atan2"))
            brute = normalized_counts(histograms, (order, size, mask_name, brute_estimator))
            if steer.size:
                ax.step(
                    centers,
                    steer,
                    where="mid",
                    color=PLOT_COLORS["steerable_atan2"],
                    linewidth=1.8,
                    label="steerable atan2",
                )
            if brute.size:
                ax.step(
                    centers,
                    brute,
                    where="mid",
                    color=PLOT_COLORS["brute"],
                    linewidth=1.8,
                    label=brute_estimator.replace("_", " "),
                )
            ax.set_xlim(0.0, 90.0)
            ax.grid(True, color=PLOT_COLORS["grid"], linewidth=0.7)
            ax.set_title(f"d={order}, {size}px", fontsize=10)
            ax.tick_params(colors=PLOT_COLORS["text"], labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(PLOT_COLORS["text"])
            if row == len(orders) - 1:
                ax.set_xlabel("unsigned orientation error (deg)", fontsize=9)
            if col == 0:
                ax.set_ylabel("fraction of pixels", fontsize=9)
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f"GT orientation-error histogram. mask={mask_name}. all palettes.",
        fontsize=13,
        color=PLOT_COLORS["text"],
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize(metrics: list[OrientationMetric]) -> dict[str, object]:
    summary_rows = []
    for order in sorted({metric.order for metric in metrics}):
        for estimator in sorted({metric.estimator for metric in metrics if metric.order == order}):
            subset = [
                metric
                for metric in metrics
                if metric.order == order and metric.estimator == estimator and metric.mask == "smooth"
            ]
            if not subset:
                continue
            total_pixels = sum(metric.n_pixels for metric in subset)
            weighted_mean = sum(metric.mean_deg * metric.n_pixels for metric in subset) / total_pixels
            worst_p95 = max(metric.p95_deg for metric in subset)
            worst_p99 = max(metric.p99_deg for metric in subset)
            summary_rows.append(
                {
                    "order": order,
                    "estimator": estimator,
                    "sample_count": subset[0].sample_count,
                    "smooth_weighted_mean_deg": weighted_mean,
                    "smooth_worst_p95_deg": worst_p95,
                    "smooth_worst_p99_deg": worst_p99,
                }
            )
    return {"summary": summary_rows}


def remove_appledouble_sidecars(root: Path) -> None:
    for path in root.rglob("._*"):
        if path.is_file():
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("example_images/synthetic_nested_shapes/manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_wvf_orientation_gt"))
    parser.add_argument("--sizes", type=parse_int_list, default=[1024, 2048, 4096])
    parser.add_argument("--orders", type=parse_int_list, default=[2, 4, 6, 8])
    parser.add_argument("--radius", type=int, default=25)
    parser.add_argument("--sample-counts", type=parse_int_list, default=[16, 32, 64, 180])
    parser.add_argument("--edge-band-px", type=int, default=2)
    parser.add_argument("--vertex-exclude-px", type=int, default=25)
    parser.add_argument("--backend", choices=["auto", "cpu", "metal"], default="auto")
    parser.add_argument("--tile-size", type=int, default=150)
    args = parser.parse_args()

    backend = "metal" if args.backend == "auto" and metal_backend_available() else args.backend
    if backend == "auto":
        backend = "cpu"
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    image_specs = [item for item in manifest["images"] if int(item["size"]) in set(args.sizes)]

    metrics: list[OrientationMetric] = []
    hist_bins = np.linspace(0.0, 90.0, 181)
    histograms: dict[tuple[int, int, str, str], np.ndarray] = {}
    figure_rows: dict[int, list[tuple[str, list[Image.Image]]]] = {order: [] for order in args.orders}
    preview_palettes = {"dark_light_steps", "low_contrast_mixed_chroma", "garnet_atlantic_grass"}
    preview_size = min(args.sizes)
    brute_preview_samples = max(args.sample_counts)
    brute_hist_estimator = f"brute_{brute_preview_samples}"

    for spec in image_specs:
        image_path = Path(str(spec["path"]))
        image, rgb_image = load_luminance(image_path)
        gt_angle, all_mask, smooth_mask = build_gt_orientation(spec, args.edge_band_px, args.vertex_exclude_px)
        masks = {"all": all_mask, "smooth": smooth_mask}

        for order in args.orders:
            gx, gy = wvf_component_gradients(
                image,
                radius=args.radius,
                order=order,
                backend=backend,
                output_dtype=np.float32,
            )
            strength = np.hypot(gx, gy)
            steer_angle = wrap_pi(np.arctan2(gy, gx)).astype(np.float32)
            steer_error = unsigned_angle_error_deg(steer_angle, gt_angle)
            steer_smooth_mean = float(np.mean(steer_error[smooth_mask]))

            for mask_name, mask in masks.items():
                metrics.append(
                    metric_from_error(
                        spec,
                        order,
                        args.radius,
                        "steerable_atan2",
                        0,
                        mask_name,
                        mask,
                        steer_error,
                        strength,
                    )
                )
                update_histogram(
                    histograms,
                    hist_bins,
                    order,
                    int(spec["size"]),
                    mask_name,
                    "steerable_atan2",
                    steer_error[mask],
                )

            brute_errors: dict[int, np.ndarray] = {}
            for sample_count in args.sample_counts:
                brute_angle = brute_grid_angle(steer_angle, sample_count)
                brute_error = unsigned_angle_error_deg(brute_angle, gt_angle)
                brute_errors[sample_count] = brute_error
                brute_vs_steer = unsigned_angle_error_deg(brute_angle, steer_angle)
                for mask_name, mask in masks.items():
                    metrics.append(
                        metric_from_error(
                            spec,
                            order,
                            args.radius,
                            f"brute_{sample_count}",
                            sample_count,
                            mask_name,
                            mask,
                            brute_error,
                            strength,
                        )
                    )
                    if sample_count == brute_preview_samples:
                        update_histogram(
                            histograms,
                            hist_bins,
                            order,
                            int(spec["size"]),
                            mask_name,
                            f"brute_{sample_count}",
                            brute_error[mask],
                        )
                    metrics.append(
                        metric_from_error(
                            spec,
                            order,
                            args.radius,
                            f"brute_{sample_count}_vs_steerable",
                            sample_count,
                            mask_name,
                            mask,
                            brute_vs_steer,
                            strength,
                        )
                    )

            if int(spec["size"]) == preview_size and str(spec["palette"]) in preview_palettes:
                brute_angle = brute_grid_angle(steer_angle, brute_preview_samples)
                brute_error = brute_errors[brute_preview_samples]
                tiles = [
                    resize_tile(rgb_image, args.tile_size),
                    resize_tile(hue_orientation_image(gt_angle, all_mask), args.tile_size),
                    resize_tile(hue_orientation_image(steer_angle, all_mask), args.tile_size),
                    resize_tile(hue_orientation_image(brute_angle, all_mask), args.tile_size),
                    resize_tile(error_image(steer_error, all_mask), args.tile_size),
                    resize_tile(error_image(brute_error, all_mask), args.tile_size),
                ]
                label = f"{spec['palette']}\nsize {spec['size']}"
                figure_rows[order].append((label, tiles))

            print(
                f"size={spec['size']} palette={spec['palette']} d={order} "
                f"steer_smooth_mean={steer_smooth_mean:.3f}",
                flush=True,
            )

    write_metrics(args.output_dir, metrics)
    summary = summarize(metrics)
    summary.update(
        {
            "radius": args.radius,
            "orders": args.orders,
            "sizes": args.sizes,
            "sample_counts": args.sample_counts,
            "backend": backend,
            "edge_band_px": args.edge_band_px,
            "vertex_exclude_px": args.vertex_exclude_px,
            "n_images": len(image_specs),
        }
    )
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    headers = [
        "input",
        "GT angle",
        "steer angle",
        f"brute {brute_preview_samples}",
        "steer error",
        f"brute {brute_preview_samples} error",
    ]
    for order, rows in figure_rows.items():
        make_contact_sheet(
            args.output_dir / "figures" / f"orientation_gt_d{order}_r{args.radius}.png",
            rows,
            headers,
            title=f"WVF orientation vs GT. d={order}, radius={args.radius}, brute={brute_preview_samples}",
            tile_size=args.tile_size,
        )

    for mask_name in masks:
        make_histogram_grid(
            args.output_dir / "figures" / f"orientation_error_hist_{mask_name}_{brute_hist_estimator}.png",
            histograms,
            hist_bins,
            args.orders,
            args.sizes,
            mask_name,
            brute_hist_estimator,
        )

    readme = (
        "# Synthetic WVF orientation ground-truth comparison\n\n"
        "This run compares closed-form steerable WVF orientation against brute sampled directional-search orientation and analytic synthetic-shape ground truth.\n\n"
        "`steerable_atan2` uses `atan2(Gy, Gx)`. `brute_N` quantizes the same steerable direction onto `N` sampled orientations in `[0, pi)`, which is equivalent to rotating the isotropic WVF directional filter bank after the direct-vs-steered kernel equivalence has been established.\n\n"
        "Primary metrics use the `smooth` boundary mask, which excludes polygon vertices and square corners by `vertex_exclude_px` because corner orientation is not single-valued.\n\n"
        "The histogram figures overlay steerable orientation error and the densest brute-grid error. The `all` histogram uses every GT-labeled boundary pixel, including corners and vertices. Pixels away from shape boundaries are not included because they do not have a defined GT edge orientation.\n"
    )
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    remove_appledouble_sidecars(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
