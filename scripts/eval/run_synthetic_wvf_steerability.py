"""Run WVF steerability checks on the nested synthetic image set."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from edgecritic.core.taylor import build_taylor_matrix
from edgecritic.wvf import build_wvf_radius_kernels, wvf_component_gradients
from edgecritic.wvf._metal import metal_backend_available, wvf_radius_gradients_metal
from edgecritic.wvf._radius_kernels import WVFRadiusKernels


BRAND = {
    "garnet": (115, 0, 10),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "black90": (54, 54, 54),
    "black70": (92, 92, 92),
    "black50": (162, 162, 162),
    "black30": (199, 199, 199),
    "black10": (235, 235, 235),
    "rose": (204, 46, 64),
    "atlantic": (70, 106, 159),
    "honeycomb": (164, 145, 55),
}


@dataclass(frozen=True)
class RunMetric:
    image: str
    palette: str
    size: int
    order: int
    radius: int
    support_size: int
    angle_deg: float
    backend: str
    kernel_max_abs_error: float
    kernel_rmse: float
    response_max_abs_error: float
    response_mean_abs_error: float
    response_rmse: float
    response_p99_abs_error: float
    response_relative_max_error: float
    response_correlation: float
    time_gxy_s: float
    time_direct_s: float


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def parse_paths(root: Path, sizes: list[int]) -> list[Path]:
    paths: list[Path] = []
    for size in sizes:
        size_dir = root / str(size)
        paths.extend(sorted(path for path in size_dir.glob("*.png") if not path.name.startswith("._")))
    return paths


def image_palette_name(path: Path, size: int) -> str:
    stem = path.stem
    prefix = "nested_star_square_oval_"
    suffix = f"_{size}"
    if stem.startswith(prefix) and stem.endswith(suffix):
        return stem[len(prefix) : -len(suffix)]
    return stem


def load_luminance(path: Path) -> tuple[np.ndarray, Image.Image]:
    rgb_image = Image.open(path).convert("RGB")
    rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0
    luminance = 0.2126 * rgb[:, :, 0] + 0.7152 * rgb[:, :, 1] + 0.0722 * rgb[:, :, 2]
    return np.ascontiguousarray(luminance, dtype=np.float32), rgb_image


def direct_directional_weights(
    kernels: WVFRadiusKernels,
    angle_rad: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    design = build_taylor_matrix(kernels.offsets_xy, order=kernels.order)
    pinv = np.linalg.pinv(design)
    target = np.zeros(pinv.shape[0], dtype=np.float64)
    target[1] = math.cos(angle_rad)
    target[2] = math.sin(angle_rad)

    direct = np.ascontiguousarray(target @ pinv, dtype=np.float64)
    steered = np.ascontiguousarray(
        math.cos(angle_rad) * kernels.weights_x + math.sin(angle_rad) * kernels.weights_y,
        dtype=np.float64,
    )
    delta = direct - steered
    return direct, steered, float(np.max(np.abs(delta))), float(np.sqrt(np.mean(delta * delta)))


def direct_directional_response(
    image: np.ndarray,
    kernels: WVFRadiusKernels,
    weights: np.ndarray,
    backend: str,
) -> np.ndarray:
    if backend == "metal":
        direct_kernels = WVFRadiusKernels(
            radius=kernels.radius,
            order=kernels.order,
            offsets_xy=kernels.offsets_xy,
            weights_x=np.ascontiguousarray(weights, dtype=np.float64),
            weights_y=np.zeros_like(weights, dtype=np.float64),
            kernel_x=np.empty((0, 0), dtype=np.float64),
            kernel_y=np.empty((0, 0), dtype=np.float64),
        )
        response, _ = wvf_radius_gradients_metal(image, direct_kernels, output_dtype=np.float32)
        return response

    dense = np.zeros((2 * kernels.radius + 1, 2 * kernels.radius + 1), dtype=np.float64)
    for index, (dx, dy) in enumerate(kernels.offsets_xy.astype(np.int64)):
        dense[dy + kernels.radius, dx + kernels.radius] += float(weights[index])
    return ndimage.correlate(np.asarray(image, dtype=np.float64), dense, mode="reflect").astype(np.float32)


def robust_abs_limit(*arrays: np.ndarray) -> float:
    values = np.concatenate([np.ravel(np.abs(array)) for array in arrays])
    limit = float(np.percentile(values, 99.7))
    if not np.isfinite(limit) or limit <= 0.0:
        limit = float(np.max(values)) if values.size else 1.0
    return max(limit, 1e-12)


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int], t: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return aa + (bb - aa) * t[..., None]


def signed_response_image(values: np.ndarray, limit: float) -> Image.Image:
    scaled = np.clip(values / limit, -1.0, 1.0)
    positive = np.clip(scaled, 0.0, 1.0)
    negative = np.clip(-scaled, 0.0, 1.0)
    base = np.zeros((*scaled.shape, 3), dtype=np.float32)
    pos_rgb = lerp_color(BRAND["white"], BRAND["garnet"], positive)
    neg_rgb = lerp_color(BRAND["white"], BRAND["atlantic"], negative)
    zero = np.asarray(BRAND["white"], dtype=np.float32)
    base[:] = zero
    base = np.where(positive[..., None] > 0.0, pos_rgb, base)
    base = np.where(negative[..., None] > 0.0, neg_rgb, base)
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")


def abs_error_image(values: np.ndarray, limit: float) -> Image.Image:
    scaled = np.clip(values / max(limit, 1e-12), 0.0, 1.0)
    rgb = lerp_color(BRAND["black"], BRAND["rose"], scaled)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def add_label(tile: Image.Image, label: str) -> Image.Image:
    font = ImageFont.load_default()
    label_h = 24
    output = Image.new("RGB", (tile.width, tile.height + label_h), BRAND["black"])
    output.paste(tile, (0, label_h))
    draw = ImageDraw.Draw(output)
    draw.text((6, 7), label, fill=BRAND["white"], font=font)
    return output


def resize_tile(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def make_contact_sheet(
    output_path: Path,
    rows: list[tuple[str, list[Image.Image]]],
    headers: list[str],
    title: str,
    tile_size: int,
) -> None:
    font = ImageFont.load_default()
    gutter = 10
    title_h = 38
    label_w = 210
    label_h = 24
    tile_h = tile_size + label_h
    width = label_w + len(headers) * tile_size + (len(headers) + 1) * gutter
    height = title_h + len(rows) * tile_h + (len(rows) + 1) * gutter
    sheet = Image.new("RGB", (width, height), BRAND["black"])
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, 12), title, fill=BRAND["white"], font=font)

    for col, header in enumerate(headers):
        x = label_w + gutter + col * (tile_size + gutter)
        draw.text((x + 4, title_h - 18), header, fill=BRAND["black10"], font=font)

    for row_index, (row_label, tiles) in enumerate(rows):
        y = title_h + gutter + row_index * (tile_h + gutter)
        draw.text((gutter, y + tile_size // 2), row_label, fill=BRAND["white"], font=font)
        for col, tile in enumerate(tiles):
            x = label_w + gutter + col * (tile_size + gutter)
            sheet.paste(add_label(tile, headers[col]), (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def response_metrics(direct: np.ndarray, steered: np.ndarray) -> tuple[float, float, float, float, float]:
    diff = np.asarray(direct - steered, dtype=np.float64)
    abs_diff = np.abs(diff)
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    p99 = float(np.percentile(abs_diff, 99.0))
    denom = max(float(np.max(np.abs(direct))), 1e-12)
    return max_abs, mean_abs, rmse, p99, max_abs / denom


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.ravel(np.asarray(a, dtype=np.float64))
    bb = np.ravel(np.asarray(b, dtype=np.float64))
    aa = aa - float(np.mean(aa))
    bb = bb - float(np.mean(bb))
    denom = math.sqrt(float(np.sum(aa * aa)) * float(np.sum(bb * bb)))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(aa * bb) / denom)


def write_metrics(output_dir: Path, metrics: list[RunMetric]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "metrics.csv"
    json_path.write_text(json.dumps([asdict(metric) for metric in metrics], indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        for metric in metrics:
            writer.writerow(asdict(metric))


def remove_appledouble_sidecars(root: Path) -> None:
    for path in root.rglob("._*"):
        if path.is_file():
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("example_images/synthetic_nested_shapes/clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_wvf_steerability"))
    parser.add_argument("--sizes", type=parse_int_list, default=[1024, 2048, 4096])
    parser.add_argument("--orders", type=parse_int_list, default=[2, 4])
    parser.add_argument("--radius", type=int, default=25)
    parser.add_argument("--angle-deg", type=float, default=23.5)
    parser.add_argument("--backend", choices=["auto", "cpu", "metal"], default="auto")
    parser.add_argument("--tile-size", type=int, default=148)
    args = parser.parse_args()

    if args.backend == "auto":
        backend = "metal" if metal_backend_available() else "cpu"
    else:
        backend = args.backend

    if backend == "cpu" and max(args.sizes) >= 2048:
        raise SystemExit("CPU backend is too slow for this full-resolution run. Use --backend metal.")

    angle_rad = math.radians(args.angle_deg)
    cos_t = math.cos(angle_rad)
    sin_t = math.sin(angle_rad)
    paths = parse_paths(args.input_root, args.sizes)
    metrics: list[RunMetric] = []
    sheet_rows: dict[tuple[int, int], list[tuple[str, list[Image.Image]]]] = {
        (order, size): [] for order in args.orders for size in args.sizes
    }
    headers = ["input", "Gx", "Gy", f"steer {args.angle_deg:g}", f"direct {args.angle_deg:g}", "abs diff"]

    for order in args.orders:
        kernels = build_wvf_radius_kernels(radius=args.radius, order=order)
        direct_weights, _, kernel_max, kernel_rmse = direct_directional_weights(kernels, angle_rad)
        for image_path in paths:
            with Image.open(image_path) as probe:
                size = int(probe.size[0])
            if size not in args.sizes:
                continue

            palette = image_palette_name(image_path, size)
            image, rgb_image = load_luminance(image_path)

            start = time.perf_counter()
            gx, gy = wvf_component_gradients(
                image,
                radius=args.radius,
                order=order,
                backend=backend,
                output_dtype=np.float32,
            )
            time_gxy = time.perf_counter() - start

            steered = np.asarray(cos_t * gx + sin_t * gy, dtype=np.float32)
            start = time.perf_counter()
            direct = direct_directional_response(image, kernels, direct_weights, backend)
            time_direct = time.perf_counter() - start

            max_abs, mean_abs, rmse, p99_abs, rel_max = response_metrics(direct, steered)
            corr = correlation(direct, steered)
            metrics.append(
                RunMetric(
                    image=str(image_path),
                    palette=palette,
                    size=size,
                    order=order,
                    radius=args.radius,
                    support_size=kernels.support_size,
                    angle_deg=args.angle_deg,
                    backend=backend,
                    kernel_max_abs_error=kernel_max,
                    kernel_rmse=kernel_rmse,
                    response_max_abs_error=max_abs,
                    response_mean_abs_error=mean_abs,
                    response_rmse=rmse,
                    response_p99_abs_error=p99_abs,
                    response_relative_max_error=rel_max,
                    response_correlation=corr,
                    time_gxy_s=time_gxy,
                    time_direct_s=time_direct,
                )
            )

            signed_limit = robust_abs_limit(gx, gy, steered, direct)
            diff = np.abs(direct - steered)
            diff_limit = max(float(np.percentile(diff, 99.9)), 1e-8)
            tiles = [
                resize_tile(rgb_image, args.tile_size),
                resize_tile(signed_response_image(gx, signed_limit), args.tile_size),
                resize_tile(signed_response_image(gy, signed_limit), args.tile_size),
                resize_tile(signed_response_image(steered, signed_limit), args.tile_size),
                resize_tile(signed_response_image(direct, signed_limit), args.tile_size),
                resize_tile(abs_error_image(diff, diff_limit), args.tile_size),
            ]
            label = f"{palette}\nmax diff {max_abs:.2e}\nr {corr:.8f}"
            sheet_rows[(order, size)].append((label, tiles))
            print(
                f"d={order} size={size} palette={palette} max_diff={max_abs:.3e} "
                f"rel={rel_max:.3e} corr={corr:.10f} gxy={time_gxy:.3f}s direct={time_direct:.3f}s",
                flush=True,
            )

    figures_dir = args.output_dir / "figures"
    for order in args.orders:
        for size in args.sizes:
            rows = sorted(sheet_rows[(order, size)], key=lambda item: item[0])
            make_contact_sheet(
                figures_dir / f"wvf_steerability_d{order}_r{args.radius}_{size}.png",
                rows,
                headers,
                title=(
                    f"WVF steerability check. d={order}, radius={args.radius}, "
                    f"theta={args.angle_deg:g} deg, size={size}"
                ),
                tile_size=args.tile_size,
            )

    write_metrics(args.output_dir, metrics)
    summary = {
        "radius": args.radius,
        "angle_deg": args.angle_deg,
        "backend": backend,
        "orders": args.orders,
        "sizes": args.sizes,
        "n_images": len(paths),
        "max_response_abs_error": max(metric.response_max_abs_error for metric in metrics),
        "max_response_relative_error": max(metric.response_relative_max_error for metric in metrics),
        "min_response_correlation": min(metric.response_correlation for metric in metrics),
        "max_kernel_abs_error": max(metric.kernel_max_abs_error for metric in metrics),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    remove_appledouble_sidecars(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
