"""Visualize the WVF kernels used in the synthetic steerability run."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from edgecritic.core.taylor import build_taylor_matrix
from edgecritic.wvf import build_wvf_radius_kernels
from edgecritic.wvf._radius_kernels import WVFRadiusKernels


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


@dataclass(frozen=True)
class KernelMetric:
    name: str
    order: int
    radius: int
    support_size: int
    sum_weights: float
    max_abs_weight: float
    l1_norm: float
    l2_norm: float
    first_moment_x: float
    first_moment_y: float


def parse_int_list(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return values


def dense_from_weights(kernels: WVFRadiusKernels, weights: np.ndarray) -> np.ndarray:
    radius = kernels.radius
    dense = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    for index, (dx, dy) in enumerate(kernels.offsets_xy.astype(np.int64)):
        dense[dy + radius, dx + radius] = float(weights[index])
    return dense


def direct_directional_weights(kernels: WVFRadiusKernels, angle_rad: float) -> np.ndarray:
    design = build_taylor_matrix(kernels.offsets_xy, order=kernels.order)
    pinv = np.linalg.pinv(design)
    target = np.zeros(pinv.shape[0], dtype=np.float64)
    target[1] = math.cos(angle_rad)
    target[2] = math.sin(angle_rad)
    return np.ascontiguousarray(target @ pinv, dtype=np.float64)


def kernel_metrics(
    name: str,
    kernels: WVFRadiusKernels,
    weights: np.ndarray,
) -> KernelMetric:
    offsets = kernels.offsets_xy
    return KernelMetric(
        name=name,
        order=kernels.order,
        radius=kernels.radius,
        support_size=kernels.support_size,
        sum_weights=float(np.sum(weights)),
        max_abs_weight=float(np.max(np.abs(weights))),
        l1_norm=float(np.sum(np.abs(weights))),
        l2_norm=float(np.sqrt(np.sum(weights * weights))),
        first_moment_x=float(np.sum(weights * offsets[:, 0])),
        first_moment_y=float(np.sum(weights * offsets[:, 1])),
    )


def blend(a: tuple[int, int, int], b: tuple[int, int, int], t: np.ndarray) -> np.ndarray:
    aa = np.asarray(a, dtype=np.float32)
    bb = np.asarray(b, dtype=np.float32)
    return aa + (bb - aa) * t[..., None]


def signed_kernel_image(
    kernel: np.ndarray,
    support_mask: np.ndarray,
    limit: float,
    cell: int,
    grid: bool,
) -> Image.Image:
    scaled = np.clip(kernel / max(limit, 1e-30), -1.0, 1.0)
    positive = np.clip(scaled, 0.0, 1.0)
    negative = np.clip(-scaled, 0.0, 1.0)
    rgb = np.zeros((*kernel.shape, 3), dtype=np.float32)
    zero = np.asarray(BRAND["white"], dtype=np.float32)
    pos_rgb = blend(BRAND["white"], BRAND["garnet"], positive)
    neg_rgb = blend(BRAND["white"], BRAND["atlantic"], negative)
    rgb[:] = zero
    rgb = np.where(positive[..., None] > 0.0, pos_rgb, rgb)
    rgb = np.where(negative[..., None] > 0.0, neg_rgb, rgb)
    rgb = np.where(support_mask[..., None], rgb, np.asarray(BRAND["black"], dtype=np.float32))
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    image = image.resize((kernel.shape[1] * cell, kernel.shape[0] * cell), Image.Resampling.NEAREST)

    if grid:
        draw = ImageDraw.Draw(image)
        grid_color = BRAND["black30"]
        width, height = image.size
        for x in range(0, width + 1, cell):
            draw.line((x, 0, x, height), fill=grid_color)
        for y in range(0, height + 1, cell):
            draw.line((0, y, width, y), fill=grid_color)
    return image


def abs_kernel_image(kernel: np.ndarray, support_mask: np.ndarray, limit: float, cell: int) -> Image.Image:
    scaled = np.clip(np.abs(kernel) / max(limit, 1e-30), 0.0, 1.0)
    rgb = blend(BRAND["black"], BRAND["rose"], scaled)
    rgb = np.where(support_mask[..., None], rgb, np.asarray(BRAND["black"], dtype=np.float32))
    image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
    return image.resize((kernel.shape[1] * cell, kernel.shape[0] * cell), Image.Resampling.NEAREST)


def add_tile_label(tile: Image.Image, title: str, detail: str) -> Image.Image:
    font = ImageFont.load_default()
    label_h = 46
    output = Image.new("RGB", (tile.width, tile.height + label_h), BRAND["black"])
    output.paste(tile, (0, label_h))
    draw = ImageDraw.Draw(output)
    draw.text((8, 8), title, fill=BRAND["white"], font=font)
    draw.text((8, 26), detail, fill=BRAND["black10"], font=font)
    return output


def make_sheet(
    output_path: Path,
    rows: list[tuple[str, list[tuple[str, str, Image.Image]]]],
    title: str,
) -> None:
    font = ImageFont.load_default()
    gutter = 14
    title_h = 42
    row_label_w = 130
    tile_w = rows[0][1][0][2].width
    tile_h = rows[0][1][0][2].height + 46
    width = row_label_w + len(rows[0][1]) * tile_w + (len(rows[0][1]) + 1) * gutter
    height = title_h + len(rows) * tile_h + (len(rows) + 1) * gutter
    sheet = Image.new("RGB", (width, height), BRAND["black"])
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, 13), title, fill=BRAND["white"], font=font)

    for row_index, (row_label, tiles) in enumerate(rows):
        y = title_h + gutter + row_index * (tile_h + gutter)
        draw.text((gutter, y + tile_h // 2), row_label, fill=BRAND["white"], font=font)
        for col_index, (tile_title, detail, tile) in enumerate(tiles):
            x = row_label_w + gutter + col_index * (tile_w + gutter)
            sheet.paste(add_tile_label(tile, tile_title, detail), (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def support_mask(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    return (xx * xx + yy * yy <= radius * radius) & ~((xx == 0) & (yy == 0))


def remove_appledouble_sidecars(root: Path) -> None:
    for path in root.rglob("._*"):
        if path.is_file():
            path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/synthetic_wvf_steerability/kernels"))
    parser.add_argument("--radius", type=int, default=25)
    parser.add_argument("--orders", type=parse_int_list, default=[2, 4])
    parser.add_argument("--angle-deg", type=float, default=23.5)
    parser.add_argument("--cell", type=int, default=8)
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    angle_rad = math.radians(args.angle_deg)
    mask = support_mask(args.radius)
    by_order: dict[int, dict[str, np.ndarray]] = {}
    metric_rows: list[KernelMetric] = []

    for order in args.orders:
        kernels = build_wvf_radius_kernels(radius=args.radius, order=order)
        direct_weights = direct_directional_weights(kernels, angle_rad)
        steered_weights = math.cos(angle_rad) * kernels.weights_x + math.sin(angle_rad) * kernels.weights_y
        by_order[order] = {
            "Gx": kernels.kernel_x,
            "Gy": kernels.kernel_y,
            f"direct {args.angle_deg:g} deg": dense_from_weights(kernels, direct_weights),
            f"steered {args.angle_deg:g} deg": math.cos(angle_rad) * kernels.kernel_x
            + math.sin(angle_rad) * kernels.kernel_y,
            "abs direct-steered": np.abs(dense_from_weights(kernels, direct_weights) - dense_from_weights(kernels, steered_weights)),
        }
        metric_rows.extend(
            [
                kernel_metrics("Gx", kernels, kernels.weights_x),
                kernel_metrics("Gy", kernels, kernels.weights_y),
                kernel_metrics(f"direct {args.angle_deg:g} deg", kernels, direct_weights),
                kernel_metrics(f"steered {args.angle_deg:g} deg", kernels, steered_weights),
            ]
        )

    signed_names = ["Gx", "Gy", f"direct {args.angle_deg:g} deg", f"steered {args.angle_deg:g} deg"]
    signed_limit = max(float(np.max(np.abs(by_order[order][name]))) for order in args.orders for name in signed_names)
    diff_limit = max(
        float(np.max(np.abs(by_order[args.orders[-1]][name] - by_order[args.orders[0]][name])))
        for name in signed_names[:3]
    )
    steer_diff_limit = max(
        float(np.max(by_order[order]["abs direct-steered"])) for order in args.orders
    )

    basis_rows: list[tuple[str, list[tuple[str, str, Image.Image]]]] = []
    for order in args.orders:
        tiles: list[tuple[str, str, Image.Image]] = []
        for name in signed_names:
            values = by_order[order][name]
            detail = f"max |w| {np.max(np.abs(values)):.2e}"
            tiles.append((name, detail, signed_kernel_image(values, mask, signed_limit, args.cell, not args.no_grid)))
        steer_abs = by_order[order]["abs direct-steered"]
        detail = f"max {np.max(steer_abs):.2e}"
        tiles.append(("abs direct-steered", detail, abs_kernel_image(steer_abs, mask, max(steer_diff_limit, 1e-30), args.cell)))
        basis_rows.append((f"d={order}", tiles))

    make_sheet(
        args.output_dir / f"wvf_kernel_basis_r{args.radius}_theta{args.angle_deg:g}.png",
        basis_rows,
        title=f"WVF derivative kernels. radius={args.radius}, theta={args.angle_deg:g} deg",
    )

    if len(args.orders) >= 2:
        low_order = args.orders[0]
        high_order = args.orders[-1]
        diff_tiles: list[tuple[str, str, Image.Image]] = []
        for name in signed_names[:3]:
            values = by_order[high_order][name] - by_order[low_order][name]
            detail = f"max |delta| {np.max(np.abs(values)):.2e}"
            diff_tiles.append((f"d{high_order} - d{low_order} {name}", detail, signed_kernel_image(values, mask, diff_limit, args.cell, not args.no_grid)))
        make_sheet(
            args.output_dir / f"wvf_kernel_d{high_order}_minus_d{low_order}_r{args.radius}.png",
            [(f"d{high_order}-d{low_order}", diff_tiles)],
            title=f"WVF kernel order difference. radius={args.radius}",
        )

    summary = {
        "radius": args.radius,
        "angle_deg": args.angle_deg,
        "orders": args.orders,
        "support_size": build_wvf_radius_kernels(args.radius, args.orders[0]).support_size,
        "signed_color_limit": signed_limit,
        "order_difference_color_limit": diff_limit,
        "max_direct_steered_abs_kernel_error": steer_diff_limit,
        "metrics": [asdict(row) for row in metric_rows],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "kernel_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    remove_appledouble_sidecars(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
