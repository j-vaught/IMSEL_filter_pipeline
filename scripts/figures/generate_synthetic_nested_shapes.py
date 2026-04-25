"""Generate nested star/square/oval synthetic edge-stress images."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BRAND = {
    "garnet": "#73000A",
    "black": "#000000",
    "white": "#FFFFFF",
    "black90": "#363636",
    "black70": "#5C5C5C",
    "black50": "#A2A2A2",
    "black30": "#C7C7C7",
    "black10": "#ECECEC",
    "warm_grey": "#676156",
    "sandstorm": "#FFF2E3",
    "rose": "#CC2E40",
    "atlantic": "#466A9F",
    "congaree": "#1F414D",
    "horseshoe": "#65780B",
    "grass": "#CED318",
    "honeycomb": "#A49137",
}


PALETTES = {
    "garnet_atlantic_grass": {
        "background": BRAND["white"],
        "layers": [
            BRAND["garnet"],
            BRAND["atlantic"],
            BRAND["grass"],
            BRAND["congaree"],
            BRAND["rose"],
            BRAND["honeycomb"],
            BRAND["horseshoe"],
            BRAND["black"],
        ],
    },
    "dark_light_steps": {
        "background": BRAND["white"],
        "layers": [
            BRAND["black"],
            BRAND["black10"],
            BRAND["black90"],
            BRAND["black30"],
            BRAND["garnet"],
            BRAND["white"],
            BRAND["black70"],
            BRAND["rose"],
        ],
    },
    "low_contrast_gray": {
        "background": "#A8A8A8",
        "layers": [
            "#858585",
            "#BEBEBE",
            "#949494",
            "#C9C9C9",
            "#7A7A7A",
            "#D2D2D2",
            "#909090",
            "#BCBCBC",
        ],
    },
    "low_contrast_cool_colors": {
        "background": "#9EADAE",
        "layers": [
            "#8FA2A8",
            "#A8B7B6",
            "#93A9B5",
            "#B0B4A2",
            "#8799A4",
            "#AEB1B8",
            "#98A7A0",
            "#A9B8AA",
        ],
    },
    "low_contrast_warm_colors": {
        "background": "#B9AFA4",
        "layers": [
            "#A89991",
            "#C0B1A4",
            "#B1A08E",
            "#C4B8A9",
            "#A89185",
            "#BCA69C",
            "#B7AC91",
            "#AFA2A0",
        ],
    },
    "low_contrast_mixed_chroma": {
        "background": "#A9ACA0",
        "layers": [
            "#96A6A0",
            "#B6AA9F",
            "#A4A2B3",
            "#B2B896",
            "#9BA5B7",
            "#B7A1AA",
            "#AAB28F",
            "#A0B3AD",
        ],
    },
    "warm_cool_midtones": {
        "background": BRAND["sandstorm"],
        "layers": [
            BRAND["congaree"],
            BRAND["honeycomb"],
            BRAND["atlantic"],
            BRAND["horseshoe"],
            BRAND["rose"],
            BRAND["warm_grey"],
            BRAND["garnet"],
            BRAND["grass"],
        ],
    },
}


NOISE_PREVIEWS = {
    "clean": {"kind": "clean", "amount": 0.0},
    "gaussian_sigma20": {"kind": "gaussian", "amount": 20.0},
    "saltpepper_2pct": {"kind": "saltpepper", "amount": 0.02},
}


@dataclass(frozen=True)
class LayerSpec:
    shape: str
    center_x: float
    center_y: float
    width: float
    height: float
    rotation_deg: float
    color: str

    @property
    def min_dimension(self) -> float:
        return min(self.width, self.height)


def parse_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise argparse.ArgumentTypeError("at least one size is required")
    if any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("sizes must be positive integers")
    return sizes


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def star_points(
    center: tuple[float, float],
    outer_radius: float,
    inner_radius: float,
    rotation_deg: float,
) -> list[tuple[float, float]]:
    cx, cy = center
    points: list[tuple[float, float]] = []
    start = math.radians(rotation_deg - 90.0)
    for index in range(10):
        radius = outer_radius if index % 2 == 0 else inner_radius
        angle = start + index * math.pi / 5.0
        points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
    return points


def render_layers(size: int, palette_name: str, min_dimension: int) -> tuple[Image.Image, list[LayerSpec]]:
    palette = PALETTES[palette_name]
    image = Image.new("RGB", (size, size), hex_to_rgb(palette["background"]))
    draw = ImageDraw.Draw(image)
    layers = build_layers(size, palette["layers"], min_dimension)

    for layer in layers:
        fill = hex_to_rgb(layer.color)
        cx = layer.center_x
        cy = layer.center_y
        if layer.shape == "star":
            points = star_points(
                (cx, cy),
                outer_radius=layer.width / 2.0,
                inner_radius=layer.width * 0.34,
                rotation_deg=layer.rotation_deg,
            )
            draw.polygon(points, fill=fill)
        elif layer.shape == "square":
            half = layer.width / 2.0
            draw.rectangle((cx - half, cy - half, cx + half, cy + half), fill=fill)
        elif layer.shape == "oval":
            half_w = layer.width / 2.0
            half_h = layer.height / 2.0
            draw.ellipse((cx - half_w, cy - half_h, cx + half_w, cy + half_h), fill=fill)
        else:
            raise ValueError(f"unsupported shape: {layer.shape}")

    return image, layers


def build_layers(size: int, colors: list[str], min_dimension: int) -> list[LayerSpec]:
    center = float(size) / 2.0
    layers: list[LayerSpec] = []
    shape = "star"
    width = height = 0.82 * float(size)
    rotation = 0.0
    index = 0

    while min(width, height) >= min_dimension:
        layers.append(
            LayerSpec(
                shape=shape,
                center_x=center,
                center_y=center,
                width=width,
                height=height,
                rotation_deg=rotation,
                color=colors[index % len(colors)],
            )
        )
        index += 1

        if shape == "star":
            next_shape = "square"
            next_width = next_height = 0.41 * width
            rotation = 0.0
            if next_width < min_dimension:
                break
        elif shape == "square":
            next_shape = "oval"
            desired_h = 0.66 * height
            desired_w = 1.28 * desired_h
            if desired_h < min_dimension:
                desired_h = float(min_dimension)
                desired_w = 1.28 * desired_h
            if desired_h > height or desired_w > width:
                break
            next_width = desired_w
            next_height = desired_h
            rotation = 0.0
        elif shape == "oval":
            next_shape = "star"
            next_width = next_height = 0.78 * min(width, height)
            rotation = 18.0
            if next_width < min_dimension:
                if 0.78 * min(width, height) >= min_dimension:
                    next_width = next_height = float(min_dimension)
                else:
                    break
        else:
            raise ValueError(f"unsupported shape: {shape}")

        shape = next_shape
        width = next_width
        height = next_height

    return layers


def apply_noise(image: Image.Image, kind: str, amount: float, seed: int) -> Image.Image:
    if kind == "clean":
        return image.copy()

    rng = np.random.default_rng(seed)
    arr = np.asarray(image, dtype=np.float32)
    if kind == "gaussian":
        arr = arr + rng.normal(0.0, amount, size=arr.shape).astype(np.float32)
    elif kind == "saltpepper":
        mask = rng.random(arr.shape[:2])
        salt = mask < amount / 2.0
        pepper = (mask >= amount / 2.0) & (mask < amount)
        arr[salt] = 255.0
        arr[pepper] = 0.0
    else:
        raise ValueError(f"unsupported noise kind: {kind}")

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def make_contact_sheet(
    output_path: Path,
    images: dict[tuple[str, int], Image.Image],
    sizes: list[int],
    palettes: list[str],
    title: str,
    thumb_size: int = 320,
) -> None:
    font = ImageFont.load_default()
    label_h = 28
    title_h = 42
    gutter = 16
    left_w = 180
    cell_w = thumb_size
    cell_h = thumb_size + label_h
    width = left_w + len(sizes) * cell_w + (len(sizes) + 1) * gutter
    height = title_h + len(palettes) * cell_h + (len(palettes) + 1) * gutter
    sheet = Image.new("RGB", (width, height), hex_to_rgb(BRAND["black"]))
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, 12), title, fill=hex_to_rgb(BRAND["white"]), font=font)

    for col, size in enumerate(sizes):
        x = left_w + gutter + col * (cell_w + gutter)
        draw.text((x, title_h - 20), f"{size} x {size}", fill=hex_to_rgb(BRAND["black10"]), font=font)

    for row, palette in enumerate(palettes):
        y = title_h + gutter + row * (cell_h + gutter)
        draw.text((gutter, y + thumb_size // 2 - 6), palette, fill=hex_to_rgb(BRAND["white"]), font=font)
        for col, size in enumerate(sizes):
            x = left_w + gutter + col * (cell_w + gutter)
            thumb = images[(palette, size)].resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
            sheet.paste(thumb, (x, y))
            draw.text((x, y + thumb_size + 8), palette, fill=hex_to_rgb(BRAND["black10"]), font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("example_images/synthetic_nested_shapes"))
    parser.add_argument("--sizes", type=parse_sizes, default=[1024, 2048, 4096])
    parser.add_argument("--min-dimension", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--write-full-noise", action="store_true")
    args = parser.parse_args()

    output_dir: Path = args.output_dir
    clean_dir = output_dir / "clean"
    contact_dir = output_dir / "contact_sheets"
    clean_images: dict[tuple[str, int], Image.Image] = {}
    manifest: dict[str, object] = {
        "sizes": args.sizes,
        "min_dimension_px": args.min_dimension,
        "palettes": PALETTES,
        "noise_previews": NOISE_PREVIEWS,
        "images": [],
    }

    palette_names = list(PALETTES)
    for size in args.sizes:
        for palette_name in palette_names:
            image, layers = render_layers(size, palette_name, args.min_dimension)
            clean_images[(palette_name, size)] = image
            clean_path = clean_dir / str(size) / f"nested_star_square_oval_{palette_name}_{size}.png"
            clean_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(clean_path)
            manifest["images"].append(
                {
                    "path": str(clean_path),
                    "size": size,
                    "palette": palette_name,
                    "layers": [asdict(layer) for layer in layers],
                }
            )

    for noise_name, config in NOISE_PREVIEWS.items():
        preview_images: dict[tuple[str, int], Image.Image] = {}
        for size in args.sizes:
            for palette_index, palette_name in enumerate(palette_names):
                image = clean_images[(palette_name, size)]
                noisy = apply_noise(
                    image,
                    config["kind"],
                    float(config["amount"]),
                    seed=args.seed + size + palette_index * 97 + len(noise_name) * 13,
                )
                preview_images[(palette_name, size)] = noisy
                if args.write_full_noise and noise_name != "clean":
                    noisy_path = (
                        output_dir
                        / "noisy"
                        / noise_name
                        / str(size)
                        / f"nested_star_square_oval_{palette_name}_{size}_{noise_name}.png"
                    )
                    noisy_path.parent.mkdir(parents=True, exist_ok=True)
                    noisy.save(noisy_path)

        make_contact_sheet(
            contact_dir / f"contact_{noise_name}.png",
            preview_images,
            args.sizes,
            palette_names,
            title=f"Nested star / square / oval synthetic set. {noise_name}",
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
