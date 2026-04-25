"""Image-domain extraction for the NMS/GMM edge pipeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def as_float_image(image: np.ndarray) -> np.ndarray:
    """Return an image as ``float64`` without changing its layout."""
    arr = np.asarray(image)
    if arr.ndim not in (2, 3):
        raise ValueError("image must be a grayscale or channel-last RGB array")

    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        return arr.astype(np.float64) / float(info.max)

    out = arr.astype(np.float64, copy=False)
    if out.size and np.nanmax(out) > 1.5 and np.nanmin(out) >= 0.0:
        out = out / 255.0
    return out


def _rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc

    hue = np.zeros_like(maxc)
    nonzero = delta > np.finfo(np.float64).eps

    red = nonzero & (maxc == r)
    green = nonzero & (maxc == g)
    blue = nonzero & (maxc == b)

    hue[red] = ((g[red] - b[red]) / delta[red]) % 6.0
    hue[green] = ((b[green] - r[green]) / delta[green]) + 2.0
    hue[blue] = ((r[blue] - g[blue]) / delta[blue]) + 4.0
    hue /= 6.0

    saturation = np.zeros_like(maxc)
    bright = maxc > np.finfo(np.float64).eps
    saturation[bright] = delta[bright] / maxc[bright]

    value = maxc
    return hue, saturation, value


def normalize_domain_names(domains: str | Sequence[str], image_ndim: int) -> tuple[str, ...]:
    """Normalize a domain selector into concrete channel names."""
    if isinstance(domains, str):
        if domains == "auto":
            return ("gray",) if image_ndim == 2 else ("gray", "red", "green", "blue", "hue")
        if domains == "rgb":
            return ("red", "green", "blue")
        return tuple(part.strip().lower() for part in domains.split(",") if part.strip())

    return tuple(str(domain).strip().lower() for domain in domains)


def extract_domains(
    image: np.ndarray,
    domains: str | Sequence[str] = "auto",
) -> dict[str, np.ndarray]:
    """Extract grayscale, color, and HSV-derived domains from an image.

    Parameters
    ----------
    image:
        A grayscale ``(H, W)`` image or channel-last ``(H, W, C)`` image.
        Alpha channels are ignored.
    domains:
        ``"auto"``, ``"rgb"``, a comma-separated string, or a sequence of
        names. Supported names are ``gray``, ``red``, ``green``, ``blue``,
        ``hue``, ``saturation``, and ``value``.
    """
    arr = as_float_image(image)
    names = normalize_domain_names(domains, arr.ndim)

    if arr.ndim == 2:
        available = {"gray": arr}
    else:
        if arr.shape[2] < 3:
            raise ValueError("channel-last images must have at least three channels")
        rgb = arr[..., :3]
        hue, saturation, value = _rgb_to_hsv(rgb)
        gray = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        available = {
            "gray": gray,
            "red": rgb[..., 0],
            "green": rgb[..., 1],
            "blue": rgb[..., 2],
            "hue": hue,
            "saturation": saturation,
            "value": value,
        }

    missing = [name for name in names if name not in available]
    if missing:
        supported = ", ".join(sorted(available))
        raise ValueError(f"unsupported domain(s): {missing}; supported domains: {supported}")

    return {name: available[name].astype(np.float64, copy=False) for name in names}
