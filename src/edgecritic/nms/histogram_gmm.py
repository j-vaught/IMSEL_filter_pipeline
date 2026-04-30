"""Weighted orientation-histogram GMM fusion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianMixtureFit:
    """Two-component weighted Gaussian mixture fit over orientation bins."""

    means: np.ndarray
    sigmas: np.ndarray
    mixing: np.ndarray
    selected: int
    total_weight: float
    iterations: int


@dataclass(frozen=True)
class GMMFusionResult:
    """Fused gradient maps from multi-scale, multi-domain inputs."""

    magnitude: np.ndarray
    angle: np.ndarray
    edge_probability: np.ndarray
    selected_component: np.ndarray
    secondary_magnitude: np.ndarray
    secondary_angle: np.ndarray


def weighted_orientation_histogram(
    angles: np.ndarray,
    magnitudes: np.ndarray,
    n_bins: int = 36,
    period: float = np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a magnitude-weighted axial orientation histogram."""
    theta = np.asarray(angles, dtype=np.float64).ravel()
    weights = np.asarray(magnitudes, dtype=np.float64).ravel()
    if theta.shape != weights.shape:
        raise ValueError("angles and magnitudes must have matching shapes")
    if n_bins < 4:
        raise ValueError("n_bins must be at least 4")

    centers = (np.arange(n_bins, dtype=np.float64) + 0.5) * period / n_bins
    hist = np.zeros(n_bins, dtype=np.float64)

    valid = np.isfinite(theta) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        return centers, hist

    bin_index = np.floor((theta[valid] % period) / period * n_bins).astype(np.int64)
    bin_index = np.clip(bin_index, 0, n_bins - 1)
    np.add.at(hist, bin_index, weights[valid])
    return centers, hist


def _normal_pdf(x: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    sigma = max(float(sigma), np.finfo(np.float64).eps)
    z = (x - mean) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def fit_weighted_two_gaussian(
    centers: np.ndarray,
    weights: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
    min_sigma: float | None = None,
    period: float = np.pi,
    component_selector: str = "mixing",
) -> GaussianMixtureFit:
    """Fit a two-component Gaussian mixture to weighted orientation bins."""
    x0 = np.asarray(centers, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if x0.ndim != 1 or w.ndim != 1 or len(x0) != len(w):
        raise ValueError("centers and weights must be one-dimensional arrays of the same length")

    total = float(np.sum(w))
    if total <= np.finfo(np.float64).eps:
        return GaussianMixtureFit(
            means=np.zeros(2, dtype=np.float64),
            sigmas=np.ones(2, dtype=np.float64),
            mixing=np.zeros(2, dtype=np.float64),
            selected=0,
            total_weight=0.0,
            iterations=0,
        )

    n_bins = max(len(x0), 1)
    bin_width = period / n_bins
    floor_sigma = min_sigma if min_sigma is not None else max(bin_width * 0.5, 1e-4)

    peak_center = float(x0[int(np.argmax(w))])
    delta = ((x0 - peak_center + period / 2.0) % period) - period / 2.0
    x = peak_center + delta

    weighted_mean = float(np.sum(w * x) / total)
    means = np.array([peak_center, weighted_mean], dtype=np.float64)
    sigmas = np.array([max(bin_width, floor_sigma), max(period / 3.0, floor_sigma)], dtype=np.float64)
    mixing = np.array([0.65, 0.35], dtype=np.float64)

    last_ll = -np.inf
    iterations = 0
    for iterations in range(1, max_iter + 1):
        pdf0 = _normal_pdf(x, means[0], sigmas[0])
        pdf1 = _normal_pdf(x, means[1], sigmas[1])
        weighted_pdf0 = mixing[0] * pdf0
        weighted_pdf1 = mixing[1] * pdf1
        denom = weighted_pdf0 + weighted_pdf1 + np.finfo(np.float64).eps

        resp0 = weighted_pdf0 / denom
        resp1 = weighted_pdf1 / denom

        nk0 = float(np.sum(w * resp0))
        nk1 = float(np.sum(w * resp1))
        if nk0 <= np.finfo(np.float64).eps or nk1 <= np.finfo(np.float64).eps:
            break

        mixing = np.array([nk0 / total, nk1 / total], dtype=np.float64)
        means = np.array([
            float(np.sum(w * resp0 * x) / nk0),
            float(np.sum(w * resp1 * x) / nk1),
        ])
        sigmas = np.array([
            np.sqrt(float(np.sum(w * resp0 * (x - means[0]) ** 2) / nk0)),
            np.sqrt(float(np.sum(w * resp1 * (x - means[1]) ** 2) / nk1)),
        ])
        sigmas = np.maximum(sigmas, floor_sigma)

        likelihood = mixing[0] * _normal_pdf(x, means[0], sigmas[0])
        likelihood += mixing[1] * _normal_pdf(x, means[1], sigmas[1])
        ll = float(np.sum(w * np.log(likelihood + np.finfo(np.float64).eps)))
        if abs(ll - last_ll) < tol:
            break
        last_ll = ll

    if component_selector == "sharpness":
        selected = int(np.argmax(mixing / sigmas))
    elif component_selector == "mixing":
        selected = int(np.argmax(mixing))
    else:
        raise ValueError("component_selector must be 'mixing' or 'sharpness'")

    return GaussianMixtureFit(
        means=means % period,
        sigmas=sigmas,
        mixing=mixing,
        selected=selected,
        total_weight=total,
        iterations=iterations,
    )


def _top_two(
    magnitudes: np.ndarray,
    angles: np.ndarray,
) -> tuple[float, float, float, float]:
    if len(magnitudes) == 0:
        return 0.0, 0.0, 0.0, 0.0

    order = np.argsort(magnitudes)
    top = int(order[-1])
    if len(order) == 1:
        return float(magnitudes[top]), float(angles[top]), 0.0, 0.0
    second = int(order[-2])
    return (
        float(magnitudes[top]),
        float(angles[top]),
        float(magnitudes[second]),
        float(angles[second]),
    )


def fuse_gradient_stack_gmm(
    magnitude_stack: np.ndarray,
    angle_stack: np.ndarray,
    n_bins: int = 36,
    max_iter: int = 50,
    component_selector: str = "mixing",
) -> GMMFusionResult:
    """Fuse multi-scale, multi-domain gradients with a two-component GMM."""
    mags = np.asarray(magnitude_stack, dtype=np.float64)
    angles = np.asarray(angle_stack, dtype=np.float64)
    if mags.shape != angles.shape or mags.ndim != 3:
        raise ValueError("magnitude_stack and angle_stack must both have shape (H, W, K)")

    h, w, _ = mags.shape
    fused_mag = np.zeros((h, w), dtype=np.float64)
    fused_angle = np.zeros((h, w), dtype=np.float64)
    edge_probability = np.zeros((h, w), dtype=np.float64)
    selected_component = np.zeros((h, w), dtype=np.int8)
    secondary_magnitude = np.zeros((h, w), dtype=np.float64)
    secondary_angle = np.zeros((h, w), dtype=np.float64)

    for y in range(h):
        for x in range(w):
            pixel_mags = np.nan_to_num(mags[y, x, :], nan=0.0, posinf=0.0, neginf=0.0)
            pixel_angles = np.nan_to_num(angles[y, x, :], nan=0.0, posinf=0.0, neginf=0.0)
            positive = pixel_mags > 0.0
            if not np.any(positive):
                continue

            _, _, second_mag, second_angle = _top_two(pixel_mags, pixel_angles)
            secondary_magnitude[y, x] = second_mag
            secondary_angle[y, x] = second_angle

            centers, hist = weighted_orientation_histogram(
                pixel_angles[positive],
                pixel_mags[positive],
                n_bins=n_bins,
            )
            fit = fit_weighted_two_gaussian(
                centers,
                hist,
                max_iter=max_iter,
                component_selector=component_selector,
            )
            selected = fit.selected
            fused_mag[y, x] = fit.total_weight * fit.mixing[selected]
            fused_angle[y, x] = fit.means[selected]
            edge_probability[y, x] = fit.mixing[selected]
            selected_component[y, x] = selected

    return GMMFusionResult(
        magnitude=fused_mag,
        angle=fused_angle,
        edge_probability=edge_probability,
        selected_component=selected_component,
        secondary_magnitude=secondary_magnitude,
        secondary_angle=secondary_angle,
    )
