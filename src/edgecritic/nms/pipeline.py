"""Paper-level NMS edge detection pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from edgecritic.nms.domains import extract_domains
from edgecritic.nms.filters import line_filter_response_stack
from edgecritic.nms.histogram_gmm import GMMFusionResult, fuse_gradient_stack_gmm
from edgecritic.nms.enhanced import (
    automatic_hysteresis_thresholds,
    enhanced_nonmax_suppression,
    hysteresis_threshold,
    link_short_gaps,
    remove_small_components,
)
from edgecritic.nms.spline import spline_orientation_map


@dataclass(frozen=True)
class NMSConfig:
    """Configuration for the multi-scale, multi-domain NMS pipeline."""

    half_widths: tuple[int, ...] = (3, 7, 11)
    np_count: int = 15
    order: int = 4
    n_orientations: int = 36
    domains: str | Sequence[str] = "auto"
    neighbor_type: str = "circular"
    spline_range_threshold_rel: float = 0.01
    refine_spline: bool = True
    gmm_bins: int = 36
    gmm_max_iter: int = 50
    gmm_component_selector: str = "mixing"
    nms_directions: int = 8
    preserve_secondary: bool = True
    secondary_ratio: float = 0.85
    high_threshold: float | None = None
    low_threshold: float | None = None
    high_quantile: float = 0.90
    low_ratio: float = 0.40
    link_gaps: bool = False
    max_link_gap: int = 3
    link_candidate_ratio: float = 0.75
    link_iterations: int = 1
    min_component_size: int = 0
    mode: str = "reflect"
    suppress_border: bool = True

    @classmethod
    def aquatic(cls, **overrides) -> "NMSConfig":
        """Preset for low-contrast aquatic scenes with textured water."""
        values = {
            "half_widths": (3, 7, 11),
            "domains": "auto",
            "np_count": 15,
            "order": 4,
            "n_orientations": 36,
            "high_quantile": 0.95,
            "low_ratio": 0.40,
            "link_gaps": True,
            "max_link_gap": 6,
            "link_candidate_ratio": 0.60,
            "link_iterations": 1,
            "min_component_size": 12,
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class GradientStackResult:
    """Multi-scale, multi-domain gradient stack before GMM fusion."""

    magnitude_stack: np.ndarray
    angle_stack: np.ndarray
    labels: tuple[str, ...]
    border: int


@dataclass(frozen=True)
class NMSResult:
    """Complete edge detection result for the NMS pipeline."""

    magnitude: np.ndarray
    angle: np.ndarray
    nms: np.ndarray
    edges: np.ndarray
    edge_probability: np.ndarray
    low_threshold: float
    high_threshold: float
    labels: tuple[str, ...]
    fusion: GMMFusionResult
    magnitude_stack: np.ndarray | None = None
    angle_stack: np.ndarray | None = None


def compute_gradient_stack(
    image: np.ndarray,
    config: NMSConfig | None = None,
) -> GradientStackResult:
    """Compute LF/spline gradients for each selected scale and domain."""
    cfg = config or NMSConfig()
    if not cfg.half_widths:
        raise ValueError("at least one half_width is required")

    domains = extract_domains(image, cfg.domains)
    magnitudes: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    labels: list[str] = []
    max_border = 0

    for domain_name, domain_image in domains.items():
        for half_width in cfg.half_widths:
            responses = line_filter_response_stack(
                domain_image,
                half_width=half_width,
                np_count=cfg.np_count,
                order=cfg.order,
                n_orientations=cfg.n_orientations,
                neighbor_type=cfg.neighbor_type,
                mode=cfg.mode,
                zero_border=cfg.suppress_border,
            )
            orientation = spline_orientation_map(
                responses.responses,
                responses.angles,
                range_threshold_rel=cfg.spline_range_threshold_rel,
                refine=cfg.refine_spline,
            )
            magnitudes.append(orientation.magnitude)
            angles.append(orientation.angle)
            labels.append(f"{domain_name}:lf_m{half_width}")
            max_border = max(max_border, responses.border)

    magnitude_stack = np.stack(magnitudes, axis=-1)
    angle_stack = np.stack(angles, axis=-1)
    return GradientStackResult(
        magnitude_stack=magnitude_stack,
        angle_stack=angle_stack,
        labels=tuple(labels),
        border=max_border,
    )


def _resolve_thresholds(
    nms: np.ndarray,
    config: NMSConfig,
) -> tuple[float, float]:
    if config.high_threshold is None:
        high, low = automatic_hysteresis_thresholds(
            nms,
            high_quantile=config.high_quantile,
            low_ratio=config.low_ratio,
        )
    else:
        high = float(config.high_threshold)
        low = float(config.low_threshold) if config.low_threshold is not None else high * config.low_ratio

    if config.low_threshold is not None:
        low = float(config.low_threshold)
    return high, low


def detect_edges(
    image: np.ndarray,
    config: NMSConfig | None = None,
    return_stack: bool = False,
) -> NMSResult:
    """Run the proposed spline-orientation, GMM-fusion, enhanced-NMS method."""
    cfg = config or NMSConfig()
    stack = compute_gradient_stack(image, cfg)

    fusion = fuse_gradient_stack_gmm(
        stack.magnitude_stack,
        stack.angle_stack,
        n_bins=cfg.gmm_bins,
        max_iter=cfg.gmm_max_iter,
        component_selector=cfg.gmm_component_selector,
    )

    secondary_mag = fusion.secondary_magnitude if cfg.preserve_secondary else None
    secondary_angle = fusion.secondary_angle if cfg.preserve_secondary else None
    nms = enhanced_nonmax_suppression(
        fusion.magnitude,
        fusion.angle,
        n_directions=cfg.nms_directions,
        secondary_magnitude=secondary_mag,
        secondary_angle=secondary_angle,
        secondary_ratio=cfg.secondary_ratio,
    )

    high, low = _resolve_thresholds(nms, cfg)
    edges = hysteresis_threshold(nms, low_threshold=low, high_threshold=high)

    if cfg.link_gaps:
        candidate = nms >= low * float(cfg.link_candidate_ratio)
        edges = link_short_gaps(
            edges,
            fusion.angle,
            candidate=candidate,
            max_gap=cfg.max_link_gap,
            n_directions=cfg.nms_directions,
            iterations=cfg.link_iterations,
        )

    if cfg.min_component_size > 1:
        edges = remove_small_components(edges, min_size=cfg.min_component_size)

    if cfg.suppress_border and stack.border > 0:
        b = stack.border
        nms[:b, :] = 0.0
        nms[-b:, :] = 0.0
        nms[:, :b] = 0.0
        nms[:, -b:] = 0.0
        edges[:b, :] = False
        edges[-b:, :] = False
        edges[:, :b] = False
        edges[:, -b:] = False

    return NMSResult(
        magnitude=fusion.magnitude,
        angle=fusion.angle,
        nms=nms,
        edges=edges,
        edge_probability=fusion.edge_probability,
        low_threshold=low,
        high_threshold=high,
        labels=stack.labels,
        fusion=fusion,
        magnitude_stack=stack.magnitude_stack if return_stack else None,
        angle_stack=stack.angle_stack if return_stack else None,
    )


nms_edges = detect_edges
