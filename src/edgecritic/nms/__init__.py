"""NMS multi-scale, multi-domain edge detection.

This package implements the paper pipeline around LF/WVF gradients:
periodic spline orientation estimation, weighted orientation-histogram
GMM fusion, enhanced non-maximum suppression, and hysteresis thresholding.
"""

from edgecritic.nms.domains import as_float_image, extract_domains
from edgecritic.nms.filters import (
    LineFilterKernels,
    LineFilterResponses,
    build_line_filter_kernels,
    line_filter_response_stack,
)
from edgecritic.nms.histogram_gmm import (
    GMMFusionResult,
    GaussianMixtureFit,
    fit_weighted_two_gaussian,
    fuse_gradient_stack_gmm,
    weighted_orientation_histogram,
)
from edgecritic.nms.enhanced import (
    automatic_hysteresis_thresholds,
    enhanced_nonmax_suppression,
    hysteresis_threshold,
    link_short_gaps,
    remove_small_components,
)
from edgecritic.nms.pipeline import (
    GradientStackResult,
    NMSConfig,
    NMSResult,
    compute_gradient_stack,
    detect_edges,
    nms_edges,
)
from edgecritic.nms.spline import (
    SplineOrientationResult,
    spline_orientation,
    spline_orientation_map,
)

__all__ = [
    "GMMFusionResult",
    "GaussianMixtureFit",
    "GradientStackResult",
    "LineFilterKernels",
    "LineFilterResponses",
    "NMSConfig",
    "NMSResult",
    "SplineOrientationResult",
    "as_float_image",
    "automatic_hysteresis_thresholds",
    "build_line_filter_kernels",
    "compute_gradient_stack",
    "detect_edges",
    "enhanced_nonmax_suppression",
    "extract_domains",
    "fit_weighted_two_gaussian",
    "fuse_gradient_stack_gmm",
    "hysteresis_threshold",
    "line_filter_response_stack",
    "link_short_gaps",
    "nms_edges",
    "remove_small_components",
    "spline_orientation",
    "spline_orientation_map",
    "weighted_orientation_histogram",
]
