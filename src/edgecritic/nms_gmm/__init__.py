"""NMS/GMM multi-scale, multi-domain edge detection.

This package implements the paper pipeline around LF/WVF gradients:
periodic spline orientation estimation, weighted orientation-histogram
GMM fusion, enhanced non-maximum suppression, and hysteresis thresholding.
"""

from edgecritic.nms_gmm._domains import as_float_image, extract_domains
from edgecritic.nms_gmm._filters import (
    LineFilterKernels,
    LineFilterResponses,
    build_line_filter_kernels,
    line_filter_response_stack,
)
from edgecritic.nms_gmm._gmm import (
    GMMFusionResult,
    GaussianMixtureFit,
    fit_weighted_two_gaussian,
    fuse_gradient_stack_gmm,
    weighted_orientation_histogram,
)
from edgecritic.nms_gmm._nms import (
    automatic_hysteresis_thresholds,
    enhanced_nonmax_suppression,
    hysteresis_threshold,
)
from edgecritic.nms_gmm._pipeline import (
    GradientStackResult,
    NMSGMMConfig,
    NMSGMMResult,
    compute_gradient_stack,
    detect_edges,
    nms_gmm_edges,
)
from edgecritic.nms_gmm._spline import (
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
    "NMSGMMConfig",
    "NMSGMMResult",
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
    "nms_gmm_edges",
    "spline_orientation",
    "spline_orientation_map",
    "weighted_orientation_histogram",
]
