"""Core WVF/LF, orientation recovery, c-GMM fusion, and NMS APIs."""

__version__ = "0.1.0"

from edgecritic._types import EdgeResult
from edgecritic.cgmm import (
    cgmm_backend_available,
    cgmm_fuse_two_pass,
    cgmm_fuse_two_pass_metal,
    theta_M_to_phi_w,
)
from edgecritic.lf import lf_image
from edgecritic.lf import lf_length_stack, lf_response, lf_response_batch, lf_stack
from edgecritic.nms import NMSConfig, detect_edges, enhanced_nonmax_suppression
from edgecritic.pipeline import pipeline_backend_available, wvf_lf_recover_metal
from edgecritic.orientation import (
    find_two_peaks,
    recover_two_peaks_metal,
    recovery_backend_available,
)
from edgecritic.wvf import build_wvf_radius_kernels, wvf_component_gradients, wvf_image

__all__ = [
    "EdgeResult",
    "NMSConfig",
    "build_wvf_radius_kernels",
    "cgmm_backend_available",
    "cgmm_fuse_two_pass",
    "cgmm_fuse_two_pass_metal",
    "detect_edges",
    "enhanced_nonmax_suppression",
    "find_two_peaks",
    "lf_image",
    "lf_length_stack",
    "lf_response",
    "lf_response_batch",
    "lf_stack",
    "pipeline_backend_available",
    "recover_two_peaks_metal",
    "recovery_backend_available",
    "theta_M_to_phi_w",
    "wvf_component_gradients",
    "wvf_image",
    "wvf_lf_recover_metal",
]
