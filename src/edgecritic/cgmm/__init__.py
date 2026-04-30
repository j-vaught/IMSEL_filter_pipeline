"""Multi-channel circular GMM fusion."""

from edgecritic.cgmm.metal import (
    CGMMResult as CGMMMetalResult,
    cgmm_backend_available,
    cgmm_fuse_two_pass_metal,
)
from edgecritic.cgmm.reference import (
    CGMMResult,
    cgmm_em,
    cgmm_fuse_two_pass,
    theta_M_to_phi_w,
)

__all__ = [
    "CGMMResult",
    "CGMMMetalResult",
    "cgmm_backend_available",
    "cgmm_em",
    "cgmm_fuse_two_pass",
    "cgmm_fuse_two_pass_metal",
    "theta_M_to_phi_w",
]
