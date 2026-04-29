"""Circular GMM fusion backends."""

from edgecritic.cgmm._metal import (
    CGMMResult,
    cgmm_backend_available,
    cgmm_fuse_two_pass_metal,
)

__all__ = [
    "CGMMResult",
    "cgmm_backend_available",
    "cgmm_fuse_two_pass_metal",
]
