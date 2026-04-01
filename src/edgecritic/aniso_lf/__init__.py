"""Anisotropic Line Filter: fused stencil reformulation.

Replaces the (2m+1)-virtual-point iteration with a single precomputed
stencil per orientation, applied via conv2d or a custom Triton kernel.
"""

from edgecritic.aniso_lf._stencil import build_anisotropic_stencils
from edgecritic.aniso_lf._conv2d import aniso_lf_conv2d
from edgecritic.aniso_lf._triton import aniso_lf_triton

__all__ = [
    "build_anisotropic_stencils",
    "aniso_lf_conv2d",
    "aniso_lf_triton",
]
