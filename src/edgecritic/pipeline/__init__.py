"""Fused processing pipelines."""

from edgecritic.pipeline._metal import (
    pipeline_backend_available,
    wvf_lf_recover_metal,
)

__all__ = [
    "pipeline_backend_available",
    "wvf_lf_recover_metal",
]
