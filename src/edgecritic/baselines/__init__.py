"""Classical edge detection baselines."""

from edgecritic.baselines.filters import (
    canny_edges,
    compute_edges_at_threshold,
    extended_sobel_gradients,
    prewitt_gradients,
    sobel_gradients,
)

__all__ = [
    "sobel_gradients",
    "prewitt_gradients",
    "extended_sobel_gradients",
    "canny_edges",
    "compute_edges_at_threshold",
]

# GPU-accelerated classical methods (lazy import — requires torch)
def run_all_classical(image, device="cuda"):
    from edgecritic.baselines.classical_gpu import run_all_classical as _run
    return _run(image, device=device)
