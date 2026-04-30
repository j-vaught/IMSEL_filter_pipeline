"""Result types for edge detection outputs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class EdgeResult:
    """Unified result type for all edge detection methods.

    Supports tuple unpacking for backward compatibility:
        mag, angle, extra = wvf_image(...)
    """

    gradient_mag: np.ndarray
    """Gradient magnitude, shape (H, W)."""

    gradient_angle: np.ndarray
    """Gradient angle in radians, shape (H, W)."""

    condition_numbers: np.ndarray | None = None
    """Condition number at each pixel, shape (H, W). None if not computed."""

    all_gradients: np.ndarray | None = None
    """Per-orientation gradient magnitudes, shape (H, W, N_orient). None if not computed."""

    backend: str = "cpu"
    """Backend used: 'cpu' or 'cuda'."""

    def __iter__(self):
        """Support legacy tuple unpacking: mag, angle, extra = result."""
        yield self.gradient_mag
        yield self.gradient_angle
        if self.condition_numbers is not None:
            yield self.condition_numbers
        elif self.all_gradients is not None:
            yield self.all_gradients
        else:
            yield np.zeros_like(self.gradient_mag)
