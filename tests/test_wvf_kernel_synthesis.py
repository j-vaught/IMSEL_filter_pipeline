from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from core.taylor import build_taylor_matrix
from wvf.radius import build_wvf_radius_kernels, disk_offsets


TEST_CASES = ((5, 3), (9, 3), (15, 5))


def _legacy_kernel_weights(
    radius: int,
    order: int,
    normalize_coords: bool,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = disk_offsets(radius, include_center=False)
    design = build_taylor_matrix(
        offsets,
        order=order,
        normalize_radius=radius if normalize_coords else None,
    )
    pinv = np.linalg.inv(design.T @ design) @ design.T
    derivative_scale = 1.0 / float(radius) if normalize_coords else 1.0
    weights_x = np.asarray(pinv[1, :] * derivative_scale, dtype=np.float64)
    weights_y = np.asarray(pinv[2, :] * derivative_scale, dtype=np.float64)
    return weights_x, weights_y


class KernelSynthesisParityTests(unittest.TestCase):
    def test_solver_parity_matches_legacy_in_easy_regime(self) -> None:
        for radius, order in TEST_CASES:
            with self.subTest(radius=radius, order=order):
                current = build_wvf_radius_kernels(radius, order, normalize_coords=False)
                legacy_x, legacy_y = _legacy_kernel_weights(radius, order, False)
                self.assertLessEqual(
                    float(np.max(np.abs(current.weights_x - legacy_x))),
                    1.0e-6,
                )
                self.assertLessEqual(
                    float(np.max(np.abs(current.weights_y - legacy_y))),
                    1.0e-6,
                )

    def test_normalized_and_unnormalized_match_in_easy_regime(self) -> None:
        for radius, order in TEST_CASES:
            with self.subTest(radius=radius, order=order):
                raw = build_wvf_radius_kernels(radius, order, normalize_coords=False)
                normalized = build_wvf_radius_kernels(radius, order, normalize_coords=True)
                self.assertLessEqual(
                    float(np.max(np.abs(raw.weights_x - normalized.weights_x))),
                    1.0e-6,
                )
                self.assertLessEqual(
                    float(np.max(np.abs(raw.weights_y - normalized.weights_y))),
                    1.0e-6,
                )


if __name__ == "__main__":
    unittest.main()
