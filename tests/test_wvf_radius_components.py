import inspect

import numpy as np
import pytest

from edgecritic.wvf import (
    build_wvf_radius_kernels,
    disk_offsets,
    wvf_component_gradients,
)
from edgecritic.wvf._metal import metal_backend_available


def test_disk_offsets_are_radius_defined():
    offsets = disk_offsets(2)

    assert offsets.shape == (12, 2)
    assert not np.any(np.all(offsets == 0, axis=1))
    assert np.all(np.sum(offsets * offsets, axis=1) <= 4)


def test_component_api_does_not_accept_np_count():
    signature = inspect.signature(wvf_component_gradients)

    assert "radius" in signature.parameters
    assert "np_count" not in signature.parameters


def test_radius_kernel_rejects_undersized_support():
    with pytest.raises(ValueError):
        build_wvf_radius_kernels(radius=1, order=2)


def test_radius_component_cpu_detects_vertical_step():
    image = np.zeros((48, 48), dtype=np.float64)
    image[:, 24:] = 1.0

    gx, gy = wvf_component_gradients(image, radius=3, order=2, backend="cpu")

    assert gx.shape == image.shape
    assert gy.shape == image.shape
    assert float(np.max(np.abs(gx))) > 10.0 * float(np.max(np.abs(gy)))


def test_radius_component_metal_matches_cpu_when_available():
    if not metal_backend_available():
        pytest.skip("Metal backend is unavailable")

    rng = np.random.default_rng(123)
    image = rng.normal(size=(40, 44)).astype(np.float64)
    cpu = wvf_component_gradients(image, radius=4, order=2, backend="cpu")
    metal = wvf_component_gradients(image, radius=4, order=2, backend="metal")

    assert np.allclose(metal[0], cpu[0], rtol=2e-5, atol=2e-5)
    assert np.allclose(metal[1], cpu[1], rtol=2e-5, atol=2e-5)
