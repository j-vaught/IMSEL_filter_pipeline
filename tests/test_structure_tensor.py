import numpy as np

from edgecritic.structure_tensor import (
    disk_kernel,
    structure_tensor_orientation,
    wvf_component_gradients,
)
from edgecritic.synthetic import create_step_edge_image


def _axial_error_deg(a, b):
    diff = (a - b + np.pi / 2.0) % np.pi - np.pi / 2.0
    return np.abs(np.rad2deg(diff))


def test_disk_kernel_is_normalized_and_square():
    kernel = disk_kernel(4)

    assert kernel.shape == (9, 9)
    assert np.isclose(kernel.sum(), 1.0)
    assert np.all(kernel >= 0.0)


def test_wvf_component_gradients_return_matching_fields():
    image = np.zeros((32, 32), dtype=np.float64)
    image[:, 16:] = 1.0

    gx, gy = wvf_component_gradients(image, np_count=8, order=2)

    assert gx.shape == image.shape
    assert gy.shape == image.shape
    assert np.max(np.abs(gx)) > np.max(np.abs(gy))


def test_structure_tensor_orientation_matches_clean_step_edge_normal():
    image, _, true_angle = create_step_edge_image(
        size=64,
        edge_angle_deg=35,
        snr=0,
        high_val=1.0,
        low_val=0.0,
    )

    result = structure_tensor_orientation(
        image,
        radius=5,
        weight_type="gaussian",
        np_count=15,
        order=2,
    )
    gate = result.magnitude_b > 0.5 * np.max(result.magnitude_b)
    error = _axial_error_deg(result.orientation[gate], np.deg2rad(true_angle))

    assert float(np.median(error)) < 3.0
    assert np.all(result.coherence >= -1e-12)
    assert np.all(result.coherence <= 1.0 + 1e-12)
