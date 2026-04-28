import numpy as np
import pytest

from edgecritic.lf._metal import (
    lf_orientation_length_stack_metal,
    lf_orientation_stack_metal,
    lf_response_metal_batch,
    metal_backend_available,
)


def _require_metal() -> None:
    if not metal_backend_available():
        pytest.skip("Metal backend is unavailable")


def _reference_orientation_stack(g_x, g_y, m: int, n_orientations: int) -> np.ndarray:
    gx = np.asarray(g_x, dtype=np.float32)
    gy = np.asarray(g_y, dtype=np.float32)
    h, w = gx.shape
    out = np.empty((n_orientations, h, w), dtype=np.float64)
    yy, xx = np.indices((h, w))

    for theta_idx, theta in enumerate(np.linspace(0.0, np.pi, n_orientations, endpoint=False)):
        cos_t = float(np.cos(theta))
        sin_t = float(np.sin(theta))
        if m <= 0:
            out[theta_idx] = np.abs(-sin_t * gx + cos_t * gy)
            continue

        max_trig = max(abs(cos_t), abs(sin_t))
        step = 1.0 / max_trig if max_trig > 0 else 1.0
        j_offsets = np.arange(-m, m + 1)
        weights = np.exp(-0.5 * (j_offsets / (m / 2.0)) ** 2)
        ix = np.round(j_offsets * step * cos_t).astype(np.int32)
        iy = np.round(j_offsets * step * sin_t).astype(np.int32)
        num = np.zeros((h, w), dtype=np.float64)
        den = np.zeros((h, w), dtype=np.float64)

        for dx, dy, weight in zip(ix, iy, weights):
            sample_x = xx + dx
            sample_y = yy + dy
            valid = (sample_x >= 0) & (sample_x < w) & (sample_y >= 0) & (sample_y < h)
            safe_x = np.clip(sample_x, 0, w - 1)
            safe_y = np.clip(sample_y, 0, h - 1)
            sample = -sin_t * gx[safe_y, safe_x] + cos_t * gy[safe_y, safe_x]
            num[valid] += sample[valid] * weight
            den[valid] += weight

        out[theta_idx] = np.abs(num / np.maximum(den, 1e-12))

    return out


@pytest.mark.parametrize("m,n_orientations", [(0, 1), (1, 2), (3, 5), (6, 8)])
def test_lf_orientation_stack_metal_matches_reference(m, n_orientations):
    _require_metal()

    rng = np.random.default_rng(5000 + m * 31 + n_orientations)
    g_x = rng.normal(size=(17, 19)).astype(np.float32)
    g_y = rng.normal(size=(17, 19)).astype(np.float32)

    got = lf_orientation_stack_metal(
        g_x, g_y, m=m, n_orientations=n_orientations, method="exact"
    )
    expected = _reference_orientation_stack(g_x, g_y, m=m, n_orientations=n_orientations)

    assert got.dtype == np.float32
    assert got.shape == expected.shape
    assert np.allclose(got, expected, rtol=2e-5, atol=3e-5)


def test_lf_orientation_stack_metal_constant_field_boundary_normalization():
    _require_metal()

    n_orientations = 16
    g_x = np.zeros((13, 17), dtype=np.float32)
    g_y = np.ones((13, 17), dtype=np.float32)

    got = lf_orientation_stack_metal(
        g_x, g_y, m=8, n_orientations=n_orientations, method="exact"
    )
    expected_values = np.abs(np.cos(np.linspace(0.0, np.pi, n_orientations, endpoint=False)))

    assert np.allclose(got, expected_values[:, None, None], rtol=2e-5, atol=2e-5)


def test_lf_orientation_stack_metal_matches_sparse_batch_all_pixels():
    _require_metal()

    rng = np.random.default_rng(987)
    h, w = 11, 14
    n_orientations = 7
    m = 5
    g_x = rng.normal(size=(h, w)).astype(np.float32)
    g_y = rng.normal(size=(h, w)).astype(np.float32)
    yy, xx = np.indices((h, w))
    px = xx.reshape(-1).astype(np.int32)
    py = yy.reshape(-1).astype(np.int32)
    thetas = np.linspace(0.0, np.pi, n_orientations, endpoint=False)

    stack = lf_orientation_stack_metal(
        g_x, g_y, m=m, n_orientations=n_orientations, method="exact"
    )
    sparse = lf_response_metal_batch(g_x, g_y, px, py, thetas, np.array([m], dtype=np.int32))
    sparse_stack = sparse[:, 0, :].reshape(n_orientations, h, w)

    assert np.allclose(stack, sparse_stack, rtol=2e-5, atol=3e-5)


def test_lf_orientation_stack_projected_matches_direct():
    _require_metal()

    rng = np.random.default_rng(24601)
    g_x = rng.normal(size=(23, 29)).astype(np.float32)
    g_y = rng.normal(size=(23, 29)).astype(np.float32)

    direct = lf_orientation_stack_metal(
        g_x, g_y, m=7, n_orientations=9, method="exact", execution="direct"
    )
    projected = lf_orientation_stack_metal(
        g_x, g_y, m=7, n_orientations=9, method="exact", execution="projected"
    )

    assert np.allclose(projected, direct, rtol=2e-5, atol=3e-5)


def test_lf_orientation_stack_metal_output_dtype_and_validation():
    _require_metal()

    g_x = np.zeros((5, 6), dtype=np.float64)
    g_y = np.ones((5, 6), dtype=np.float64)
    got = lf_orientation_stack_metal(g_x, g_y, m=0, n_orientations=4, output_dtype=np.float64)

    assert got.dtype == np.float64
    assert got.shape == (4, 5, 6)

    reusable = np.empty((4, 5, 6), dtype=np.float32)
    reused = lf_orientation_stack_metal(g_x, g_y, m=0, n_orientations=4, out=reusable)
    assert reused is reusable
    assert reused.dtype == np.float32

    with pytest.raises(ValueError, match="n_orientations"):
        lf_orientation_stack_metal(g_x, g_y, m=0, n_orientations=0)
    with pytest.raises(ValueError, match="method"):
        lf_orientation_stack_metal(g_x, g_y, m=0, n_orientations=4, method="unknown")
    with pytest.raises(ValueError, match="execution"):
        lf_orientation_stack_metal(g_x, g_y, m=0, n_orientations=4, execution="unknown")
    with pytest.raises(ValueError, match="out"):
        lf_orientation_stack_metal(
            g_x,
            g_y,
            m=0,
            n_orientations=4,
            out=np.empty((4, 5, 6), dtype=np.float64),
        )


def test_lf_orientation_stack_box_matches_projection_for_zero_m():
    _require_metal()

    rng = np.random.default_rng(440)
    g_x = rng.normal(size=(17, 21)).astype(np.float32)
    g_y = rng.normal(size=(17, 21)).astype(np.float32)

    exact = lf_orientation_stack_metal(g_x, g_y, m=0, n_orientations=8, method="exact")
    box = lf_orientation_stack_metal(g_x, g_y, m=0, n_orientations=8, method="box")

    assert np.allclose(box, exact, rtol=2e-5, atol=3e-5)


def test_lf_orientation_stack_box_constant_field_boundary_normalization():
    _require_metal()

    n_orientations = 16
    g_x = np.zeros((19, 23), dtype=np.float32)
    g_y = np.ones((19, 23), dtype=np.float32)

    got = lf_orientation_stack_metal(
        g_x, g_y, m=12, n_orientations=n_orientations, method="box", box_passes=6
    )
    expected_values = np.abs(np.cos(np.linspace(0.0, np.pi, n_orientations, endpoint=False)))

    assert np.allclose(got, expected_values[:, None, None], rtol=2e-5, atol=2e-5)


def test_lf_orientation_stack_default_uses_one_pass_box():
    _require_metal()

    rng = np.random.default_rng(452)
    g_x = rng.normal(size=(13, 15)).astype(np.float32)
    g_y = rng.normal(size=(13, 15)).astype(np.float32)

    default = lf_orientation_stack_metal(g_x, g_y, m=6, n_orientations=7)
    box = lf_orientation_stack_metal(
        g_x, g_y, m=6, n_orientations=7, method="box", box_passes=1
    )

    assert np.allclose(default, box, rtol=0.0, atol=0.0)


def test_lf_orientation_length_stack_box_matches_loop():
    _require_metal()

    rng = np.random.default_rng(453)
    g_x = rng.normal(size=(17, 19)).astype(np.float32)
    g_y = rng.normal(size=(17, 19)).astype(np.float32)
    ms = np.array([0, 4, 9], dtype=np.int32)
    n_orientations = 7

    got = lf_orientation_length_stack_metal(
        g_x, g_y, ms=ms, n_orientations=n_orientations
    )

    assert got.dtype == np.float32
    assert got.shape == (n_orientations, ms.size, *g_x.shape)
    for m_idx, m_value in enumerate(ms):
        expected = lf_orientation_stack_metal(
            g_x,
            g_y,
            m=int(m_value),
            n_orientations=n_orientations,
            method="box",
            box_passes=1,
        )
        assert np.allclose(got[:, m_idx], expected, rtol=5e-5, atol=5e-5)


def test_lf_orientation_length_stack_validation_and_reusable_output():
    _require_metal()

    g_x = np.zeros((5, 6), dtype=np.float32)
    g_y = np.ones((5, 6), dtype=np.float32)
    ms = np.array([1, 3], dtype=np.int32)
    out = np.empty((4, 2, 5, 6), dtype=np.float32)

    reused = lf_orientation_length_stack_metal(g_x, g_y, ms=ms, n_orientations=4, out=out)

    assert reused is out
    assert np.isfinite(reused).all()

    with pytest.raises(ValueError, match="method"):
        lf_orientation_length_stack_metal(g_x, g_y, ms=ms, n_orientations=4, method="exact")
    with pytest.raises(ValueError, match="box_passes"):
        lf_orientation_length_stack_metal(g_x, g_y, ms=ms, n_orientations=4, box_passes=2)
    with pytest.raises(ValueError, match="ms"):
        lf_orientation_length_stack_metal(g_x, g_y, ms=np.zeros((1, 2), dtype=np.int32))
    with pytest.raises(ValueError, match="out"):
        lf_orientation_length_stack_metal(
            g_x, g_y, ms=ms, n_orientations=4, out=np.empty((4, 5, 6), dtype=np.float32)
        )


def test_lf_orientation_stack_box_reusable_output_and_validation():
    _require_metal()

    g_x = np.zeros((7, 8), dtype=np.float32)
    g_y = np.ones((7, 8), dtype=np.float32)
    out = np.empty((5, 7, 8), dtype=np.float32)

    reused = lf_orientation_stack_metal(
        g_x, g_y, m=5, n_orientations=5, method="box", out=out, box_passes=3
    )

    assert reused is out
    assert np.isfinite(reused).all()

    with pytest.raises(ValueError, match="execution"):
        lf_orientation_stack_metal(
            g_x, g_y, m=5, n_orientations=5, method="box", execution="projected"
        )
    with pytest.raises(ValueError, match="box_passes"):
        lf_orientation_stack_metal(g_x, g_y, m=5, n_orientations=5, method="box", box_passes=0)
    with pytest.raises(ValueError, match="box_radius"):
        lf_orientation_stack_metal(
            g_x, g_y, m=5, n_orientations=5, method="box", box_radius=-1
        )


def test_lf_orientation_stack_box_tracks_exact_on_smooth_fields():
    _require_metal()

    h, w = 47, 53
    y, x = np.indices((h, w), dtype=np.float32)
    g_x = (np.sin(x * 0.11) + 0.5 * np.cos(y * 0.07)).astype(np.float32)
    g_y = (np.cos(x * 0.05) - 0.4 * np.sin(y * 0.13)).astype(np.float32)

    exact = lf_orientation_stack_metal(
        g_x, g_y, m=10, n_orientations=12, method="exact", execution="projected"
    )
    box = lf_orientation_stack_metal(
        g_x, g_y, m=10, n_orientations=12, method="box", box_passes=6
    )
    diff = box - exact
    rel_rmse = np.sqrt(np.mean(diff * diff)) / max(float(np.sqrt(np.mean(exact * exact))), 1e-12)
    corr = np.corrcoef(exact.ravel(), box.ravel())[0, 1]

    assert rel_rmse < 0.2
    assert corr > 0.95


def test_lf_orientation_stack_scanline_matches_projection_for_zero_m():
    _require_metal()

    rng = np.random.default_rng(441)
    g_x = rng.normal(size=(17, 21)).astype(np.float32)
    g_y = rng.normal(size=(17, 21)).astype(np.float32)

    exact = lf_orientation_stack_metal(g_x, g_y, m=0, n_orientations=8, method="exact")
    scanline = lf_orientation_stack_metal(
        g_x, g_y, m=0, n_orientations=8, method="scanline"
    )

    assert np.allclose(scanline, exact, rtol=2e-5, atol=3e-5)


def test_lf_orientation_stack_scanline_constant_field_boundary_normalization():
    _require_metal()

    n_orientations = 16
    g_x = np.zeros((19, 23), dtype=np.float32)
    g_y = np.ones((19, 23), dtype=np.float32)

    got = lf_orientation_stack_metal(
        g_x, g_y, m=12, n_orientations=n_orientations, method="scanline"
    )
    expected_values = np.abs(np.cos(np.linspace(0.0, np.pi, n_orientations, endpoint=False)))

    assert np.allclose(got, expected_values[:, None, None], rtol=2e-5, atol=2e-5)


def test_lf_orientation_stack_scanline_tracks_exact_on_smooth_fields():
    _require_metal()

    h, w = 47, 53
    y, x = np.indices((h, w), dtype=np.float32)
    g_x = (np.sin(x * 0.11) + 0.5 * np.cos(y * 0.07)).astype(np.float32)
    g_y = (np.cos(x * 0.05) - 0.4 * np.sin(y * 0.13)).astype(np.float32)

    exact = lf_orientation_stack_metal(
        g_x, g_y, m=10, n_orientations=12, method="exact", execution="projected"
    )
    scanline = lf_orientation_stack_metal(
        g_x, g_y, m=10, n_orientations=12, method="scanline"
    )
    diff = scanline - exact
    rel_rmse = np.sqrt(np.mean(diff * diff)) / max(float(np.sqrt(np.mean(exact * exact))), 1e-12)
    corr = np.corrcoef(exact.ravel(), scanline.ravel())[0, 1]

    assert rel_rmse < 0.15
    assert corr > 0.97
