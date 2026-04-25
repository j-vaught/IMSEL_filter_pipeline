import numpy as np

from edgecritic.nms_gmm import (
    NMSGMMConfig,
    detect_edges,
    enhanced_nonmax_suppression,
    extract_domains,
    fit_weighted_two_gaussian,
    link_short_gaps,
    remove_small_components,
    weighted_orientation_histogram,
)


def test_extract_domains_rgb_auto_includes_gray_and_hue():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[..., 0] = 255

    domains = extract_domains(image, "auto")

    assert set(domains) == {"gray", "red", "green", "blue", "hue"}
    assert domains["red"].max() == 1.0
    assert domains["gray"].shape == (4, 5)


def test_weighted_gmm_selects_dominant_orientation():
    angles = np.deg2rad(np.array([84.0, 88.0, 91.0, 95.0, 20.0, 145.0]))
    magnitudes = np.array([4.0, 7.0, 8.0, 5.0, 0.4, 0.3])

    centers, hist = weighted_orientation_histogram(angles, magnitudes, n_bins=36)
    fit = fit_weighted_two_gaussian(centers, hist, component_selector="mixing")

    selected_mean = fit.means[fit.selected]
    assert abs(np.rad2deg(selected_mean) - 90.0) < 8.0
    assert fit.mixing[fit.selected] > 0.5


def test_enhanced_nms_keeps_local_maximum_and_suppresses_neighbor():
    magnitude = np.zeros((7, 7), dtype=np.float64)
    magnitude[3, 2] = 1.0
    magnitude[3, 3] = 5.0
    magnitude[3, 4] = 2.0
    angle = np.zeros_like(magnitude)

    nms = enhanced_nonmax_suppression(magnitude, angle, n_directions=8)

    assert nms[3, 3] == 5.0
    assert nms[3, 2] == 0.0
    assert nms[3, 4] == 0.0


def test_detect_edges_smoke_on_step_image():
    image = np.zeros((28, 28), dtype=np.float64)
    image[:, 14:] = 1.0
    config = NMSGMMConfig(
        half_widths=(1,),
        np_count=8,
        order=2,
        n_orientations=8,
        domains=("gray",),
        spline_range_threshold_rel=0.0,
        refine_spline=False,
        gmm_bins=16,
        high_quantile=0.75,
    )

    result = detect_edges(image, config=config, return_stack=True)

    assert result.edges.shape == image.shape
    assert result.nms.max() > 0.0
    assert result.magnitude_stack is not None
    assert result.angle_stack is not None
    assert result.edges[:, 12:16].sum() > 0


def test_postprocessing_removes_small_components_and_links_gap():
    edges = np.zeros((9, 11), dtype=bool)
    edges[4, 2:5] = True
    edges[4, 7:10] = True
    edges[1, 1] = True
    angle = np.full(edges.shape, np.pi / 2.0)
    candidate = np.zeros_like(edges)
    candidate[4, 2:10] = True
    candidate[1, 1] = True

    linked = link_short_gaps(edges, angle, candidate=candidate, max_gap=3)
    cleaned = remove_small_components(linked, min_size=4)

    assert linked[4, 5]
    assert linked[4, 6]
    assert not cleaned[1, 1]
