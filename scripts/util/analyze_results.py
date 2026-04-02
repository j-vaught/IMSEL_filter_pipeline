"""Run all critique experiments and generate the final report.

This script orchestrates:
1. Synthetic image generation
2. WVF/LF vs baseline gradient comparisons
3. Cubic spline vs arctan angle accuracy test
4. Condition number analysis
5. Runtime benchmarking
6. Report generation with figures
"""

import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from edgecritic.wvf import wvf_image
from edgecritic.wvf._cpu import wvf_single_pixel
from edgecritic.lf import lf_image
from edgecritic.core.taylor import build_taylor_matrix, get_circular_neighbors, rotate_coordinates
from edgecritic.angles import cubic_spline_angle, arctan_angle
from edgecritic.baselines import sobel_gradients, prewitt_gradients, canny_edges
from edgecritic.evaluation import (
    compute_ods_ois, angular_error_deg, runtime_comparison,
    analyze_condition_numbers, wvf_orientation_profile,
)
from edgecritic.synthetic import (
    create_multi_angle_line_image, create_step_edge_image,
    create_parallel_line_image,
)


RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def sample_parallel_line_points(size, angle_deg, n_points=7, offset_step=10):
    """Sample points along the center line used by create_parallel_line_image."""
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    cx = cy = size // 2

    points = []
    for offset in range(-(n_points // 2), n_points // 2 + 1):
        px = int(round(cx + offset * offset_step * cos_a))
        py = int(round(cy + offset * offset_step * sin_a))
        points.append((px, py))
    return points


def test_condition_numbers():
    """Test 1: Analyze numerical stability of WVF least-squares system."""
    print("=" * 60)
    print("TEST 1: Condition Number Analysis")
    print("=" * 60)

    np_counts = [15, 25, 50, 100, 150, 250]
    results = analyze_condition_numbers(np_counts, order=4, n_orientations=72)

    fig, ax = plt.subplots(figsize=(10, 6))
    for np_count in np_counts:
        conds = results[np_count]
        angles = np.linspace(0, 360, len(conds))
        ax.semilogy(angles, conds, label=f'Np={np_count}')

    ax.set_xlabel('Orientation (degrees)')
    ax.set_ylabel('Condition Number of A^T A')
    ax.set_title('WVF System Conditioning vs Np and Orientation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(RESULTS_DIR / 'condition_numbers.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    report_lines = ["## Test 1: Condition Number Analysis\n"]
    report_lines.append("| Np | Min Cond | Max Cond | Mean Cond | Ratio Max/Min |")
    report_lines.append("|---:|--------:|---------:|----------:|--------------:|")
    for np_count in np_counts:
        c = results[np_count]
        report_lines.append(
            f"| {np_count} | {c.min():.2e} | {c.max():.2e} | "
            f"{c.mean():.2e} | {c.max()/c.min():.2f} |"
        )

    report_lines.append("")
    report_lines.append("**Finding:** " + (
        "Condition numbers stay well below catastrophic failure, but they grow "
        "substantially with large Np and reach the 1e5 range at Np=250. "
        "That is compatible with a solvable double-precision system, yet it is "
        "large enough to merit discussion as a numerical-stability concern."
    ))

    for line in report_lines:
        print(line)
    return "\n".join(report_lines)


def test_angle_accuracy():
    """Test 2: Cubic spline vs arctan angle detection."""
    print("\n" + "=" * 60)
    print("TEST 2: Angle Detection Accuracy (Spline vs Arctan)")
    print("=" * 60)
    np.random.seed(1234)

    report_lines = ["## Test 2: Angle Detection Accuracy\n"]
    test_angles = [0, 23, 63.5, 90, 135, 174]
    snr_levels = [2.0, 1.0, 0.75]
    size = 128

    representative_img, representative_clean, _ = create_multi_angle_line_image(
        size=256, angles_deg=test_angles, snr=2.0
    )
    np.save(RESULTS_DIR / 'angle_test_image.npy', representative_img)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(representative_clean, cmap='gray')
    axes[0].set_title('Clean Multi-Angle Image')
    axes[1].imshow(representative_img, cmap='gray')
    axes[1].set_title('Noisy Multi-Angle Image (SNR=2.0)')
    fig.savefig(RESULTS_DIR / 'angle_test_images.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    per_snr = []
    detailed_rows = []

    print("Computing direct spline-vs-arctan comparisons...")
    start = time.time()
    for snr in snr_levels:
        sobel_errors = []
        spline_errors = []
        improvements = []

        for angle_deg in test_angles:
            img, _, true_angle = create_parallel_line_image(
                size=size, n_lines=1, spacing=30, angle_deg=angle_deg, snr=snr
            )
            _, _, sobel_mag, sobel_angle = sobel_gradients(img)
            sample_points = sample_parallel_line_points(size=size, angle_deg=angle_deg, n_points=7)

            point_sobel_errors = []
            point_spline_errors = []
            for px, py in sample_points:
                y0 = max(py - 1, 0)
                y1 = min(py + 2, size)
                x0 = max(px - 1, 0)
                x1 = min(px + 2, size)
                local_mag = sobel_mag[y0:y1, x0:x1]
                local_idx = np.argmax(local_mag)
                ly, lx = np.unravel_index(local_idx, local_mag.shape)
                best_x = x0 + lx
                best_y = y0 + ly

                sobel_pred_deg = np.degrees(sobel_angle[best_y, best_x])
                sobel_err = angular_error_deg(sobel_pred_deg, true_angle)
                point_sobel_errors.append(sobel_err)

                angles_rad, profile = wvf_orientation_profile(
                    img, best_x, best_y, np_count=15, order=4, n_orientations=18
                )
                spline_angle_val, _, _ = cubic_spline_angle(profile, angles_rad)
                spline_pred_deg = np.degrees(spline_angle_val)
                spline_err = angular_error_deg(spline_pred_deg, true_angle)
                point_spline_errors.append(spline_err)

            sobel_med = float(np.median(point_sobel_errors))
            spline_med = float(np.median(point_spline_errors))
            sobel_errors.append(sobel_med)
            spline_errors.append(spline_med)
            improvements.append(sobel_med - spline_med)

            if snr == 2.0:
                detailed_rows.append((true_angle, sobel_med, spline_med, sobel_med - spline_med))
            print(
                f"  SNR={snr:.2f} angle={true_angle:.1f}: "
                f"Sobel={sobel_med:.3f}, Spline={spline_med:.3f}"
            )

        per_snr.append({
            "snr": snr,
            "sobel_mean": float(np.mean(sobel_errors)),
            "spline_mean": float(np.mean(spline_errors)),
            "improvement_mean": float(np.mean(improvements)),
            "sobel_max": float(np.max(sobel_errors)),
            "spline_max": float(np.max(spline_errors)),
        })

    elapsed = time.time() - start
    print(f"  Completed direct angle comparison in {elapsed:.1f}s")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(detailed_rows))
    width = 0.35
    ax.bar(x - width/2, [row[1] for row in detailed_rows], width, label='Sobel arctan')
    ax.bar(x + width/2, [row[2] for row in detailed_rows], width, label='WVF + spline')
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row[0]:.1f}" for row in detailed_rows], rotation=45, ha='right')
    ax.set_ylabel('Median Angular Error (degrees)')
    ax.set_title('Angle Error by True Edge Normal (SNR=2.0, Np=15)')
    ax.grid(axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'angle_error_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    report_lines.append("### Direct WVF Spline vs Sobel Arctan Comparison")
    report_lines.append("| True Normal Angle | Sobel Median Error | WVF+Spline Median Error | Improvement |")
    report_lines.append("|-----------------:|-------------------:|------------------------:|------------:|")
    for true_angle, sobel_err, spline_err, improvement in detailed_rows:
        report_lines.append(
            f"| {true_angle:.1f} | {sobel_err:.3f} | {spline_err:.3f} | {improvement:.3f} |"
        )

    report_lines.append("")
    report_lines.append("### Error Summary by SNR")
    report_lines.append("| SNR | Mean Sobel Error | Mean WVF+Spline Error | Mean Improvement | Max Sobel Error | Max WVF+Spline Error |")
    report_lines.append("|---:|------------------:|----------------------:|-----------------:|----------------:|---------------------:|")
    for row in per_snr:
        report_lines.append(
            f"| {row['snr']:.2f} | {row['sobel_mean']:.3f} | {row['spline_mean']:.3f} | "
            f"{row['improvement_mean']:.3f} | {row['sobel_max']:.3f} | {row['spline_max']:.3f} |"
        )

    report_lines.append("")
    report_lines.append("### Key Finding on Angle Accuracy")
    report_lines.append(
        "Unlike the earlier draft, this test directly evaluates the spline estimator. "
        "At the tractable setting used here (Np=15, 18 orientations), we do not reproduce "
        "the paper's claimed angular advantage: WVF+Spline is usually worse than Sobel arctan "
        "on this test and does not support a sub-degree accuracy claim. That may reflect the "
        "reduced Np/orientation setting, but it means the strong angle-accuracy claim should "
        "not be presented as independently verified in this repo."
    )

    for line in report_lines:
        print(line)
    return "\n".join(report_lines)


def test_snr_robustness():
    """Test 3: Edge detection at different SNR levels."""
    print("\n" + "=" * 60)
    print("TEST 3: SNR Robustness")
    print("=" * 60)
    np.random.seed(2345)

    snr_levels = [0.5, 0.75, 1.0, 2.0]
    fig, axes = plt.subplots(len(snr_levels), 4, figsize=(16, 4 * len(snr_levels)))

    report_lines = ["## Test 3: SNR Robustness\n"]
    report_lines.append("| SNR | Sobel Edge Pixels | Prewitt Edge Pixels | Canny Edge Pixels |")
    report_lines.append("|----:|------------------:|--------------------:|------------------:|")

    for si, snr in enumerate(snr_levels):
        img, clean, angle_map = create_multi_angle_line_image(
            size=256, snr=snr
        )

        _, _, sobel_mag, _ = sobel_gradients(img)
        _, _, prewitt_mag, _ = prewitt_gradients(img)
        canny = canny_edges(img, sigma=1.0)

        sobel_thresh = np.percentile(sobel_mag, 95)
        prewitt_thresh = np.percentile(prewitt_mag, 95)
        sobel_edges = sobel_mag > sobel_thresh
        prewitt_edges = prewitt_mag > prewitt_thresh

        axes[si, 0].imshow(img, cmap='gray')
        axes[si, 0].set_title(f'SNR={snr}')
        axes[si, 0].axis('off')

        axes[si, 1].imshow(sobel_edges, cmap='gray')
        axes[si, 1].set_title('Sobel Edges')
        axes[si, 1].axis('off')

        axes[si, 2].imshow(prewitt_edges, cmap='gray')
        axes[si, 2].set_title('Prewitt Edges')
        axes[si, 2].axis('off')

        axes[si, 3].imshow(canny, cmap='gray')
        axes[si, 3].set_title('Canny Edges')
        axes[si, 3].axis('off')

        report_lines.append(
            f"| {snr} | {np.sum(sobel_edges)} | {np.sum(prewitt_edges)} | {np.sum(canny)} |"
        )
        print(f"  SNR={snr}: Sobel={np.sum(sobel_edges)}, "
              f"Prewitt={np.sum(prewitt_edges)}, Canny={np.sum(canny)}")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / 'snr_comparison.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    report_lines.append("")
    report_lines.append(
        "**Finding:** At low SNR (0.5-0.75), Sobel and Prewitt produce many false edges "
        "due to noise amplification, consistent with Bagan's critique. The WVF's larger "
        "neighborhood averaging should theoretically suppress this noise. However, we note "
        "that simply using a larger Gaussian pre-filter with Sobel would also achieve this "
        "effect -- the WVF's advantage is specifically in combining noise suppression with "
        "orientation-specific gradient computation."
    )

    for line in report_lines:
        print(line)
    return "\n".join(report_lines)


def test_runtime_scaling():
    """Test 4: Runtime comparison."""
    print("\n" + "=" * 60)
    print("TEST 4: Runtime Comparison")
    print("=" * 60)
    np.random.seed(3456)

    report_lines = ["## Test 4: Runtime Comparison\n"]
    same_region = np.random.rand(32, 32) * 255
    same_region_results = runtime_comparison(same_region, n_runs=10)
    region_pixels = same_region.size

    report_lines.append("### Same-Region CPU Timing (32x32 Input for Every Method)")
    report_lines.append("| Method | Mean Time (s) | Std Time (s) | us / input pixel |")
    report_lines.append("|--------|-------------:|-------------:|-----------------:|")

    for name, data in same_region_results.items():
        us_per_pixel = data['mean_time'] * 1e6 / region_pixels
        report_lines.append(
            f"| {name} | {data['mean_time']:.6f} | {data['std_time']:.6f} | {us_per_pixel:.2f} |"
        )
        print(f"  32x32 {name}: {data['mean_time']:.6f}s ({us_per_pixel:.2f} us/pixel)")

    wvf_start = time.time()
    wvf_image(same_region, np_count=15, order=4, n_orientations=18, backend="cpu")
    wvf_time = time.time() - wvf_start

    lf_start = time.time()
    lf_image(
        same_region, half_width=3, np_count=15,
        order=4, n_orientations=18, subsample=2, backend="cpu"
    )
    lf_time = time.time() - lf_start

    report_lines.append(
        f"| WVF (Np=15, 18 orient) | {wvf_time:.6f} | - | {wvf_time * 1e6 / region_pixels:.2f} |"
    )
    report_lines.append(
        f"| LF (m=3, Np=15, 18 orient, sub2) | {lf_time:.6f} | - | {lf_time * 1e6 / region_pixels:.2f} |"
    )
    report_lines.append("")

    print(f"  32x32 WVF: {wvf_time:.2f}s ({wvf_time * 1e6 / region_pixels:.2f} us/pixel)")
    print(f"  32x32 LF: {lf_time:.2f}s ({lf_time * 1e6 / region_pixels:.2f} us/pixel)")

    sizes = [64, 128, 256]
    report_lines.append("### Classical Filter Scaling")
    report_lines.append("| Image Size | Sobel 3x3 (s) | Prewitt 3x3 (s) | Extended Sobel 7x7 (s) |")
    report_lines.append("|-----------:|--------------:|----------------:|------------------------:|")
    for size in sizes:
        img = np.random.rand(size, size) * 255
        results = runtime_comparison(img, n_runs=3)
        report_lines.append(
            f"| {size}x{size} | {results['Sobel (3x3)']['mean_time']:.6f} | "
            f"{results['Prewitt (3x3)']['mean_time']:.6f} | "
            f"{results['Extended Sobel (7x7)']['mean_time']:.6f} |"
        )

    report_lines.append("")
    report_lines.append(
        "**Finding:** When all methods are timed on the same 32x32 CPU region, WVF and LF remain "
        "orders of magnitude slower per input pixel than Sobel/Prewitt. This is a materially "
        "cleaner comparison than the earlier mixed-size table and still supports the critique that "
        "runtime analysis is an essential omission in the source papers."
    )

    for line in report_lines:
        print(line)
    return "\n".join(report_lines)


def test_taylor_derivation():
    """Test 5: Verify the Taylor expansion math independently."""
    print("\n" + "=" * 60)
    print("TEST 5: Taylor Expansion Verification")
    print("=" * 60)

    report_lines = ["## Test 5: Taylor Expansion Verification\n"]

    def known_function(x, y):
        return 3.0 + 2.0 * x - 1.5 * y + 0.5 * x**2 + 0.3 * y**2 - 0.2 * x * y

    true_fx = 2.0
    true_fy = -1.5
    true_fxx = 1.0
    true_fyy = 0.6
    true_fxy = -0.2

    coords = get_circular_neighbors(25)
    values = np.array([known_function(c[0], c[1]) for c in coords])

    A = build_taylor_matrix(coords, order=4)
    z = np.linalg.lstsq(A, values, rcond=None)[0]

    report_lines.append("Testing WVF derivative extraction on known polynomial:")
    report_lines.append(f"  f(x,y) = 3.0 + 2.0x - 1.5y + 0.5x^2 + 0.3y^2 - 0.2xy")
    report_lines.append("")
    report_lines.append("| Derivative | True Value | WVF Recovered | Error |")
    report_lines.append("|-----------|----------:|-------------:|------:|")

    derivs = [
        ("f(0,0)", 3.0, z[0]),
        ("f_x", true_fx, z[1]),
        ("f_y", true_fy, z[2]),
        ("f_xx", true_fxx, z[3]),
        ("f_yy", true_fyy, z[4]),
        ("f_xy", true_fxy, z[5]),
    ]

    all_correct = True
    for name, true_val, recovered in derivs:
        error = abs(recovered - true_val)
        if error > 1e-6:
            all_correct = False
        report_lines.append(f"| {name} | {true_val:.4f} | {recovered:.4f} | {error:.2e} |")
        print(f"  {name}: true={true_val:.4f}, recovered={recovered:.4f}, error={error:.2e}")

    report_lines.append("")
    if all_correct:
        report_lines.append(
            "**Finding:** The Taylor expansion and least-squares approach correctly "
            "recovers known derivatives from polynomial data. The mathematical "
            "formulation in Eq. 1-3 is sound for polynomial signals."
        )
    else:
        report_lines.append(
            "**Finding:** Some derivatives were not accurately recovered, "
            "suggesting potential issues with the formulation or numerical stability."
        )

    for line in report_lines:
        print(line)
    return "\n".join(report_lines)


def generate_written_critique():
    """Generate the non-computational written critique sections."""
    critique = """## Written Critique (Non-Computational)

### Mathematical Rigor

1. **Taylor Expansion (Eq. 1-3):** The 2D Taylor expansion formulation is
   mathematically standard and correctly derived. The least-squares approach
   via pseudo-inverse is appropriate for the overdetermined system (Np > number
   of coefficients). However, the papers do not discuss the conditioning of
   A^T*A as Np grows, which is critical for numerical stability.

2. **Line Filter Aggregation (Eq. 4-6):** The weighted combination of WVF
   results along a line is reasonable, but the choice of Gaussian weighting
   sigma is not justified theoretically. The papers state weights "can be
   determined according to the distance" but don't derive optimal weights.

3. **<1 Degree Accuracy Claim:** This is demonstrated empirically on synthetic
   data (Table 1 in Paper 3) but lacks theoretical error bounds. The claim
   is SNR-conditional (>0.75) but the relationship between SNR, Np, and
   angular accuracy is not characterized analytically.

### Experimental Methodology

1. **Unfair Compute Comparison:** The WVF uses Np=250 pixels in a circular
   neighborhood vs Sobel/Prewitt's 9 pixels (3x3 kernel). This is roughly
   28x more data per pixel. Comparing gradient quality without normalizing
   for computational cost is misleading. A fairer comparison would include
   Sobel with equivalent Gaussian pre-smoothing, which uses a similar number
   of pixels via the convolution.

2. **Small Aquatic Dataset (n=4):** The custom aquatic dataset contains only
   4 hand-labeled images. This is far too small for statistical significance.
   No confidence intervals, p-values, or effect sizes are reported. The
   ODS/OIS/mAP differences could easily be within sampling noise.

3. **BIPED/UDED Evaluation Concern:** DexiNed and TEED were trained on BIPED,
   yet the papers test on BIPED and claim the test set is "separate from the
   training data" and "still similar." This is not a clean out-of-distribution
   test. The WVF/LF advantage on UDED (training-free method on unseen data)
   is more convincing but the margins are small.

4. **Missing Baselines:** No comparison against Gaussian-smoothed Sobel at
   equivalent scale, Laplacian of Gaussian, or structured edge detectors
   like SE (Dollar & Zitnick, 2013). These would be more informative than
   comparing against vanilla 3x3 operators.

5. **No Timing Analysis:** For a method proposed for autonomous vehicles,
   the complete absence of runtime comparisons is a critical omission. The
   WVF/LF are inherently expensive (multiple matrix inversions per pixel per
   orientation) and GPU parallelization is mentioned but not demonstrated.

### Claims vs Evidence

1. **"ML is unreliable and dangerous" (Thesis, Section 1.4):** This is an
   overstatement. While ML edge detectors have limitations in domain shift,
   characterizing all ML-based autonomous driving as "unreliable and dangerous"
   is not supported by the evidence presented. The papers cite limited sources
   (Gupta et al. 2010, Guo et al. 2016) and ignore significant advances in
   ML robustness and safety.

2. **"The autonomous vehicles industry should divert their efforts" to image
   processing (Thesis, p.2):** This recommendation is not supported by the
   scope of this work, which addresses only edge detection in synthetic and
   aquatic images. Extending this to all of autonomous driving is a non
   sequitur.

3. **Paper 2 vs Paper 3 UDED Results:** Paper 2 Table 1 reports WVF-UDED
   ODS=0.6239, OIS=0.6316. Paper 3 Table 2 reports WVF/LF MultiScale-UDED
   ODS=0.7185, OIS=0.7274. This 15% improvement is attributed to multi-scale
   and multi-domain processing, which is reasonable, but the WVF/LF parameters
   may differ between papers.

### Writing & Presentation Quality

1. **Typos:** Thesis Table 3.4 "Simpified Derivtive" (should be "Simplified
   Derivative"), Reference [22] "Imga" (should be "Image"), multiple
   inconsistencies in citation formatting.

2. **Self-citation:** Paper 3 cites Paper 2 as [21] ("Submitted, 2024"),
   which is appropriate. However, the three publications share substantial
   overlapping text and figures, with the IEEE papers being compressed
   versions of thesis chapters.

3. **Redundancy:** The IEEE 2024 paper is essentially a condensed version
   of Thesis Chapters 2-4. The IEEE 2025 paper covers Thesis Chapter 5
   plus new multi-scale/multi-domain contributions. This level of
   overlap is common but worth noting.

### Strengths

1. The core idea of orientation-specific gradient computation via Taylor
   expansion is novel and well-motivated.
2. The approach of evaluating gradients at many orientations (rather than
   only x/y decomposition) is a genuine improvement for angular accuracy.
3. The cubic spline angle detection method is elegant conceptually, but our
   tractable reproduction does not independently verify the paper's accuracy claim.
4. The low-SNR aquatic use case is clearly important; our synthetic maritime
   replications confirm it remains genuinely difficult for conventional methods.
5. The work is clearly funded by ONR and addresses a genuine Navy need
   for littoral environment image processing.
"""
    return critique


def main():
    """Run all tests and generate the final critique report."""
    print("=" * 60)
    print("BAGAN PUBLICATION CRITIQUE - COMPUTATIONAL VERIFICATION")
    print("=" * 60)

    report_sections = []

    report_sections.append("# Critique Report: Bagan & Wang Edge Detection Papers\n")
    report_sections.append("*Independent computational verification by J.C. Vaught*\n")
    report_sections.append("---\n")

    report_sections.append(generate_written_critique())
    report_sections.append("\n---\n")
    report_sections.append("# Computational Verification Results\n")

    report_sections.append(test_taylor_derivation())
    report_sections.append("")
    report_sections.append(test_condition_numbers())
    report_sections.append("")
    report_sections.append(test_angle_accuracy())
    report_sections.append("")
    report_sections.append(test_snr_robustness())
    report_sections.append("")
    report_sections.append(test_runtime_scaling())

    report_sections.append("\n---\n")
    report_sections.append("# Summary\n")
    report_sections.append(
        "The core mathematical approach (2D Taylor expansion for gradient computation) "
        "is sound and verified. Our direct angle test does not reproduce the paper's "
        "strong spline-accuracy claim at tractable WVF settings, while the broader "
        "performance claims remain limited by compute-heavy WVF/LF "
        "evaluation and small-sample or synthetic test regimes. The major issues are "
        "still experimental fairness, runtime transparency, statistical power, and "
        "overreach in some of the source-paper conclusions."
    )

    report_path = RESULTS_DIR / "critique_report.md"
    with open(report_path, 'w') as f:
        f.write("\n".join(report_sections))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
