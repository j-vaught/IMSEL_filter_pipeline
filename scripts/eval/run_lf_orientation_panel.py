"""Eight-orientation Line Filter response panel on a synthetic image.

Uses the modern steerable-WVF LF formulation. The pipeline is:
  1. Compute the WVF gradient maps G_x, G_y once via a single image-wide
     pass (precomputed pseudo-inverse, dense convolution).
  2. For each candidate edge orientation theta in {0, 22.5, 45, 67.5,
     90, 112.5, 135, 157.5} degrees:
       a. Form the steered perpendicular derivative
            G_perp(p) = -sin(theta) * G_x(p) + cos(theta) * G_y(p)
          at every integer pixel.
       b. Sample 2m+1 virtual pixels along the line through each
          target pixel in direction theta, with step
            Delta_theta = 1 / max(|cos|, |sin|)
          so consecutive virtual pixels differ by exactly one unit on
          the lattice's dominant axis. Snap each virtual pixel to the
          nearest integer position.
       c. Read G_perp at those integer positions from the precomputed
          map; combine with Gaussian weights w_j = exp(-j^2/(2 sigma^2))
          and renormalize over contributing samples.
  3. Render the eight |L_theta| magnitude maps as a 2x4 grid PNG using
     a single global color scale for cross-panel comparability.

This is a corner-stress test. At sharp polygon vertices the gradient
direction is multi-valued; the LF response at each orientation should
show a distinctive leak pattern around corners that complements the
edge-segment responses.

Usage::

    python scripts/eval/run_lf_orientation_panel.py \
        --image example_images/synthetic_nested_shapes/clean/4096/...png \
        --out-dir outputs/lf_orientation_panel/run-001
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from edgecritic.wvf._radius_kernels import wvf_radius_gradients_cpu


def lf_response_at_orientation(
    g_x: np.ndarray,
    g_y: np.ndarray,
    theta: float,
    half_width: int,
    sigma: float | None = None,
) -> np.ndarray:
    """Compute the LF response at one orientation from precomputed gradients.

    Implements the modern steerable formulation:
      L_theta(x_0, y_0) = sum_j w_j * G_perp(snapped(p_j))
                       / sum_j w_j (over contributing samples)
    where the line p_j is in direction theta and G_perp is the
    perpendicular directional derivative obtained from the steerability
    identity. Returns the signed response (caller can take abs).
    """
    H, W = g_x.shape
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    max_trig = max(abs(cos_t), abs(sin_t))
    step = 1.0 / max_trig if max_trig > 0 else 1.0

    # Perpendicular directional derivative at every integer pixel.
    g_perp = -sin_t * g_x + cos_t * g_y

    if sigma is None:
        sigma = half_width / 2.0
    j_offsets = np.arange(-half_width, half_width + 1)
    weights = np.exp(-0.5 * (j_offsets / sigma) ** 2)

    response = np.zeros((H, W), dtype=np.float64)
    weight_sum = np.zeros((H, W), dtype=np.float64)

    for w_j, j in zip(weights, j_offsets):
        ix = int(round(j * step * cos_t))
        iy = int(round(j * step * sin_t))

        # Source / destination valid windows when reading
        # response[y, x] += w_j * g_perp[y + iy, x + ix].
        y_src_lo = max(0,  iy)
        y_src_hi = min(H, H + iy)
        x_src_lo = max(0,  ix)
        x_src_hi = min(W, W + ix)
        y_dst_lo = max(0, -iy)
        y_dst_hi = y_dst_lo + (y_src_hi - y_src_lo)
        x_dst_lo = max(0, -ix)
        x_dst_hi = x_dst_lo + (x_src_hi - x_src_lo)

        if y_src_hi <= y_src_lo or x_src_hi <= x_src_lo:
            continue

        response[y_dst_lo:y_dst_hi, x_dst_lo:x_dst_hi] += (
            w_j * g_perp[y_src_lo:y_src_hi, x_src_lo:x_src_hi]
        )
        weight_sum[y_dst_lo:y_dst_hi, x_dst_lo:x_dst_hi] += w_j

    valid = weight_sum > 0
    response[valid] = response[valid] / weight_sum[valid]
    return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--radius", type=int, default=7, help="WVF radius.")
    parser.add_argument("--order", type=int, default=4, help="WVF polynomial degree.")
    parser.add_argument("--half-width", type=int, default=7, help="LF half-width m.")
    parser.add_argument(
        "--n-orientations", type=int, default=8,
        help="Number of orientations sampled on [0, pi).")
    parser.add_argument(
        "--clip-percentile", type=float, default=99.5,
        help="Percentile for the global vmax of the heatmap colorbar.")
    parser.add_argument(
        "--crop-size", type=int, default=0,
        help=("If > 0, crop the rendered panel and the input thumbnail "
              "to a square of this size centered on the image. The WVF "
              "and LF are still computed on the full image so the crop "
              "is free of boundary artifacts."))
    parser.add_argument(
        "--pixel", type=str, default=None,
        help=("'x,y' image-coordinate pair. If provided, also produce "
              "a single-pixel orientation-response curve (|L_theta| vs "
              "theta) at that pixel, sampled at --pixel-n-orientations "
              "candidate orientations on [0, pi)."))
    parser.add_argument(
        "--multi-n-orientations", type=str, default="8,16,32,64,128",
        help=("Comma-separated list of orientation counts to overlay "
              "in the single-pixel curve plot."))
    parser.add_argument(
        "--gt-angles", type=str, default="",
        help=("Comma-separated ground-truth tangent angles in degrees "
              "(modulo 180) to mark with vertical lines on the curve."))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    pil = Image.open(args.image).convert("L")
    image = np.asarray(pil, dtype=np.float64)
    h, w = image.shape
    print(f"Loaded {args.image.name}: {h} x {w}")

    print(f"Computing WVF gradient maps (radius={args.radius}, "
          f"order={args.order})...")
    t0 = time.perf_counter()
    g_x, g_y = wvf_radius_gradients_cpu(
        image, radius=args.radius, order=args.order)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    angles = np.linspace(0, np.pi, args.n_orientations, endpoint=False)
    print(f"Sampling {args.n_orientations} orientations: "
          f"{[f'{np.degrees(a):.1f}°' for a in angles]}")

    response_stack = np.zeros((h, w, args.n_orientations), dtype=np.float32)
    for k, theta in enumerate(angles):
        t0 = time.perf_counter()
        l_theta = lf_response_at_orientation(
            g_x, g_y, theta, half_width=args.half_width)
        response_stack[..., k] = np.abs(l_theta).astype(np.float32)
        print(f"  theta={np.degrees(theta):5.1f}deg  "
              f"({time.perf_counter() - t0:.1f}s)")

    np.save(args.out_dir / f"lf_response_stack_n{args.n_orientations}.npy",
            response_stack)

    crop_label = ""
    image_for_thumb = image
    stack_for_render = response_stack
    if args.crop_size > 0:
        c = args.crop_size
        cy = h // 2
        cx = w // 2
        y0 = max(0, cy - c // 2)
        x0 = max(0, cx - c // 2)
        y1 = min(h, y0 + c)
        x1 = min(w, x0 + c)
        stack_for_render = response_stack[y0:y1, x0:x1, :]
        image_for_thumb = image[y0:y1, x0:x1]
        crop_label = f"_crop{c}"
        print(f"Cropping render to {y1 - y0} x {x1 - x0} centered "
              f"at ({cy}, {cx})")

    rows = 2
    cols = (args.n_orientations + rows - 1) // rows
    fig, axes = plt.subplots(
        rows, cols, figsize=(3.2 * cols, 3.2 * rows), squeeze=False)

    vmax = float(np.percentile(stack_for_render, args.clip_percentile))

    for k in range(args.n_orientations):
        ax = axes[k // cols, k % cols]
        ax.imshow(stack_for_render[..., k], cmap="gray_r",
                  vmin=0.0, vmax=vmax)
        deg = np.degrees(angles[k])
        ax.set_title(rf"$\theta = {deg:.1f}^\circ$", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    for k in range(args.n_orientations, rows * cols):
        axes[k // cols, k % cols].axis("off")

    plt.tight_layout()
    panel_path = (args.out_dir
                  / f"lf_orientations_n{args.n_orientations}{crop_label}.png")
    plt.savefig(panel_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {panel_path}")

    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.imshow(image_for_thumb, cmap="gray")
    ax2.axis("off")
    ax2.set_title(args.image.name + crop_label, fontsize=10)
    input_path = args.out_dir / f"input{crop_label}.png"
    plt.savefig(input_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {input_path}")

    if args.pixel is not None:
        px, py = (int(v.strip()) for v in args.pixel.split(","))
        n_list = [int(v.strip()) for v in
                  args.multi_n_orientations.split(",") if v.strip()]
        gt_angles_deg = [float(v.strip()) for v in
                         args.gt_angles.split(",") if v.strip()]
        print(f"Single-pixel curves at (x={px}, y={py}) for "
              f"orientation counts {n_list}")

        sigma_g = args.half_width / 2.0
        j_offsets = np.arange(-args.half_width, args.half_width + 1)
        weights = np.exp(-0.5 * (j_offsets / sigma_g) ** 2)

        def response_at(theta: float) -> float:
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            max_trig = max(abs(cos_t), abs(sin_t))
            step = 1.0 / max_trig if max_trig > 0 else 1.0
            acc = 0.0
            wsum = 0.0
            for w_j, j in zip(weights, j_offsets):
                ix = int(round(j * step * cos_t))
                iy = int(round(j * step * sin_t))
                yy = py + iy
                xx = px + ix
                if 0 <= yy < h and 0 <= xx < w:
                    g_perp = -sin_t * g_x[yy, xx] + cos_t * g_y[yy, xx]
                    acc += w_j * g_perp
                    wsum += w_j
            return abs(acc / wsum) if wsum > 0 else 0.0

        # Compute one response curve per N, sharing colors.
        cmap_n = plt.get_cmap("viridis")
        curves: list[tuple[int, np.ndarray, np.ndarray]] = []
        for idx, n in enumerate(n_list):
            angles_n = np.linspace(0, np.pi, n, endpoint=False)
            resp = np.array([response_at(t) for t in angles_n])
            curves.append((n, angles_n, resp))

        fig3, (ax_lin, ax_pol_holder) = plt.subplots(
            1, 2, figsize=(11, 4.8))
        ax_pol_holder.remove()
        ax_pol = fig3.add_subplot(1, 2, 2, projection="polar")

        for idx, (n, ang, resp) in enumerate(curves):
            color = cmap_n(idx / max(1, len(curves) - 1))
            ax_lin.plot(np.degrees(ang), resp, marker="o",
                        markersize=3, linewidth=1.4,
                        color=color, label=f"N = {n}")
            full_a = np.concatenate([ang, ang + np.pi])
            full_r = np.concatenate([resp, resp])
            ax_pol.plot(full_a, full_r, marker="o",
                        markersize=2, linewidth=1.2, color=color)

        for gt in gt_angles_deg:
            ax_lin.axvline(gt % 180, color="#cc2e40",
                           linestyle="--", linewidth=1.1, alpha=0.8,
                           label="GT" if gt == gt_angles_deg[0] else None)
            theta_rad = np.radians(gt % 180)
            ax_pol.plot([theta_rad, theta_rad],
                        [0, max(r.max() for _, _, r in curves)],
                        color="#cc2e40", linestyle="--",
                        linewidth=1.1, alpha=0.8)
            ax_pol.plot([theta_rad + np.pi, theta_rad + np.pi],
                        [0, max(r.max() for _, _, r in curves)],
                        color="#cc2e40", linestyle="--",
                        linewidth=1.1, alpha=0.8)

        ax_lin.set_xlabel(r"orientation $\theta$ (deg)")
        ax_lin.set_ylabel(r"$|L_\theta|$")
        ax_lin.set_title(f"Pixel ({px}, {py}) response curves")
        ax_lin.set_xlim(0, 180)
        ax_lin.grid(True, alpha=0.3)
        ax_lin.legend(loc="lower center", fontsize=9, ncol=3)

        ax_pol.set_theta_zero_location("E")
        ax_pol.set_theta_direction(-1)
        ax_pol.set_title("Polar view (mirrored to full circle)")

        plt.tight_layout()
        n_label = "_".join(str(n) for n in n_list)
        curve_path = (args.out_dir
                      / f"pixel_response_x{px}_y{py}_multi_{n_label}.png")
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig3)
        print(f"Saved {curve_path}")

        # Also save a small input thumbnail with the pixel marked.
        fig4, ax4 = plt.subplots(figsize=(5, 5))
        ax4.imshow(image_for_thumb, cmap="gray")
        if args.crop_size > 0:
            c = args.crop_size
            cy = h // 2
            cx = w // 2
            y0 = max(0, cy - c // 2)
            x0 = max(0, cx - c // 2)
            mark_x = px - x0
            mark_y = py - y0
        else:
            mark_x = px
            mark_y = py
        ax4.scatter([mark_x], [mark_y], s=80, c="#cc2e40",
                    marker="o", edgecolors="#363636", linewidths=1.0)
        ax4.set_title(f"Pixel ({px}, {py}){crop_label}")
        ax4.axis("off")
        marked_path = (args.out_dir
                       / f"input_marked_x{px}_y{py}{crop_label}.png")
        plt.savefig(marked_path, dpi=150, bbox_inches="tight")
        plt.close(fig4)
        print(f"Saved {marked_path}")


if __name__ == "__main__":
    main()
