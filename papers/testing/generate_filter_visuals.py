#!/usr/bin/env python3
"""
Generate publication-quality visualizations comparing three oriented edge
detection kernel designs:
  1. Original fused anisotropic stencil (virtual pixels + polynomial fit)
  2. Elliptical Gaussian kernel
  3. Rectangular Gaussian kernel

Produces:
  fig1b_kernel_comparison.pdf/png  — neighborhood / kernel structure (3 panels)
  fig3_stencil_orientations.pdf/png — weight maps at 6 angles (3x6 grid)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# ── Import the real stencil computation from edgecritic ───────────────────────
_project_root = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "edge-detection-filter-critique",
)
sys.path.insert(0, os.path.join(_project_root, "src"))

from edgecritic.core.taylor import (
    get_circular_neighbors as _impl_get_circular_neighbors,
)
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "edgecritic.aniso_lf._stencil",
    os.path.join(_project_root, "src", "edgecritic", "aniso_lf", "_stencil.py"),
)
_stencil_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_stencil_mod)
_compute_pfx_row = _stencil_mod._compute_pfx_row

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Style constants ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# Diverging colormap: Congaree (neg) → neutral → Garnet (pos)
WEIGHT_CMAP = LinearSegmentedColormap.from_list("uofsc_div", [
    "#1F414D",   # Congaree (strong negative)
    "#466A9F",   # Atlantic (mild negative)
    "#C7C7C7",   # 30% Black (zero)
    "#CC2E40",   # Rose (mild positive)
    "#73000A",   # Garnet (strong positive)
])


def _save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    print(f"  Saved {name}.pdf / .png")
    plt.close(fig)


def _draw_pixel_grid(ax, grid_half):
    for i in range(-grid_half, grid_half + 1):
        for j in range(-grid_half, grid_half + 1):
            ax.add_patch(plt.Rectangle(
                (i - 0.45, j - 0.45), 0.9, 0.9,
                fill=True, facecolor="#f8f9fa", edgecolor="#dee2e6",
                linewidth=0.3, zorder=0,
            ))


# ══════════════════════════════════════════════════════════════════════════════
# Kernel weight computation for each design
# ══════════════════════════════════════════════════════════════════════════════

def compute_original_stencil(theta, m=7, n_neighbors=20, order=4):
    """Compute fused stencil weights using the real edgecritic internals."""
    neighbors_arr = _impl_get_circular_neighbors(n_neighbors)
    pfx = _compute_pfx_row(neighbors_arr, theta, order)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    sigma = m / 2.0
    j_range = np.arange(-m, m + 1, dtype=np.float64)
    gauss_weights = np.exp(-0.5 * (j_range / sigma) ** 2)
    gauss_weights /= gauss_weights.sum()

    stencil_map = defaultdict(float)
    for ji, j in enumerate(range(-m, m + 1)):
        vdx = int(round(j * cos_t))
        vdy = int(round(j * sin_t))
        w_j = gauss_weights[ji]
        for ni in range(n_neighbors):
            ndx = int(neighbors_arr[ni, 0])
            ndy = int(neighbors_arr[ni, 1])
            stencil_map[(vdx + ndx, vdy + ndy)] += w_j * pfx[ni]

    return dict(stencil_map)


def compute_elliptical_kernel(theta, sigma_along=2.0, sigma_across=1.2,
                              length=15):
    """Compute elliptical Gaussian edge kernel weights on the pixel grid."""
    half = length // 2
    stencil_map = {}

    ct, st = np.cos(theta), np.sin(theta)

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            u = dx * ct + dy * st
            v = -dx * st + dy * ct

            ellipse_arg = (u**2 / sigma_along**2) + (v**2 / sigma_across**2)
            if ellipse_arg > 9.0:
                continue

            gauss = np.exp(-0.5 * ellipse_arg)
            weight = -v * gauss

            if abs(weight) > 1e-15:
                stencil_map[(dx, dy)] = weight

    # Zero-DC and normalise (same as the runtime kernel)
    mean_w = np.mean(list(stencil_map.values()))
    total = 0.0
    for k in stencil_map:
        stencil_map[k] -= mean_w
        total += abs(stencil_map[k])
    for k in stencil_map:
        stencil_map[k] /= total + 1e-12

    return stencil_map


def compute_rectangular_kernel(theta, half_along=6.0, half_across=3.6,
                               sigma_along=2.0, sigma_across=1.2,
                               length=15):
    """Compute rectangular Gaussian edge kernel weights on the pixel grid."""
    half = length // 2
    stencil_map = {}

    ct, st = np.cos(theta), np.sin(theta)

    for dy in range(-half, half + 1):
        for dx in range(-half, half + 1):
            u = dx * ct + dy * st
            v = -dx * st + dy * ct

            if abs(u) > half_along or abs(v) > half_across:
                continue

            gauss = np.exp(-0.5 * (u**2 / sigma_along**2 +
                                    v**2 / sigma_across**2))
            weight = -v * gauss

            if abs(weight) > 1e-15:
                stencil_map[(dx, dy)] = weight

    mean_w = np.mean(list(stencil_map.values()))
    total = 0.0
    for k in stencil_map:
        stencil_map[k] -= mean_w
        total += abs(stencil_map[k])
    for k in stencil_map:
        stencil_map[k] /= total + 1e-12

    return stencil_map


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1b : Kernel Structure Comparison (3 panels at theta=30)
# ══════════════════════════════════════════════════════════════════════════════

def figure1b():
    print("Figure 1b: Kernel Structure Comparison")
    theta = np.radians(30)
    grid_half = 12

    # Match elliptical/rectangular to original's actual extent:
    #   Original at theta=0 spans x in [-9, 9], y in [-2, 2]
    #   -> sigma_along = 9/3 = 3.0,  sigma_across = 2/3 ≈ 0.7
    sigma_a, sigma_c = 3.0, 0.7
    half_a, half_c = 3.0 * sigma_a, 3.0 * sigma_c  # 9.0, 2.1
    klen = 2 * int(np.ceil(3 * sigma_a)) + 1        # 19

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = [
        "Original Fused Stencil",
        "Elliptical Gaussian",
        "Rectangular Gaussian",
    ]
    stencils = [
        compute_original_stencil(theta, m=7, n_neighbors=20),
        compute_elliptical_kernel(theta, sigma_along=sigma_a,
                                  sigma_across=sigma_c, length=klen),
        compute_rectangular_kernel(theta, half_along=half_a,
                                   half_across=half_c,
                                   sigma_along=sigma_a, sigma_across=sigma_c,
                                   length=klen),
    ]

    # Global weight range across all three for consistent coloring
    all_w = []
    for s in stencils:
        all_w.extend(s.values())
    vmax = max(abs(min(all_w)), abs(max(all_w)))

    for idx, (ax, title, stencil) in enumerate(zip(axes, titles, stencils)):
        ax.set_xlim(-grid_half - 0.7, grid_half + 0.7)
        ax.set_ylim(-grid_half - 0.7, grid_half + 0.7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        _draw_pixel_grid(ax, grid_half)

        # Draw kernel weights as colored squares
        cell_size = 0.8
        for (px, py), weight in stencil.items():
            norm_w = weight / vmax
            color = WEIGHT_CMAP(0.5 + 0.5 * norm_w)
            rect = plt.Rectangle((px - cell_size / 2, py - cell_size / 2),
                                 cell_size, cell_size,
                                 facecolor=color, edgecolor="none",
                                 alpha=0.9, zorder=3)
            ax.add_patch(rect)

        # Center marker
        ax.plot(0, 0, "+", color="black", ms=10, mew=2, zorder=7)

        # Orientation line
        line_len = grid_half - 1
        ax.plot([-line_len * np.cos(theta), line_len * np.cos(theta)],
                [-line_len * np.sin(theta), line_len * np.sin(theta)],
                "--", color="#CED318", linewidth=0.8, zorder=1)

        # Normal direction arrow
        norm_ang = theta + np.pi / 2
        arrow_len = 3
        ax.annotate("", xy=(arrow_len * np.cos(norm_ang),
                             arrow_len * np.sin(norm_ang)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5),
                    zorder=4)

        # For the original stencil, also draw virtual pixel markers
        if idx == 0:
            m_val = 7
            for j in range(-m_val, m_val + 1):
                vx = int(round(j * np.cos(theta)))
                vy = int(round(j * np.sin(theta)))
                ax.add_patch(plt.Circle((vx, vy), 0.18, facecolor="#65780B",
                                        edgecolor="white", linewidth=0.4,
                                        zorder=6))

        # For elliptical, draw the 3-sigma ellipse outline
        if idx == 1:
            from matplotlib.patches import Ellipse
            ell = Ellipse((0, 0),
                          width=2 * 3 * sigma_a,
                          height=2 * 3 * sigma_c,
                          angle=np.degrees(theta),
                          fill=False, edgecolor="#65780B",
                          linewidth=1.5, linestyle="--", zorder=5)
            ax.add_patch(ell)

        # For rectangular, draw the rectangle outline
        if idx == 2:
            from matplotlib.patches import FancyBboxPatch
            import matplotlib.transforms as mtransforms
            ha, hc = half_a, half_c
            corners = np.array([[-ha, -hc], [ha, -hc],
                                [ha, hc], [-ha, hc], [-ha, -hc]])
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta),  np.cos(theta)]])
            rotated = corners @ rot.T
            ax.plot(rotated[:, 0], rotated[:, 1], "--", color="#65780B",
                    linewidth=1.5, zorder=5)

        n_pts = len(stencil)
        ax.set_title(f"{title}\n({n_pts} pixels, "
                     f"$\\theta = 30^\\circ$)", pad=10, fontsize=11)

    # Panel labels
    for i, letter in enumerate("ABC"):
        axes[i].text(-0.04, 1.06, letter, transform=axes[i].transAxes,
                     fontsize=14, fontweight="bold", va="bottom")

    # Shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap=WEIGHT_CMAP,
                               norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Kernel weight", fontsize=10)

    fig.tight_layout(rect=[0, 0, 0.88, 1], w_pad=1.5)
    _save(fig, "fig1b_kernel_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 : Stencil Shapes Across Orientations (3 rows x 6 cols)
# ══════════════════════════════════════════════════════════════════════════════

def figure3():
    print("Figure 3: Stencil Shape Across Orientations (all 3 designs)")
    angles_deg = [0, 30, 60, 90, 120, 150]
    n_angles = len(angles_deg)
    design_names = ["Original Fused Stencil", "Elliptical Gaussian",
                    "Rectangular Gaussian"]

    # Match elliptical/rectangular to original's actual extent:
    #   Original at theta=0 spans x in [-9,9], y in [-2,2]
    sigma_a, sigma_c = 3.0, 0.7
    half_a, half_c = 3.0 * sigma_a, 3.0 * sigma_c
    klen = 2 * int(np.ceil(3 * sigma_a)) + 1

    # Precompute all stencils
    all_stencils = []  # [design_idx][angle_idx]
    for design_idx in range(3):
        row = []
        for ang in angles_deg:
            rad = np.radians(ang)
            if design_idx == 0:
                s = compute_original_stencil(rad, m=7, n_neighbors=20)
            elif design_idx == 1:
                s = compute_elliptical_kernel(rad, sigma_along=sigma_a,
                                              sigma_across=sigma_c,
                                              length=klen)
            else:
                s = compute_rectangular_kernel(rad, half_along=half_a,
                                               half_across=half_c,
                                               sigma_along=sigma_a,
                                               sigma_across=sigma_c,
                                               length=klen)
            row.append(s)
        all_stencils.append(row)

    # Global weight range across everything for consistent coloring
    all_w = []
    for row in all_stencils:
        for s in row:
            all_w.extend(s.values())
    vmax = max(abs(min(all_w)), abs(max(all_w)))

    grid_half = 9

    fig, axes = plt.subplots(3, n_angles, figsize=(20, 10.5))

    for row_idx in range(3):
        for col_idx in range(n_angles):
            ax = axes[row_idx, col_idx]
            stencil = all_stencils[row_idx][col_idx]
            ang = angles_deg[col_idx]
            rad = np.radians(ang)

            ax.set_xlim(-grid_half - 0.5, grid_half + 0.5)
            ax.set_ylim(-grid_half - 0.5, grid_half + 0.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            _draw_pixel_grid(ax, grid_half)

            # Draw kernel weights
            cell_size = 0.8
            for (px, py), weight in stencil.items():
                norm_w = weight / vmax
                color = WEIGHT_CMAP(0.5 + 0.5 * norm_w)
                rect = plt.Rectangle(
                    (px - cell_size / 2, py - cell_size / 2),
                    cell_size, cell_size,
                    facecolor=color, edgecolor="none",
                    alpha=0.9, zorder=3)
                ax.add_patch(rect)

            # Center marker
            ax.plot(0, 0, "+", color="black", ms=8, mew=1.5, zorder=7)

            # Orientation line
            line_len = grid_half - 1
            ax.plot([-line_len * np.cos(rad), line_len * np.cos(rad)],
                    [-line_len * np.sin(rad), line_len * np.sin(rad)],
                    "--", color="#CED318", linewidth=0.7, zorder=1)

            # Virtual pixel markers for original only
            if row_idx == 0:
                m_val = 7
                for j in range(-m_val, m_val + 1):
                    vx = int(round(j * np.cos(rad)))
                    vy = int(round(j * np.sin(rad)))
                    ax.add_patch(plt.Circle(
                        (vx, vy), 0.15, facecolor="#65780B",
                        edgecolor="white", linewidth=0.3, zorder=6))

            # Column titles (angle) on top row only
            if row_idx == 0:
                n_pts = len(stencil)
                ax.set_title(f"$\\theta = {ang}^\\circ$  ({n_pts} pts)",
                             pad=8, fontsize=10)

        # Row labels on leftmost column
        axes[row_idx, 0].set_ylabel(design_names[row_idx], fontsize=11,
                                     rotation=90, labelpad=15)

    # Shared colorbar
    fig.subplots_adjust(right=0.92, hspace=0.15, wspace=0.08)
    cbar_ax = fig.add_axes([0.935, 0.15, 0.012, 0.7])
    sm = plt.cm.ScalarMappable(cmap=WEIGHT_CMAP,
                               norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Kernel weight", fontsize=10)

    _save(fig, "fig3_stencil_orientations")


if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")
    figure1b()
    figure3()
    print("\nDone.")
