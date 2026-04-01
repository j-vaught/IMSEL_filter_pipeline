#!/usr/bin/env python3
"""
Generate publication-quality visualizations comparing edge detection filter
approaches: Point Filter (WVF), Line Filter (LF), and Anisotropic Stencil
Filter (ASF).

Requirements: numpy, matplotlib (Python 3.9+)
No external dependencies beyond standard scientific Python.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import PatchCollection
from collections import defaultdict
import os
import sys

# Ensure the project's src/ is on the path so we import the real implementation
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_project_root, "src"))

from edgecritic.core.taylor import (
    get_circular_neighbors as _impl_get_circular_neighbors,
)
# Import _compute_pfx_row directly from the module to avoid pulling in
# torch-dependent siblings via the aniso_lf package __init__.py.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "edgecritic.aniso_lf._stencil",
    os.path.join(_project_root, "src", "edgecritic", "aniso_lf", "_stencil.py"),
)
_stencil_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_stencil_mod)
_compute_pfx_row = _stencil_mod._compute_pfx_row

# ── Output directory ──────────────────────────────────────────────────────────
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "papers", "anisotropic-lf-proposal", "figures",
)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style constants ───────────────────────────────────────────────────────────
COLOR_ORIGINAL = "#2c3e50"
COLOR_CONV2D = "#e74c3c"
COLOR_TRITON = "#27ae60"

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


def get_circular_neighbors(n_neighbors=15):
    """Return the n_neighbors closest integer grid points to origin.

    Delegates to the real implementation in edgecritic.core.taylor to ensure
    the figure visualisations always match the actual filter code.
    """
    coords = _impl_get_circular_neighbors(n_neighbors)  # ndarray (Np, 2)
    return [(int(c[0]), int(c[1])) for c in coords]


def _save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    print(f"  Saved {name}.pdf / .png")
    plt.close(fig)


def _draw_pixel_grid(ax, grid_half):
    """Draw a light background pixel grid."""
    for i in range(-grid_half, grid_half + 1):
        for j in range(-grid_half, grid_half + 1):
            ax.add_patch(plt.Rectangle(
                (i - 0.45, j - 0.45), 0.9, 0.9,
                fill=True, facecolor="#f8f9fa", edgecolor="#dee2e6",
                linewidth=0.3, zorder=0,
            ))


def _draw_filled_pixel(ax, x, y, color, alpha=1.0, zorder=3):
    """Draw a pixel as a filled circle (disc) centered on the grid point."""
    circle = plt.Circle((x, y), 0.38, facecolor=color, edgecolor="none",
                        alpha=alpha, zorder=zorder)
    ax.add_patch(circle)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 : Neighborhood Comparison (3 panels)
# ══════════════════════════════════════════════════════════════════════════════
def figure1():
    print("Figure 1: Neighborhood Comparison")
    neighbors = get_circular_neighbors(20)
    theta = np.radians(30)
    m = 2

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    grid_half = 7
    for ax in axes:
        ax.set_xlim(-grid_half - 0.7, grid_half + 0.7)
        ax.set_ylim(-grid_half - 0.7, grid_half + 0.7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        _draw_pixel_grid(ax, grid_half)
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ── Panel A: Point Filter ─────────────────────────────────────────────────
    ax = axes[0]
    ax.set_title("Point Filter ($N_p$=20)", pad=10, fontsize=12)
    _draw_filled_pixel(ax, 0, 0, "#73000A", zorder=5)
    for dx, dy in neighbors:
        _draw_filled_pixel(ax, dx, dy, "#466A9F", zorder=4)
    # Draw circle outline showing the circular selection region
    r = max(np.sqrt(dx**2 + dy**2) for dx, dy in neighbors) + 0.3
    ax.add_patch(plt.Circle((0, 0), r, fill=False, edgecolor="#466A9F",
                             linewidth=1.2, linestyle="--", zorder=2, alpha=0.5))

    # ── Panel B: Line Filter ──────────────────────────────────────────────────
    ax = axes[1]
    ax.set_title("Line Filter ($m$=2, $N_p$=20)", pad=10, fontsize=12)
    _draw_filled_pixel(ax, 0, 0, "#73000A", zorder=5)
    # dashed line
    line_len = 6
    ax.plot(
        [-line_len * np.cos(theta), line_len * np.cos(theta)],
        [-line_len * np.sin(theta), line_len * np.sin(theta)],
        "--", color="#676156", linewidth=1.2, zorder=2,
    )
    # virtual points and their neighborhoods
    for j in range(-m, m + 1):
        vx = round(j * np.cos(theta))
        vy = round(j * np.sin(theta))
        # Neighbors around this virtual point (semi-transparent to show overlap)
        for dx, dy in neighbors:
            _draw_filled_pixel(ax, vx + dx, vy + dy, "#466A9F", alpha=0.25, zorder=3)
        # Draw dashed circle outline for this virtual point's neighborhood
        ax.add_patch(plt.Circle((vx, vy), r, fill=False, edgecolor="#466A9F",
                                linewidth=1.2, linestyle="--", zorder=2, alpha=0.5))
        # Virtual point marker (diamond)
        ax.plot(vx, vy, "D", color="#65780B", ms=9, zorder=6,
                markeredgecolor="white", markeredgewidth=1)

    # ── Panel C: Fused Stencil ────────────────────────────────────────────────
    ax = axes[2]
    hit_count = defaultdict(int)
    total_samples = 0
    for j in range(-m, m + 1):
        vx = round(j * np.cos(theta))
        vy = round(j * np.sin(theta))
        for dx, dy in neighbors:
            hit_count[(vx + dx, vy + dy)] += 1
            total_samples += 1

    n_unique = len(hit_count)
    ax.set_title(f"Fused Stencil ({n_unique} unique of {total_samples})",
                 pad=10, fontsize=12)
    _draw_filled_pixel(ax, 0, 0, "#73000A", zorder=5)

    max_hits = max(hit_count.values())
    # Custom garnet colormap: light garnet → dark garnet
    from matplotlib.colors import LinearSegmentedColormap
    _garnet_cmap = LinearSegmentedColormap.from_list(
        "garnet", ["#F2D4D6", "#CC2E40", "#73000A", "#570008"])
    # Color scale: 1 hit = light, max hits = dark
    for (px, py), hits in hit_count.items():
        frac = hits / max_hits
        color = _garnet_cmap(0.15 + 0.85 * frac)
        _draw_filled_pixel(ax, px, py, color, zorder=4)

    # Legend
    legend_elements = []
    for h_val, label in [(1, "1 hit"), (2, "2 hits"), (max_hits, f"{max_hits}+ hits")]:
        if h_val > max_hits:
            continue
        c = _garnet_cmap(0.15 + 0.85 * h_val / max_hits)
        legend_elements.append(mpatches.Patch(facecolor=c, edgecolor="none", label=label))
    ax.legend(handles=legend_elements, loc="upper right", frameon=False,
              fontsize=8, handletextpad=0.3)

    # Panel labels
    for i, letter in enumerate("ABC"):
        axes[i].text(-0.04, 1.06, letter, transform=axes[i].transAxes,
                     fontsize=14, fontweight="bold", va="bottom")

    fig.tight_layout(w_pad=1.5)
    _save(fig, "fig1_neighborhood_comparison")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 : Computation Flow Comparison (2 panels)
# ══════════════════════════════════════════════════════════════════════════════
def figure2():
    print("Figure 2: Computation Flow Comparison")

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.5))

    def _draw_flow(ax, boxes, title, complexity, panel_label):
        ax.set_xlim(-0.5, len(boxes) + 0.5)
        ax.set_ylim(-1.2, 2.0)
        ax.set_aspect("auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(0.0, 1.75, f"{panel_label}  {title}",
                fontsize=11, fontweight="bold", va="center")
        box_w, box_h = 0.85, 0.7
        y_center = 0.5
        colors_cycle = ["#ecf0f1", "#d5dbdb"]
        for i, txt in enumerate(boxes):
            x = i + 0.5
            rect = FancyBboxPatch(
                (x - box_w / 2, y_center - box_h / 2), box_w, box_h,
                boxstyle="round,pad=0.08",
                facecolor=colors_cycle[i % 2],
                edgecolor="#2c3e50", linewidth=1.0,
            )
            ax.add_patch(rect)
            ax.text(x, y_center, txt, ha="center", va="center",
                    fontsize=7, fontfamily="serif")
            if i < len(boxes) - 1:
                ax.annotate("", xy=(x + box_w / 2 + 0.15, y_center),
                           xytext=(x + box_w / 2 + 0.01, y_center),
                           arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.2))
        ax.text(len(boxes) / 2 + 0.5, -0.65, complexity,
                ha="center", va="center", fontsize=9, style="italic", color="#7f8c8d")

    _draw_flow(axes[0],
               ["For each\n$\\theta$", "For each\nvirtual pt $j$",
                "Gather\n$N_p$ nbrs", "Matmul\nwith $P_\\theta$",
                "Weight\n$w_j$", "Sum\nover $j$", "$|R_\\theta|$",
                "max\nover $\\theta$"],
               "Naive Line Filter",
               "Complexity:  $\\mathcal{O}(N_s \\times L \\times N_p)$  per pixel", "A")

    _draw_flow(axes[1],
               ["Precompute\nstencils\n(one-time)", "For each\n$\\theta$",
                "Single gather\n$N'$ unique\npositions",
                "Dot product\nwith $\\alpha_\\theta$", "$|R_\\theta|$",
                "max\nover $\\theta$"],
               "Anisotropic Stencil Filter",
               "Complexity:  $\\mathcal{O}(N_s \\times N')$  per pixel", "B")

    fig.tight_layout(h_pad=1.0)
    _save(fig, "fig2_computation_flow")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 : Stencil Shape Across Orientations (2x3 grid)
# ══════════════════════════════════════════════════════════════════════════════
def compute_stencil(theta, m=7, n_neighbors=15, order=4):
    """Compute fused stencil weights using the real implementation internals.

    Uses _compute_pfx_row (same pseudoinverse code as the deployed filter)
    and the same Gaussian weighting + deduplication loop as
    build_anisotropic_stencils, ensuring the figure exactly matches.

    Returns a dict {(x, y): weight} for matplotlib plotting.
    """
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
            # Store as (x, y) for matplotlib: x = col = vdx+ndx, y = row = vdy+ndy
            stencil_map[(vdx + ndx, vdy + ndy)] += w_j * pfx[ni]

    return dict(stencil_map)


def figure3():
    print("Figure 3: Stencil Shape Across Orientations")
    angles_deg = [0, 30, 60, 90, 120, 150]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.5))

    stencils = []
    for ang in angles_deg:
        s = compute_stencil(np.radians(ang), m=7, n_neighbors=20)
        stencils.append(s)

    # Global weight range for consistent colormap
    all_w = []
    for s in stencils:
        all_w.extend(s.values())
    vmax = max(abs(min(all_w)), abs(max(all_w)))

    # Custom diverging colormap: Congaree (negative) → 30% Black (zero) → Garnet (positive)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("uofsc_div", [
        "#1F414D",   # Congaree (strong negative)
        "#466A9F",   # Atlantic (mild negative)
        "#C7C7C7",   # 30% Black (zero — visible neutral)
        "#CC2E40",   # Rose (mild positive)
        "#73000A",   # Garnet (strong positive)
    ])
    grid_half = 12

    for idx, (ang, stencil) in enumerate(zip(angles_deg, stencils)):
        ax = axes[idx // 3, idx % 3]
        ax.set_xlim(-grid_half - 0.5, grid_half + 0.5)
        ax.set_ylim(-grid_half - 0.5, grid_half + 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Draw pixel grid cells
        _draw_pixel_grid(ax, grid_half)

        # Draw stencil as uniform filled squares centered in grid cells
        cell_size = 0.8
        for (px, py), weight in stencil.items():
            norm_w = weight / vmax  # in [-1, 1]
            color = cmap(0.5 + 0.5 * norm_w)  # map to colormap
            rect = plt.Rectangle((px - cell_size / 2, py - cell_size / 2),
                                 cell_size, cell_size,
                                 facecolor=color, edgecolor="none",
                                 alpha=0.9, zorder=3)
            ax.add_patch(rect)

        # Virtual point markers (small dark green dots)
        m_val = 7
        rad = np.radians(ang)
        for j in range(-m_val, m_val + 1):
            vx = int(round(j * np.cos(rad)))
            vy = int(round(j * np.sin(rad)))
            ax.add_patch(plt.Circle((vx, vy), 0.18, facecolor="#65780B",
                                    edgecolor="white", linewidth=0.4,
                                    zorder=6))

        # Center marker
        ax.plot(0, 0, "+", color="black", ms=10, mew=2, zorder=7)

        # Draw the line direction (tangent)
        length = grid_half - 1
        rad = np.radians(ang)
        ax.plot([-length * np.cos(rad), length * np.cos(rad)],
                [-length * np.sin(rad), length * np.sin(rad)],
                "--", color="#bdc3c7", linewidth=0.8, zorder=1)

        # Draw normal direction arrow (perpendicular to line)
        norm_ang = rad + np.pi / 2
        arrow_len = 3
        ax.annotate("", xy=(arrow_len * np.cos(norm_ang), arrow_len * np.sin(norm_ang)),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.5),
                    zorder=4)
        ax.text(arrow_len * np.cos(norm_ang) * 1.3,
                arrow_len * np.sin(norm_ang) * 1.3,
                "$f_x$", fontsize=9, ha="center", va="center",
                fontweight="bold", zorder=5)

        n_unique = len(stencil)
        ax.set_title(f"$\\theta = {ang}^\\circ$  ({n_unique} pts)", pad=8, fontsize=11)

    # Shared colorbar
    fig.subplots_adjust(right=0.87, hspace=0.35, wspace=0.15)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-vmax, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.set_label("Fused weight $\\alpha$", fontsize=10)

    _save(fig, "fig3_stencil_orientations")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 : Speedup and VRAM Comparison (2-panel bar chart)
# ══════════════════════════════════════════════════════════════════════════════
def figure4():
    print("Figure 4: Speedup and VRAM Comparison")
    configs = ["$m$=1", "$m$=2", "$m$=7", "$m$=14"]
    speedup_conv2d = [6.55, 10.24, 13.84, 15.02]
    speedup_triton = [7.99, 13.68, 18.43, 24.29]
    vram_original = [5452, 6225, 6223, 6222]
    vram_conv2d = [157, 158, 158, 289]
    vram_triton = [20, 20, 20, 20]
    x = np.arange(len(configs))
    bar_w = 0.30

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))

    b1 = ax1.bar(x - bar_w / 2, speedup_conv2d, bar_w,
                 color=COLOR_CONV2D, label="Conv2d", zorder=3)
    b2 = ax1.bar(x + bar_w / 2, speedup_triton, bar_w,
                 color=COLOR_TRITON, label="Triton", zorder=3)
    ax1.set_ylabel("Speedup vs Original LF")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs)
    ax1.set_ylim(0, 28)
    ax1.legend(frameon=False, loc="upper left")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.set_title("Speedup vs Original LF", pad=10)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                     f"{h:.1f}x", ha="center", va="bottom", fontsize=7)

    bar_w2 = 0.22
    b1 = ax2.bar(x - bar_w2, vram_original, bar_w2,
                 color=COLOR_ORIGINAL, label="Original LF", zorder=3)
    b2 = ax2.bar(x, vram_conv2d, bar_w2,
                 color=COLOR_CONV2D, label="Conv2d", zorder=3)
    b3 = ax2.bar(x + bar_w2, vram_triton, bar_w2,
                 color=COLOR_TRITON, label="Triton", zorder=3)
    ax2.set_ylabel("Peak VRAM (MB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs)
    ax2.legend(frameon=False, loc="upper left", fontsize=7)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.set_title("Peak VRAM (MB)", pad=10)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 50,
                     f"{int(h)}", ha="center", va="bottom", fontsize=6,
                     rotation=45)

    ax1.text(-0.08, 1.08, "A", transform=ax1.transAxes,
             fontsize=13, fontweight="bold", va="bottom")
    ax2.text(-0.08, 1.08, "B", transform=ax2.transAxes,
             fontsize=13, fontweight="bold", va="bottom")

    fig.tight_layout(w_pad=2.5)
    _save(fig, "fig4_speedup_vram")


if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")
    figure1()
    figure2()
    figure3()
    figure4()
    print("\nDone.")
