#!/usr/bin/env python3
"""
Generate publication-quality figures for the Anisotropic Stencil Filter paper.
All math implemented inline -- no external edge-detection libraries.
Only dependencies: numpy, matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from collections import defaultdict
import os

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "papers", "anisotropic-lf-proposal", "figures_v2",
)
os.makedirs(OUT_DIR, exist_ok=True)

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# ── Math helpers ─────────────────────────────────────────────────────────────

def get_circular_neighbors(n_neighbors=15, radius_max=6):
    """Return the n_neighbors closest integer grid points to origin (excl. origin)."""
    candidates = []
    for dx in range(-radius_max, radius_max + 1):
        for dy in range(-radius_max, radius_max + 1):
            if dx == 0 and dy == 0:
                continue
            candidates.append((dx, dy, np.sqrt(dx**2 + dy**2)))
    candidates.sort(key=lambda t: (t[2], t[0], t[1]))
    return [(c[0], c[1]) for c in candidates[:n_neighbors]]


def build_taylor_matrix(coords, order=4):
    """Taylor monomial design matrix up to given order."""
    x = coords[:, 0]
    y = coords[:, 1]
    cols = [np.ones_like(x)]
    if order >= 1:
        cols += [x, y]
    if order >= 2:
        cols += [x**2 / 2, y**2 / 2, x * y]
    if order >= 3:
        cols += [x**3 / 6, y**3 / 6, x**2 * y / 2, x * y**2 / 2]
    if order >= 4:
        cols += [x**4 / 24, y**4 / 24, x**3 * y / 6,
                 x**2 * y**2 / 4, x * y**3 / 6]
    return np.column_stack(cols)


def compute_stencil(theta, m=7, n_neighbors=15, order=4):
    """Compute fused stencil weights for orientation theta.

    Returns dict (dx, dy) -> fused weight for the f_x derivative.
    Uses Gaussian weighting along the line (sigma = m/2).
    """
    neighbors = get_circular_neighbors(n_neighbors)
    nbr_arr = np.array(neighbors, dtype=float)

    cos_t, sin_t = np.cos(theta), np.sin(theta)
    local_x = nbr_arr[:, 0] * cos_t + nbr_arr[:, 1] * sin_t
    local_y = -nbr_arr[:, 0] * sin_t + nbr_arr[:, 1] * cos_t
    local_coords = np.column_stack([local_x, local_y])

    A = build_taylor_matrix(local_coords, order)
    P = np.linalg.pinv(A)
    p_fx = P[1, :]  # f_x extraction row

    # Gaussian weights along line
    js = np.arange(-m, m + 1)
    sigma = m / 2.0
    w = np.exp(-0.5 * (js / sigma) ** 2)
    w /= w.sum()

    fused = defaultdict(float)
    for j_idx, j in enumerate(js):
        vx = round(j * cos_t)
        vy = round(j * sin_t)
        for n_idx, (dx, dy) in enumerate(neighbors):
            fused[(vx + dx, vy + dy)] += w[j_idx] * p_fx[n_idx]

    return fused


def _save(fig, name):
    fig.savefig(os.path.join(OUT_DIR, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT_DIR, f"{name}.png"))
    print(f"  Saved {name}.pdf / .png")
    plt.close(fig)


def _draw_grid(ax, half, lw=0.3, color="#d5d8dc"):
    """Draw faint pixel gridlines."""
    for i in range(-half, half + 2):
        ax.axhline(i - 0.5, color=color, linewidth=lw, zorder=0)
        ax.axvline(i - 0.5, color=color, linewidth=lw, zorder=0)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1a: How a Point Filter Works
# ══════════════════════════════════════════════════════════════════════════════
def figure1a():
    print("Figure 1a: How a Point Filter Works")
    neighbors = get_circular_neighbors(34)
    theta_axis = np.radians(30)
    grid_half = 7

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-grid_half - 0.7, grid_half + 0.7)
    ax.set_ylim(-grid_half - 0.7, grid_half + 0.7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    _draw_grid(ax, grid_half)

    # Dashed circle showing neighborhood boundary
    max_r = max(np.sqrt(dx**2 + dy**2) for dx, dy in neighbors)
    circle = plt.Circle((0, 0), max_r + 0.15, fill=False, linestyle="--",
                         edgecolor="#7f8c8d", linewidth=1.0, zorder=2)
    ax.add_patch(circle)

    # Neighbor pixels as blue circles
    for dx, dy in neighbors:
        ax.plot(dx, dy, "o", color="#3498db", ms=8, zorder=4,
                markeredgecolor="#2980b9", markeredgewidth=0.5)

    # Center pixel as red square
    ax.add_patch(plt.Rectangle((-0.4, -0.4), 0.8, 0.8,
                                facecolor="#e74c3c", edgecolor="#c0392b",
                                linewidth=1.2, zorder=5))

    # Arrows from a few neighbors toward center with labels
    # Use well-spaced actual neighbors from Np=15
    arrow_sources = [(-2, 1), (2, 0), (1, -1), (-1, -2)]
    label_offsets = [(-1.2, 0.6), (1.2, 0.6), (1.2, -0.6), (-1.2, -0.6)]
    for i, ((sx, sy), (lox, loy)) in enumerate(zip(arrow_sources, label_offsets)):
        if (sx, sy) not in neighbors:
            continue
        # Arrow from neighbor toward center, stopping short
        dx_n = -sx / np.sqrt(sx**2 + sy**2)
        dy_n = -sy / np.sqrt(sx**2 + sy**2)
        ax.annotate(
            "", xy=(sx + dx_n * 0.6, sy + dy_n * 0.6),
            xytext=(sx, sy),
            arrowprops=dict(arrowstyle="->,head_width=0.15,head_length=0.12",
                            color="#2c3e50", lw=1.3),
            zorder=6,
        )
        # Label offset from the source pixel
        ax.text(sx + lox, sy + loy,
                f"$f(x_{{{i+1}}}, y_{{{i+1}}})$", fontsize=8.5, color="#2c3e50",
                ha="center", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="none", alpha=0.7))

    # Local coordinate axes (rotated ~30 deg)
    axis_len = 3.5
    cos_a, sin_a = np.cos(theta_axis), np.sin(theta_axis)
    # x-axis (normal direction) - green
    ax.annotate("", xy=(axis_len * cos_a, axis_len * sin_a), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.2",
                                color="#27ae60", lw=2.0), zorder=8)
    ax.text(axis_len * cos_a + 0.3, axis_len * sin_a + 0.3,
            "normal ($\\hat{x}$)", fontsize=9, color="#27ae60",
            fontweight="bold", zorder=8)
    # y-axis (tangent direction) - green
    ax.annotate("", xy=(-axis_len * sin_a, axis_len * cos_a), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.2",
                                color="#27ae60", lw=2.0), zorder=8)
    ax.text(-axis_len * sin_a - 0.3, axis_len * cos_a + 0.4,
            "tangent ($\\hat{y}$)", fontsize=9, color="#27ae60",
            fontweight="bold", ha="right", zorder=8)

    ax.set_title("Point Filter: Polynomial Fit over Circular Neighborhood", pad=12)
    fig.tight_layout()
    _save(fig, "fig1a_point_filter")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1b: How the Line Extension Works
# ══════════════════════════════════════════════════════════════════════════════
def figure1b():
    print("Figure 1b: How the Line Extension Works")
    neighbors = get_circular_neighbors(15)
    theta = np.radians(30)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    m = 2
    grid_half = 7

    # Specified display weights for j=-2..2
    js = np.arange(-m, m + 1)
    display_w = np.array([0.14, 0.61, 1.00, 0.61, 0.14])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-grid_half - 0.7, grid_half + 0.7)
    ax.set_ylim(-grid_half - 0.7, grid_half + 0.7)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    _draw_grid(ax, grid_half, lw=0.2, color="#e0e0e0")

    # Line through center
    line_len = grid_half
    ax.plot([-line_len * cos_t, line_len * cos_t],
            [-line_len * sin_t, line_len * sin_t],
            "--", color="#555555", linewidth=1.2, zorder=2)

    # Pastel colors for each virtual point's neighborhood
    pastel_colors = ["#ffb3ba", "#ffcba4", "#ffffba", "#baffc9", "#bae1ff"]
    max_r = max(np.sqrt(dx**2 + dy**2) for dx, dy in neighbors)

    # Draw semi-transparent circular regions and neighbor dots per virtual point
    for j_idx, j in enumerate(js):
        vx = round(j * cos_t)
        vy = round(j * sin_t)
        # exact position on line for circle center
        exact_x = j * cos_t
        exact_y = j * sin_t

        # Semi-transparent circular region
        circle = plt.Circle((exact_x, exact_y), max_r + 0.3,
                             facecolor=pastel_colors[j_idx], alpha=0.25,
                             edgecolor=pastel_colors[j_idx], linewidth=1.0,
                             linestyle="-", zorder=1)
        ax.add_patch(circle)

        # Neighbor dots
        for dx, dy in neighbors:
            ax.plot(vx + dx, vy + dy, "o",
                    color=pastel_colors[j_idx], ms=4, alpha=0.6, zorder=3,
                    markeredgecolor="none")

    # Virtual points as green diamonds (on top)
    for j_idx, j in enumerate(js):
        vx = round(j * cos_t)
        vy = round(j * sin_t)
        ax.plot(vx, vy, "D", color="#27ae60", ms=9, zorder=6,
                markeredgecolor="#1e8449", markeredgewidth=1.0)

        # Weight annotations using specified display values
        w_val = display_w[j_idx]
        # Stagger labels to avoid overlap: alternate above/below
        if j == -2:
            label_offset_x, label_offset_y = -1.8, -1.0
        elif j == -1:
            label_offset_x, label_offset_y = -1.8, 1.2
        elif j == 0:
            label_offset_x, label_offset_y = 0.0, -1.8
        elif j == 1:
            label_offset_x, label_offset_y = 1.8, 1.2
        else:  # j == 2
            label_offset_x, label_offset_y = 1.8, -1.0
        ax.text(vx + label_offset_x, vy + label_offset_y,
                f"$w_{{{j}}} = {w_val:.2f}$", fontsize=9,
                color="#2c3e50", ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="#bdc3c7", alpha=0.85),
                zorder=7)

    # Center pixel in red
    ax.add_patch(plt.Rectangle((-0.4, -0.4), 0.8, 0.8,
                                facecolor="#e74c3c", edgecolor="#c0392b",
                                linewidth=1.2, zorder=5))

    ax.set_title("Line Extension: Weighted Average of Multiple Polynomial Fits", pad=12)
    fig.tight_layout()
    _save(fig, "fig1b_line_extension")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1c: Stencil Fusion / Deduplication (two sub-panels)
# ══════════════════════════════════════════════════════════════════════════════
def figure1c():
    print("Figure 1c: Stencil Fusion / Deduplication")
    neighbors = get_circular_neighbors(15)
    theta = np.radians(30)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    m = 2
    js = np.arange(-m, m + 1)

    # Pastel colors matching fig1b
    pastel_colors = ["#ffb3ba", "#ffcba4", "#ffffba", "#baffc9", "#bae1ff"]

    # Collect all accesses with source info
    all_accesses = []  # list of ((px, py), j_idx)
    hit_count = defaultdict(int)
    total_accesses = 0
    for j_idx, j in enumerate(js):
        vx = round(j * cos_t)
        vy = round(j * sin_t)
        for dx, dy in neighbors:
            px, py = vx + dx, vy + dy
            all_accesses.append(((px, py), j_idx))
            hit_count[(px, py)] += 1
            total_accesses += 1

    n_unique = len(hit_count)
    reduction = 100.0 * (1.0 - n_unique / total_accesses)

    grid_half = 8

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    for ax in [ax1, ax2]:
        ax.set_xlim(-grid_half - 0.7, grid_half + 0.7)
        ax.set_ylim(-grid_half - 0.7, grid_half + 0.7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        _draw_grid(ax, grid_half, lw=0.2, color="#e8e8e8")

    # ── Left: Before Deduplication ───────────────────────────────────────────
    ax1.set_title("Before Deduplication", pad=10)

    # Draw dots with slight jitter for overlapping ones, colored by source
    # Group by position to apply jitter
    pos_groups = defaultdict(list)
    for (px, py), j_idx in all_accesses:
        pos_groups[(px, py)].append(j_idx)

    for (px, py), j_idxs in pos_groups.items():
        n = len(j_idxs)
        if n == 1:
            ax1.plot(px, py, "o", color=pastel_colors[j_idxs[0]], ms=5,
                     markeredgecolor="#666", markeredgewidth=0.3, zorder=3)
        else:
            # Fan out slightly so overlapping dots are visible
            angles_fan = np.linspace(0, 2 * np.pi, n, endpoint=False)
            jitter_r = 0.18
            for k, j_idx in enumerate(j_idxs):
                jx = px + jitter_r * np.cos(angles_fan[k])
                jy = py + jitter_r * np.sin(angles_fan[k])
                ax1.plot(jx, jy, "o", color=pastel_colors[j_idx], ms=4.5,
                         markeredgecolor="#666", markeredgewidth=0.3, zorder=3)

    ax1.text(0, -grid_half + 0.3, f"{total_accesses} total accesses",
             fontsize=11, ha="center", va="bottom", fontweight="bold",
             color="#c0392b",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#c0392b", alpha=0.9), zorder=8)

    # ── Big arrow between panels ─────────────────────────────────────────────
    fig.text(0.50, 0.50, "Fuse &\nDeduplicate",
             ha="center", va="center", fontsize=11, fontweight="bold",
             color="#2c3e50")
    # Arrow drawn in figure coordinates
    arrow = FancyArrowPatch(
        (0.46, 0.45), (0.54, 0.45),
        transform=fig.transFigure,
        arrowstyle="->,head_width=8,head_length=6",
        color="#2c3e50", linewidth=2.5,
        mutation_scale=1, zorder=10,
    )
    fig.patches.append(arrow)

    # ── Right: After Deduplication ───────────────────────────────────────────
    ax2.set_title("After Deduplication", pad=10)

    max_hits = max(hit_count.values())
    blues = plt.cm.Blues
    for (px, py), hits in hit_count.items():
        # Map 1..max_hits to 0.3..1.0 in the Blues colormap
        frac = 0.3 + 0.7 * (hits / max_hits)
        color = blues(frac)
        ax2.plot(px, py, "o", color=color, ms=7,
                 markeredgecolor="#333", markeredgewidth=0.4, zorder=4)

    # Legend for hit counts
    for h in sorted(set(hit_count.values())):
        frac = 0.3 + 0.7 * (h / max_hits)
        label = f"{h} access" if h == 1 else f"{h} accesses"
        ax2.plot([], [], "o", color=blues(frac), ms=7,
                 markeredgecolor="#333", markeredgewidth=0.4, label=label)
    ax2.legend(loc="upper right", frameon=True, framealpha=0.9,
               edgecolor="#ccc", fontsize=9)

    ax2.text(0, -grid_half + 0.3,
             f"{n_unique} unique positions ({reduction:.0f}% reduction)",
             fontsize=11, ha="center", va="bottom", fontweight="bold",
             color="#27ae60",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                       edgecolor="#27ae60", alpha=0.9), zorder=8)

    fig.suptitle("Stencil Fusion: Eliminating Redundant Memory Accesses",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93], w_pad=4.0)
    _save(fig, "fig1c_stencil_fusion")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Stencil Weights at Multiple Orientations (3x2 grid, bigger)
# ══════════════════════════════════════════════════════════════════════════════
def figure3():
    print("Figure 3: Stencil Weights at Multiple Orientations")
    angles_deg = [0, 30, 60, 90, 120, 150]

    # Compute all stencils
    stencils = []
    for ang in angles_deg:
        s = compute_stencil(np.radians(ang), m=7, n_neighbors=15, order=4)
        stencils.append(s)

    # Global weight range for consistent colormap
    all_weights = []
    for s in stencils:
        all_weights.extend(s.values())
    vmax = max(abs(min(all_weights)), abs(max(all_weights)))

    cmap = plt.cm.RdBu_r
    grid_half = 10

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))

    sc = None  # for colorbar reference
    for idx, (ang, stencil) in enumerate(zip(angles_deg, stencils)):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        ax.set_xlim(-grid_half - 0.7, grid_half + 0.7)
        ax.set_ylim(-grid_half - 0.7, grid_half + 0.7)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Pixel gridlines
        _draw_grid(ax, grid_half, lw=0.2, color="#eaecee")

        # Stencil dots
        xs = [p[0] for p in stencil]
        ys = [p[1] for p in stencil]
        ws = [stencil[p] for p in stencil]

        sc = ax.scatter(xs, ys, c=ws, cmap=cmap, vmin=-vmax, vmax=vmax,
                        s=70, edgecolors="none", zorder=3)

        # Center cross
        ax.plot(0, 0, "+", color="black", ms=10, mew=2.0, zorder=5)

        # Normal direction arrow at center
        arr_len = 3.0
        rad = np.radians(ang)
        ax.annotate("",
                     xy=(arr_len * np.cos(rad), arr_len * np.sin(rad)),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.2",
                                     color="#2c3e50", lw=1.8),
                     zorder=6)

        n_unique = len(stencil)
        ax.set_title(f"$\\theta = {ang}^\\circ$ ({n_unique} unique pts)", pad=8)

    # Shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.25, wspace=0.12)
    cbar_ax = fig.add_axes([0.90, 0.12, 0.02, 0.76])
    cb = fig.colorbar(sc, cax=cbar_ax)
    cb.set_label("Fused weight $\\alpha$", fontsize=11)

    _save(fig, "fig3_stencil_orientations")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"Output directory: {OUT_DIR}\n")
    figure1a()
    figure1b()
    figure1c()
    figure3()
    print("\nDone.")
