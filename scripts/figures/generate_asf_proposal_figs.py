"""Improved paper figures v3 — denser grids, better colors for stencil visualization."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.colors import Normalize
from matplotlib import cm
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "papers", "anisotropic-lf-proposal", "figures_v3")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.facecolor': 'white',
})


# =====================================================================
# Math helpers
# =====================================================================
def get_circular_neighbors(Np, radius=None):
    if radius is None:
        radius = int(np.ceil(np.sqrt(Np / np.pi))) + 1
    candidates = []
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            dist = np.sqrt(dx**2 + dy**2)
            if dist <= radius:
                candidates.append((dx, dy, dist))
    candidates.sort(key=lambda c: c[2])
    return np.array([(c[0], c[1]) for c in candidates[:Np]], dtype=float)


def rotate_coords(coords, theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, s], [-s, c]])
    return coords @ R.T


def build_taylor_matrix(coords, order=4):
    x, y = coords[:, 0], coords[:, 1]
    n = len(x)
    cols = [np.ones(n)]
    if order >= 1:
        cols += [x, y]
    if order >= 2:
        cols += [x**2/2, y**2/2, x*y]
    if order >= 3:
        cols += [x**3/6, y**3/6, x**2*y/2, x*y**2/2]
    if order >= 4:
        cols += [x**4/24, y**4/24, x**3*y/6, x**2*y**2/4, x*y**3/6]
    return np.column_stack(cols)


def build_stencil(theta, half_width, Np, order=4):
    neighbors = get_circular_neighbors(Np)
    local = rotate_coords(neighbors, theta)
    A = build_taylor_matrix(local, order)
    P = np.linalg.pinv(A)
    pfx = P[1, :]

    sigma = half_width / 2.0
    j_range = np.arange(-half_width, half_width + 1, dtype=float)
    gauss_w = np.exp(-0.5 * (j_range / sigma) ** 2)
    gauss_w /= gauss_w.sum()

    stencil = {}
    for ji, j in enumerate(range(-half_width, half_width + 1)):
        vdx = int(round(j * np.cos(theta)))
        vdy = int(round(j * np.sin(theta)))
        for ni in range(Np):
            ndx, ndy = int(neighbors[ni, 0]), int(neighbors[ni, 1])
            key = (vdy + ndy, vdx + ndx)
            alpha = gauss_w[ji] * pfx[ni]
            stencil[key] = stencil.get(key, 0.0) + alpha

    offsets = np.array(list(stencil.keys()))
    weights = np.array(list(stencil.values()))
    return offsets, weights


# =====================================================================
# Figure 1a: Point Filter — DENSE pixel grid
# =====================================================================
def make_fig1a():
    fig, ax = plt.subplots(figsize=(7, 7))

    # Dense grid — show ALL pixels as small gray dots
    grid_r = 8
    for gx in range(-grid_r, grid_r + 1):
        for gy in range(-grid_r, grid_r + 1):
            ax.plot(gx, gy, 'o', color='#d0d0d0', markersize=4, zorder=1)

    # Circular neighbors (Np=24 for a nicer circle)
    Np = 24
    neighbors = get_circular_neighbors(Np)
    radius = np.max(np.sqrt(neighbors[:, 0]**2 + neighbors[:, 1]**2)) + 0.5

    # Draw neighborhood circle
    circle = Circle((0, 0), radius, fill=False, edgecolor='#888888',
                     linestyle='--', linewidth=1.5, zorder=2)
    ax.add_patch(circle)

    # Draw neighbor pixels
    ax.scatter(neighbors[:, 0], neighbors[:, 1], s=120, c='#3498db',
               edgecolors='#2c3e50', linewidths=0.8, zorder=4)

    # Center pixel
    ax.plot(0, 0, 's', color='#e74c3c', markersize=14, markeredgecolor='#c0392b',
            markeredgewidth=1.5, zorder=5)

    # Rotated coordinate axes at 30 degrees
    theta = np.radians(30)
    ax_len = 4.5
    # Normal (x-hat)
    ax.annotate('', xy=(ax_len*np.cos(theta), ax_len*np.sin(theta)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
    ax.text(ax_len*np.cos(theta) + 0.3, ax_len*np.sin(theta) + 0.3,
            r'normal ($\hat{x}$)', color='#27ae60', fontsize=11, fontweight='bold')
    # Tangent (y-hat)
    ax.annotate('', xy=(-ax_len*np.sin(theta), ax_len*np.cos(theta)),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2.5))
    ax.text(-ax_len*np.sin(theta) - 0.5, ax_len*np.cos(theta) + 0.5,
            r'tangent ($\hat{y}$)', color='#27ae60', fontsize=11, fontweight='bold')

    # Intensity gather arrows for a few pixels
    arrow_indices = [0, 5, 12, 20]
    for idx in arrow_indices:
        nx, ny = neighbors[idx]
        ax.annotate('', xy=(nx*0.3, ny*0.3), xytext=(nx, ny),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50',
                                    lw=1.2, shrinkA=6, shrinkB=3))
    # Labels
    ax.text(neighbors[0, 0] - 1.8, neighbors[0, 1] + 0.3,
            r'$f(x_i, y_i)$', fontsize=10, color='#2c3e50')

    ax.set_xlim(-grid_r - 0.5, grid_r + 0.5)
    ax.set_ylim(-grid_r - 0.5, grid_r + 0.5)
    ax.set_aspect('equal')
    ax.set_title(r'Point Filter: Polynomial Fit ($N_p = 24$)', fontsize=14, pad=15)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig1a_point_filter.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'fig1a_point_filter.pdf'), bbox_inches='tight')
    plt.close()
    print("  fig1a done")


# =====================================================================
# Figure 1b: Line Extension — dense grid, colored neighborhoods
# =====================================================================
def make_fig1b():
    fig, ax = plt.subplots(figsize=(9, 7))

    grid_r = 10
    for gx in range(-grid_r, grid_r + 1):
        for gy in range(-grid_r, grid_r + 1):
            ax.plot(gx, gy, 'o', color='#e0e0e0', markersize=3, zorder=1)

    theta = np.radians(30)
    Np = 24
    neighbors = get_circular_neighbors(Np)
    nb_radius = np.max(np.sqrt(neighbors[:, 0]**2 + neighbors[:, 1]**2)) + 0.5
    m = 3  # half_width

    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']
    sigma = m / 2.0

    # Draw line
    line_ext = 8
    ax.plot([-line_ext*np.cos(theta), line_ext*np.cos(theta)],
            [-line_ext*np.sin(theta), line_ext*np.sin(theta)],
            '--', color='#7f8c8d', linewidth=1.5, zorder=2)

    # Virtual points and their neighborhoods
    for ji, j in enumerate(range(-m, m + 1)):
        vx = j * np.cos(theta)
        vy = j * np.sin(theta)
        vx_r, vy_r = round(vx), round(vy)
        w = np.exp(-0.5 * (j / sigma) ** 2)
        w_norm = w / np.exp(0)  # normalize so center = 1

        color = colors[ji % len(colors)]

        # Draw neighborhood circle (transparency scales with weight)
        circle = Circle((vx_r, vy_r), nb_radius, fill=True,
                         facecolor=color, alpha=0.12 + 0.10 * w_norm,
                         edgecolor=color, linewidth=1.2, linestyle='-',
                         zorder=3)
        ax.add_patch(circle)

        # Draw neighbor pixels
        for ni in range(Np):
            nx = vx_r + neighbors[ni, 0]
            ny = vy_r + neighbors[ni, 1]
            ax.plot(nx, ny, 'o', color=color, markersize=4,
                    alpha=0.3 + 0.4 * w_norm, zorder=4)

        # Virtual point diamond
        ax.plot(vx_r, vy_r, 'D', color=color, markersize=10,
                markeredgecolor='#2c3e50', markeredgewidth=1.2, zorder=6)

        # Weight label
        w_display = f'{w / np.exp(0):.2f}'
        offset_y = 1.8 if j % 2 == 0 else -1.8
        ax.text(vx_r + 0.3, vy_r + offset_y,
                f'$w_{{{j}}}={w_display}$', fontsize=8.5,
                ha='center', color='#2c3e50',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='#bdc3c7', alpha=0.9))

    # Center pixel
    ax.plot(0, 0, 's', color='#e74c3c', markersize=12,
            markeredgecolor='#c0392b', markeredgewidth=1.5, zorder=7)

    ax.set_xlim(-grid_r - 0.5, grid_r + 0.5)
    ax.set_ylim(-grid_r + 2, grid_r - 2)
    ax.set_aspect('equal')
    ax.set_title(r'Line Extension: Weighted Average of Polynomial Fits ($m=3$, $N_p=24$)',
                 fontsize=13, pad=15)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig1b_line_extension.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'fig1b_line_extension.pdf'), bbox_inches='tight')
    plt.close()
    print("  fig1b done")


# =====================================================================
# Figure 1c: Stencil Fusion — before/after dedup with dense grid
# =====================================================================
def make_fig1c():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    theta = np.radians(30)
    Np = 24
    m = 3
    neighbors = get_circular_neighbors(Np)
    sigma = m / 2.0
    colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6', '#1abc9c']

    # Collect all accesses with source info
    all_positions = []  # (dy, dx, source_j)
    access_count = {}
    for ji, j in enumerate(range(-m, m + 1)):
        vdx = int(round(j * np.cos(theta)))
        vdy = int(round(j * np.sin(theta)))
        for ni in range(Np):
            ndx, ndy = int(neighbors[ni, 0]), int(neighbors[ni, 1])
            pos = (vdy + ndy, vdx + ndx)
            all_positions.append((pos[1], pos[0], ji))
            access_count[pos] = access_count.get(pos, 0) + 1

    # Grid background
    all_coords = [(p[0], p[1]) for p in all_positions]
    all_xs = [c[0] for c in all_coords]
    all_ys = [c[1] for c in all_coords]
    pad = 2
    xmin, xmax = min(all_xs) - pad, max(all_xs) + pad
    ymin, ymax = min(all_ys) - pad, max(all_ys) + pad

    for ax in [ax1, ax2]:
        for gx in range(int(xmin), int(xmax) + 1):
            for gy in range(int(ymin), int(ymax) + 1):
                ax.plot(gx, gy, 'o', color='#e8e8e8', markersize=3, zorder=1)

    # Panel 1: Before dedup — color by source virtual point
    for x, y, ji in all_positions:
        ax1.plot(x, y, 'o', color=colors[ji % len(colors)],
                 markersize=7, alpha=0.6, zorder=3)

    total = len(all_positions)
    ax1.set_title('Before Deduplication', fontsize=13, fontweight='bold')
    ax1.text(0.5, -0.06, f'{total} total memory accesses',
             transform=ax1.transAxes, ha='center', fontsize=12,
             color='#c0392b', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeaea',
                       edgecolor='#e74c3c'))

    # Panel 2: After dedup — color by access count
    unique_count = len(access_count)
    max_count = max(access_count.values())
    cmap = cm.get_cmap('YlOrRd')

    for (dy, dx), count in access_count.items():
        norm_count = count / max_count
        color = cmap(0.2 + 0.7 * norm_count)
        size = 6 + 4 * norm_count
        ax2.plot(dx, dy, 'o', color=color, markersize=size,
                 markeredgecolor='#2c3e50', markeredgewidth=0.5, zorder=3)

    reduction = (1 - unique_count / total) * 100
    ax2.set_title('After Deduplication', fontsize=13, fontweight='bold')
    ax2.text(0.5, -0.06, f'{unique_count} unique positions ({reduction:.0f}% reduction)',
             transform=ax2.transAxes, ha='center', fontsize=12,
             color='#27ae60', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaea',
                       edgecolor='#27ae60'))

    # Arrow between panels
    fig.text(0.50, 0.52, 'Fuse &\nDeduplicate', ha='center', va='center',
             fontsize=13, fontweight='bold', color='#2c3e50')
    ax_arrow = fig.add_axes([0.46, 0.45, 0.08, 0.06])
    ax_arrow.annotate('', xy=(1, 0.5), xytext=(0, 0.5),
                      arrowprops=dict(arrowstyle='->', lw=2.5, color='#2c3e50'))
    ax_arrow.axis('off')

    # Legend for panel 2
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.25),
               markersize=8, label='1 access'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.55),
               markersize=10, label='2 accesses'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.9),
               markersize=12, label=f'{max_count}+ accesses'),
    ]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9,
               framealpha=0.9)

    for ax in [ax1, ax2]:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect('equal')
        ax.axis('off')

    plt.suptitle(r'Stencil Fusion: Eliminating Redundant Memory Accesses ($\theta=30°$, $m=3$, $N_p=24$)',
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig1c_stencil_fusion.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'fig1c_stencil_fusion.pdf'), bbox_inches='tight')
    plt.close()
    print("  fig1c done")


# =====================================================================
# Figure 3: Stencil orientations — use MAGNITUDE coloring, not signed
# =====================================================================
def make_fig3():
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    angles_deg = [0, 30, 60, 90, 120, 150]
    m = 7
    Np = 15
    order = 4

    # Compute all stencils first to get global max for consistent colorbar
    all_data = []
    for deg in angles_deg:
        theta = np.radians(deg)
        offsets, weights = build_stencil(theta, m, Np, order)
        all_data.append((offsets, weights))

    global_max_mag = max(np.max(np.abs(w)) for _, w in all_data)

    for idx, (deg, (offsets, weights)) in enumerate(zip(angles_deg, all_data)):
        ax = axes[idx // 3, idx % 3]
        theta = np.radians(deg)

        # Background grid
        if len(offsets) > 0:
            pad = 3
            gmin = min(offsets.min() - pad, -pad)
            gmax = max(offsets.max() + pad, pad)
        else:
            gmin, gmax = -10, 10

        for gx in range(int(gmin), int(gmax) + 1):
            for gy in range(int(gmin), int(gmax) + 1):
                ax.plot(gx, gy, '.', color='#f0f0f0', markersize=1.5, zorder=1)

        # Color by WEIGHT MAGNITUDE — single-hue colormap
        magnitudes = np.abs(weights)
        norm = Normalize(vmin=0, vmax=global_max_mag)
        cmap = cm.get_cmap('inferno')

        # Sort by magnitude so bright ones render on top
        sort_idx = np.argsort(magnitudes)
        for si in sort_idx:
            dy, dx = offsets[si]
            mag = magnitudes[si]
            color = cmap(norm(mag))
            size = 4 + 8 * (mag / global_max_mag)
            ax.plot(dx, dy, 'o', color=color, markersize=size,
                    markeredgecolor='none', zorder=3)

        # Center crosshair
        ax.plot(0, 0, '+', color='black', markersize=8, markeredgewidth=1.5, zorder=5)

        # Normal direction arrow
        arrow_len = 4
        ax.annotate('', xy=(arrow_len * np.cos(theta), arrow_len * np.sin(theta)),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))

        ax.set_title(f'$\\theta = {deg}°$  ({len(offsets)} pts)', fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlim(gmin + 1, gmax - 1)
        ax.set_ylim(gmin + 1, gmax - 1)
        ax.axis('off')

    # Shared colorbar
    sm = cm.ScalarMappable(cmap='inferno', norm=Normalize(0, global_max_mag))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label(r'Weight magnitude $|\alpha_{k,\ell}|$', fontsize=11)

    plt.suptitle(r'Fused Stencil Shape Across Orientations ($m=7$, $N_p=15$, $d=4$)',
                 fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'fig3_stencil_orientations.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(OUT, 'fig3_stencil_orientations.pdf'), bbox_inches='tight')
    plt.close()
    print("  fig3 done")


# =====================================================================
# Run all
# =====================================================================
print("Generating v3 figures...")
make_fig1a()
make_fig1b()
make_fig1c()
make_fig3()
print(f"All saved to {OUT}")
