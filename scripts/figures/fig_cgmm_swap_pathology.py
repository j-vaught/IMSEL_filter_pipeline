"""Figure C: §7.1 swap-pathology schematic.

Three-panel illustration of why K=2 fails and K=3 succeeds at a
junction pixel.  Synthetic LF response curves (vM-shaped, matched to
the real corner_v0 m-sweep statistics) make the mechanism visible
without relying on whichever real configurations happen to swap
on this particular test image.

Panel 1: LF response at one configuration -- two peaks at the two
         real edge directions; primary = larger, secondary = smaller.
Panel 2: same junction at a different configuration where the two
         peaks' magnitudes have flipped -- primary/secondary labels
         now sit on the opposite peaks.
Panel 3: pooled primary-set across N configurations on the
         doubled-angle domain; stems clustered around the two real
         directions plus noise scatter, illustrating the three
         populations a K=3 mixture is designed to separate.

Output: cetz_figures/pdfs/fig_cgmm_swap_pathology.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BRAND = {
    "atlantic":   "#466A9F",
    "rose":       "#CC2E40",
    "honeycomb":  "#A49137",
    "garnet":     "#73000A",
    "warmgrey":   "#676156",
    "blk-90":     "#363636",
    "blk-70":     "#5C5C5C",
    "blk-50":     "#A2A2A2",
    "blk-30":     "#C7C7C7",
    "blk-10":     "#ECECEC",
    "horseshoe":  "#65780B",
}


def vm_density(theta_rad, mu_rad, kappa):
    """Unnormalised vM density on theta-space (theta in [0, pi))."""
    # On the half-circle we use phi = 2 theta to avoid wrap issues.
    return np.exp(kappa * (np.cos(2 * theta_rad - 2 * mu_rad) - 1))


def synthetic_lf(theta_grid_rad, mu_a_deg, mu_b_deg, kappa,
                 amp_a, amp_b, noise_floor, rng):
    a = amp_a * vm_density(theta_grid_rad, np.deg2rad(mu_a_deg), kappa)
    b = amp_b * vm_density(theta_grid_rad, np.deg2rad(mu_b_deg), kappa)
    noise = noise_floor * rng.normal(scale=0.4, size=theta_grid_rad.size)
    return a + b + np.maximum(noise, 0)


def main():
    repo = Path(__file__).resolve().parents[2]
    out = repo.parent / "New project" / "cetz_figures" / "pdfs" / "fig_cgmm_swap_pathology.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)

    # Real corner_v0 GT directions (degrees).
    GT_A_DEG = 64.0      # primary edge in most real configs
    GT_B_DEG = 152.0     # secondary edge in most real configs

    # Realistic magnitudes from the m-sweep data (channel L, m=40):
    # M_hat=42.5, M_sec=36.4 -> ratio ~ 0.86 (config 1).
    # We construct a swap by swapping the two amplitudes for config 2.
    AMP_HIGH = 42.5
    AMP_LOW  = 36.4
    KAPPA    = 70.0
    NOISE    = 1.6

    theta_deg = np.linspace(0, 180, 1000, endpoint=False)
    theta_rad = np.deg2rad(theta_deg)

    # Two configurations with magnitudes swapped.
    resp_1 = synthetic_lf(theta_rad, GT_A_DEG, GT_B_DEG, KAPPA,
                          AMP_HIGH, AMP_LOW, NOISE, rng)
    resp_2 = synthetic_lf(theta_rad, GT_A_DEG, GT_B_DEG, KAPPA,
                          AMP_LOW, AMP_HIGH, NOISE, rng)

    # ------------------ figure ------------------
    fig = plt.figure(figsize=(10.0, 3.4))
    gs = fig.add_gridspec(1, 3, wspace=0.32,
                          left=0.06, right=0.98,
                          top=0.86, bottom=0.21)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # --- Panel 1: config A ---
    ax1.plot(theta_deg, resp_1, color=BRAND["blk-70"], linewidth=1.2)
    ax1.fill_between(theta_deg, 0, resp_1, color=BRAND["blk-30"],
                     alpha=0.25)
    ax1.axvline(GT_A_DEG, color=BRAND["atlantic"], linewidth=0.8,
                linestyle="--", alpha=0.5)
    ax1.axvline(GT_B_DEG, color=BRAND["rose"], linewidth=0.8,
                linestyle="--", alpha=0.5)
    # Annotate peaks: primary at A (larger), secondary at B (smaller)
    p1_y = resp_1[np.argmin(np.abs(theta_deg - GT_A_DEG))]
    p2_y = resp_1[np.argmin(np.abs(theta_deg - GT_B_DEG))]
    ax1.annotate("primary", xy=(GT_A_DEG, p1_y),
                 xytext=(GT_A_DEG, p1_y * 1.30),
                 ha="center", fontsize=9, color=BRAND["atlantic"],
                 weight="bold",
                 arrowprops=dict(arrowstyle="-", color=BRAND["atlantic"],
                                  linewidth=0.8))
    ax1.annotate("secondary", xy=(GT_B_DEG, p2_y),
                 xytext=(GT_B_DEG, p2_y * 1.30),
                 ha="center", fontsize=9, color=BRAND["rose"],
                 weight="bold",
                 arrowprops=dict(arrowstyle="-", color=BRAND["rose"],
                                  linewidth=0.8))
    ax1.set_xlim(0, 180)
    ax1.set_ylim(0, max(resp_1.max(), resp_2.max()) * 1.55)
    ax1.set_xticks([0, 45, 90, 135, 180])
    ax1.set_xticklabels(["0°", "45°", "90°", "135°", "180°"], fontsize=8)
    ax1.set_ylabel("LF response (a.u.)", fontsize=9)
    ax1.set_xlabel(r"orientation $\theta$", fontsize=9)
    ax1.set_title("config A", fontsize=10, color=BRAND["blk-90"])
    ax1.tick_params(labelsize=8)
    for s in ("top", "right"): ax1.spines[s].set_visible(False)

    # --- Panel 2: config B (swap) ---
    ax2.plot(theta_deg, resp_2, color=BRAND["blk-70"], linewidth=1.2)
    ax2.fill_between(theta_deg, 0, resp_2, color=BRAND["blk-30"],
                     alpha=0.25)
    ax2.axvline(GT_A_DEG, color=BRAND["atlantic"], linewidth=0.8,
                linestyle="--", alpha=0.5)
    ax2.axvline(GT_B_DEG, color=BRAND["rose"], linewidth=0.8,
                linestyle="--", alpha=0.5)
    p1b = resp_2[np.argmin(np.abs(theta_deg - GT_A_DEG))]
    p2b = resp_2[np.argmin(np.abs(theta_deg - GT_B_DEG))]
    ax2.annotate("secondary", xy=(GT_A_DEG, p1b),
                 xytext=(GT_A_DEG, p1b * 1.45),
                 ha="center", fontsize=9, color=BRAND["rose"],
                 weight="bold",
                 arrowprops=dict(arrowstyle="-", color=BRAND["rose"],
                                  linewidth=0.8))
    ax2.annotate("primary", xy=(GT_B_DEG, p2b),
                 xytext=(GT_B_DEG, p2b * 1.30),
                 ha="center", fontsize=9, color=BRAND["atlantic"],
                 weight="bold",
                 arrowprops=dict(arrowstyle="-", color=BRAND["atlantic"],
                                  linewidth=0.8))

    # Inter-panel "swap" annotation drawn directly on the figure canvas.
    from matplotlib.patches import FancyArrowPatch
    swap_arrow = FancyArrowPatch(
        (0.345, 0.55), (0.385, 0.55),
        transform=fig.transFigure,
        arrowstyle="->", color=BRAND["garnet"], linewidth=1.5,
        mutation_scale=14)
    fig.patches.append(swap_arrow)
    fig.text(0.365, 0.62, "labels swap", color=BRAND["garnet"],
             ha="center", fontsize=8.5, weight="bold")

    ax2.set_xlim(0, 180)
    ax2.set_ylim(0, max(resp_1.max(), resp_2.max()) * 1.55)
    ax2.set_xticks([0, 45, 90, 135, 180])
    ax2.set_xticklabels(["0°", "45°", "90°", "135°", "180°"], fontsize=8)
    ax2.set_xlabel(r"orientation $\theta$", fontsize=9)
    ax2.set_title("config B (magnitudes flipped)", fontsize=10,
                  color=BRAND["blk-90"])
    ax2.tick_params(labelsize=8)
    for s in ("top", "right"): ax2.spines[s].set_visible(False)

    # --- Panel 3: pooled primary-set on doubled-angle ---
    # 40 measurements pooled across N=40 configs (4 channels x 10 m).
    # In a true junction with swaps, half the configs report edge A as
    # primary and half report edge B; plus a few noise outliers.
    n_per_edge = 16
    n_noise    = 8

    # Per-config primary samples drawn around the two real edges.
    a_samples = rng.normal(loc=GT_A_DEG, scale=1.5, size=n_per_edge)
    b_samples = rng.normal(loc=GT_B_DEG, scale=2.0, size=n_per_edge)
    noise_samples = rng.uniform(0, 180, size=n_noise)
    primary_samples_deg = np.concatenate([a_samples, b_samples,
                                           noise_samples])
    # Magnitudes drawn around the realistic AMP_HIGH for "real" edges,
    # smaller for the noise outliers.
    a_M = rng.normal(loc=AMP_HIGH, scale=4.0, size=n_per_edge)
    b_M = rng.normal(loc=AMP_HIGH, scale=4.0, size=n_per_edge)
    noise_M = rng.uniform(8, 22, size=n_noise)
    M_samples = np.concatenate([a_M, b_M, noise_M])
    # Doubled-angle phi for the abscissa.
    phi_samples_deg = (2.0 * primary_samples_deg) % 360.0

    # Stem plot.
    ax3.set_xlim(0, 360)
    ax3.set_ylim(0, max(M_samples.max(), AMP_HIGH * 1.25))
    # GT verticals on doubled-angle.
    for gt in (GT_A_DEG, GT_B_DEG):
        ax3.axvline((2 * gt) % 360, color=BRAND["horseshoe"],
                    linestyle="--", linewidth=0.8, alpha=0.6)
    # Color samples by which population they belong to.
    colors = ([BRAND["atlantic"]] * n_per_edge
              + [BRAND["rose"]] * n_per_edge
              + [BRAND["warmgrey"]] * n_noise)
    for x, m, c in zip(phi_samples_deg, M_samples, colors):
        ax3.vlines(x, 0, m, color=c, linewidth=1.0)
        ax3.plot([x], [m], "o", color=c, markersize=2.8)
    # Population labels.
    ax3.text((2 * GT_A_DEG) % 360, M_samples.max() * 1.13,
             "edge A samples", color=BRAND["atlantic"], fontsize=8.5,
             ha="center", weight="bold")
    ax3.text((2 * GT_B_DEG) % 360, M_samples.max() * 1.13,
             "edge B samples", color=BRAND["rose"], fontsize=8.5,
             ha="center", weight="bold")
    ax3.text(15, M_samples.max() * 0.55, "noise\nscatter",
             color=BRAND["warmgrey"], fontsize=8, ha="left")

    ax3.set_xticks([0, 90, 180, 270, 360])
    ax3.set_xticklabels(["0°", "90°", "180°", "270°", "360°"], fontsize=8)
    ax3.set_xlabel(r"doubled angle $\phi = 2\theta$", fontsize=9)
    ax3.set_ylabel(r"$M_n$", fontsize=9)
    ax3.set_title("pooled primary-set across N configs",
                  fontsize=10, color=BRAND["blk-90"])
    ax3.tick_params(labelsize=8)
    for s in ("top", "right"): ax3.spines[s].set_visible(False)

    fig.savefig(out, format="pdf", bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
