"""Figure B: vMM hard-EM convergence trace at three canonical pixels.

Renders 3 rows (edge / corner / off-edge) by 2 columns (mu_k vs iter,
kappa_k vs iter, log-y).  Each panel has 3 lines (one per component)
in atlantic / rose / honeycomb.  Vertical reference lines at iter 10
(empirical convergence horizon, dashed) and iter 30 (cap, solid).

Data: outputs/cgmm_vmm_trace/trace_K3.npz (run cgmm_vmm_trace.py first).
Output: cetz_figures/pdfs/fig_cgmm_em_trace.pdf
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BRAND = {
    "atlantic":  "#466A9F",
    "rose":      "#CC2E40",
    "honeycomb": "#A49137",
    "warmgrey":  "#676156",
    "blk-90":    "#363636",
    "blk-50":    "#A2A2A2",
    "blk-30":    "#C7C7C7",
}
COMP_COLORS = (BRAND["atlantic"], BRAND["rose"], BRAND["honeycomb"])
ROW_TITLES = {
    "edge":   "edge (unimodal)",
    "corner": "corner (bimodal)",
    "blank":  "off-edge (no consensus)",
}


def main():
    repo = Path(__file__).resolve().parents[2]
    npz_path = repo / "outputs/cgmm_vmm_trace/trace_K3.npz"
    out_path = repo.parent / "New project" / "cetz_figures" / "pdfs" / "fig_cgmm_em_trace.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = np.load(npz_path, allow_pickle=False)
    labels   = [str(s) for s in d["labels"]]
    mu       = d["hard_mu_trace"]            # (n_iters+1, P, K)
    kappa    = d["hard_kappa_trace"]
    n_iters_p1, P, K = mu.shape
    iters = np.arange(n_iters_p1)            # 0..30

    # Convert mu (radians, [0, 2pi)) to degrees on doubled-angle space
    mu_deg = np.degrees(mu) % 360.0

    fig, axes = plt.subplots(P, 2, figsize=(7.0, 6.6),
                             sharex=True, gridspec_kw=dict(
                                 hspace=0.35, wspace=0.30,
                                 left=0.10, right=0.97,
                                 top=0.93, bottom=0.08))

    for r, label in enumerate(labels):
        # Left column: mu_k vs iter
        ax_mu = axes[r, 0]
        for k in range(K):
            ax_mu.plot(iters, mu_deg[:, r, k],
                       color=COMP_COLORS[k], linewidth=1.6,
                       marker="o", markersize=2.5,
                       label=f"k={k+1}")
        ax_mu.axvline(10, color=BRAND["blk-50"], linestyle="--",
                      linewidth=0.8, zorder=0)
        ax_mu.axvline(30, color=BRAND["blk-30"], linestyle="-",
                      linewidth=0.8, zorder=0)
        ax_mu.set_ylim(0, 360)
        ax_mu.set_yticks([0, 90, 180, 270, 360])
        ax_mu.set_ylabel(r"$\mu_k$ (deg)", fontsize=9)
        if r == 0:
            ax_mu.set_title(r"component mean $\mu_k$ (doubled-angle)",
                            fontsize=10, color=BRAND["blk-90"])
        ax_mu.tick_params(labelsize=8)
        ax_mu.spines["top"].set_visible(False)
        ax_mu.spines["right"].set_visible(False)

        # Right column: kappa_k vs iter, log y
        ax_ka = axes[r, 1]
        for k in range(K):
            ax_ka.plot(iters, np.maximum(kappa[:, r, k], 1.0),
                       color=COMP_COLORS[k], linewidth=1.6,
                       marker="o", markersize=2.5)
        ax_ka.axvline(10, color=BRAND["blk-50"], linestyle="--",
                      linewidth=0.8, zorder=0)
        ax_ka.axvline(30, color=BRAND["blk-30"], linestyle="-",
                      linewidth=0.8, zorder=0)
        ax_ka.set_yscale("log")
        ax_ka.set_ylim(1, 1e3)
        ax_ka.set_ylabel(r"$\kappa_k$ (log)", fontsize=9)
        if r == 0:
            ax_ka.set_title(r"component concentration $\kappa_k$",
                            fontsize=10, color=BRAND["blk-90"])
        ax_ka.tick_params(labelsize=8)
        ax_ka.spines["top"].set_visible(False)
        ax_ka.spines["right"].set_visible(False)

        # Row label on the left side
        ax_mu.text(-0.32, 0.5, ROW_TITLES[label], transform=ax_mu.transAxes,
                   fontsize=10, color=BRAND["blk-90"], rotation=90,
                   ha="center", va="center", weight="bold")

    axes[-1, 0].set_xlabel("EM iteration", fontsize=9)
    axes[-1, 1].set_xlabel("EM iteration", fontsize=9)
    axes[-1, 0].set_xlim(0, n_iters_p1 - 1)

    # Top-right legend (component colors), placed once for the whole grid
    handles = [plt.Line2D([0], [0], color=COMP_COLORS[k], lw=1.8,
                          marker="o", markersize=3,
                          label=f"component {k + 1}")
               for k in range(K)]
    handles += [
        plt.Line2D([0], [0], color=BRAND["blk-50"], lw=0.8,
                   linestyle="--", label="iter 10 (empirical)"),
        plt.Line2D([0], [0], color=BRAND["blk-30"], lw=0.8,
                   linestyle="-",  label="iter 30 (cap)"),
    ]
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, 1.005), ncol=5, frameon=False,
               fontsize=8.5)

    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
