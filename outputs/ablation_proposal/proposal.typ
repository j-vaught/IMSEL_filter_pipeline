#set page(paper: "us-letter", margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#set figure(placement: auto)

#align(center)[
  #text(size: 16pt, weight: "bold")[WVF/LF Ablation Study Proposal]
  #v(0.3em)
  #text(size: 12pt)[Parameter Sweeps for Edge Detection Quality on BSDS500]
  #v(0.5em)
  J. C. Vaught #sym.dot.c Edge-Detection Filter Critique Project #sym.dot.c March 2026
]

#v(1em)

= Motivation

The initial single-image ablation on BSDS500 image \#100007 revealed that support size $N_p$ dominates edge quality while orientation count $N_s$ has negligible effect beyond 18 discrete samples. However, that study used coarse parameter steps ($N_p in {15, 50, 100, 250}$, $N_s in {18, 36, 72}$) and evaluated only a single test image. Two questions remain open: whether the single-image finding ($N_p = 50$ optimal, $F = 0.806$) generalizes across the full BSDS500 test set, and where exactly the performance peak lies when the parameter space is sampled at finer resolution.

This document proposes two ablation tiers to answer these questions. Recent infrastructure improvements make both tiers computationally tractable. The GPU-accelerated filter implementations in the `edgecritic` package reduce WVF runtime to 0.009 seconds per full-resolution image and LF to 0.05--0.3 seconds, while a vectorized evaluation pipeline using distance transforms and sorted-array binary search achieves a 120$times$ speedup over the original dilation-based threshold sweep. Together, these optimizations reduce the full 56-configuration, 200-image ablation from an estimated 5.2 hours to under 5 minutes.

= Mathematical Background

This section presents the WVF and LF operators with the ablation parameters explicitly identified. Boxed quantities are those varied in the sweeps; all other terms are fixed or derived.

== Wide View Filter

For a grayscale image $I$ and a target pixel $bold(p)$, the WVF selects the $N_p$ nearest integer-coordinate pixels within a circular neighborhood, excluding the origin. At each candidate orientation $theta_k in {0, 2 pi slash N_s, dots, 2 pi (N_s - 1) slash N_s}$, the neighbor positions are rotated into a local coordinate frame via

$ bold(q)_i^((k)) = mat(cos theta_k, sin theta_k; -sin theta_k, cos theta_k) (bold(r)_i - bold(p)), quad i = 1, dots, N_p $

where $bold(r)_i$ are the global neighbor positions. A design matrix $A in bb(R)^(N_p times M)$ is constructed from the 2D Taylor monomials up to order $d$, where $M = (d+1)(d+2) slash 2$ is the number of monomial terms. For $d = 4$, $M = 15$; the columns correspond to

$ 1, quad x, quad y, quad x^2/2, quad y^2/2, quad x y, quad x^3/6, quad y^3/6, quad x^2 y/2, quad x y^2/2, quad x^4/24, quad y^4/24, quad x^3 y/6, quad x^2 y^2/4, quad x y^3/6 $

evaluated at each neighbor's local coordinates $(x_i, y_i) = bold(q)_i^((k))$. The derivative coefficients are recovered by least squares:

$ hat(bold(c))^((k)) = (A^top A)^(-1) A^top bold(f) in bb(R)^M $

where $bold(f) = [I(bold(r)_1), dots, I(bold(r)_N_p)]^top$ is the vector of neighbor intensities. The edge response at orientation $theta_k$ is the magnitude of the estimated normal derivative $hat(c)_2^((k))$ (the $f_x$ coefficient). The WVF output at pixel $bold(p)$ is

$ R(bold(p)) = max_(k = 1, dots, N_s) |hat(c)_2^((k))|, quad quad theta^*(bold(p)) = theta_(arg max_k |hat(c)_2^((k))|) $

The three parameters varied in the ablation are therefore: *$N_p$*, which controls the spatial extent of the neighborhood and the row dimension of $A$; *$N_s$*, which controls the angular resolution of the orientation sweep; and *$d$*, which controls the column dimension $M$ of $A$ and the expressiveness of the local polynomial model.

== Line Filter

The LF extends the WVF by chaining $2m + 1$ WVF applications along a line centered on $bold(p)$ at each orientation $theta_k$. Virtual expansion points are placed at

$ bold(v)_j^((k)) = bold(p) + j (cos theta_k, sin theta_k), quad j = -m, dots, m $

and each virtual point produces a WVF normal-derivative estimate $hat(c)_(2,j)^((k))$. These are combined via a Gaussian-weighted average:

$ R_"LF"^((k))(bold(p)) = lr(|sum_(j = -m)^(m) w_j hat(c)_(2,j)^((k))|), quad quad w_j = exp(-j^2 slash 2 sigma^2) slash sum_(ell=-m)^(m) exp(-ell^2 slash 2 sigma^2) $

where $sigma$ controls the weighting bandwidth. The LF output is the orientation-maximized response, as with the WVF. In addition to the WVF parameters $N_p$, $N_s$, and $d$, the LF introduces two additional ablation parameters: the *half-width $m$*, which determines the number of virtual points along the line, and the *Gaussian bandwidth $sigma$*, which controls how rapidly the weights decay away from the center.

== Evaluation Metric

Edge quality is measured by the Optimal Dataset Scale (ODS) F-score. A continuous gradient magnitude map is binarized at threshold $t$ and matched against human-annotated ground truth within a spatial tolerance of $r$ pixels. Precision and recall are computed as

$ P(t) = "TP"(t) / ("TP"(t) + "FP"(t)), quad quad R(t) = "TP"(t) / ("TP"(t) + "FN"(t)) $

where a predicted edge pixel counts as a true positive if it falls within distance $r$ of any ground-truth edge, and a ground-truth pixel counts as a false negative if no prediction falls within distance $r$. The ODS is $max_t F(t)$ where $F = 2 P R slash (P + R)$, evaluated over the full dataset at a single global threshold. All ablation results report ODS at $r = 3$ pixels with 300 uniformly spaced thresholds.

= Tier 1: Full BSDS500 ODS Sweep

The goal of Tier 1 is to compute dataset-wide ODS and OIS for each filter--parameter configuration across all 200 BSDS500 test images. This provides statistically meaningful results that are directly comparable to published benchmarks.

The sweep covers both the Wide View Filter and the Line Filter. The WVF is parameterized by support size $N_p$ and orientation count $N_s$, with polynomial order fixed at $d = 4$. The LF adds a half-width parameter $m$ controlling the line extent, with Gaussian weighting at $sigma = m slash 2$.

#figure(
  align(center,
    table(
      columns: 4,
      align: (left, left, left, left),
      stroke: none,
      table.hline(),
      table.header([*Filter*], [*Parameters*], [*Line half-width $m$*], [*Notes*]),
      table.hline(),
      [WVF], [$N_p$, $N_s$, $d = 4$], [---], [Base filter],
      [LF], [$N_p$, $N_s$, $d = 4$, $m$], [3, 7, 14], [Gaussian-weighted line],
      table.hline(),
    )
  ),
  caption: [Filter configurations included in the Tier 1 sweep.],
)

The parameter grid crosses seven support sizes with two orientation counts for WVF, and additionally three line half-widths for LF, as shown in @tab:grid. This yields 14 WVF and 42 LF configurations for a total of 56 settings, each evaluated on all 200 test images (11,200 filter applications). Every configuration is scored at 300 uniformly spaced thresholds.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [$N_p$ (support size)], [15, 25, 50, 75, 100, 150, 250], [7],
      [$N_s$ (orientations)], [18, 36], [2],
      [$m$ (LF half-width)], [3, 7, 14], [3],
      table.hline(),
    )
  ),
  caption: [Parameter grid for the Tier 1 dataset-wide sweep.],
) <tab:grid>

The estimated runtime on an NVIDIA RTX 6000 Ada is broken down in @tab:runtime-t1. The filter computation is GPU-bound and completes in under 30 minutes even for LF, while the evaluation step---previously the dominant cost at 30--60 minutes with per-threshold binary dilation---now finishes in approximately 2.5 minutes thanks to the vectorized pipeline. The total wall-clock time for Tier 1 is estimated at 32 minutes.

#figure(
  align(center,
    table(
      columns: 4,
      align: (left, right, right, right),
      stroke: none,
      table.hline(),
      table.header([*Component*], [*Configs*], [*Per-image*], [*Total*]),
      table.hline(),
      [WVF filter (GPU)], [14], [0.009 s], [25 s],
      [LF filter (GPU)], [42], [0.05--0.3 s], [28 min],
      [Evaluation (vectorized)], [56], [0.014 s], [2.6 min],
      table.hline(),
      [*Total*], [], [], [*~32 min*],
      table.hline(),
    )
  ),
  caption: [Estimated Tier 1 runtime on RTX 6000 Ada (48 GB VRAM).],
) <tab:runtime-t1>

The outputs of Tier 1 are a full ODS/OIS table with 56 rows, heatmaps of ODS as a function of $N_p$ and $N_s$ for WVF and each LF half-width, precision--recall curves at the best threshold per configuration, and a comparison row with classical baselines (Sobel 3$times$3, Sobel 15$times$15, and Canny).

#pagebreak()

= Tier 2: High-Fidelity Single-Image Ablation

The goal of Tier 2 is to densely sample the parameter space on a single representative image in order to map the exact performance surface. The test image is BSDS500 \#100007 (polar bear on ice, 321$times$481), chosen for continuity with the initial ablation.

== WVF Dense Sweep

The WVF sweep varies three parameters simultaneously: support size $N_p$ at 20 levels spanning 5 to 500, orientation count $N_s$ at 10 levels from 9 to 180, and polynomial order $d$ from 2 to 5. This produces 800 configurations, each evaluated at 500 thresholds uniformly distributed in $[0.005, 0.995]$.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Range*], [*Count*]),
      table.hline(),
      [$N_p$], [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500], [20],
      [$N_s$], [9, 12, 18, 24, 36, 48, 72, 90, 120, 180], [10],
      [Polynomial order $d$], [2, 3, 4, 5], [4],
      table.hline(),
    )
  ),
  caption: [Parameter grid for the Tier 2 WVF dense sweep.],
)

The filter computation takes approximately $800 times 0.01 = 8$ seconds on GPU. The evaluation step, at 500 thresholds per configuration, would have required over an hour with the original dilation-based approach; the vectorized pipeline handles it in roughly 11 seconds ($800 times 0.014$ s). Total Tier 2 WVF time is under 20 seconds.

This sweep is designed to answer three questions. First, it locates the optimal $N_p$ to within $plus.minus 5$ pixels rather than the coarse 50-pixel jumps of the initial study. Second, it tests whether polynomial order $d$ matters---prior work uses $d = 4$ exclusively, but $d = 3$ may suffice (fewer monomials, better conditioning) and $d = 5$ may overfit. Third, it identifies the precise $N_s$ at which angular resolution saturates.

== LF Dense Sweep

The LF sweep holds polynomial order fixed at $d = 4$ and instead varies line half-width $m$ at 8 levels and Gaussian weighting bandwidth $sigma$ at 4 levels (expressed as a fraction of $m$, plus uniform weighting). Combined with 7 support sizes and 3 orientation counts, this produces 672 configurations.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Range*], [*Count*]),
      table.hline(),
      [$N_p$], [15, 25, 50, 75, 100, 150, 250], [7],
      [$N_s$], [18, 36, 72], [3],
      [$m$ (half-width)], [1, 2, 3, 5, 7, 10, 14, 20], [8],
      [Gaussian $sigma$ (fraction of $m$)], [0.25, 0.5, 1.0, $infinity$ (uniform)], [4],
      table.hline(),
    )
  ),
  caption: [Parameter grid for the Tier 2 LF dense sweep.],
)

At 0.2 seconds per image on average, the 672 LF filter runs complete in approximately 134 seconds. Evaluation adds another 9 seconds. Total Tier 2 LF time is under 2.5 minutes.

This sweep tests whether there is a better line length than the default $m = 7$, whether the Gaussian weighting bandwidth matters or uniform weighting is competitive, and how LF compares to WVF at matched $N_p$ and $N_s$.

== Visualizations

The Tier 2 results will be presented as six figure types: a 3D surface plot of best F-score as a function of $N_p$ and $N_s$ at each polynomial order $d$; a heatmap grid of $N_p$ versus $m$ for LF at fixed $N_s = 36$ with one panel per $sigma$ fraction; cross-section curves of F-score versus $N_p$ at fixed $N_s = 36$ overlaying WVF and LF at $m in {3, 7, 14}$; an order comparison plot of F-score versus $N_p$ for $d in {2, 3, 4, 5}$; side-by-side edge maps at the best configuration per filter type, the worst configuration, and the ground truth; and precision--recall curves for the top 5 and bottom 5 configurations.

= Runtime Summary

@tab:runtime-summary compares the estimated wall-clock time for each tier under the original evaluation pipeline (per-threshold binary dilation) and the optimized pipeline (distance transforms with vectorized threshold sweep).

#figure(
  align(center,
    table(
      columns: 4,
      align: (left, right, right, right),
      stroke: none,
      table.hline(),
      table.header([*Tier*], [*Filter time*], [*Eval (old)*], [*Eval (new)*]),
      table.hline(),
      [Tier 1 (56 configs $times$ 200 images)], [28 min], [5.2 h], [2.6 min],
      [Tier 2 WVF (800 configs $times$ 1 image)], [8 s], [56 min], [11 s],
      [Tier 2 LF (672 configs $times$ 1 image)], [2.2 min], [47 min], [9 s],
      table.hline(),
      [*Combined total*], [*31 min*], [*6.5 h*], [*3 min*],
      table.hline(),
    )
  ),
  caption: [Runtime comparison: old dilation-based evaluation versus vectorized pipeline. Filter times are identical in both cases (GPU-accelerated). The 120$times$ evaluation speedup reduces the combined runtime from approximately 7 hours to 34 minutes, with filter computation now dominating.],
) <tab:runtime-summary>

= Execution Plan

Tier 2 will be run first on the single test image, as the results are available in under 3 minutes and provide immediate feedback on the parameter surface shape. Once verified, Tier 1 will be launched on the full 200-image test set, requiring approximately 32 minutes. Both tiers run on an NVIDIA RTX 6000 Ada Generation (48 GB VRAM) accessed via Tailscale SSH. The `edgecritic` package provides a unified API: `wvf_image(image, backend="auto")` dispatches to the GPU backend automatically when CUDA is available, and `compute_ods_ois` uses the vectorized evaluation pipeline without any configuration. The final results will be compiled into a Typst ablation report with embedded figures.

= Expected Outcomes

Based on the initial coarse ablation, the $N_p$ sweet spot for WVF is expected to lie in $[40, 80]$, with LF potentially tolerating a wider range due to its line-averaging structure suppressing off-edge contamination. Orientation saturation is expected at $N_s approx 18$--$24$, consistent with the initial data showing no improvement beyond 18 samples. Polynomial order $d = 4$ is expected to remain optimal: lower orders underfit edge structure, while $d = 5$ adds 6 additional monomials (21 total) that worsen conditioning without contributing meaningful curvature signal at typical image resolutions. These are hypotheses to be tested, not conclusions.
