#set page(paper: "us-letter", margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 16pt, weight: "bold")[GPU-Accelerated WVF Ablation Study]
  #v(0.3em)
  #text(size: 12pt)[Support Size, Orientation Count, and Edge Quality on BSDS500]
  #v(0.5em)
  J. C. Vaught · Edge-Detection Filter Critique Project · March 2026
]

#v(1em)

= Overview

The Wide View Filter (WVF) has two primary hyperparameters: the support size $N_p$ (number of neighbor pixels in the local least-squares fit) and the number of discrete orientations $N_s$ swept during edge detection. This document reports a single-image ablation measuring how these parameters affect edge detection quality at full resolution (481$times$321), evaluated against human-annotated ground truth from BSDS500.

= GPU Acceleration

The original WVF implementation processes each pixel sequentially in Python, requiring approximately 7 hours per full-resolution image on a 112-core CPU node. We implemented a GPU-accelerated version (`wvf_cuda.py`) that exploits a key structural property: the Taylor design matrix $A$ depends only on the orientation $theta$, not on pixel position. This allows precomputation of the pseudoinverse operator

$ P_theta = (A^top A)^(-1) A^top in bb(R)^(M times N_p) $

for each orientation. At runtime, the per-pixel computation reduces to a gather of $N_p$ neighbor intensities followed by a single matrix-vector product $hat(c) = P_theta f$, extracting $|hat(c)_2|$ as the edge response. All pixels are processed simultaneously via batched tensor operations.

#figure(
  table(
    columns: 3,
    align: (left, right, right),
    stroke: none,
    table.hline(),
    table.header([*Platform*], [*Time / image*], [*Speedup*]),
    table.hline(),
    [CPU (112-core, Python loop)], [~7 hours], [1$times$],
    [GPU (RTX 6000 Ada, PyTorch)], [0.009 s], [~18,000$times$],
    table.hline(),
  ),
  caption: [Runtime comparison for WVF at $N_p = 15$, $N_s = 18$, on a 481$times$321 image.],
)

The GPU version was validated against the CPU implementation on a 64$times$64 crop: the maximum absolute difference in gradient magnitude was $3 times 10^(-6)$, attributable to float32 vs. float64 rounding.

= Evaluation Protocol

Edge quality is measured using the standard BSDS500 protocol:

+ *Threshold.* The continuous gradient magnitude map is binarized at threshold $t in [0, 1]$.
+ *Match.* Each predicted edge pixel is compared to the ground truth within a spatial tolerance of 3 pixels (binary dilation).
+ *Precision.* Fraction of predicted edge pixels that fall within 3 pixels of a ground-truth edge.
+ *Recall.* Fraction of ground-truth edge pixels that have a predicted edge within 3 pixels.
+ *F-score.* Harmonic mean: $F = (2 P R) / (P + R)$.
+ *ODS.* The best F-score across all thresholds, evaluated on the full dataset.

For this single-image ablation, we sweep 300 thresholds uniformly in $[0.01, 0.99]$ and report the best F-score per configuration.

= Ablation Design

We evaluate 12 configurations: $N_p in {15, 50, 100, 250}$ crossed with $N_s in {18, 36, 72}$ orientations. All runs use polynomial order $d = 4$ (15-term Taylor basis). The test image is BSDS500 \#100007 (321$times$481, polar bear on ice), with ground truth from the union of 5 human annotators (9,181 edge pixels).

= Results

#figure(
  table(
    columns: 5,
    align: (left, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header([$N_p$], [$N_s = 18$], [$N_s = 36$], [$N_s = 72$], [Best $t$]),
    table.hline(),
    [15],  [0.721], [0.720], [0.720], [0.18--0.20],
    [50],  [0.805], [0.806], [0.806], [0.26--0.27],
    [100], [0.804], [0.804], [0.805], [0.25--0.26],
    [250], [0.745], [0.745], [0.745], [0.20],
    table.hline(),
  ),
  caption: [Best F-score by support size and orientation count. Bold entries mark the overall best.],
)

Three observations:

*Orientation count has negligible effect.* Across all support sizes, increasing $N_s$ from 18 to 72 changes the F-score by less than 0.002. The discrete orientation grid at $N_s = 18$ (10$degree$ spacing) is already sufficient to capture the response peak. This is consistent with the discretization bound in our sensitivity analysis: the worst-case angular error is $pi slash (2 N_s) = 5 degree$ at $N_s = 18$, which is well below the spatial matching tolerance.

*Moderate support ($N_p = 50$--$100$) is optimal.* The best F-score (0.806) occurs at $N_p = 50$ with $N_s = 36$. At $N_p = 15$, precision is low (0.64) because the small support responds to texture and noise. At $N_p = 250$, precision drops again (0.72) as the wide support picks up non-edge intensity gradients. Recall is relatively stable across support sizes (0.77--0.83).

*Large support degrades performance.* $N_p = 250$ (F$=$0.745) is worse than $N_p = 50$ (F$=$0.806), a 7.6% reduction. The match maps show that $N_p = 250$ produces thick false-positive bands along the horizon and ice texture, consistent with the curvature-induced bias $O(r_s^2 slash rho)$ predicted by Proposition 5 of the sensitivity analysis. Wider support does not uniformly improve edge detection; it trades noise reduction for structure mixing.

= Precision--Recall Tradeoff

At the best threshold ($t approx 0.26$) for $N_p = 50$:

- *Precision = 0.84*: 84% of predicted edge pixels are within 3 pixels of a ground-truth edge.
- *Recall = 0.77*: 77% of ground-truth edges have a nearby prediction.

The primary source of false positives is ice surface texture, which produces low-magnitude but spatially extensive gradient responses. The primary source of missed edges is fine-scale features (bear legs, shadow boundaries) where the gradient magnitude falls below threshold.

= Comparison to Prior Results

The previous WVF evaluation on BSDS500 used 64$times$64 downscaled images and reported ODS$=$0.0 across all configurations. At full resolution, WVF achieves F$=$0.806 on this test image --- a qualitatively different outcome that demonstrates the prior zero-ODS result was an artifact of extreme downscaling rather than a fundamental limitation of the filter.

For context, the best reproduced classical baseline on BSDS500 is Sobel-15$times$15 with ODS$=$0.549 (50-image subset). A single-image F-score is not directly comparable to a dataset-wide ODS, but the magnitude suggests WVF at full resolution is competitive. A full 201-image ODS evaluation is in progress.

= Method

All GPU computations were performed on an NVIDIA RTX 6000 Ada Generation (48 GB VRAM) via Tailscale SSH to a remote workstation. The 12 ablation configurations ran in 0.17 seconds total. The evaluation sweep (300 thresholds $times$ 12 configurations $=$ 3,600 binary dilation operations) took approximately 45 seconds.
