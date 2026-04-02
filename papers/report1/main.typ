#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Parameter Optimality of the Wide View and Line Filters for Edge Detection],
  abstract: [
    The Wide View Filter (WVF) and Line Filter (LF) are training-free edge detectors based on local polynomial fitting that were recently proposed as alternatives to classical gradient operators. Their authors recommend specific parameter settings---support size $N_p = 250$, orientation count $N_s = 18$, and polynomial order $d = 4$---derived from maritime imaging applications. We conduct a systematic ablation study spanning 1,206 WVF and 168 LF configurations across four diverse edge detection benchmarks totaling 330 images. Our results demonstrate that the published parameters are substantially suboptimal on clean imagery: reducing the support size to $N_p = 25$--$100$ and lowering the polynomial order to $d = 2$ consistently improves the Optimal Dataset Scale (ODS) F-score by 0.01--0.05 for the WVF and 0.06--0.16 for the LF. Furthermore, orientation count saturates at $N_s = 4$--$6$, far below the recommended 18. These findings suggest that the published parameters were tuned for noise-dominated conditions and are unnecessarily conservative for standard computer vision benchmarks.
  ],
  authors: (
    (
      name: "J. C. Vaught",
      organization: [],
      email: ""
    ),
  ),
  index-terms: ("Edge detection", "Parameter optimization", "Wide View Filter", "Line Filter", "Ablation study"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction

Edge detection is a foundational operation in computer vision, serving as a prerequisite for object recognition, image segmentation, and scene understanding. Classical approaches based on first-order gradient operators---Sobel @sobel2014history, Prewitt @prewitt1970gradient, and Canny @canny1986edge --- remain widely deployed due to their simplicity and computational efficiency. More recent learning-based methods such as Structured Forests @dollar2013structured and Holistically-Nested Edge Detection @xie2015hed have pushed the state of the art on standard benchmarks but require large annotated training sets and substantial computational resources.

Bagan and Wang proposed the Wide View Filter (WVF) @bagan2021wvf and its extension, the Line Filter (LF) @bagan2023lf, as training-free edge detectors that exploit local polynomial fitting over configurable neighborhoods. Unlike fixed-kernel operators, the WVF and LF adapt their spatial support and angular resolution through explicit parameters: the number of neighborhood pixels $N_p$, the number of sampled orientations $N_s$, and the polynomial order $d$. The LF further introduces a line half-width $m$ that extends the fitting along the estimated edge direction. The original publications recommend $N_p = 250$, $N_s = 18$, and $d = 4$ for the WVF, with the LF additionally using $m = 14$ and Gaussian weighting bandwidth $sigma = m\/2$.

These parameter choices were developed in the context of maritime image processing, where significant noise, low contrast, and color distortion are prevalent. A natural question arises: are these parameters also optimal for standard, clean-imagery benchmarks commonly used in the edge detection literature? If not, what parameters _are_ optimal, and by how much do they improve performance?

In this paper, we address these questions through a comprehensive ablation study. We evaluate 1,206 WVF configurations and 168 LF configurations across four diverse datasets: UDED (30 images), BIPED v1 (50 images) @poma2020biped, BIPED v2 (50 images) @soria2023bipedv2, and BSDS500 (200 images) @arbelaez2011bsds500. Our principal findings are as follows.

First, smaller support is better on clean data. The optimal $N_p$ lies in the range 25--100, depending on the dataset, yielding ODS improvements of 0.01--0.05 over $N_p = 250$ for the WVF and 0.06--0.16 for the LF. Second, lower polynomial order suffices for gradient estimation. Quadratic fitting ($d = 2$) consistently outperforms the quartic model ($d = 4$) recommended by Bagan and Wang. Third, orientation count saturates early. Performance plateaus at $N_s = 4$--$6$; the recommended $N_s = 18$ provides no additional benefit while increasing computation. Fourth, the LF's line extension does not help on clean data. The WVF alone matches or exceeds the LF across all four datasets. Finally, non-maximum suppression degrades performance. Applying standard NMS post-processing to the raw gradient magnitude maps reduces ODS by 0.06--0.09.

These results have practical implications for practitioners seeking to deploy WVF/LF-based edge detection: using smaller, computationally cheaper configurations yields both faster execution and higher accuracy on clean imagery.


= Mathematical Background

This section presents the WVF and LF operators with the ablation parameters explicitly identified, followed by the evaluation metric used throughout this study.

== Wide View Filter <sec:wvf>

For a grayscale image $I$ and a target pixel $bold(p)$, the WVF selects the $N_p$ nearest integer-coordinate pixels within a circular neighborhood, excluding the origin. At each candidate orientation $theta_k in {0, 2 pi \/ N_s, dots, 2 pi (N_s - 1) \/ N_s}$, the neighbor positions are rotated into a local coordinate frame via

$ bold(q)_i^((k)) = mat(cos theta_k, sin theta_k; -sin theta_k, cos theta_k) (bold(r)_i - bold(p)), quad i = 1, dots, N_p $ <eq:rotation>

where $bold(r)_i$ are the global neighbor positions sorted by Euclidean distance from $bold(p)$.

A design matrix $A in bb(R)^(N_p times M)$ is constructed from the 2D Taylor monomials up to total degree $d$, where $M = (d+1)(d+2) \/ 2$ is the number of monomial terms. For $d = 2$, $M = 6$; for $d = 4$, $M = 15$. The columns of $A$ correspond to the normalized monomials

$ 1, quad x, quad y, quad x^2\/2, quad y^2\/2, quad x y, quad dots $ <eq:monomials>

evaluated at each neighbor's local coordinates $(x_i, y_i) = bold(q)_i^((k))$. The full set of monomials for degree $d$ includes all terms $x^a y^b \/ (a! b!)$ with $a + b <= d$.

The derivative coefficients are recovered by ordinary least squares:

$ hat(bold(c))^((k)) = (A^top A)^(-1) A^top bold(f) in bb(R)^M $ <eq:lstsq>

where $bold(f) = [I(bold(r)_1), dots, I(bold(r)_(N_p))]^top$ is the vector of neighbor intensities. The coefficient $hat(c)_2^((k))$ corresponds to the first derivative in the $x$-direction (normal to the candidate edge orientation $theta_k$), while $hat(c)_3^((k))$ corresponds to the tangential derivative.

The WVF edge response at pixel $bold(p)$ is computed as the maximum absolute normal derivative over all orientations:

$ R_"WVF"(bold(p)) = max_(k = 1, dots, N_s) |hat(c)_2^((k))| $ <eq:wvf_response>

with the estimated edge orientation given by

$ theta^*(bold(p)) = theta_(arg max_k |hat(c)_2^((k))|) $ <eq:wvf_orientation>

The three parameters varied in our ablation are: $N_p$, which controls the spatial extent of the neighborhood and the row dimension of $A$; $N_s$, which controls the angular resolution of the orientation sweep; and $d$, which controls the column dimension $M$ and the expressiveness of the local polynomial model. The system in @eq:lstsq requires $N_p >= M$, which imposes a minimum support size that grows with polynomial order.

== Line Filter <sec:lf>

The LF extends the WVF by chaining $2m + 1$ WVF applications along a line centered on $bold(p)$ at each candidate orientation $theta_k$. Virtual expansion points are placed at

$ bold(v)_j^((k)) = bold(p) + j (cos theta_k, sin theta_k), quad j = -m, dots, m $ <eq:virtual_points>

Each virtual point $bold(v)_j^((k))$ produces a WVF normal-derivative estimate $hat(c)_(2,j)^((k))$ via @eq:lstsq applied to the neighborhood of $bold(v)_j^((k))$. These estimates are combined via a Gaussian-weighted average:

$ R_"LF"^((k))(bold(p)) = lr(|sum_(j = -m)^(m) w_j hat(c)_(2,j)^((k))|) $ <eq:lf_response>

where the weights are

$ w_j = exp(-j^2 \/ 2 sigma^2) slash sum_(ell=-m)^(m) exp(-ell^2 \/ 2 sigma^2) $ <eq:lf_weights>

and $sigma$ controls the Gaussian bandwidth. The LF output is the orientation-maximized response, analogous to @eq:wvf_response. In addition to $N_p$, $N_s$, and $d$, the LF introduces the half-width $m$ as an additional parameter. Following Bagan and Wang, we fix $sigma = m\/2$ throughout this study and vary $m$ explicitly.

The line extension is motivated by the hypothesis that averaging derivative estimates along the edge tangent direction suppresses noise while preserving the edge signal. This hypothesis is well-founded for noisy imagery but, as we demonstrate, does not yield improvements on clean data.

== Evaluation Metrics <sec:metrics>

Edge quality is measured by the Optimal Dataset Scale (ODS) F-score, the standard metric for edge detection benchmarks @martin2004evaluation @arbelaez2011bsds500. A continuous gradient magnitude map is binarized at threshold $t$ and matched against human-annotated ground truth within a spatial tolerance of $r$ pixels. Precision and recall are computed as

$ P(t) = "TP"(t) / ("TP"(t) + "FP"(t)), quad quad R(t) = "TP"(t) / ("TP"(t) + "FN"(t)) $ <eq:precision_recall>

where a predicted edge pixel counts as a true positive if it falls within distance $r$ of any ground-truth edge pixel, and a ground-truth pixel counts as a false negative if no predicted edge pixel falls within distance $r$. The F-score at threshold $t$ is the harmonic mean

$ F(t) = (2 P(t) R(t)) / (P(t) + R(t)) $ <eq:fscore>

The ODS is the maximum F-score over all thresholds, evaluated at a single global threshold applied to the entire dataset:

$ "ODS" = max_t F(t) $ <eq:ods>

We also report the Optimal Image Scale (OIS), which allows a per-image optimal threshold: $"OIS" = 1\/N sum_(i=1)^N max_t F_i (t)$. Throughout this study, we use a match radius of $r = 3$ pixels and evaluate over 1,001 uniformly spaced thresholds.


= Experimental Setup <sec:setup>

== Datasets

We evaluate on four edge detection benchmarks spanning a range of image content, resolution, and annotation style. The evaluation corpus comprises 330 images with diverse content including natural scenes, urban environments, animals, and maritime imagery. @tab:datasets provides a summary of the datasets used in this study.

#figure(
  placement: top,
  table(
    columns: 5,
    align: (left, center, center, left, left),
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header(
      [*Dataset*], [*Images*], [*Resolution*], [*Description*], [*Ref.*],
    ),
    [UDED], [30], [variable], [Unified Dataset for Edge Detection; aggregates images from 15 sources including BIPED, BSDS, and others.], [@soria2023bipedv2],
    [BIPED v1], [50], [1280 $times$ 720], [Barcelona Images for Perceptual Edge Detection dataset containing high-resolution outdoor scenes with carefully annotated edges.], [@poma2020biped],
    [BIPED v2], [50], [1280 $times$ 720], [Updated version of BIPED with refined annotations and additional scenes. Maintains the same resolution as v1.], [@soria2023bipedv2],
    [BSDS500], [200], [481 $times$ 321], [Berkeley Segmentation Dataset, the most widely used edge detection benchmark. Contains 200 test images, each annotated by multiple human subjects. Standard test split is used.], [@arbelaez2011bsds500],
  ),
  caption: [Summary of the four edge detection datasets used in the evaluation. Total corpus comprises 330 images.],
) <tab:datasets>

== Parameter Grid

The WVF parameter space is defined by three axes. Support size $N_p$ is sampled at 20 levels: $N_p in {5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500}$. Orientation count $N_s$ is sampled at 18 levels: $N_s in {3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 24, 30, 36, 48, 60, 72, 90, 120}$. Polynomial order $d$ takes values in ${2, 3, 4, 5}$. Not all combinations are valid: the constraint $N_p >= (d+1)(d+2)\/2$ eliminates configurations where the least-squares system is underdetermined. After removing invalid configurations, 1,206 valid WVF settings remain.

The LF parameter space adds the line half-width $m in {1, 2, 3, 5, 7, 10, 14, 20}$ to a subset of the WVF grid. With $N_p in {15, 25, 50, 75, 100, 150, 250}$, $N_s in {18, 36, 72}$, and $d = 4$ fixed, the LF grid contains 168 configurations. The Gaussian bandwidth is set to $sigma = m\/2$ throughout.

The total number of distinct filter evaluations across all datasets is $(1206 + 168) times 330 = 453{,}420$ filter applications.

== Evaluation Protocol

Each filter configuration produces a continuous gradient magnitude map for every image. These maps are evaluated against ground-truth annotations using the protocol of @arbelaez2011bsds500 with two modifications: we use 1,001 thresholds (instead of the standard 99) for finer resolution of the ODS, and a match radius of $r = 3$ pixels. No post-processing (non-maximum suppression, hysteresis thresholding, or morphological thinning) is applied to the raw filter output unless explicitly stated otherwise.

== Implementation

All filter computations are GPU-accelerated using a custom CUDA implementation. The WVF processes a full-resolution BSDS500 image in approximately 0.009 seconds; the LF requires 0.05--0.3 seconds depending on $m$. The evaluation pipeline uses distance transforms and vectorized threshold sweeps, achieving a 120$times$ speedup over naive per-threshold binary dilation. The full ablation across all four datasets completed in approximately 12.4 hours of wall-clock time on an NVIDIA RTX 6000 Ada (48 GB VRAM).


= Single-Image Ablation <sec:single_image>

Before committing to the full dataset evaluation, we conducted dense single-image ablations on representative images from BSDS500 and BIPED to map the parameter response surface at high resolution. These preliminary experiments guided the design of the full-dataset grid and provided initial evidence for the suboptimality of the published parameters.

== Effect of Support Size $N_p$

@fig:wvf_ods_vs_np shows the ODS F-score as a function of $N_p$ for the WVF at polynomial order $d = 2$ on a single BSDS500 test image (\#100007, 321$times$481 pixels). Performance rises steeply from $N_p = 5$ (ODS $approx$ 0.73) to $N_p = 25$ (ODS $approx$ 0.847), peaks in the range $N_p = 30$--$50$ (ODS $approx$ 0.856--0.860), and then monotonically declines. At $N_p = 250$ (the published setting), ODS drops to approximately 0.71, and at $N_p = 500$ it falls further to 0.65.

#figure(
  image("figures/fig_wvf_ods_vs_np.png", width: 95%),
  caption: [WVF ODS F-score versus support size $N_p$ on BSDS500 image \#100007 at $d = 2$, showing a clear peak near $N_p = 40$--$50$ and monotonic decline beyond. The published parameter $N_p = 250$ falls well below the optimum.],
) <fig:wvf_ods_vs_np>

The intuition behind this result is straightforward. A small $N_p$ constrains the polynomial fit to a tight neighborhood where the intensity surface is well-approximated by low-order polynomials, yielding sharp, localized gradient estimates. As $N_p$ increases, the support region grows and the polynomial must fit intensity variations across increasingly diverse structures---textures, corners, and junctions---leading to blurred, delocalized edge responses. The resulting gradient magnitude maps have wider edge ridges that reduce precision at any given recall level.

== Interaction of $N_p$ and $N_s$

@fig:wvf_heatmap shows the ODS as a joint function of $N_p$ (vertical axis) and $N_s$ (horizontal axis) at $d = 2$. The dominant structure is vertical: performance varies strongly with $N_p$ and weakly with $N_s$. The optimal region forms a horizontal band at $N_p approx 25$--$50$ that is largely invariant to orientation count.

#figure(
  image("figures/fig_wvf_heatmap_np_ns.png", width: 95%),
  caption: [Heatmap of WVF ODS as a function of $N_p$ (rows) and $N_s$ (columns) at $d = 2$ on BSDS500 image \#100007. The dominant variation is along the $N_p$ axis; $N_s$ has minimal effect beyond approximately 4--6 orientations.],
) <fig:wvf_heatmap>

== Orientation Count Saturation

@fig:ns_saturation isolates the effect of $N_s$ by plotting ODS versus $N_s$ at several fixed values of $N_p$. At all support sizes, ODS rises from $N_s = 3$ to $N_s = 4$--$6$ and then plateaus. The difference between $N_s = 6$ and $N_s = 120$ is typically less than 0.005 in ODS. This saturation occurs because the polynomial fit already captures the edge orientation implicitly: the maximum normal-derivative response over even a few orientations is sufficient to locate the dominant gradient direction. Sampling more orientations refines the angle estimate but does not substantially change the magnitude, which is what determines the ODS.

#figure(
  image("figures/fig_ns_saturation.png", width: 95%),
  caption: [ODS versus orientation count $N_s$ at several fixed $N_p$ values ($d = 2$). Performance saturates by $N_s = 4$--$6$, with negligible improvement from the published $N_s = 18$ or beyond.],
) <fig:ns_saturation>

This finding has significant computational implications. The WVF runtime scales linearly with $N_s$ because each orientation requires constructing and solving a separate least-squares system. Reducing $N_s$ from 18 to 4 yields a $4.5 times$ speedup with no loss in edge quality.

== Effect of Polynomial Order $d$

The single-image ablation reveals that $d = 2$ (quadratic, $M = 6$ monomials) consistently outperforms $d = 4$ (quartic, $M = 15$ monomials) across the full range of $N_p$ and $N_s$. On the BSDS500 test image, the best $d = 2$ configuration achieves ODS = 0.860, compared to approximately 0.834 for the best $d = 4$ configuration. The quadratic model's advantage is twofold. First, with fewer parameters to estimate, the least-squares fit is better conditioned, particularly at small $N_p$ where the ratio $N_p \/ M$ is low. At $N_p = 25$ and $d = 4$, the system has only $25\/15 approx 1.7$ observations per parameter, which leads to noisy coefficient estimates. At $d = 2$, the same $N_p$ gives $25\/6 approx 4.2$ observations per parameter. Second, the higher-order monomials in the quartic model capture curvature and higher-frequency intensity variations that are not relevant to first-derivative (edge) estimation---they add noise to the gradient estimate without improving it.


= Full-Dataset Generalization <sec:full_dataset>

The single-image findings generalize robustly across all four evaluation datasets. @tab:full_dataset_results presents the best-found and published-parameter ODS for both the WVF and LF on each dataset.

#figure(
  placement: top,
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header(
      [*Dataset*], [*Best WVF*], [*Bagan WVF*], [*$Delta$*],
      [*Best LF*], [*Bagan LF*], [*$Delta$*],
    ),
    [UDED (30)],
    [0.899], [0.847], [+0.052],
    [0.887], [0.729], [+0.158],

    [BIPED v1 (50)],
    [0.812], [0.767], [+0.045],
    [0.802], [0.643], [+0.159],

    [BIPED v2 (50)],
    [0.830], [0.783], [+0.047],
    [0.819], [0.658], [+0.161],

    [BSDS500 (200)],
    [0.682], [0.671], [+0.011],
    [0.671], [0.615], [+0.056],
  ),
  caption: [ODS F-score comparison between optimized and published (Bagan) parameters across all four datasets. The $Delta$ column shows the improvement from parameter optimization. The WVF gains 0.01--0.05, while the LF gains 0.06--0.16.],
) <tab:full_dataset_results>

Several patterns are noteworthy.

*Consistent WVF optimum.* The best WVF configuration is $(N_p = 25, N_s = 4, d = 2)$ on UDED, BIPED v1, and BIPED v2. On BSDS500, the optimum shifts to $(N_p = 100, N_s = 4, d = 2)$, likely reflecting the lower resolution and more diverse content of the BSDS500 images. In all cases, $d = 2$ and $N_s = 4$ are selected, confirming the single-image findings.

*Larger LF improvement.* The gap between the best and published LF parameters is substantially larger than for the WVF (0.06--0.16 versus 0.01--0.05). This is because the published LF uses $m = 14$, which extends the line support over 29 pixels and averages derivative estimates across structures that may not share a common edge orientation. On clean data, this averaging blurs the edge response without the compensating benefit of noise suppression.

*BSDS500 is the hardest dataset.* All configurations achieve lower ODS on BSDS500 than on the other three datasets, consistent with the known difficulty of this benchmark's diverse content and multiple annotators with varying judgment. The improvement from parameter optimization is correspondingly smaller (0.011 for WVF, 0.056 for LF), but still statistically meaningful across 200 images.

@fig:cross_dataset shows the best-found ODS for each filter type and dataset alongside the published parameter performance.

#figure(
  image("figures/fig_cross_dataset_best.png", width: 95%),
  caption: [Cross-dataset comparison of best-optimized versus published (Bagan) parameters for WVF and LF. The optimized parameters consistently outperform the published settings, with particularly large gains for the LF.],
) <fig:cross_dataset>

@tab:best_params details the optimal parameter configurations found for each dataset.

#figure(
  placement: top,
  table(
    columns: 6,
    align: (left, left, center, center, center, center),
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header(
      [*Dataset*], [*Filter*], [*$N_p$*], [*$N_s$*], [*$d$*], [*ODS*],
    ),
    [UDED], [WVF], [25], [4], [2], [0.899],
    [UDED], [LF], [25], [36], [4], [0.887],
    [BIPED v1], [WVF], [25], [4], [2], [0.812],
    [BIPED v1], [LF], [100], [18], [4], [0.802],
    [BIPED v2], [WVF], [25], [4], [2], [0.830],
    [BIPED v2], [LF], [100], [18], [4], [0.819],
    [BSDS500], [WVF], [100], [4], [2], [0.682],
    [BSDS500], [LF], [250], [18], [4], [0.671],
  ),
  caption: [Optimal parameter configurations for each dataset and filter type. The WVF consistently selects $d = 2$ and $N_s = 4$. The LF optimal parameters show more variation but always prefer smaller $m$ than the published $m = 14$.],
) <tab:best_params>

The OIS scores follow the same pattern. For example, on UDED the best WVF achieves OIS = 0.905 versus 0.862 for the published parameters; on BSDS500, the best WVF achieves OIS = 0.695 versus 0.686. The OIS--ODS gap is small (0.006--0.014), indicating that a single global threshold performs nearly as well as per-image optimization---the gradient magnitude maps are well-calibrated across images within each dataset.


= WVF versus LF <sec:wvf_vs_lf>

A central claim of Bagan and Wang @bagan2023lf is that the LF improves upon the WVF by exploiting edge continuity through the line extension mechanism. Our ablation results challenge this claim for clean imagery.

On all four datasets, the best WVF configuration outperforms the best LF configuration. The margins are modest---0.012 on UDED, 0.010 on BIPED v1, 0.011 on BIPED v2, and 0.011 on BSDS500---but they are consistent. The WVF achieves this advantage despite having no line-averaging mechanism and despite the LF search space including $m = 1$ (which makes the LF nearly equivalent to the WVF, with only 3 virtual points).

@fig:lf_ods_vs_m shows the LF ODS as a function of line half-width $m$. On all datasets where data is available, the best ODS occurs at $m = 1$--$3$, the shortest line extensions. As $m$ increases beyond 3, ODS monotonically declines. The published parameter $m = 14$ yields the worst or near-worst performance. This pattern is consistent with the clean-data hypothesis: longer lines average over more spatial context, which helps suppress noise but harms edge localization when noise is absent.

#figure(
  image("figures/fig_lf_ods_vs_m.png", width: 95%),
  caption: [LF ODS versus line half-width $m$. Shorter line extensions ($m = 1$--$3$) yield higher ODS than the published $m = 14$. On clean data, the line-averaging mechanism harms rather than helps.],
) <fig:lf_ods_vs_m>

The computational cost of the LF scales as $O((2m+1) dot N_p dot N_s)$ compared to $O(N_p dot N_s)$ for the WVF. At $m = 14$, the LF requires 29 WVF evaluations per pixel per orientation, making it approximately 29 times slower than the WVF for the same $N_p$ and $N_s$. Given that the LF provides no accuracy benefit on clean data, the WVF is strictly preferable in this regime: it is both faster and more accurate.


= Post-Processing Effects <sec:postprocessing>

Standard edge detection pipelines typically apply non-maximum suppression (NMS) to thin the gradient magnitude map to single-pixel-wide ridges before thresholding @canny1986edge. We investigated whether NMS improves the ODS of WVF and LF outputs by testing both 4-directional and 8-directional NMS variants.

@fig:postprocessing summarizes the results across multiple configurations on the BIPED dataset. In all cases, NMS reduces the ODS compared to the raw gradient magnitude map. The degradation ranges from 0.06 to 0.09 for 4-directional NMS and is similar for 8-directional NMS. For example, the best WVF configuration ($N_p = 25$, $d = 2$, $N_s = 3$) achieves ODS = 0.844 without NMS but drops to 0.754 with 4-directional NMS and 0.760 with 8-directional NMS. The Bagan WVF configuration ($N_p = 250$, $d = 4$) drops from 0.766 to 0.674 with NMS.

#figure(
  image("figures/fig_postprocessing.png", width: 95%),
  caption: [Effect of non-maximum suppression (NMS) on ODS for various WVF and LF configurations. Both 4-directional and 8-directional NMS consistently reduce ODS compared to the raw gradient magnitude output.],
) <fig:postprocessing>

The LF shows the same pattern. The best LF ($N_p = 75$, $m = 1$) achieves ODS = 0.837 without NMS, falling to 0.746 with 4-directional NMS. The Bagan LF ($N_p = 250$, $m = 14$) drops from 0.621 to 0.563.

This counterintuitive result arises because the ODS evaluation protocol already optimizes the binarization threshold. The raw gradient magnitude map contains soft, multi-pixel-wide edge responses that, when thresholded appropriately, produce edge predictions that match ground truth within the $r = 3$ pixel tolerance. NMS thins these responses to single-pixel ridges, which can shift the detected edge position away from the ground-truth annotation, reducing true positives within the match radius. The 3-pixel match tolerance is generous enough that the soft gradient ridges are not penalized for their width, making NMS counterproductive.


= Discussion and Conclusions <sec:discussion>

Our comprehensive ablation study demonstrates that the parameter settings published by Bagan and Wang for the WVF ($N_p = 250$, $N_s = 18$, $d = 4$) and LF ($N_p = 250$, $N_s = 18$, $d = 4$, $m = 14$) are substantially suboptimal for clean-imagery edge detection benchmarks. Across four diverse datasets totaling 330 images, we find consistent and significant improvements from three parameter changes: reducing the support size to $N_p = 25$--$100$, lowering the polynomial order to $d = 2$, and reducing the orientation count to $N_s = 4$.

== Why Are the Published Parameters Suboptimal?

The most likely explanation is that Bagan and Wang optimized their parameters for maritime imagery, where noise levels are substantially higher than in standard benchmarks. In noisy conditions, a larger support size $N_p$ provides more observations for the least-squares fit, improving the signal-to-noise ratio of the gradient estimate at the cost of spatial resolution. Similarly, a higher polynomial order $d = 4$ may be needed to capture intensity variations in the larger neighborhood without attributing them to the gradient. The line extension of the LF ($m = 14$) provides additional noise suppression through averaging. On clean data, these noise-mitigation mechanisms become liabilities: the large support blurs edges, the high polynomial order overfits to local texture, and the line extension averages across distinct structures. We investigate this noise-robustness hypothesis in a companion study.

== Practical Recommendations

Based on our findings, we offer the following recommendations for practitioners deploying WVF-based edge detection on clean imagery.

Practitioners should use the WVF rather than the LF, as the line extension adds computational cost without improving accuracy on clean data. For the support size parameter, set $N_p = 25$--$50$ for high-resolution images (e.g., 1280 $times$ 720) and $N_p = 50$--$100$ for lower-resolution images (e.g., 481 $times$ 321). The optimal support size appears to scale with image resolution. The polynomial order should be set to $d = 2$, as quadratic fitting provides sufficient expressiveness for gradient estimation while maintaining good conditioning of the least-squares system. For orientation count, use $N_s = 4$--$6$; finer angular sampling provides negligible benefit while increasing computation linearly. Finally, do not apply non-maximum suppression post-processing if using the ODS evaluation protocol with a match radius of $r >= 3$ pixels. The raw gradient magnitude map achieves higher ODS than the NMS-thinned version.

== Computational Savings

The recommended parameters ($N_p = 25$, $N_s = 4$, $d = 2$) offer substantial computational savings over the published parameters ($N_p = 250$, $N_s = 18$, $d = 4$). The WVF computation scales as $O(N_s dot N_p dot M)$ where $M = (d+1)(d+2)\/2$. The recommended configuration has $N_s dot N_p dot M = 4 times 25 times 6 = 600$, compared to $18 times 250 times 15 = 67{,}500$ for the published parameters---a $112 times$ reduction in the dominant computational cost. In practice, the speedup is less dramatic due to memory access patterns and fixed overheads, but it remains significant.

== Limitations and Future Work

Several limitations of this study should be noted and addressed in future work. This study evaluated only the ODS and OIS metrics with a fixed match radius of $r = 3$ pixels; different match radii or alternative metrics such as average precision might favor different parameter settings. The LF ablation fixed $d = 4$ for computational tractability; exploring $d = 2$ for the LF could reveal further improvements. Furthermore, we did not evaluate performance on genuinely noisy imagery, where the published parameters may indeed be optimal. This gap is the subject of a companion study on noise robustness.

The broader implication of this work is that filter parameters should not be treated as universal constants. The WVF and LF are flexible operators with a rich parameter space, and the optimal configuration depends on the operating conditions---image quality, resolution, and the downstream task. Our ablation framework, implemented as a GPU-accelerated open-source tool, enables practitioners to efficiently tune these parameters for their specific application.
