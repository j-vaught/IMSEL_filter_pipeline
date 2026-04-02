#set page(paper: "us-letter", margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 16pt, weight: "bold")[Classical Edge Detector Ablation Study]
  #v(0.3em)
  #text(size: 12pt)[Comprehensive Parameter Sweep Across Noise Regimes]
  #v(0.5em)
  J. C. Vaught #sym.dot.c Edge-Detection Filter Critique Project #sym.dot.c March 2026
]

#v(1em)

= Motivation

Prior experiments compared the WVF/LF against four deep learning edge detectors (DexiNed, TEED, pidinet, NBED) and found that the WVF with appropriately tuned parameters beats every DL model at all tested noise levels below SNR $approx 4$. However, the study omitted classical edge detectors beyond the basic Sobel operator. Several classical methods---particularly the Canny pipeline, Laplacian of Gaussian, and Gaussian derivative filters at multiple scales---are more natural comparisons for the WVF because they share its core principle: convolution-based gradient estimation with scale as a tunable parameter.

This study adds 10 classical edge detectors to the comparison, each swept across its parameter space at the same 8 SNR levels and 5 noise types used in the existing ablation. The goal is to determine whether the WVF's advantage over DL models extends to well-tuned classical methods, or whether simpler operators with appropriate scale parameters already achieve comparable performance.

= Methods and Parameter Grids

== Gradient Operators (1st Derivative)

=== Sobel

The Sobel operator computes horizontal and vertical image gradients using separable convolution kernels. The standard 3$times$3 kernel is widely used but its fixed scale makes it noise-sensitive. Extended Sobel operators at larger kernel sizes are equivalent to Gaussian derivative filters and provide a direct comparison to the WVF's multi-scale support.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [Kernel size $k$], [3, 5, 7, 9, 11, 15, 21, 31, 41, 51], [10],
      table.hline(),
    )
  ),
  caption: [Sobel parameter grid. Kernel size $k$ gives an effective support of $approx pi (k slash 2)^2$ pixels. Sobel-9 $approx$ WVF $N_p = 64$; Sobel-31 $approx N_p = 755$; Sobel-51 $approx N_p = 2043$, well beyond the WVF's maximum of 500.],
)

=== Prewitt

Identical structure to Sobel but with equal-weight kernels (no center-weighted smoothing). Included to test whether the Sobel weighting scheme matters under noise.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [Kernel size $k$], [3, 5, 7, 9, 11, 15, 21, 31, 41], [9],
      table.hline(),
    )
  ),
  caption: [Prewitt parameter grid. Extended to match Sobel's spatial range.],
)

=== Scharr

The Scharr operator uses optimized 3$times$3 coefficients for improved rotational symmetry compared to Sobel. It has no scale parameter---only one configuration is tested.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [Kernel size $k$], [3 (fixed)], [1],
      table.hline(),
    )
  ),
  caption: [Scharr parameter grid.],
)

=== Roberts Cross

The Roberts Cross operator uses 2$times$2 diagonal difference kernels. It is the smallest possible edge detector and provides a lower bound on performance at every noise level.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [Kernel size $k$], [2 (fixed)], [1],
      table.hline(),
    )
  ),
  caption: [Roberts Cross parameter grid.],
)

== 2nd Derivative Methods

=== Laplacian of Gaussian (LoG)

The LoG convolves the image with the Laplacian of a Gaussian kernel, producing zero-crossings at edges. The Gaussian $sigma$ controls the scale. Edge maps are obtained by detecting zero-crossings in the LoG response and thresholding the gradient magnitude at the crossing points.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [$sigma$ (Gaussian scale)], [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0], [10],
      table.hline(),
    )
  ),
  caption: [LoG parameter grid. $sigma = 4$ $approx$ WVF $N_p = 200$ (effective radius $approx 3 sigma = 12$px). $sigma = 16$ far exceeds the WVF range.],
)

=== Difference of Gaussians (DoG)

The DoG approximates the LoG by subtracting two Gaussian-blurred images at different scales. The ratio $k = sigma_2 slash sigma_1$ controls the bandwidth.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [$sigma_1$], [0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0], [8],
      [$k = sigma_2 slash sigma_1$], [1.6 (standard)], [1],
      table.hline(),
    )
  ),
  caption: [DoG parameter grid. $sigma_2 = k dot sigma_1$.],
)

== Compass / Directional Operators

=== Kirsch

The Kirsch compass operator applies 8 directional 3$times$3 kernels and takes the maximum response. It provides an explicit multi-orientation search similar in spirit to the WVF's orientation sweep, but at fixed 3$times$3 scale.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [Directions], [8 (fixed, 45$degree$ spacing)], [1],
      table.hline(),
    )
  ),
  caption: [Kirsch parameter grid. Single configuration --- 8 fixed 3$times$3 compass masks.],
)

=== Robinson

Similar to Kirsch but with different kernel coefficients (based on the Sobel template rotated to 8 directions). Provides a second fixed-scale directional comparison.

=== Frei-Chen

The Frei-Chen edge detector projects the local 3$times$3 neighborhood onto an orthogonal basis of 9 masks, then measures the projection onto the 4-dimensional edge subspace. It separates edge, line, and average components analytically.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [Basis size], [9 masks (4 edge, fixed)], [1],
      table.hline(),
    )
  ),
  caption: [Frei-Chen parameter grid. Single configuration using the standard 9-mask orthogonal basis.],
)

== Multi-Step Pipelines

=== Canny

The Canny edge detector applies Gaussian smoothing, Sobel gradient computation, non-maximum suppression, and hysteresis thresholding. Two parameters are swept: the Gaussian $sigma$ and the hysteresis threshold ratio. The low threshold is set as a fraction of the high threshold, which is determined by the gradient magnitude distribution.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [$sigma$ (Gaussian smoothing)], [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0], [10],
      [Low/high threshold ratio], [0.3, 0.4, 0.5], [3],
      table.hline(),
    )
  ),
  caption: [Canny parameter grid. 30 configurations total. The high threshold is set at the 90th percentile of the NMS output; the low threshold is the high threshold multiplied by the ratio. $sigma = 16$ produces extremely blurred gradients, matching a WVF with $N_p > 1000$.],
)

Note: Canny produces binary edge maps, not continuous gradient magnitudes. ODS evaluation sweeps thresholds on the hysteresis output, which is already binary---effectively testing only the single threshold defined by the parameters. To make Canny comparable to the other methods, we also evaluate the NMS output (before hysteresis) as a continuous map, sweeping thresholds in the standard way.

=== Marr-Hildreth

The Marr-Hildreth detector applies LoG followed by zero-crossing detection. It produces binary edges and is parameterized by the LoG $sigma$.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [$sigma$], [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0], [10],
      table.hline(),
    )
  ),
  caption: [Marr-Hildreth parameter grid.],
)

== Scale-Space / Steerable Methods

=== Gaussian Derivatives

First-order Gaussian derivative filters at arbitrary scale. This is the continuous-scale generalization of Sobel and the most direct classical analogue to the WVF: both compute directional derivatives at a controllable spatial scale. The key difference is that Gaussian derivatives use isotropic smoothing while the WVF uses polynomial fitting with orientation-specific support.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [$sigma$ (Gaussian scale)], [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0], [11],
      table.hline(),
    )
  ),
  caption: [Gaussian derivative parameter grid. The WVF at $N_p = n$ has effective radius $r approx sqrt(n slash pi)$; the comparable Gaussian $sigma approx r slash 2$. At $N_p = 50$, $sigma approx 2.0$; at $N_p = 250$, $sigma approx 4.5$; at $N_p = 500$, $sigma approx 6.3$. Values of $sigma = 12$--$20$ exceed the WVF range, testing whether even larger support helps.],
)

=== Steerable Filters

Steerable filters (Freeman and Adelson, 1991) compute oriented gradient responses by analytically rotating a basis set of filters. They enable continuous orientation estimation without discrete sweeping---unlike the WVF which tests $N_s$ discrete angles.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, right),
      stroke: none,
      table.hline(),
      table.header([*Parameter*], [*Values*], [*Count*]),
      table.hline(),
      [$sigma$ (Gaussian scale)], [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0], [8],
      [Order], [1st derivative, 2nd derivative], [2],
      table.hline(),
    )
  ),
  caption: [Steerable filter parameter grid. 16 configurations total. $sigma = 16$ gives an effective diameter of $approx 96$ pixels, far exceeding the WVF's largest support.],
)

= Summary of Configurations

#figure(
  align(center,
    table(
      columns: 4,
      align: (left, right, left, left),
      stroke: none,
      table.hline(),
      table.header([*Method*], [*Configs*], [*Parameters*], [*Type*]),
      table.hline(),
      [Sobel (extended)], [10], [$k in {3 dots 51}$], [1st derivative],
      [Prewitt (extended)], [9], [$k in {3 dots 41}$], [1st derivative],
      [Scharr], [1], [fixed 3$times$3], [1st derivative],
      [Roberts Cross], [1], [fixed 2$times$2], [1st derivative],
      [Kirsch], [1], [8 compass masks], [Directional],
      [Robinson], [1], [8 compass masks], [Directional],
      [Frei-Chen], [1], [9 basis masks], [Subspace],
      [Laplacian of Gaussian], [10], [$sigma in {0.5 dots 16}$], [2nd derivative],
      [Difference of Gaussians], [8], [$sigma_1 in {0.5 dots 12}$], [2nd derivative],
      [Canny NMS], [10], [$sigma in {0.5 dots 16}$], [Pipeline],
      [Gaussian derivatives], [11], [$sigma in {0.5 dots 20}$], [Scale-space],
      [Steerable 2nd order], [8], [$sigma in {1 dots 16}$], [Scale-space],
      table.hline(),
      [*Total classical*], [*71*], [], [],
      table.hline(),
      [WVF (from prior ablation)], [462], [$N_p times N_s times d$], [Polynomial fit],
      [LF (from prior ablation)], [112], [$N_p times N_s times m$], [Line averaging],
      [DL models], [5], [fixed weights], [Learned],
      table.hline(),
      [*Grand total*], [*650*], [], [],
      table.hline(),
    )
  ),
  caption: [Complete configuration count. Classical methods add 93 configurations per image per noise condition. Combined with existing WVF/LF and DL results, the full comparison spans 672 method-parameter combinations.],
)

= Scale Equivalence Reference

To compare classical methods with the WVF at matched spatial scale, @tab:equiv maps WVF support sizes to equivalent classical filter parameters. The effective support of the WVF is a circle of radius $r = sqrt(N_p slash pi)$; a Gaussian derivative filter with $sigma$ has 99% of its energy within radius $3 sigma$; a Sobel kernel of size $k$ covers a square of side $k$.

#figure(
  align(center,
    table(
      columns: 5,
      align: (right, right, right, right, right),
      stroke: none,
      table.hline(),
      table.header([*WVF $N_p$*], [*Eff. radius*], [*Eff. diameter*], [*Equiv. Gaussian $sigma$*], [*Equiv. Sobel $k$*]),
      table.hline(),
      [15], [2.2 px], [4 px], [0.7], [5],
      [25], [2.8 px], [6 px], [0.9], [5],
      [50], [4.0 px], [8 px], [1.3], [7],
      [100], [5.6 px], [11 px], [1.9], [11],
      [250], [8.9 px], [18 px], [3.0], [17],
      [500], [12.6 px], [25 px], [4.2], [25],
      table.hline(),
      [LF $m = 7$], [---], [15 px (line)], [2.5], [15],
      [LF $m = 14$], [---], [29 px (line)], [4.8], [29],
      [LF $m = 20$], [---], [41 px (line)], [6.8], [41],
      table.hline(),
    )
  ),
  caption: [Scale equivalence between WVF/LF support and classical filter parameters. "Equiv. Gaussian $sigma$" is computed as $r slash 3$ to match 99% energy. "Equiv. Sobel $k$" is $approx 2r$. The classical sweep extends to Sobel-51 ($approx N_p = 2043$) and Gaussian $sigma = 20$ ($approx N_p = 11{,}310$), well beyond the WVF's maximum.],
) <tab:equiv>

= Noise Conditions

The same noise conditions from the existing WVF/LF and DL ablation studies are used, ensuring direct comparability.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, left),
      stroke: none,
      table.hline(),
      table.header([*Noise Type*], [*Model*], [*Relevance*]),
      table.hline(),
      [Gaussian], [$I' = I + cal(N)(0, sigma^2)$], [Sensor thermal noise (baseline)],
      [Salt \& pepper], [$P("pixel" = 0 "or" 255) prop 1/(1+"SNR")$], [Dead pixels, bit errors],
      [Poisson], [$I' = "Pois"(I dot s)/s$], [Low-light shot noise],
      [Speckle], [$I' = I dot (1 + cal(N)(0, 1/"SNR"^2))$], [Sonar, radar, underwater],
      [Uniform], [$I' = I + cal(U)(-a, a)$], [ADC quantization noise],
      table.hline(),
    )
  ),
  caption: [Five noise types at 8 SNR levels (0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, $infinity$).],
)

= Evaluation Protocol

Identical to prior ablation studies: 1001 thresholds, 3-pixel match radius, dataset-wide ODS, no post-processing. Classical methods that produce continuous gradient magnitudes (Sobel, Prewitt, Gaussian derivatives, etc.) are evaluated by threshold sweep. Methods that produce binary outputs (Canny, Marr-Hildreth) are evaluated both as binary maps and---where applicable---as the continuous intermediate representation (e.g., Canny's NMS output before hysteresis).

= Computational Budget

All classical edge detectors are implemented as GPU-accelerated convolutions via PyTorch's `torch.nn.functional.conv2d`. Even the largest kernels (Sobel-51, Gaussian $sigma = 20$) run in $<$5 ms per image on an A100. The full sweep of 71 configurations on one 720$times$1280 image completes in $<$0.5 seconds.

#figure(
  align(center,
    table(
      columns: 4,
      align: (left, right, right, right),
      stroke: none,
      table.hline(),
      table.header([*Component*], [*Configs*], [*Images $times$ conditions*], [*Est. time*]),
      table.hline(),
      [Classical filters (GPU)], [71], [$20 times 36 = 720$], [$approx 6$ min],
      [ODS evaluation (CPU)], [71], [720], [$approx 15$ min],
      table.hline(),
      [*Total*], [], [], [*$approx 21$ min*],
      table.hline(),
    )
  ),
  caption: [Estimated runtime for the classical ablation on one A100 GPU. All convolutions are GPU-accelerated via PyTorch. ODS evaluation uses the vectorized CPU pipeline.],
)

= Expected Outcomes

*Hypothesis 1: Gaussian derivatives at matched scale will match WVF performance.* The WVF at $N_p = 50$ has an effective support radius of $r approx 4$ pixels. A Gaussian derivative filter at $sigma approx 2$ uses a comparable spatial scale with isotropic smoothing. If the WVF's advantage is primarily due to its larger support (noise averaging), then Gaussian derivatives at $sigma = 2$--$4$ should perform similarly. If the WVF still wins, the advantage is in the polynomial model or orientation-specific fitting.

*Hypothesis 2: Extended Sobel with large kernels will approach WVF at low SNR.* Sobel-31$times$31 uses 961 pixels per gradient estimate, comparable to WVF at $N_p = 300$. At low SNR where noise averaging dominates, these should converge.

*Hypothesis 3: Canny will underperform raw gradient magnitudes.* Our earlier Canny NMS test showed that NMS uniformly degrades ODS by 0.03--0.14. This should hold for all classical methods: the continuous gradient magnitude outperforms the thinned/thresholded version under the ODS metric.

= Relationship to WVF

The most important comparison is between the WVF and Gaussian derivatives at matched scale. The WVF fits a 4th-order polynomial to $N_p$ neighbors at each orientation, extracting the normal derivative. A Gaussian derivative filter convolves with the derivative of a Gaussian, producing an equivalent directional derivative with isotropic support. The WVF has three potential advantages:

1. *Orientation-specific support*: the WVF evaluates the polynomial along the edge normal, while Gaussian derivatives use circular support. This should matter for anisotropic features.

2. *Higher-order fitting*: the WVF fits a 4th-order surface, potentially capturing curvature information that 1st-order Gaussian derivatives miss.

3. *Non-uniform neighbor weighting*: the WVF's least-squares fit implicitly weights neighbors by their position in the Taylor expansion, while Gaussian derivatives use symmetric Gaussian weighting.

If Gaussian derivatives at $sigma = 2$--$4$ match WVF at $N_p = 50$--$250$, then advantages 1--3 are negligible and the WVF reduces to an expensive reimplementation of a standard technique. If the WVF consistently outperforms, the polynomial fitting provides genuine value beyond scale selection.
