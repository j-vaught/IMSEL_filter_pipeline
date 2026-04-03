= Parameter Analysis <sec:parameter-analysis>

The algebraic equivalences established in Sections 3--5 guarantee that the fused stencil, geometric kernels, and naive line filter produce identical or near-identical gradient maps for matched parameters. This section evaluates how sensitive edge detection accuracy is to those parameters and identifies optimal operating points for each filter variant. All ablations reported here were conducted using the naive (unfused) WVF and LF implementations, prior to the derivation of the fused stencil and geometric kernel formulations. Because the fused stencil is mathematically equivalent to the naive LF, and the geometric kernels are evaluated separately in @sec:experiments, the parameter findings transfer directly to the accelerated variants.

== Methodology and Scope <sec:param-methodology>

The naive implementation's computational cost was the primary constraint on ablation scope. Each full-dataset LF evaluation at the published parameters ($N_p = 250$, $N_s = 18$, $m = 14$, $d = 4$) required approximately 2.4 seconds per image on an NVIDIA A100 GPU, making exhaustive grid search over the full joint parameter space infeasible. The fused stencil and geometric kernels, derived after these ablations were complete, would have reduced per-evaluation cost by 18--54$times$ and enabled a substantially denser search grid. We note this as a limitation and identify priority ablations for future work in @sec:ablations-not-conducted.

The total experimental scope comprised 1,206 WVF configurations and 168 LF configurations evaluated across four datasets. UDED contributed 30 images, BIPED v1 contributed 50, BIPED v2 contributed 50, and BSDS500 contributed 200, totaling 330 images. Across all parameter-dataset combinations, over 500,000 individual filter evaluations were performed. All accuracy measurements use the Optimal Dataset Scale (ODS) F-measure, evaluated at a fixed threshold per dataset.

== Support Size <sec:param-np>

The support size $N_p$ is the most computationally consequential parameter. It directly controls the number of memory reads per pixel per orientation, which is the dominant cost term in the gather-dot-product inner loop, and it determines the physical size of the circular neighborhood from which intensity samples are drawn. The published recommendation of $N_p = 250$ implies a circular neighborhood of radius approximately 9 pixels. If smaller values suffice for accurate gradient estimation, both accuracy and speed improve simultaneously, a situation that is unusual because most filter parameters trade one for the other.

We tested $N_p in {10, 15, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500}$ for the WVF and a representative subset for the LF, where the cost of each evaluation scales with $N_p$ through the design matrix dimensions. On clean imagery, the optimal $N_p$ lies in the range 25--100 depending on the dataset. ODS improvements over the published $N_p = 250$ ranged from 0.01--0.05 for the WVF and 0.06--0.16 for the LF. Larger neighborhoods average over too many pixels, smoothing out fine edge structure that the evaluation protocol rewards.

Under additive noise, the optimal $N_p$ grows monotonically with noise severity. At high signal-to-noise ratio (SNR), small neighborhoods ($N_p = 25$--$50$) are optimal because they preserve fine spatial detail. As SNR decreases, the noise-averaging benefit of larger neighborhoods outweighs the loss of spatial resolution, and the optimal $N_p$ shifts to 250--500. This monotonic relationship between noise level and optimal filter size is consistent with classical results in statistical estimation. The bias-variance tradeoff shifts toward variance reduction, favoring larger support, as observation noise increases. The published large-support parameterization reflects this. The maritime application for which the filter was originally developed operates in a regime where noise dominates and large $N_p$ is appropriate.

#figure(
  image("../figures/fig_sec07_ods_vs_np_by_degree.pdf", width: 85%),
  caption: [ODS F-score versus support size $N_p$ for polynomial degrees $d = 2, 3, 4, 5$ at fixed $N_s = 18$. Lower-order polynomials ($d = 2$) consistently outperform higher orders across all support sizes, with peak performance at $N_p approx 40$--$65$. The overdetermination ratio $N_p slash M$ is higher for $d = 2$, providing more stable gradient estimates.],
) <fig:ods-vs-np-degree>

== Polynomial Order <sec:param-d>

The polynomial order $d$ controls the expressiveness of the local intensity model fitted within each circular neighborhood. Higher orders capture more complex local structure, including curvature and inflection, but require more unknown coefficients. For a bivariate polynomial of order $d$, the number of unknowns is $M = (d+1)(d+2) slash 2$, yielding $M = 15$ at $d = 4$ and $M = 6$ at $d = 2$. The overdetermination ratio $N_p slash M$ governs the degree of noise averaging in the least-squares fit. We hypothesized that $d = 2$ would suffice for gradient estimation because the first-order partial derivative $hat(f)_x$ depends only on the linear coefficients of the fitted polynomial. Higher-order terms do not contribute to the gradient coefficient itself; they affect only the conditioning of the system.

We tested $d in {2, 3, 4}$ at each $(N_p, N_s)$ combination. The finding was consistent across all four datasets. The second-order polynomial ($d = 2$) outperforms $d = 4$ on clean imagery by a small but persistent margin of 0.005--0.02 ODS. The mechanism is straightforward. At $d = 2$ with $N_p = 25$, the system is $25 slash 6 approx 4.2 times$ overdetermined. At $d = 4$ with the same $N_p = 25$, it is only $25 slash 15 approx 1.67 times$ overdetermined, providing far less noise averaging and yielding a less stable gradient estimate.

We did not test $d = 1$. A first-order polynomial ($M = 3$) reduces the Taylor model to a plane fit, which is equivalent to a weighted average of finite differences. This loses the ability to distinguish edge curvature from noise and collapses the method to something resembling a weighted Sobel operator. We considered this outside the parameter range of interest for the polynomial fitting framework.

== Orientation Count <sec:param-ns>

The orientation count $N_s$ determines the angular resolution of the discrete orientation sweep. The published setting of $N_s = 18$ spaces candidate orientations at $20 degree$ intervals. Additional orientations improve angular precision but multiply computational cost linearly, since the gather-dot-product must be repeated for each orientation. If performance saturates at low $N_s$, the computational savings are substantial.

We tested $N_s in {2, 4, 6, 8, 12, 18, 24, 36}$. Performance saturated at $N_s = 4$--$6$ on all four datasets. Increasing from $N_s = 6$ to $N_s = 18$ yielded less than 0.002 ODS improvement while tripling computation. This is consistent with the angular smoothness of the polynomial fit. The least-squares derivative estimate varies smoothly over orientation angle $theta$, so the discrete maximum over $theta_k$ is well-resolved even with coarse angular sampling.

We did not test $N_s = 1$. A single orientation reduces the filter to a fixed-direction gradient operator, losing the orientation-selective property entirely. The classical baselines (Sobel, Prewitt) serve as implicit references for this regime, since they are effectively $N_s = 2$ operators combined via $arctan$.

#figure(
  image("../figures/fig_sec07_ns_cliff.pdf", width: 85%),
  caption: [ODS F-score versus number of orientations $N_s$ at various support sizes $N_p$. A critical jump occurs at $N_s = 3$ (dashed line), where ODS increases by 0.20--0.24 relative to $N_s = 2$. Beyond $N_s = 3$, performance plateaus. The cliff indicates that three orientations suffice for accurate edge detection; additional orientations provide negligible benefit while multiplying computational cost.],
) <fig:ns-cliff>

== Line Half-Width <sec:param-m>

The line half-width $m$ controls the spatial extent of the LF's line extension and is the parameter unique to the LF. The WVF is the $m = 0$ special case. The published setting of $m = 14$ creates a line of $2 m + 1 = 29$ virtual evaluation points, each spawning a full polynomial fit over its own circular neighborhood. This is the most expensive parameter because it multiplies the number of polynomial fits per pixel per orientation by $(2 m + 1)$. Kruskal-Wallis effect size analysis identified $m$ as the most influential parameter, with $eta^2 = 0.34$, exceeding the effect sizes of $N_p$, $N_s$, and $d$.

We tested $m in {0, 1, 2, 3, 5, 7, 10, 14}$ at selected $(N_p, N_s, d)$ combinations. On clean data, the WVF ($m = 0$) matched or exceeded the full LF across all four datasets. The line extension provides no benefit on clean imagery. The additional spatial context it introduces is unnecessary when noise is low and edges are well-defined. Under noise, moderate values ($m = 2$--$7$) are beneficial, but $m = 14$ is excessive even at low SNR.

The LF ablation grid is substantially sparser than the WVF grid (168 versus 1,206 configurations). Each LF evaluation is $(2 m + 1) times$ more expensive than the corresponding WVF evaluation. At $m = 14$, a single full-dataset LF run requires approximately 36 seconds per image. Exhaustive search was computationally prohibitive with the naive implementation, so we sampled a representative subset of $(N_p, N_s, d)$ combinations at each $m$ value.

#figure(
  image("../figures/fig_sec07_lf_heatmap.pdf", width: 90%),
  caption: [LF ODS heatmap as a function of support size $N_p$ and line half-width $m$. Each cell shows the ODS averaged across $N_s in {18, 36, 72}$. The optimal region (ODS $approx 0.84$) occurs at moderate $N_p = 50$--$100$ and small $m = 1$--$3$. Performance degrades with both excessive support size and excessive line extension, confirming that the published parameters ($N_p = 250$, $m = 14$) are overspecified for clean imagery.],
) <fig:lf-heatmap>

== Geometric Kernel Parameters <sec:param-geometric>

The geometric kernels parameterize spatial extent through the standard deviations $sigma_u$ and $sigma_v$ of the Gaussian envelope along and across the edge direction, respectively. These play an analogous role to $N_p$ and $m$ in the polynomial filter but with cleaner semantics. The parameter $sigma_u$ directly sets the edge-parallel smoothing extent, $sigma_v$ sets the edge-normal extent, and the aspect ratio $sigma_u slash sigma_v$ controls the degree of anisotropy. The grid resolution in each direction is fixed at $ceil(3 sigma)$ pixels, ensuring that the Gaussian envelope is sampled out to three standard deviations.

We tested $sigma_u in {1.0, 1.5, 2.0, 3.0, 5.0, 7.0}$ and $sigma_v in {0.8, 1.0, 1.2, 1.5, 2.0, 2.5}$, forming a $6 times 6$ grid of 36 configurations per dataset. On clean data, $sigma_u = 2.0$ and $sigma_v = 1.2$ were optimal across all datasets, producing a compact kernel that matches the stencil footprint of the optimized polynomial filter. Under noise, both parameters grow. At SNR $approx 0.5$ dB, the optimal values shift to $sigma_u approx 7.2$ and $sigma_v approx 2.5$, reflecting the same noise-driven expansion observed in the polynomial parameter sweep.

An approximate equivalence mapping connects the two parameterizations. The edge-parallel extent satisfies $sigma_u approx m$ in pixel units, while the edge-normal extent satisfies $sigma_v approx sqrt(N_p slash pi)$, corresponding to the effective radius of the circular neighborhood. This mapping allows direct comparison between geometric and polynomial parameter sweeps and confirms that the two formulations explore the same underlying tradeoff between spatial resolution and noise averaging from different parameterization perspectives.

#figure(
  image("../figures/fig_sec07_ods_vs_snr.pdf", width: 85%),
  caption: [ODS F-score versus signal-to-noise ratio (SNR) for the ASF and five deep learning models under Gaussian noise. The ASF (garnet) degrades gracefully as noise increases, maintaining ODS $> 0.5$ even at SNR $= 0.3$. Deep learning models collapse below approximately SNR $= 2$, falling to near-random baseline ($"ODS" approx 0.33$). At SNR $= 1$, the ASF achieves ODS $= 0.72$ while all DL models are at 0.33--0.35.],
) <fig:ods-vs-snr>

#figure(
  image("../figures/fig_sec07_optimal_params_vs_snr.pdf", width: 85%),
  caption: [Optimal LF parameters versus SNR, aggregated across all noise types and datasets. As noise increases (SNR decreases), both optimal support size $N_p$ (garnet) and optimal line half-width $m$ (atlantic) grow monotonically. At clean conditions, optimal parameters are $N_p approx 53$, $m approx 2$. At SNR $= 0.3$, they shift to $N_p approx 190$, $m approx 16$. This validates the noise gain theory of @sec:noise-rejection: larger kernels become optimal under noise because they average over more pixels.],
) <fig:optimal-params-snr>

== Published Versus Optimal Parameters <sec:published-vs-optimal>

The published parameter recommendations ($N_p = 250$, $N_s = 18$, $d = 4$, $m = 14$) are suboptimal on every clean-imagery benchmark tested, by margins of 0.01--0.16 ODS. The optimal clean-data configuration ($N_p = 25$--$50$, $N_s = 4$--$6$, $d = 2$, $m = 0$) is simultaneously more accurate and approximately $112 times$ cheaper to compute. This disparity is the central finding of the parameter analysis. The filter is _highly_ parameter-sensitive, and the optimal settings are condition-dependent. No single configuration performs well across all noise regimes.

Under noise, the published parameters are partially vindicated. Large $N_p$ and nonzero $m$ become beneficial below approximately SNR $approx 5$--$10$ dB, where the noise-averaging capacity of the large support region compensates for the loss of spatial precision. However, even under noise, $d = 4$ does not outperform $d = 2$, and $N_s = 18$ remains unnecessary. The polynomial order and orientation count are consistently over-specified regardless of operating conditions.

The interpretation is not that the published parameters are wrong but that they were tuned for a different operating regime. The maritime imagery application for which the filter was developed features high noise, low contrast, and large-scale edge structures. For that regime, conservative over-smoothing is a defensible choice. For the standard computer vision benchmarks evaluated here, where edges are sharp and noise is minimal, the published parameters are unnecessarily conservative and the filter's potential is substantially underrealized.

== Ablations Not Conducted <sec:ablations-not-conducted>

Several parameter dimensions remain unexplored due to the computational constraints of the naive implementation. We identify six categories of ablation that were not conducted and assess their priority in light of the fused stencil and geometric kernel speedups now available.

_Polynomial basis type._ All experiments used the standard monomial basis ($1, x, y, x^2 slash 2, dots$). Orthogonal polynomial bases such as Legendre, Chebyshev, or Zernike polynomials could improve the conditioning of the design matrix $bold(A)$ and potentially change the optimal $d$. This is particularly relevant for large $N_p$, where the monomial Vandermonde matrix becomes ill-conditioned. We did not test alternative bases because the published formulation specifies the monomial basis and our goal was to evaluate the method as published before proposing alternatives.

_Weighted least squares._ The current formulation uses uniform (unweighted) least squares within the circular neighborhood. Distance-weighted fitting, as in locally weighted scatterplot smoothing (LOWESS), would downweight distant neighbors and create a smooth taper rather than the hard circular cutoff at radius $sqrt(N_p slash pi)$. This could improve accuracy at the boundary of the support region. We did not test this modification because it would introduce additional hyperparameters (kernel bandwidth, kernel shape) and because the geometric kernel variants already achieve the desired smooth taper through their Gaussian envelope.

_Adaptive parameter selection._ All results reported here use a fixed parameter setting across all pixels in an image. Local adaptation, for instance selecting $N_p$ or $sigma_v$ based on an estimate of local SNR, could improve performance in images with spatially varying noise or mixed edge scales. We did not test adaptive selection due to the complexity of reliable local SNR estimation and the additional per-pixel computational overhead it would entail.

_Joint parameter optimization._ Our ablation varied parameters semi-independently due to the cost of the naive implementation. A full joint grid search over $N_p times N_s times d times m$ at the full-dataset level would require approximately 50,000 LF evaluations per dataset. At 2.4 seconds per evaluation with the naive implementation, this amounts to roughly 33 GPU-hours per dataset. The fused stencil at 0.13 seconds per evaluation would reduce this to approximately 1.8 GPU-hours, and the geometric kernels at 0.045 seconds per evaluation would reduce it further to approximately 0.6 GPU-hours. Exhaustive joint optimization is now feasible with the accelerated implementations and is a high-priority future experiment. We expect the fused stencil and geometric kernels to reach equivalent throughput as dataset size increases, since the precomputed stencils and kernels are constructed once per orientation and reused across all images.

_Geometric kernel ablation under noise._ The $sigma_u$, $sigma_v$ sweep under noise was conducted at a smaller scale than the polynomial parameter sweep. A full noise-by-geometric-parameter ablation matching the scope of the polynomial noise study (five noise types at seven SNR levels) is a clear next step, now tractable with the geometric kernel implementation.

_Post-processing interaction._ We tested non-maximum suppression (NMS) and hysteresis thresholding on the raw gradient magnitude maps and found that both degrade ODS by 0.06--0.09. We did not exhaustively search post-processing parameters (NMS radius, hysteresis thresholds) because the degradation was consistent across all settings tested. This suggests a fundamental mismatch between the thick gradient maps produced by large-support filters and the thin-edge assumption underlying NMS, rather than a failure to find the right post-processing configuration.
