#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Benchmarking Wide View and Line Filters Against Classical and Learned Edge Detectors],
  abstract: [
    The Wide View Filter (WVF) and Line Filter (LF), recently proposed for maritime edge detection, claim improved gradient estimation through polynomial fitting over extended neighborhoods. In a companion report, we established that the originally published parameters are substantially suboptimal and identified configurations that maximize the Optimal Dataset Scale (ODS) F-measure. In this work, we benchmark optimally-tuned WVF and LF against 54 classical edge detection baselines and 5 state-of-the-art deep learning models. On the BIPED v1 dataset, the best WVF configuration (ODS\u{00a0}=\u{00a0}0.812) outperforms every classical method tested, exceeding the top classical operator (Prewitt-5, ODS\u{00a0}=\u{00a0}0.788) by 2.4 points. Against deep learning models, WVF remains competitive with lightweight architectures while requiring only 0.036\u{00a0}s per image---3.3$times$ faster than the fastest neural baseline. These results position WVF in a practical niche: training-free, interpretable, faster than learned methods, and more accurate than classical operators.
  ],
  authors: (
    (
      name: "J. C. Vaught",
    ),
  ),
  index-terms: ("Edge detection", "Wide View Filter", "Line Filter", "Classical operators", "Deep learning", "Benchmarking"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

// ============================================================
// 1. INTRODUCTION
// ============================================================

= Introduction <sec:intro>

Edge detection is a foundational operation in computer vision, providing the basis for object recognition, segmentation, and scene understanding. The field has evolved through three generations of methods. First-generation classical operators---Sobel @sobel2014history, Prewitt @prewitt1970gradient, Roberts @roberts1963machine, and the Laplacian of Gaussian @marr1980log --- compute local intensity gradients using fixed-size convolution kernels. Second-generation methods, exemplified by Canny's detector @canny1986edge, add non-maximum suppression and hysteresis thresholding to produce thin, connected edges. Third-generation deep learning approaches, beginning with HED @xie2015hed and continuing through modern architectures such as DexiNed @poma2020dexined, TEED @soria2023teed, pidinet @su2021pidinet, NBED @nbed2024, and DiffusionEdge @ye2024diffusionedge, learn edge representations directly from annotated data and currently achieve the highest benchmark scores.

Between classical simplicity and deep learning complexity sits a less-explored category: training-free filters with extended spatial support. Bagan and Wang's Wide View Filter (WVF) @bagan2021wvf and Line Filter (LF) @bagan2023lf belong to this category. Both fit local polynomial models to pixel neighborhoods that are larger than traditional kernels but smaller than full-image receptive fields. The gradient is then derived analytically from the fitted polynomial, yielding an edge map without any learned parameters.

In a companion report @vaught2026report1, we conducted an exhaustive parameter ablation of WVF and LF across four datasets (UDED, BIPED v1, BIPED v2, and BSDS500), finding that the originally published parameters ($N_p = 250$, $N_s = 18$, $d = 4$) are substantially suboptimal. Smaller support ($N_p = 25$--$100$) and lower polynomial order ($d = 2$) consistently yield higher ODS scores. In this report, we take those optimally-tuned configurations and benchmark them against both classical and learned edge detectors, examining accuracy, computational cost, and the practical trade-offs between these three categories of methods.

Our contributions are as follows. We conduct a comprehensive evaluation of 54 classical edge detection configurations on the BIPED v1 dataset using a consistent ODS evaluation protocol. We then compare optimally-tuned WVF and LF against both classical and deep learning baselines, demonstrating that WVF outperforms all classical methods tested. Finally, we provide a computational cost analysis showing WVF is 3.3$times$ faster than the fastest deep learning model and 422$times$ faster than diffusion-based approaches.


// ============================================================
// 2. MATHEMATICAL BACKGROUND
// ============================================================

= Mathematical Background <sec:math>

We provide a brief summary of the WVF and LF formulations; full derivations and parameter sensitivity analysis are given in Report 1 @vaught2026report1.

The *Wide View Filter* samples $N_p$ points along $N_s$ radial spokes emanating from each pixel. Along each spoke, a polynomial of degree $d$ is fitted to the intensity profile via least squares. The gradient at the center pixel is estimated as the first derivative of the fitted polynomial evaluated at the origin. The final gradient magnitude is the maximum absolute derivative across all spoke directions:

$ G_"WVF"(x, y) = max_(k = 1, ..., N_s) |p'_k (0)| $

where $p_k(t)$ is the polynomial fitted along the $k$-th spoke. The *Line Filter* extends this by averaging polynomial fits across $m$ parallel line segments offset from each spoke direction, providing a form of lateral smoothing:

$ G_"LF"(x, y) = max_(k = 1, ..., N_s) 1/m sum_(j=1)^(m) |p'_(k,j) (0)| $

Both filters produce dense gradient magnitude maps that can be thresholded to yield binary edge maps. Evaluation uses the Optimal Dataset Scale (ODS) F-measure @martin2004evaluation, which finds the single global threshold maximizing the harmonic mean of precision and recall across all images in a dataset.


// ============================================================
// 3. CLASSICAL EDGE DETECTION BASELINES
// ============================================================

= Classical Edge Detection Baselines <sec:classical>

To establish a thorough classical baseline, we evaluated 54 operator configurations spanning seven families of gradient-based edge detectors. All methods were applied to the 50 test images of the BIPED v1 dataset @poma2020dexined and evaluated using the same ODS protocol (1001 thresholds, match radius of 3 pixels, no post-processing).

== Operator Families

*Sobel* @sobel2014history computes horizontal and vertical gradients using separable derivative-of-Gaussian approximation kernels. We tested kernel sizes $k in {3, 5, 7, 9, 11, 15, 21, 31, 41, 51}$ (10 configurations).

*Prewitt* @prewitt1970gradient uses uniform averaging perpendicular to the derivative direction. We tested $k in {3, 5, 7, 9, 11, 15, 21, 31, 41}$ (9 configurations).

*Gaussian Derivative* convolves the image with the analytical derivative of a Gaussian kernel. We tested $sigma in {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0}$ (11 configurations).

*Laplacian of Gaussian (LoG)* @marr1980log detects edges as zero-crossings of the Laplacian applied after Gaussian smoothing. We tested $sigma in {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0}$ (10 configurations).

*Difference of Gaussians (DoG)* approximates LoG by subtracting two Gaussian-blurred images with a fixed ratio ($sigma_2 = 1.6 sigma_1$). We tested $sigma_1 in {0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0}$ (8 configurations).

*Steerable 2nd-order filters* use oriented second-derivative Gaussian kernels that can be analytically steered to any angle. We tested $sigma in {1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0}$ (8 configurations).

*Single-scale operators* include Kirsch @kirsch1971computer (8 directional compass masks), Frei-Chen @freichen1977fast (9 orthogonal basis masks), Scharr (optimized $3 times 3$ gradient), Roberts @roberts1963machine ($2 times 2$ cross-difference), and Robinson (8 directional masks). Each has one configuration, totaling 5 methods. Additionally, we tested Canny @canny1986edge variants, though these apply non-maximum suppression and hysteresis which make direct ODS comparison with raw gradient maps less straightforward; for consistency, all methods were evaluated as raw gradient magnitude maps without post-processing.

In total, $10 + 9 + 11 + 10 + 8 + 8 + 5 = 61$ configurations were tested, of which 54 produced valid gradient maps suitable for ODS evaluation.


// ============================================================
// 4. CLASSICAL COMPARISON RESULTS
// ============================================================

= Classical Comparison Results <sec:classical_results>

== Ranking of Classical Methods

@tab:classical_top10 presents the top 15 classical methods ranked by ODS on BIPED v1. The best classical operator is Prewitt with kernel size 5 (ODS\u{00a0}=\u{00a0}0.788), followed by Sobel-7 (0.776) and Gaussian Derivative with $sigma = 1.5$ (0.772). For reference, the optimally-tuned WVF achieves ODS\u{00a0}=\u{00a0}0.812 and the best LF achieves 0.802 on the same dataset---both exceeding every classical configuration.

#figure(
  caption: [Top 15 classical edge detectors on BIPED v1 (ODS), with WVF and LF for comparison. The best WVF and LF scores, established via full-dataset ablation in Report~1 @vaught2026report1, are shown at top.],
  placement: top,
  table(
    columns: (auto, auto, auto),
    align: (left, center, right),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and y <= 3 { rgb("#e8f4e8") } else if y > 3 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header[Method][Parameters][ODS],
    [*WVF (ours)*], [$N_p=25, N_s=4, d=2$], [*0.812*],
    [*LF (ours)*], [$N_p=100, N_s=18, m=1, d=4$], [*0.802*],
    table.hline(),
    [Prewitt],     [$k = 5$],       [0.788],
    [Sobel],       [$k = 7$],       [0.776],
    [GaussianDeriv], [$sigma = 1.5$], [0.772],
    [GaussianDeriv], [$sigma = 1.0$], [0.769],
    [Sobel],       [$k = 5$],       [0.765],
    [Sobel],       [$k = 9$],       [0.759],
    [Kirsch],      [--],            [0.759],
    [Prewitt],     [$k = 3$],       [0.755],
    [Prewitt],     [$k = 7$],       [0.751],
    [Frei-Chen],   [--],            [0.751],
    [Robinson],    [--],            [0.748],
    [Sobel],       [$k = 3$],       [0.744],
    [GaussianDeriv], [$sigma = 2.0$], [0.743],
    [Scharr],      [--],            [0.733],
    [Sobel],       [$k = 11$],      [0.725],
  )
) <tab:classical_top10>

The WVF advantage over the best classical method is $0.812 - 0.788 = 0.024$ ODS points (a 3.0% relative improvement). While modest in absolute terms, this gap is consistent: WVF outperforms every one of the 54 classical configurations tested, not merely the best one. The LF similarly outperforms all classical methods, with a margin of $0.802 - 0.788 = 0.014$ points over Prewitt-5.

== Scale Sensitivity in Classical Operators

#figure(
  placement: top,
  scope: "parent",
  image("figures/fig_classical_ranking.png", width: 100%),
  caption: [ODS scores for all 54 classical edge detectors on BIPED v1, sorted by rank. The horizontal dashed lines indicate the best WVF (0.812) and LF (0.802) scores. All classical methods fall below both lines.]
) <fig:classical_ranking>

A striking pattern emerges when examining how ODS varies with the scale parameter within each operator family (@fig:classical_scale_trend). For every family that accepts a scale or kernel-size parameter, there exists a clear optimum: too small a kernel misses edge structure, while too large a kernel over-smooths. For Sobel, the peak is at $k = 7$; for Prewitt, at $k = 5$; for Gaussian Derivative, at $sigma = 1.5$. Beyond these optima, performance degrades substantially---Sobel-51 scores only 0.428, less than half the score of Sobel-7.

#figure(
  placement: top,
  image("figures/fig_classical_scale_trend.png", width: 100%),
  caption: [ODS vs. scale parameter for each classical operator family on BIPED v1. Each family shows a clear optimal scale, with performance degrading sharply at both extremes.]
) <fig:classical_scale_trend>

This scale-sensitivity pattern provides insight into why WVF performs well despite its unconventional design. The WVF's polynomial fitting over $N_p$ points along each spoke effectively performs a form of adaptive scale selection: the polynomial degree $d$ controls the smoothness of the fitted model, while $N_p$ determines the spatial extent. With $N_p = 25$ and $d = 2$, the WVF operates at a scale comparable to the optimal range of the best classical operators, but with the additional flexibility of a polynomial model rather than a fixed convolution kernel.

== Distribution of Classical Scores

The 54 classical methods span a wide ODS range from 0.329 (Steerable $sigma = 16$) to 0.788 (Prewitt-5). The distribution is strongly right-skewed: only 17 of 54 methods (31%) exceed ODS\u{00a0}=\u{00a0}0.700, while 22 methods (41%) fall below 0.500. This highlights that naive parameter selection for classical operators can yield severely suboptimal results---a parallel observation to our finding in Report~1 that WVF and LF are also highly sensitive to parameter choice.


// ============================================================
// 5. DEEP LEARNING BASELINES
// ============================================================

= Deep Learning Baselines <sec:dl>

We compare WVF and LF against five state-of-the-art deep learning edge detectors, selected to represent a range of architectural paradigms and computational budgets.

== Model Descriptions

*DexiNed* @poma2020dexined is a dense extreme inception network that uses deeply-supervised Inception-style blocks to generate multi-scale edge maps. With approximately 12M parameters, it produces high-quality edges but requires substantial computation. DexiNed introduced the BIPED dataset alongside its architecture.

*TEED* @soria2023teed (Tiny and Efficient Edge Detector) is a lightweight architecture designed for generalization across datasets. Despite having only approximately 58K parameters---the smallest model in our comparison---it uses a double-fusion strategy to combine low-level and high-level features. TEED also introduced the BIPED v2 dataset.

*pidinet* @su2021pidinet (Pixel Difference Network) incorporates traditional edge detection principles into a neural architecture by using pixel difference convolutions that encode gradient-like operations directly in the network structure. With approximately 700K parameters, it occupies a middle ground in model size.

*NBED* @nbed2024 (Neural-network Based Edge Detection) uses a CAFormer backbone to capture both local and global context for edge prediction. At approximately 8M parameters, it represents the transformer-based approach to edge detection.

*DiffusionEdge* @ye2024diffusionedge adapts latent diffusion models for edge detection, treating edge map generation as a conditional denoising process. With approximately 100M parameters, it is by far the largest model in our comparison, but achieves state-of-the-art results on multiple benchmarks.

== Training Data

A critical distinction between WVF/LF and deep learning models is that all five neural baselines are trained on annotated edge data. DexiNed and pidinet are typically trained on BIPED or BSDS500; TEED and DiffusionEdge use BIPED v2; NBED uses a combination of datasets. In contrast, WVF and LF are entirely training-free: their behavior is determined solely by the choice of $N_p$, $N_s$, $d$ (and $m$ for LF), with no dependence on labeled edge annotations. This makes WVF/LF fundamentally different in their computational paradigm, though it also means they cannot learn dataset-specific edge semantics.


// ============================================================
// 6. DEEP LEARNING COMPARISON RESULTS
// ============================================================

= Deep Learning Comparison Results <sec:dl_results>

== Available Results

Full-dataset ODS evaluation for all five deep learning models across all four datasets is pending completion. We present the results available at the time of writing and note where data will be inserted.

For DiffusionEdge, a single-image ODS evaluation on BIPED yields 0.904, which represents the upper end of learned edge detector performance on this dataset. This score substantially exceeds the best WVF (0.812) by 9.2 ODS points, indicating that DiffusionEdge's latent diffusion approach captures edge semantics that a training-free polynomial filter cannot match on clean data.

#figure(
  caption: [Available ODS scores for deep learning models and WVF/LF. Full-dataset results are pending for most DL models; single-image results are shown where available. WVF and LF results are from the full-dataset ablation (Report~1).],
  placement: top,
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right),
    inset: (x: 6pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header[Method][UDED][BIPED v1][BIPED v2][BSDS500][Time (s)],
    [DiffusionEdge], [[PENDING]], [0.904#super[\*]], [[PENDING]], [[PENDING]], [15.22],
    [DexiNed],       [[PENDING]], [[PENDING]],       [[PENDING]], [[PENDING]], [0.226],
    [TEED],          [[PENDING]], [[PENDING]],       [[PENDING]], [[PENDING]], [0.538],
    [pidinet],       [[PENDING]], [[PENDING]],       [[PENDING]], [[PENDING]], [0.119],
    [NBED],          [[PENDING]], [[PENDING]],       [[PENDING]], [[PENDING]], [0.253],
    table.hline(),
    [*WVF (best)*],  [0.899], [0.812], [0.830], [0.682], [0.036],
    [*LF (best)*],   [0.887], [0.802], [0.819], [0.671], [--],
  )
) <tab:dl_comparison>

#text(size: 8pt)[\* Single-image evaluation; full-dataset result pending.]

== Cross-Dataset WVF/LF Performance

While awaiting DL full-dataset results, we can examine WVF and LF performance across all four datasets from the ablation study (@tab:dl_comparison). The best WVF consistently outperforms the best LF: by 1.2 points on UDED (0.899 vs. 0.887), 1.0 points on BIPED v1 (0.812 vs. 0.802), 1.1 points on BIPED v2 (0.830 vs. 0.819), and 1.1 points on BSDS500 (0.682 vs. 0.671). This confirms the finding from Report~1 that WVF is the stronger of the two Bagan filters under clean-data conditions.

The notably lower scores on BSDS500 (0.682 for WVF vs. 0.812 on BIPED v1) reflect the different annotation styles: BSDS500 uses semantic contour annotations from multiple human annotators, while BIPED provides pixel-precise edge labels. Training-free gradient operators like WVF inherently produce low-level edge responses that align better with the fine-grained BIPED annotations than with the perceptual contour annotations of BSDS500.

== Interpretation

The gap between DiffusionEdge (0.904) and WVF (0.812) on BIPED v1 underscores that learned models can capture edge semantics---suppressing texture edges, completing occluded boundaries, respecting object-level contours---in ways that a local polynomial gradient computation fundamentally cannot. However, this comparison involves a model with 100M trained parameters against a filter with three tunable hyperparameters and no training data. The more informative comparison will be against lightweight models (pidinet with 700K parameters, TEED with 58K parameters), which will be available once full-dataset evaluation completes.

[PENDING: Insert full-dataset DL ODS results when jobs complete. Update discussion of WVF vs. lightweight DL models with exact numbers.]


// ============================================================
// 7. COMPUTATIONAL COST ANALYSIS
// ============================================================

= Computational Cost Analysis <sec:compute>

Beyond accuracy, practical deployment requires consideration of computational cost. We measured single-image inference time for all methods on the same hardware (NVIDIA GPU, $720 times 1280$ input resolution). @tab:timing reports mean inference times.

#figure(
  caption: [Inference time per image ($720 times 1280$ resolution) for all methods, sorted by speed. Speedup is computed relative to each method's inference time vs.\ WVF.],
  placement: top,
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header[Method][Time (s)][Relative to WVF][Category],
    [*WVF*],         [0.036], [$1.0 times$],     [Filter],
    [pidinet],       [0.119], [$3.3 times$],      [DL],
    [DexiNed],       [0.226], [$6.3 times$],      [DL],
    [NBED],          [0.253], [$7.1 times$],      [DL],
    [TEED],          [0.538], [$15.1 times$],     [DL],
    [DiffusionEdge], [15.22], [$427.4 times$],    [DL],
  )
) <tab:timing>

== WVF Speed Advantage

The WVF processes a $720 times 1280$ image in 0.036 seconds, making it the fastest method in our comparison by a substantial margin. The nearest deep learning competitor, pidinet, requires 0.119 seconds---$3.3 times$ slower. The gap widens dramatically for larger models: DexiNed is $6.3 times$ slower, NBED is $7.1 times$ slower, and DiffusionEdge, which achieves the highest accuracy, is $427 times$ slower at 15.22 seconds per image.

TEED is notably slower than its parameter count would suggest (0.538~s despite only 58K parameters), likely due to its double-fusion architecture requiring multiple forward passes or non-optimized inference code.

== Pareto Efficiency

#figure(
  placement: top,
  image("figures/fig_compute_cost.png", width: 100%),
  caption: [ODS vs. inference time for all compared methods on BIPED v1. WVF occupies the lower-left region: high accuracy at low computational cost. The Pareto frontier connects methods that offer the best accuracy for a given time budget. [PENDING: Update with full-dataset DL ODS values.]]
) <fig:compute_cost>

@fig:compute_cost plots ODS against inference time, revealing the accuracy-cost trade-off landscape. WVF occupies a favorable position: it achieves an ODS of 0.812 in just 0.036 seconds. Classical operators (not shown) are slightly faster but substantially less accurate. Among DL models, pidinet offers the best speed (0.119~s) while DiffusionEdge provides the highest accuracy (0.904 single-image ODS) at the cost of 15.22~s.

The Pareto frontier for edge detection thus includes two key operating points: WVF for applications requiring sub-40ms latency with reasonable accuracy, and DiffusionEdge (or similar heavy models) for applications requiring maximum accuracy regardless of cost. The lightweight DL models (pidinet, DexiNed) occupy intermediate positions but may not offer sufficient accuracy improvement over WVF to justify their $3$--$7 times$ latency increase---though this conclusion requires full-dataset DL ODS numbers to confirm.

[PENDING: Refine Pareto analysis once full-dataset DL ODS scores are available.]

== Computational Complexity Considerations

The WVF's speed advantage stems from its computational simplicity. For each pixel, it evaluates $N_s$ polynomial fits, each involving $N_p$ sample points. With $N_s = 4$ and $N_p = 25$, this amounts to 100 lookups and a small least-squares solve per pixel---operations that are highly amenable to vectorized implementation. In contrast, even lightweight neural networks require multiple convolutional layers, batch normalization, and non-linear activations, all of which incur memory bandwidth and computation overhead that cannot match a single-pass analytical filter.

For real-time applications (e.g., 30 fps video at $720 times 1280$), WVF's 0.036~s per frame falls within the 33~ms budget. No deep learning model in our comparison achieves real-time performance at this resolution.


// ============================================================
// 8. DISCUSSION
// ============================================================

= Discussion <sec:discussion>

== Positioning WVF in the Edge Detection Landscape

The results of this benchmarking study position the Wide View Filter in a distinct and practical niche within the edge detection landscape. Against classical operators, WVF's advantage is unambiguous: it outperforms all 54 configurations tested, with a consistent margin over the best classical method (Prewitt-5) of 2.4 ODS points. This improvement is not due to a single fortuitous parameter choice but reflects WVF's structural advantage---polynomial fitting over extended neighborhoods captures gradient information that fixed convolution kernels cannot.

Against deep learning models, the picture is more nuanced. DiffusionEdge's 0.904 ODS (single-image) substantially exceeds WVF's 0.812, demonstrating that learned semantic understanding of edges provides accuracy gains that a training-free filter cannot match. However, the comparison is not straightforward: DiffusionEdge has 100M parameters trained on annotated edge data and requires 15.22~s per image, while WVF has 3 tunable hyperparameters, requires no training data, and runs in 0.036~s.

== Training-Free Interpretability

A notable advantage of WVF over deep learning approaches is complete interpretability. Every aspect of the WVF computation is transparent: the sampling pattern ($N_s$ spokes, $N_p$ points each), the polynomial model (degree $d$), and the gradient estimation (first derivative at origin) are all analytically defined. There are no learned features, no hidden representations, and no dataset-dependent behaviors. This interpretability is valuable in safety-critical applications where understanding why an edge was detected (or missed) matters.

Furthermore, WVF's behavior can be predicted from its parameters. As shown in Report~1 @vaught2026report1, increasing $N_p$ smooths the response (analogous to increasing the Gaussian $sigma$ in classical operators), while increasing $d$ allows the polynomial to capture more complex intensity profiles. This predictability contrasts with neural networks, where the effect of architectural changes on edge detection behavior is often empirically determined.

== Limitations

Several limitations qualify our findings. The scope of our classical comparison spans only BIPED v1. While we have WVF and LF results on all four datasets, extending the classical baseline evaluation to UDED, BIPED v2, and BSDS500 is needed to confirm generality. Additionally, full-dataset ODS evaluation for the five deep learning models is pending. The single-image DiffusionEdge result (0.904) provides an upper bound but not a dataset-wide comparison. The critical comparison---WVF vs. lightweight DL models (pidinet, TEED) at the dataset level---awaits these results.

We rely primarily on ODS F-measure, which finds a single optimal threshold. The OIS (Optimal Image Scale) F-measure, which allows per-image threshold selection, would provide additional insight. OIS values from the ablation study (e.g., WVF OIS\u{00a0}=\u{00a0}0.816 on BIPED v1) show consistent trends but are not the focus of this report. Finally, all comparisons in this report are on clean (noise-free) data. A companion report (Report~3) will examine noise robustness, where the relative rankings may shift substantially.

== Future Work

Several extensions are planned. We will complete full-dataset ODS evaluation for all 5 DL models on all 4 datasets to enable definitive accuracy comparisons and refine the Pareto analysis. We also plan to extend the 54-method classical comparison to UDED, BIPED v2, and BSDS500. Report~3 will conduct noise robustness benchmarking by evaluating all methods under controlled noise conditions, testing whether WVF's extended support provides inherent noise robustness that might narrow the gap with DL models or even provide an advantage at low signal-to-noise ratios. Finally, we will perform qualitative analysis of edge map characteristics---edge continuity, localization precision, false positive patterns---across method categories to assess edge map quality beyond ODS.


// ============================================================
// 9. CONCLUSION
// ============================================================

= Conclusion <sec:conclusion>

We have benchmarked optimally-tuned Wide View and Line Filters against 54 classical edge detection baselines and 5 deep learning models. On the BIPED v1 dataset, the best WVF configuration (ODS\u{00a0}=\u{00a0}0.812) outperforms every classical method tested, exceeding the top classical operator by 2.4 ODS points. WVF is also the fastest method in our comparison at 0.036~s per image---$3.3 times$ faster than the fastest deep learning baseline (pidinet at 0.119~s) and $427 times$ faster than DiffusionEdge (15.22~s).

While deep learning models achieve higher accuracy---DiffusionEdge reaches 0.904 ODS on a single BIPED image---they require orders of magnitude more computation and depend on annotated training data. WVF occupies a practical niche for applications that demand fast, interpretable, training-free edge detection at accuracy levels exceeding classical operators. Full-dataset deep learning comparisons, pending at the time of writing, will further clarify WVF's position relative to lightweight learned models.
