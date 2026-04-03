= Related Work <sec:related>

The filters examined in this paper, the Wide View Filter (WVF) @bagan2021wvf and the Line Filter (LF) @bagan2023lf, are presented by their authors as novel contributions to edge detection. In this section we situate them within a sixty-year lineage of local polynomial fitting, orientation-selective filtering, and classical gradient estimation. The central observation is that the WVF and LF do not introduce new mathematical machinery. Rather, they assemble well-known components, specifically least-squares polynomial fitting over shaped support regions with orientation sweeping, into a particular configuration. Understanding these antecedents is essential for a fair evaluation of what, if anything, the WVF and LF add beyond existing methods.


== Local Polynomial Fitting for Derivative Estimation <sec:related:lp>

The idea of fitting a low-degree polynomial to data within a sliding window and then differentiating the polynomial to estimate derivatives dates to the foundational work of Savitzky and Golay @savitzkygolay1964. Their 1964 paper showed that the convolution weights for smoothing and differentiation can be precomputed from the pseudoinverse of a Vandermonde matrix constructed over the sample indices. For a one-dimensional window of $2m+1$ samples and a polynomial of degree $p$, the $k$-th derivative weights are the $k$-th row of $(bold(V)^top bold(V))^(-1) bold(V)^top$, where $bold(V)$ is the Vandermonde matrix. This formulation yields finite impulse response (FIR) filters whose coefficients depend only on the window size and polynomial degree, not on the data itself. The resulting filters are optimal in the least-squares sense and provide simultaneous smoothing and differentiation.

#figure(
  image("../figures/fig_sec02_savitzky_golay_1d.pdf", width: 100%),
  caption: [One-dimensional Savitzky--Golay filtering. A polynomial of degree $d$ is fitted to a sliding window of $2m+1$ samples (highlighted), and the derivative at the center point is extracted from the fitted coefficients. This example uses degree $d = 3$ and half-width $m = 3$ (window size 7).],
  placement: top,
) <fig:sg-1d>

Gorry @gorry1990sg generalized the Savitzky--Golay framework by deriving recursive relations for the convolution coefficients that extend naturally to higher-order derivatives and non-uniform sample spacing. Extensions to two-dimensional domains followed. Meer et al. @meer1991sg2d analyzed the frequency-domain properties of 2D polynomial fitting kernels in the context of image pyramids, establishing connections between polynomial order, window shape, and spectral characteristics. Luo et al. @luo2005sg2d provided a systematic analysis of the properties of Savitzky--Golay digital differentiators, including their frequency response and noise attenuation characteristics, and gave explicit constructions for 2D variants.

#figure(
  image("../figures/fig_sec02_savitzky_golay_2d.pdf", width: 100%),
  caption: [Two-dimensional Savitzky--Golay filtering on a $9 times 9$ pixel grid. Column heights represent pixel intensities. The inner $5 times 5$ window (garnet) is fitted with a degree $d = 2$ polynomial surface (green wireframe), and the partial derivatives $partial f \/ partial x$ and $partial f \/ partial y$ are extracted at the center pixel (green dot). Gray columns outside the window are not used in the fit.],
) <fig:sg-2d>

A closely related thread is the locally weighted scatterplot smoothing (LOWESS) framework of Cleveland @cleveland1979loess, which fits weighted polynomials within a neighborhood where the weights decay with distance from the center point. LOWESS differs from the standard Savitzky--Golay approach primarily in using a smooth distance-based weighting function rather than a uniform or binary window. The practical effect is that LOWESS produces derivative estimates that are influenced more strongly by nearby samples, at the cost of losing the precomputed-weight efficiency of Savitzky--Golay.

With this context, the WVF @bagan2021wvf can be understood as a 2D Savitzky--Golay filter with three specific design choices. First, the support region is circular (a disk of radius $r$) rather than the rectangular windows typical of the classical formulation. Second, the coordinate system is rotated so that one polynomial axis aligns with a candidate edge orientation $theta_k$. Third, the filter is evaluated at $K$ discrete orientations and the orientation yielding the maximum gradient response is selected. 

#figure(
  image("../figures/fig_sec02_loess_weighted_fit.pdf", width: 100%),
  caption: [LOWESS (Cleveland, 1979). Samples within the bandwidth are weighted by a distance-based kernel (tricube) from the target location; bar color encodes weight (darker garnet = higher weight, gray = low weight). This example uses bandwidth $h = 3$ and polynomial degree $d = 2$.],
) <fig:loess>

Each of these choices, circular support, coordinate rotation, and orientation sweeping, has independent precedent in the literature. The LF @bagan2023lf extends the WVF by averaging polynomial derivative estimates computed at _virtual expansion points_ distributed along a line perpendicular to the candidate edge direction, effectively replacing the single disk-shaped support with a rectangular strip sampled at multiple centers. This construction is a specific instance of line-averaged local polynomial regression, a technique that appears in the signal processing literature under various names.

#figure(
  image("../figures/fig_sec02_wvf_three_orientations.pdf", width: 100%),
  caption: [The WVF and LF as oriented 2D Savitzky--Golay filters. (a) The Wide View Filter selects a circular neighborhood of $N_p$ pixels around the target pixel, rotates local coordinates $(x', y')$ to align with a candidate edge orientation, and extracts the normal derivative $partial f \/ partial x'$ via polynomial fitting. (b) The Line Filter places $(2m+1)$ virtual evaluation points along the tangent direction $y'$, each centered on the nearest pixel. (c) At each virtual point, a full circular neighborhood is gathered and a polynomial fit applied. The derivative estimates are combined via Gaussian-weighted averaging. This example uses $m = 4$ (9 virtual points).],
  placement: top,
  scope: "parent",
) <fig:wvf-lf>

== Orientation-Selective Filtering <sec:related:oriented>

The problem of estimating image derivatives along arbitrary orientations has been addressed extensively through steerable filters. Freeman and Adelson @freeman1991steerable demonstrated that certain classes of filters, including Gaussian derivatives up to arbitrary order, can be analytically _steered_ to any orientation by taking a linear combination of a small, fixed set of basis filters. For a filter of angular order $n$, only $n + 1$ basis responses are required. The WVF's strategy of computing filter responses at $K$ discrete orientations and selecting the maximum is a brute-force discretization of the same underlying operation. Where steerable filters achieve continuous orientation interpolation via analytic basis decomposition, the WVF evaluates responses on a discrete grid, trading elegance and computational efficiency for implementation simplicity.


Phase-based methods provide an alternative route to orientation estimation. Morrone and Owens @morrone1987feature proposed detecting features at points of maximum _local energy_, defined through the analytic signal and its phase. Perona @perona1990oriented developed steerable and scalable kernels that combine orientation selectivity with scale adaptation, using the phase congruency of oriented filter responses to localize edges independently of contrast. These phase-based approaches offer a principled framework for orientation estimation that does not require the exhaustive search employed by the WVF.


Anisotropic Gaussian derivatives represent yet another strategy. Geusebroek et al. @geusebroek2003aniso developed efficient algorithms for computing derivatives of anisotropic (elongated) Gaussian kernels, enabling the construction of filters whose spatial support adapts to local image structure. The geometric relationship between their elongated Gaussian kernels and the oriented disk or strip supports of the WVF and LF is direct. Both families of methods implement the same core idea: an anisotropic, orientation-dependent smoothing kernel whose derivative provides a directional gradient estimate. The principal difference is that the Gaussian formulation admits closed-form expressions and separable implementations, while the WVF and LF rely on numerical construction of their weight matrices.

#figure(
  image("../figures/fig_sec02_anisotropic_gaussian_deriv.pdf", width: 100%),
  caption: [Anisotropic Gaussian derivatives (Geusebroek et al., 2003). An elongated Gaussian envelope aligned along the candidate edge direction is modulated by the normal derivative profile $(-v)$, producing the dipole pattern that is essentially identical to the geometric kernels proposed in this work.],
  placement: top,
) <fig:aniso-gaussian>


== Classical Edge Detection <sec:related:classical>

The earliest edge detection operators were small, hand-designed convolution masks. Roberts @roberts1963machine introduced $2 times 2$ cross-difference operators for detecting diagonal edges. Sobel @sobel2014history and Prewitt @prewitt1970gradient independently proposed $3 times 3$ masks that approximate first-order directional derivatives with a degree of smoothing orthogonal to the gradient direction. Kirsch @kirsch1971computer designed a set of eight $3 times 3$ templates, one for each compass direction, and defined the edge response as the maximum over all eight. The Frei--Chen basis @freichen1977fast decomposed the $3 times 3$ neighborhood into nine orthogonal subspaces, separating edge, line, and average components, and detected edges by projection onto the edge subspace. All of these operators share the property of fixed, small spatial support, which limits their noise robustness and their ability to adapt to edges at different scales.

#figure(
  image("../figures/fig_sec02_sobel_kirsch_combined.pdf", width: 100%),
  caption: [Classical $3 times 3$ edge operators. Left column: Sobel (1968) estimates horizontal and vertical gradients with two fixed kernels $G_x$ and $G_y$. Center and right columns: Kirsch (1971) evaluates eight compass masks (four shown) and selects the maximum response. Positive weights in garnet, negative in blue, center pixel in white.],
  placement: top,
) <fig:sobel-kirsch>


The Gaussian derivative framework placed edge detection on a firmer mathematical foundation. Marr and Hildreth @marr1980log proposed detecting zero crossings of the Laplacian of Gaussian ($nabla^2 G$), providing a principled coupling of smoothing scale and differentiation. Canny @canny1986edge formulated edge detection as an optimization problem, seeking the operator that maximizes detection probability and localization accuracy while minimizing multiple responses. His solution for step edges in white Gaussian noise is closely approximated by the first derivative of a Gaussian, and his framework introduced non-maximum suppression and hysteresis thresholding as post-processing stages that remain standard practice. Lindeberg @lindeberg1998edge extended these ideas to _automatic scale selection_, using normalized derivatives across scale space to identify the characteristic scale of each edge. A persistent limitation of all fixed-kernel methods, whether $3 times 3$ masks or Gaussian derivatives at a single scale, is that their spatial support does not adapt to the local noise level or to the spatial extent of the edge structure. The WVF and LF address this limitation by enlarging the support region, but as discussed in @sec:related:lp, the mechanism they use, polynomial fitting over large windows, is itself classical.


#figure(
  image("../figures/fig_sec02_canny_pipeline_stages.pdf", width: 100%),
  caption: [The Canny edge detection pipeline (1986). Five stages: Gaussian smoothing, gradient computation, angle estimation via arctan, non-maximum suppression, and hysteresis thresholding. The pre-smoothing step trades edge localization for noise robustness; stronger smoothing blurs weak edges.],
  placement: top,
) <fig:canny>


== Deep Learning Edge Detection <sec:related:dl>

The introduction of deep convolutional networks to edge detection marked a paradigm shift in performance on standard benchmarks. Holistically-Nested Edge Detection (HED) @xie2015hed pioneered the use of multi-scale side outputs from a VGG-style backbone, fusing predictions from multiple network depths to produce edge maps that capture both fine details and large-scale contours. Structured Forests @dollar2013structured, while not a deep method, introduced the random-forest-based approach to structured prediction for edges that motivated much subsequent work. DexiNed @poma2020dexined extended the multi-scale fusion paradigm with a dense extreme inception architecture, demonstrating strong generalization to datasets unseen during training. The Tiny and Efficient Edge Detector (TEED) @soria2023teed achieved competitive accuracy with a dramatically reduced parameter count, demonstrating that architectural efficiency and edge detection quality need not be in tension. PIDINet @su2021pidinet incorporated traditional pixel-difference operations as inductive biases within a lightweight network, explicitly bridging classical gradient computation and learned feature extraction. More recently, NBED @chen2024nbed proposed a neural-network-based architecture with attention mechanisms for boundary detection, and DiffusionEdge @ye2024diffusionedge applied diffusion probabilistic models to generate crisp edge maps through iterative denoising.

#figure(
  image("../figures/fig_sec02_hed_architecture.pdf", width: 100%),
  caption: [Holistically-Nested Edge Detection (Xie and Tu, 2015). A VGG-style encoder produces multi-scale side outputs $S_1$ through $S_5$ that are fused into a final edge map. Each side output captures edges at a different spatial scale.],
  placement: top,
) <fig:hed>

#figure(
  image("../figures/fig_sec02_pidinet_pixel_diff.pdf", width: 100%),
  caption: [PIDINet pixel difference convolutions (Su et al., 2021). Instead of multiplying pixel values by learned weights, PIDINet computes pixel _differences_ $Delta_i$ between the center and each neighbor, then applies learned weights $w_i$ to the differences. This is structurally similar to a classical gradient operator but with trainable coefficients.],
) <fig:pidinet>

These methods achieve state-of-the-art performance on benchmarks such as BSDS500 @arbelaez2011bsds500, but they share several limitations that motivate continued interest in classical approaches. First, they depend on labeled training data, and their performance degrades under _domain shift_ when the test distribution differs from the training distribution. Underwater imagery, medical imaging, and satellite data all present domains where labeled edge data is scarce. Second, the learned filters are opaque. One cannot, in general, characterize the spatial frequency response, noise sensitivity, or derivative accuracy of a trained network's edge output in the way that one can for a Savitzky--Golay filter or a Gaussian derivative. Third, deep methods provide no formal guarantees on gradient accuracy. The WVF and LF, by virtue of their polynomial fitting formulation, admit closed-form bias and variance expressions that can guide parameter selection, an advantage that no data-driven method currently shares.


== GPU Acceleration for Image Filtering <sec:related:gpu>

The computational cost of applying large-support filters at multiple orientations has historically been a barrier to deploying methods like the WVF and LF in real-time applications. Modern GPU computing frameworks have substantially lowered this barrier. The cuDNN library @chetlur2014cudnn provides highly optimized implementations of standard convolution operations, including batched and grouped convolutions that can be leveraged for multi-orientation filtering. For operations that do not map cleanly onto standard convolution, the Triton compiler @tillet2019triton enables rapid development of custom GPU kernels in a high-level Python-embedded language, with automatic tiling and memory management. CUTLASS @cutlass2023 provides composable CUDA templates for general matrix operations that can express the gather-and-dot-product structure underlying all three filter variants considered in this work.

#figure(
  image("../figures/fig_sec02_gather_dot_product.pdf", width: 100%),
  caption: [The gather-dot-product abstraction. Both regular convolution (left, as in cuDNN) and irregular stencil operations (right, as in the ASF) reduce to the same primitive: gather pixel intensities at precomputed offsets, multiply by precomputed weights, and sum. This uniformity enables a single GPU kernel to serve all three proposed filter variants.],
  placement: top,
) <fig:gather-dot>

The key observation for GPU implementation is that the WVF, LF, and the anisotropic filter variant proposed in this paper all reduce to the same computational primitive. For each pixel and each candidate orientation, a set of pixel values is gathered from the support region and multiplied by a precomputed weight vector. This gather-dot-product operation is embarrassingly parallel across pixels and orientations, making it well-suited to GPU execution. The orientation sweep, which the WVF performs sequentially, can be issued as a batched operation over the $K$ orientation channels, and the subsequent maximum selection is a trivial reduction. The practical consequence is that even the WVF's brute-force orientation search becomes tractable when implemented as a single fused GPU kernel, removing much of the computational motivation for the analytic interpolation provided by steerable filters.
