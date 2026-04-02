= Related Work <sec:related>

The filters examined in this paper, the Wide View Filter (WVF) @bagan2021wvf and the Line Filter (LF) @bagan2023lf, are presented by their authors as novel contributions to edge detection. In this section we situate them within a sixty-year lineage of local polynomial fitting, orientation-selective filtering, and classical gradient estimation. The central observation is that the WVF and LF do not introduce new mathematical machinery. Rather, they assemble well-known components, specifically least-squares polynomial fitting over shaped support regions with orientation sweeping, into a particular configuration. Understanding these antecedents is essential for a fair evaluation of what, if anything, the WVF and LF add beyond existing methods.


== Local Polynomial Fitting for Derivative Estimation <sec:related:lp>

The idea of fitting a low-degree polynomial to data within a sliding window and then differentiating the polynomial to estimate derivatives dates to the foundational work of Savitzky and Golay @savitzkygolay1964. Their 1964 paper showed that the convolution weights for smoothing and differentiation can be precomputed from the pseudoinverse of a Vandermonde matrix constructed over the sample indices. For a one-dimensional window of $2m+1$ samples and a polynomial of degree $p$, the $k$-th derivative weights are the $k$-th row of $(bold(V)^top bold(V))^(-1) bold(V)^top$, where $bold(V)$ is the Vandermonde matrix. This formulation yields finite impulse response (FIR) filters whose coefficients depend only on the window size and polynomial degree, not on the data itself. The resulting filters are optimal in the least-squares sense and provide simultaneous smoothing and differentiation.

#figure(
  image("../figures/fig_sec02_savitzky_golay_1d.pdf", width: 80%),
  caption: [One-dimensional Savitzky--Golay filtering. A polynomial of degree $d$ is fitted to a sliding window of $2m+1$ samples (highlighted), and the derivative at the center point is extracted from the fitted coefficients. This example uses degree $d = 3$ and half-width $m = 3$ (window size 7).],
  placement: top,
) <fig:sg-1d>

Gorry @gorry1990sg generalized the Savitzky--Golay framework by deriving recursive relations for the convolution coefficients that extend naturally to higher-order derivatives and non-uniform sample spacing. Extensions to two-dimensional domains followed. Meer et al. @meer1991sg2d analyzed the frequency-domain properties of 2D polynomial fitting kernels in the context of image pyramids, establishing connections between polynomial order, window shape, and spectral characteristics. Luo et al. @luo2005sg2d provided a systematic analysis of the properties of Savitzky--Golay digital differentiators, including their frequency response and noise attenuation characteristics, and gave explicit constructions for 2D variants.

#figure(
  image("../figures/fig_sec02_savitzky_golay_2d.pdf", width: 100%),
  caption: [Two-dimensional Savitzky--Golay filtering on a $9 times 9$ pixel grid. Column heights represent pixel intensities. The inner $5 times 5$ window (garnet) is fitted with a degree $d = 2$ polynomial surface (green wireframe), and the partial derivatives $partial f \/ partial x$ and $partial f \/ partial y$ are extracted at the center pixel (green dot). Gray columns outside the window are not used in the fit.],
) <fig:sg-2d>

A closely related thread is the locally weighted scatterplot smoothing (LOWESS) framework of Cleveland @cleveland1979loess, which fits weighted polynomials within a neighborhood where the weights decay with distance from the center point. LOWESS differs from the standard Savitzky--Golay approach primarily in using a smooth distance-based weighting function rather than a uniform or binary window. The practical effect is that LOWESS produces derivative estimates that are influenced more strongly by nearby samples, at the cost of losing the precomputed-weight efficiency of Savitzky--Golay.

#figure(
  image("../figures/fig_sec02_loess_weighted_fit.pdf", width: 70%),
  caption: [LOWESS (Cleveland, 1979). Data points are weighted by a distance-based kernel centered on the target point; nearer samples (larger dots) contribute more strongly to the local polynomial fit.],
) <fig:loess>

With this context, the WVF @bagan2021wvf can be understood as a 2D Savitzky--Golay filter with three specific design choices. First, the support region is circular (a disk of radius $r$) rather than the rectangular windows typical of the classical formulation. Second, the coordinate system is rotated so that one polynomial axis aligns with a candidate edge orientation $theta_k$. Third, the filter is evaluated at $K$ discrete orientations and the orientation yielding the maximum gradient response is selected. Each of these choices, circular support, coordinate rotation, and orientation sweeping, has independent precedent in the literature. The LF @bagan2023lf extends the WVF by averaging polynomial derivative estimates computed at _virtual expansion points_ distributed along a line perpendicular to the candidate edge direction, effectively replacing the single disk-shaped support with a rectangular strip sampled at multiple centers. This construction is a specific instance of line-averaged local polynomial regression, a technique that appears in the signal processing literature under various names.

#figure(
  image("../figures/fig_sec02_wvf_rotated_neighborhood.pdf", width: 75%),
  caption: [The Wide View Filter as a rotated 2D Savitzky--Golay derivative filter. A circular neighborhood of $N_p$ pixels surrounds the target pixel. Local coordinates $(x', y')$ are rotated to align $x'$ with the candidate edge normal at angle $theta$. The normal derivative $hat(f)_(x')$ is extracted via least-squares polynomial fitting.],
  placement: top,
) <fig:wvf>

#figure(
  image("../figures/fig_sec02_lf_line_averaged.pdf", width: 75%),
  caption: [The Line Filter as line-averaged Savitzky--Golay. The polynomial fit is evaluated at $(2m+1)$ virtual positions along the edge tangent, each with its own circular neighborhood. The neighborhoods overlap heavily. A Gaussian weight profile $w_j$ combines the derivative estimates into a single response $R_k$.],
  placement: top,
) <fig:lf>


== Orientation-Selective Filtering <sec:related:oriented>

The problem of estimating image derivatives along arbitrary orientations has been addressed extensively through steerable filters. Freeman and Adelson @freeman1991steerable demonstrated that certain classes of filters, including Gaussian derivatives up to arbitrary order, can be analytically _steered_ to any orientation by taking a linear combination of a small, fixed set of basis filters. For a filter of angular order $n$, only $n + 1$ basis responses are required. The WVF's strategy of computing filter responses at $K$ discrete orientations and selecting the maximum is a brute-force discretization of the same underlying operation. Where steerable filters achieve continuous orientation interpolation via analytic basis decomposition, the WVF evaluates responses on a discrete grid, trading elegance and computational efficiency for implementation simplicity.

#figure(
  image("../figures/fig_sec02_steerable_filter_basis.pdf", width: 85%),
  caption: [Steerable filters (Freeman and Adelson, 1991). Two basis kernels $G_1$ and $G_2$ (first derivatives of a Gaussian in the $x$ and $y$ directions) are combined as $R(theta) = cos(theta) R_1 + sin(theta) R_2$ to analytically interpolate the response at any orientation, without discrete evaluation.],
  placement: top,
) <fig:steerable>

Phase-based methods provide an alternative route to orientation estimation. Morrone and Owens @morrone1987feature proposed detecting features at points of maximum _local energy_, defined through the analytic signal and its phase. Perona @perona1990oriented developed steerable and scalable kernels that combine orientation selectivity with scale adaptation, using the phase congruency of oriented filter responses to localize edges independently of contrast. These phase-based approaches offer a principled framework for orientation estimation that does not require the exhaustive search employed by the WVF.

#figure(
  image("../figures/fig_sec02_oriented_energy_polar.pdf", width: 70%),
  caption: [Oriented energy (Morrone and Owens, 1987). Even-symmetric and odd-symmetric filter pairs are combined to form the local energy $E(theta) = F_"even"^2 + F_"odd"^2$, whose peak indicates the dominant edge orientation.],
) <fig:oriented-energy>

Anisotropic Gaussian derivatives represent yet another strategy. Geusebroek et al. @geusebroek2003aniso developed efficient algorithms for computing derivatives of anisotropic (elongated) Gaussian kernels, enabling the construction of filters whose spatial support adapts to local image structure. The geometric relationship between their elongated Gaussian kernels and the oriented disk or strip supports of the WVF and LF is direct. Both families of methods implement the same core idea: an anisotropic, orientation-dependent smoothing kernel whose derivative provides a directional gradient estimate. The principal difference is that the Gaussian formulation admits closed-form expressions and separable implementations, while the WVF and LF rely on numerical construction of their weight matrices.

#figure(
  image("../figures/fig_sec02_anisotropic_gaussian_deriv.pdf", width: 75%),
  caption: [Anisotropic Gaussian derivatives (Geusebroek et al., 2003). An elongated Gaussian envelope aligned along the candidate edge direction is modulated by the normal derivative profile $(-v)$, producing the dipole pattern that is essentially identical to the geometric kernels proposed in this work.],
  placement: top,
) <fig:aniso-gaussian>

#figure(
  image("../figures/fig_sec02_discrete_orientation_sweep.pdf", width: 70%),
  caption: [Discrete orientation sweep as used by the WVF and the proposed ASF. Eight candidate orientations are evaluated; the response magnitude at each angle is indicated by the bar height. The orientation with maximum response (highlighted) is selected as the edge direction. This is a brute-force approximation of the analytic steering shown in @fig:steerable.],
) <fig:discrete-sweep>


== Classical Edge Detection <sec:related:classical>

The earliest edge detection operators were small, hand-designed convolution masks. Roberts @roberts1963machine introduced $2 times 2$ cross-difference operators for detecting diagonal edges. Sobel @sobel2014history and Prewitt @prewitt1970gradient independently proposed $3 times 3$ masks that approximate first-order directional derivatives with a degree of smoothing orthogonal to the gradient direction. Kirsch @kirsch1971computer designed a set of eight $3 times 3$ templates, one for each compass direction, and defined the edge response as the maximum over all eight. The Frei--Chen basis @freichen1977fast decomposed the $3 times 3$ neighborhood into nine orthogonal subspaces, separating edge, line, and average components, and detected edges by projection onto the edge subspace. All of these operators share the property of fixed, small spatial support, which limits their noise robustness and their ability to adapt to edges at different scales.

#figure(
  image("../figures/fig_sec02_roberts_cross_kernel.pdf", width: 55%),
  caption: [Roberts Cross operator (1963). Two $2 times 2$ kernels compute diagonal intensity differences. Positive weights in garnet, negative in blue.],
) <fig:roberts>

#figure(
  image("../figures/fig_sec02_sobel_kernel.pdf", width: 80%),
  caption: [Sobel operator (1968). Two $3 times 3$ kernels $G_x$ and $G_y$ estimate horizontal and vertical gradients with center-weighted smoothing. The gradient magnitude and direction are obtained by combining both responses.],
  placement: top,
) <fig:sobel>

#figure(
  image("../figures/fig_sec02_prewitt_kernel.pdf", width: 80%),
  caption: [Prewitt operator (1970). Similar to Sobel but with uniform row weighting rather than center-weighted smoothing.],
  placement: top,
) <fig:prewitt>

#figure(
  image("../figures/fig_sec02_kirsch_compass_kernel.pdf", width: 80%),
  caption: [Kirsch compass operator (1971). Four of the eight $3 times 3$ compass masks are shown at 0°, 45°, 90°, and 135°. The edge response is the maximum over all eight orientations, an early instance of the orientation sweep that the WVF later employs with polynomial fitting.],
  placement: top,
) <fig:kirsch>

#figure(
  image("../figures/fig_sec02_freichen_basis_masks.pdf", width: 75%),
  caption: [Frei--Chen operator (1977). The $3 times 3$ neighborhood is decomposed into orthogonal subspaces. Representative edge-subspace and line-subspace basis masks are shown. Edges are detected by projecting the image patch onto the edge subspace.],
) <fig:freichen>

The Gaussian derivative framework placed edge detection on a firmer mathematical foundation. Marr and Hildreth @marr1980log proposed detecting zero crossings of the Laplacian of Gaussian ($nabla^2 G$), providing a principled coupling of smoothing scale and differentiation. Canny @canny1986edge formulated edge detection as an optimization problem, seeking the operator that maximizes detection probability and localization accuracy while minimizing multiple responses. His solution for step edges in white Gaussian noise is closely approximated by the first derivative of a Gaussian, and his framework introduced non-maximum suppression and hysteresis thresholding as post-processing stages that remain standard practice. Lindeberg @lindeberg1998edge extended these ideas to _automatic scale selection_, using normalized derivatives across scale space to identify the characteristic scale of each edge. A persistent limitation of all fixed-kernel methods, whether $3 times 3$ masks or Gaussian derivatives at a single scale, is that their spatial support does not adapt to the local noise level or to the spatial extent of the edge structure. The WVF and LF address this limitation by enlarging the support region, but as discussed in @sec:related:lp, the mechanism they use, polynomial fitting over large windows, is itself classical.

#figure(
  image("../figures/fig_sec02_gaussian_derivative_log.pdf", width: 85%),
  caption: [Gaussian derivatives. Left: the Gaussian function $G(x)$, its first derivative $G'(x)$ (odd-symmetric dipole), and its second derivative $G''(x)$ (Mexican hat). Right: the corresponding 2D kernels. The first-derivative-of-Gaussian kernel is the mathematical ancestor of the geometric kernels proposed in this work.],
  placement: top,
) <fig:gaussian-deriv>

#figure(
  image("../figures/fig_sec02_canny_pipeline_stages.pdf", width: 90%),
  caption: [The Canny edge detection pipeline (1986). Five stages: Gaussian smoothing, gradient computation, angle estimation via arctan, non-maximum suppression, and hysteresis thresholding. The pre-smoothing step trades edge localization for noise robustness; stronger smoothing blurs weak edges.],
  placement: top,
) <fig:canny>


== Deep Learning Edge Detection <sec:related:dl>

The introduction of deep convolutional networks to edge detection marked a paradigm shift in performance on standard benchmarks. Holistically-Nested Edge Detection (HED) @xie2015hed pioneered the use of multi-scale side outputs from a VGG-style backbone, fusing predictions from multiple network depths to produce edge maps that capture both fine details and large-scale contours. Structured Forests @dollar2013structured, while not a deep method, introduced the random-forest-based approach to structured prediction for edges that motivated much subsequent work. DexiNed @poma2020dexined extended the multi-scale fusion paradigm with a dense extreme inception architecture, demonstrating strong generalization to datasets unseen during training. The Tiny and Efficient Edge Detector (TEED) @soria2023teed achieved competitive accuracy with a dramatically reduced parameter count, demonstrating that architectural efficiency and edge detection quality need not be in tension. PIDINet @su2021pidinet incorporated traditional pixel-difference operations as inductive biases within a lightweight network, explicitly bridging classical gradient computation and learned feature extraction. More recently, NBED @chen2024nbed proposed a neural-network-based architecture with attention mechanisms for boundary detection, and DiffusionEdge @ye2024diffusionedge applied diffusion probabilistic models to generate crisp edge maps through iterative denoising.

#figure(
  image("../figures/fig_sec02_hed_architecture.pdf", width: 80%),
  caption: [Holistically-Nested Edge Detection (Xie and Tu, 2015). A VGG-style encoder produces multi-scale side outputs $S_1$ through $S_5$ that are fused into a final edge map. Each side output captures edges at a different spatial scale.],
  placement: top,
) <fig:hed>

#figure(
  image("../figures/fig_sec02_pidinet_pixel_diff.pdf", width: 70%),
  caption: [PIDINet pixel difference convolutions (Su et al., 2021). Instead of multiplying pixel values by learned weights, PIDINet computes pixel _differences_ $Delta_i$ between the center and each neighbor, then applies learned weights $w_i$ to the differences. This is structurally similar to a classical gradient operator but with trainable coefficients.],
) <fig:pidinet>

These methods achieve state-of-the-art performance on benchmarks such as BSDS500 @arbelaez2011bsds500, but they share several limitations that motivate continued interest in classical approaches. First, they depend on labeled training data, and their performance degrades under _domain shift_ when the test distribution differs from the training distribution. Underwater imagery, medical imaging, and satellite data all present domains where labeled edge data is scarce. Second, the learned filters are opaque. One cannot, in general, characterize the spatial frequency response, noise sensitivity, or derivative accuracy of a trained network's edge output in the way that one can for a Savitzky--Golay filter or a Gaussian derivative. Third, deep methods provide no formal guarantees on gradient accuracy. The WVF and LF, by virtue of their polynomial fitting formulation, admit closed-form bias and variance expressions that can guide parameter selection, an advantage that no data-driven method currently shares.


== GPU Acceleration for Image Filtering <sec:related:gpu>

The computational cost of applying large-support filters at multiple orientations has historically been a barrier to deploying methods like the WVF and LF in real-time applications. Modern GPU computing frameworks have substantially lowered this barrier. The cuDNN library @chetlur2014cudnn provides highly optimized implementations of standard convolution operations, including batched and grouped convolutions that can be leveraged for multi-orientation filtering. For operations that do not map cleanly onto standard convolution, the Triton compiler @tillet2019triton enables rapid development of custom GPU kernels in a high-level Python-embedded language, with automatic tiling and memory management. CUTLASS @cutlass2023 provides composable CUDA templates for general matrix operations that can express the gather-and-dot-product structure underlying all three filter variants considered in this work.

#figure(
  image("../figures/fig_sec02_gather_dot_product.pdf", width: 90%),
  caption: [The gather-dot-product abstraction. Both regular convolution (left, as in cuDNN) and irregular stencil operations (right, as in the ASF) reduce to the same primitive: gather pixel intensities at precomputed offsets, multiply by precomputed weights, and sum. This uniformity enables a single GPU kernel to serve all three proposed filter variants.],
  placement: top,
) <fig:gather-dot>

The key observation for GPU implementation is that the WVF, LF, and the anisotropic filter variant proposed in this paper all reduce to the same computational primitive. For each pixel and each candidate orientation, a set of pixel values is gathered from the support region and multiplied by a precomputed weight vector. This gather-dot-product operation is embarrassingly parallel across pixels and orientations, making it well-suited to GPU execution. The orientation sweep, which the WVF performs sequentially, can be issued as a batched operation over the $K$ orientation channels, and the subsequent maximum selection is a trivial reduction. The practical consequence is that even the WVF's brute-force orientation search becomes tractable when implemented as a single fused GPU kernel, removing much of the computational motivation for the analytic interpolation provided by steerable filters.
