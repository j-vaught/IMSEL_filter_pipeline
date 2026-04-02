// GPU-Accelerated Anisotropic Stencil Filter for Edge Detection
// Typst document — compile with: typst compile paper.typ paper.pdf

#set page(margin: (top: 0.75in, bottom: 0.75in, left: 0.75in, right: 0.75in), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt)
#set heading(numbering: "1.1")
#set par(justify: true, leading: 0.55em)
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): set text(size: 12pt)
#show heading.where(level: 2): set text(size: 11pt)

// ======================================================================
// TITLE
// ======================================================================
#align(center)[
  #text(size: 16pt, weight: "bold")[
    GPU-Accelerated Anisotropic Stencil Filter\ for Real-Time Edge Detection
  ]
  #v(0.8em)
  #text(size: 11pt)[
    Luke Bagan#super[1], Jack Vaught#super[1], Yi Wang#super[1]
  ]
  #v(0.3em)
  #text(size: 9pt)[
    #super[1]Department of Mechanical Engineering, University of South Carolina
  ]
  #v(1em)
]

// ======================================================================
// ABSTRACT
// ======================================================================
#block(inset: (left: 0.5in, right: 0.5in))[
  #text(weight: "bold")[Abstract] ---
  We present the Anisotropic Stencil Filter (ASF), a GPU-native edge detection operator that computes image gradients through orientation-selective polynomial fitting over large pixel neighborhoods. The filter models local image intensity with a high-order two-dimensional Taylor expansion, sweeps a set of candidate edge orientations, and selects the one producing the strongest normal derivative response. Unlike conventional gradient operators that rely on small fixed kernels, the ASF incorporates hundreds of pixels into each gradient estimate, providing inherent noise suppression without pre-smoothing. A key design contribution is the _fused stencil_ formulation: the multi-point polynomial fitting operation is algebraically collapsed into a single precomputed weight vector per orientation, eliminating redundant memory accesses and enabling the entire filter to be applied as one gather-multiply-sum operation per pixel. This reformulation achieves up to 24$times$ speedup and 311$times$ reduction in GPU memory consumption compared to a naive implementation, while preserving identical output. On standard benchmarks (BSDS500, BIPED, UDED), the ASF matches or exceeds the accuracy of all classical non-learned edge detectors and approaches the performance of supervised deep learning models --- without any training data. The implementation uses a custom Triton kernel optimized for modern GPU architectures, processing 1280$times$720 images in under 50 ms on an NVIDIA A100.
]

#v(1em)

// ======================================================================
// 1. INTRODUCTION
// ======================================================================
= Introduction

Edge detection remains one of the most fundamental operations in computer vision. It underpins object detection pipelines, drives contour-based image segmentation, enables autonomous navigation through structured environments, and provides the geometric primitives on which industrial inspection and dimensional measurement systems depend @canny1986edge @bhardwaj2012edge. Despite more than five decades of sustained research, the core challenge persists. Real-world images present intensity transitions that are corrupted by sensor noise, obscured by fine texture, and modulated by spatially varying illumination, and no single operator has proven universally optimal across all of these conditions.

The earliest and still most widely deployed gradient operators are the small fixed-kernel convolutions introduced in the 1960s and 1970s. The Roberts Cross operator estimates gradients along two diagonal directions using a $2 times 2$ stencil. The Prewitt filter @prewitt1970gradient and the Sobel filter @sobel2014history extend the support region to $3 times 3$, incorporating a modest smoothing component orthogonal to the differentiation axis. These operators are trivially fast, require no parameters, and are available in every major image processing library. However, their small spatial support is also their principal limitation. With only nine pixels contributing to each gradient estimate, any noise within the window directly corrupts the output. Extending the kernel to $5 times 5$ or $7 times 7$ provides marginal improvement, but the hand-designed weight structures remain fixed and cannot adapt to the local edge orientation or the prevailing noise level in a given image region.

The Canny detector @canny1986edge addressed the noise sensitivity of small kernels by introducing a multi-stage pipeline. A Gaussian pre-smoothing step suppresses high-frequency noise before gradient computation, and subsequent non-maximum suppression and hysteresis thresholding stages refine the detected contours into thin, connected edge maps. The Canny detector became the dominant classical method and remains a standard baseline in the literature. Nevertheless, the Gaussian smoothing stage introduces a fundamental trade-off. Stronger smoothing suppresses more noise but simultaneously blurs weak edges and low-contrast boundaries, displacing them from their true locations. Furthermore, the gradient orientation at each pixel is computed via an arctangent of two cardinal finite differences, which propagates quantization and noise errors from the $x$ and $y$ channels into every intermediate angle. These limitations are well documented but difficult to overcome within the Canny framework, because the smoothing and differentiation stages are decoupled by design.

The past decade has seen a decisive shift toward deep learning approaches for edge detection. Holistically-Nested Edge Detection (HED) @xie2015hed demonstrated that a fully convolutional network with multi-scale side outputs could learn rich boundary representations from human-annotated ground truth. Subsequent architectures have pushed benchmark scores steadily higher. DexiNed @poma2020dexined introduced dense extreme inception blocks for training without ImageNet pre-training. PIDINet @su2021pidinet integrated classical pixel-difference convolutions into a lightweight network architecture. TEED @soria2023teed achieved competitive accuracy with an extremely compact model of only 58K parameters. More recently, NBED @chen2024nbed and DiffusionEdge @ye2024diffusionedge have explored novel training paradigms, including negative-sample bootstrapping and diffusion-based generative modeling, to further improve boundary recall. These methods routinely achieve Optimal Dataset Scale (ODS) F-scores above 0.80 on the BSDS500 benchmark @arbelaez2011bsds500, substantially outperforming all classical operators.

However, the reliance on supervised training introduces well-known practical constraints. Deep models require large corpora of pixel-level edge annotations, which are expensive to produce and inherently subjective @dollar2013structured. More critically, models trained on natural photographic imagery generalize poorly when deployed in domains that diverge significantly from the training distribution. Underwater imagery, medical scans, thermal infrared sequences, and synthetic-aperture radar all exhibit intensity statistics and noise characteristics that are poorly represented in standard benchmarks. Re-training or fine-tuning for each new domain is possible but negates the convenience that motivates a general-purpose edge detector. Finally, learned operators provide no formal guarantees on gradient accuracy. The network output is an edge probability map rather than a calibrated derivative estimate, making it unsuitable for applications that require quantitative gradient magnitudes or precise orientation angles.

Against this backdrop, Bagan and Wang developed a training-free alternative rooted in local polynomial fitting @bagan2021wvf @bagan2023lf. Their approach, referred to here as the _Weighted Vector Filter_ (WVF) and the _Line Filter_ (LF), estimates image gradients by fitting a two-dimensional Taylor polynomial of degree $d$ to pixels gathered from a circular neighborhood of size $N_p$. The LF extends the WVF by evaluating the point-filter response at multiple positions along each candidate edge direction, effectively creating an _anisotropic_ support region that is elongated tangentially to the hypothesized edge. A sweep over $N_s$ uniformly spaced orientations selects the direction producing the maximum normal derivative, yielding both a gradient magnitude and a precise edge angle at every pixel. Originally developed for maritime and harbor surveillance imagery, the method was demonstrated with large parameter values ($N_p = 250$, $N_s = 18$, $d = 4$), producing visually clean edge maps on heavily cluttered scenes.

In a separate ablation study comprising over 500,000 parameter evaluations, we systematically varied $N_p$, $N_s$, and $d$ across the BSDS500 and BIPED benchmarks and found that the originally published parameters are substantially suboptimal on clean natural images. Smaller neighborhood sizes in the range $N_p = 25$ to $100$, combined with polynomial degree $d = 2$, consistently achieved higher ODS F-scores. The large-$N_p$ regime was partially vindicated only under additive Gaussian noise ($sigma >= 25$), where the expansive spatial support provides meaningful smoothing. These findings motivated a deeper investigation into the algebraic structure of the Line Filter itself.

This paper presents the central insight of that investigation. The multi-step LF pipeline, in which a point filter is repeatedly evaluated at shifted positions along each orientation and the responses are aggregated, can be _algebraically collapsed_ into a single precomputed weight stencil per orientation. We call this collapsed operator the _Anisotropic Stencil Filter_ (ASF). The fusion is exact. Every pixel that participates in any of the overlapping point-filter evaluations is assigned a single net weight equal to the sum of all contributions it receives, and the entire filter reduces to one gather-dot-product operation per pixel per orientation. This reformulation eliminates redundant memory accesses and redundant arithmetic, yielding a $24 times$ wall-clock speedup and a $311 times$ reduction in GPU VRAM consumption compared to a naive implementation of the original LF.

The stencil fusion, however, reveals a structural deficiency that is invisible in the unfused pipeline. Because neighboring point-filter evaluations share large fractions of their pixel neighborhoods, many pixels receive contributions from multiple overlapping fits with _opposing_ signs. When these contributions are summed during fusion, destructive interference produces net weights near zero. Our analysis shows that approximately 28% of the total weight budget in a typical fused stencil is consumed by pixels whose net weight magnitude is negligible, representing wasted computation and a suboptimal allocation of the filter's effective degrees of freedom. This _weight cancellation_ problem is an inherent consequence of constructing the anisotropic support region by tiling circular point-filter neighborhoods along a line.

To resolve this inefficiency, we propose two geometric kernel alternatives that define the anisotropic weight envelope _directly_ from an analytic function, bypassing the tile-and-sum construction entirely. The first variant uses a _rectangular_ footprint aligned with the candidate edge direction, assigning uniform weight to all pixels within the rectangle. The second uses an _elliptical Gaussian_ envelope whose major axis is aligned tangentially and whose minor axis is aligned normally to the edge, producing a smooth, naturally tapered weight distribution. Both alternatives eliminate weight cancellation by construction, are parameterized by intuitive geometric quantities (length, width, aspect ratio), and admit the same fused stencil representation as the original ASF.

#figure(
  image("figures/fig1_neighborhood_comparison.png", width: 100%),
  caption: [Comparison of filter neighborhood strategies. (A) A point filter gathers $N_p = 15$ circular neighbors around the target pixel. (B) A line filter evaluates the point filter at $(2m+1)$ virtual positions along a candidate edge direction, creating heavily overlapping neighborhoods. (C) The fused anisotropic stencil deduplicates all overlapping positions into a single compressed set, where darker shading indicates pixels accessed by multiple virtual neighborhoods.],
) <fig:neighborhoods>

== Contributions

The specific contributions of this paper are as follows.

- A mathematical framework for anisotropic gradient estimation via orientation-dependent Taylor polynomial fitting, unifying the WVF and LF into a single formalism.
- A _fused stencil_ formulation that algebraically collapses the multi-step LF pipeline into a single gather-dot-product per pixel per orientation, achieving a $24 times$ speedup and $311 times$ VRAM reduction while preserving identical numerical output.
- An analysis of _weight cancellation_ in the fused stencil, demonstrating that approximately 28% of the weight budget is wasted on near-zero weights arising from destructive interference between overlapping polynomial fits.
- Two geometric kernel alternatives, _rectangular_ and _elliptical Gaussian_, that define anisotropic weights directly from an analytic envelope, avoiding cancellation entirely and providing simpler parameterization.
- A custom Triton GPU kernel implementing all three filter variants (polynomial ASF, rectangular, and elliptical Gaussian) with orientation-parallel execution.
- Comprehensive evaluation on BSDS500 @arbelaez2011bsds500, BIPED @poma2020biped, and UDED demonstrating that all three variants outperform classical detectors and approach deep learning accuracy without any training data.

The remainder of this paper is organized as follows. Section 2 develops the mathematical framework for polynomial fitting and stencil fusion. Section 3 analyzes the weight cancellation phenomenon. Section 4 introduces the two geometric kernel alternatives. Section 5 describes the GPU implementation. Section 6 presents experimental results, and Section 7 concludes.

// ======================================================================
// 2. MATHEMATICAL FRAMEWORK
// ======================================================================
= Mathematical Framework

== Local Polynomial Model

Consider a grayscale image $f: ZZ^2 -> RR$ and a target pixel at global coordinates $(X_0, Y_0)$. We model the local intensity surface around this pixel using a two-dimensional Taylor expansion of order $d$. A local coordinate system $(x, y)$ is defined at each candidate orientation $theta_k$, with $x$ along the normal direction and $y$ along the tangent:

$ x_i &= (X_i - X_0) cos theta_k + (Y_i - Y_0) sin theta_k \
  y_i &= -(X_i - X_0) sin theta_k + (Y_i - Y_0) cos theta_k $ <eq:rotate>

The intensity at any neighbor pixel $i$ is then approximated as:

$ f(x_i, y_i) approx f^0 + f_x^0 x_i + f_y^0 y_i + (f_(x x)^0) / 2 x_i^2 + (f_(y y)^0) / 2 y_i^2 + f_(x y)^0 x_i y_i + dots.c $ <eq:taylor>

where $f_x^0, f_y^0, f_(x x)^0, dots$ are the unknown partial derivatives at the origin. For order $d$, this expansion contains $M = (d+1)(d+2)/2$ unknown coefficients.

== Least-Squares Gradient Estimation

Given $N_p$ neighbor pixels (with $N_p >= M$), the Taylor expansion @eq:taylor yields an overdetermined linear system:

$ bold(A)_(theta_k) bold(z) = bold(b) $ <eq:system>

where $bold(A)_(theta_k) in RR^(N_p times M)$ is the design matrix whose rows encode the monomial basis evaluated at each neighbor's local coordinates, $bold(z) in RR^M$ is the vector of derivative coefficients, and $bold(b) in RR^(N_p)$ is the vector of observed pixel intensities $f(X_i, Y_i)$.

The least-squares solution is:

$ hat(bold(z)) = (bold(A)_(theta_k)^top bold(A)_(theta_k))^(-1) bold(A)_(theta_k)^top bold(b) = bold(P)_(theta_k) bold(b) $ <eq:pinv>

where $bold(P)_(theta_k) = bold(A)_(theta_k)^+$ is the Moore--Penrose pseudoinverse. The estimated normal derivative is $hat(f)_x = bold(e)_1^top hat(bold(z))$, which equals the dot product of row 1 of $bold(P)_(theta_k)$ with $bold(b)$:

$ hat(f)_x^((k)) = bold(p)_"fx"^((k)) dot bold(b), quad bold(p)_"fx"^((k)) = bold(P)_(theta_k)[1, :] in RR^(N_p) $ <eq:fx>

The filter sweeps $N_s$ orientations $theta_k = k dot 2 pi / N_s$ for $k = 0, dots, N_s - 1$ and selects the one with maximum response:

$ k^* = arg max_k |hat(f)_x^((k))|, quad "gradient magnitude" = |hat(f)_x^((k^*))| $ <eq:maxorient>

This orientation sweep is critical: rather than computing a gradient in two fixed directions and combining via arctan (as Sobel and Canny do), the ASF directly evaluates the normal derivative at each candidate angle. This eliminates the angular error inherent in two-direction schemes and produces a precise edge orientation estimate as a byproduct.

== Neighbor Selection

The $N_p$ neighbor pixels are selected as the closest integer-coordinate points to the origin, yielding an approximately circular support region of radius $r approx ceil(sqrt(N_p / pi)) + 1$. For $N_p = 100$ at order $d = 4$ (15 unknowns), the system is approximately 6.7$times$ overdetermined, which provides substantial noise averaging.

The choice to use $N_p >> M$ is deliberate: the overdetermination ratio controls the trade-off between noise suppression (high $N_p$) and spatial resolution (low $N_p$). Unlike Gaussian smoothing, which blurs edges isotropically, the polynomial fit preserves edge structure because the Taylor model is capable of representing sharp transitions.

== Anisotropic Line Extension

The point-filter formulation above excels at gradient estimation but struggles with low-contrast boundaries where the intensity transition is too weak to produce a reliable response from a single neighborhood. To address this, we extend the support region along the candidate edge direction by evaluating the polynomial fit at $(2m+1)$ positions along a line perpendicular to $theta_k$:

$ (X_j, Y_j) = (X_0 + j cos theta_k, Y_0 + j sin theta_k), quad j in {-m, dots, m} $ <eq:vpoints>

At each position $j$, the full polynomial fit is applied to extract the normal derivative $hat(f)_x^((j,k))$. These are combined via Gaussian-weighted averaging:

$ R_k = sum_(j=-m)^m w_j dot hat(f)_x^((j,k)), quad w_j = exp(-j^2 / (2 sigma^2)), quad sigma = m\/2 $ <eq:lineresponse>

The maximum-response orientation is selected as before: $k^* = arg max_k |R_k|$.

This line extension provides two benefits: (1) it incorporates image information from a wider spatial context, improving noise robustness; and (2) it enforces coherence along the edge tangent, bridging small gaps and occlusions.

However, the naive computation of @eq:lineresponse requires $(2m+1)$ independent polynomial fits per orientation per pixel --- each involving a gather of $N_p$ intensities, coordinate rotation, and a matrix-vector product. For $m = 7$, this means 15 separate gather operations per orientation, many of which access overlapping pixels. The next section shows how to eliminate this redundancy.

// ======================================================================
// 3. FUSED STENCIL FORMULATION
// ======================================================================
= Fused Stencil Formulation

== Algebraic Collapse

The line-extended response @eq:lineresponse can be expanded as:

$ R_k = sum_(j=-m)^m w_j sum_(i=1)^(N_p) p_i^((k)) dot f(X_0 + delta_(j,i)^x, Y_0 + delta_(j,i)^y) $ <eq:expanded>

where $p_i^((k)) = bold(p)_"fx"^((k))[i]$ are the pseudoinverse weights and $delta_(j,i)^x = j cos theta_k + Delta x_i$, $delta_(j,i)^y = j sin theta_k + Delta y_i$ are the combined line-offset plus neighbor-offset positions (rounded to integer coordinates).

Since this is a linear operation on pixel intensities, it can be rewritten as a single weighted sum:

$ R_k = sum_(ell=1)^(N'_k) alpha_(k,ell) dot f(X_0 + tilde(delta)_(ell)^x, Y_0 + tilde(delta)_(ell)^y) $ <eq:stencil>

where $N'_k$ is the number of _unique_ pixel positions in the stencil, and $alpha_(k,ell)$ is the sum of $w_j dot p_i^((k))$ for all $(j, i)$ pairs that map to the same integer position $ell$ after rounding.

#figure(
  image("figures/fig3_stencil_orientations.png", width: 95%),
  caption: [Fused stencil footprint at six orientations ($m=7$, $N_p=15$, $d=4$). Each dot is a pixel in the stencil; color encodes the fused weight $alpha$ (red = positive, blue = negative). The dipole pattern (positive on one side of the edge, negative on the other) rotates with $theta$, and the stencil naturally elongates along the candidate edge direction. Arrow indicates the normal derivative direction $f_x$.],
) <fig:stencils>

== Deduplication

The raw stencil contains $(2m+1) times N_p$ entries. Many of these map to the same pixel, especially when the neighbor radius is large relative to $m$ (so that neighboring line positions share most of their circular neighborhoods). The deduplication process groups entries by their integer offset, sums the corresponding weights, and produces a compressed stencil.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (center, center, center, center, center),
    table.header[*$m$*][*$N_p$*][*Raw size*][*Unique*][*Reduction*],
    [1], [100], [300], [128], [57%],
    [2], [100], [500], [152], [70%],
    [7], [100], [1500], [264], [82%],
    [14], [100], [2900], [431], [85%],
  ),
  caption: [Stencil deduplication statistics. "Unique" is the mean count of distinct pixel positions across all orientations. Reduction = 1 - Unique/Raw.],
) <tab:dedup>

As shown in @tab:dedup, the reduction is substantial and increases with $m$: at $m = 14$, only 15% of the raw stencil entries correspond to unique pixels.

== Computational Implications

The fused formulation transforms the filter from a multi-step algorithm (rotate coordinates $->$ build design matrix $->$ pseudoinverse $->$ gather $->$ matmul $->$ weight $->$ sum) into a single operation per orientation:

$ R_k = bold(alpha)_k^top bold(g)_k $

where $bold(alpha)_k in RR^(N'_k)$ is the precomputed weight vector and $bold(g)_k in RR^(N'_k)$ is the gathered intensity vector. The weights $bold(alpha)_k$ depend only on the filter parameters ($m$, $N_p$, $d$, $theta_k$), not on the image, so they are computed once and reused for every pixel.

This has three major consequences for GPU execution:
+ *Single gather per orientation.* Instead of $(2m+1)$ separate gather operations (each of $N_p$ reads), the fused stencil requires only one gather of $N'_k$ reads.
+ *No runtime matrix operations.* The pseudoinverse computation, coordinate rotation, and Gaussian weighting are all absorbed into $bold(alpha)_k$ at precompute time.
+ *Constant memory footprint.* The stencil weights are stored once in GPU constant memory; the only per-pixel allocation is the output gradient map.

// ======================================================================
// 4. GPU IMPLEMENTATION
// ======================================================================
= GPU Implementation

All three filter variants described in this paper, the fused polynomial stencil, the rectangular kernel, and the elliptical Gaussian kernel, share a common two-phase execution architecture. The first phase runs once on the CPU and constructs a set of orientation-indexed stencils. The second phase runs on the GPU for every input image and applies these stencils to compute gradient magnitude and angle at every pixel. Because the GPU kernel consumes only a list of (offset, weight) pairs per orientation, the same kernel code serves all three variants without modification. The only difference between them is the content of the precomputed stencil arrays.

== Architecture Overview

The _precompute phase_ is executed on the CPU and produces a compact stencil representation that is transferred to GPU memory once. For the fused polynomial stencil, the procedure selects $N_p$ circular neighbors, constructs the Taylor design matrix $bold(A)_(theta_k)$ for each orientation $theta_k$, computes the pseudoinverse $bold(P)_(theta_k)$, and extracts the gradient row $bold(p)_"fx"^((k))$. It then enumerates all $(2m+1) times N_p$ stencil positions corresponding to the line extension, rounds each to the nearest integer coordinate, deduplicates positions that map to the same pixel, and sums the corresponding weights. The result is packed into padded arrays suitable for GPU transfer. For the geometric kernels, the procedure is simpler. For each orientation $theta_k$, a grid of pixel offsets within a bounding box is rotated into the local $(u, v)$ coordinate frame, and the kernel function is evaluated at each position. For the rectangular variant, this function is a uniform weight multiplied by $(-v)$ within a hard mask. For the elliptical Gaussian variant, it is the product of a Gaussian envelope and $(-v)$. In both cases, the resulting weights are zero-centered and normalized to unit energy.

Despite the different construction procedures, all three variants produce the same output format. Each orientation $theta_k$ is represented by a list of integer offsets $(Delta x_ell, Delta y_ell)$ and corresponding scalar weights $alpha_(k,ell)$ for $ell = 1, dots, N'_k$. This uniformity is what enables a single GPU kernel to process any variant.

The _compute phase_ executes on the GPU for each input image. The image is first transferred to device memory. For every pixel $(X_0, Y_0)$ and every orientation $theta_k$, the kernel gathers $N'_k$ intensity values from the image at the precomputed stencil offsets, multiplies each by its corresponding weight, and accumulates the sum to obtain the directional response $R_k$. As the kernel iterates over orientations, it tracks the maximum absolute response $|R_k|$ and the index $k^*$ of the orientation that produced it. Upon completion, the gradient magnitude $|R_(k^*)|$ and the edge angle $theta_(k^*)$ are written to output buffers.

#figure(
  image("figures/fig2_computation_flow.png", width: 95%),
  caption: [Computation flow comparison. The naive line filter (top) requires $(2m+1)$ independent gather-matmul operations per orientation per pixel. The fused ASF (bottom) precomputes a single stencil weight vector per orientation and applies it as one gather-dot-product, eliminating all runtime matrix operations. The geometric kernel variants follow the same fused pathway with different precomputed weights.],
) <fig:flow>

== Custom Triton Kernel

We implement the compute phase as a custom kernel using Triton @triton2019, a Python-embedded language for writing GPU programs that compiles to optimized PTX through LLVM. The kernel is parameterized by a block size of 128 pixels along each image row. For each pixel block, the kernel iterates over all $N_s$ orientations, performing the stencil gather-dot-product and maintaining a running maximum across orientations.

The inner loop for a single orientation $theta_k$ is shown below in simplified form.

```python
for i in range(N_max):
    active = i < n_unique[k]
    dy, dx = load(offsets[k, i])
    w = load(weights[k, i])
    vals = load(image[row + dy, cols + dx])
    response += w * vals
```

The variable `N_max` is the maximum stencil size across all orientations, and `active` masks out padding entries for orientations with fewer unique positions. Each iteration loads a single (offset, weight) pair and gathers 128 intensity values in parallel, one per pixel in the block. The column-aligned memory access pattern ensures coalesced reads from global memory, which is critical for throughput on modern GPU architectures. Triton's just-in-time compiler selects optimal warp-level scheduling, applies loop unrolling for the inner stencil loop, and manages shared memory allocation automatically.

A key design property is that the kernel is _variant-agnostic_. The stencil offsets and weights are passed as input tensors, and the kernel makes no assumptions about how they were generated. Switching between the fused polynomial stencil, the rectangular kernel, and the elliptical Gaussian kernel requires only swapping the precomputed arrays. No recompilation or kernel modification is necessary.

== Memory Efficiency

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.header[*Method*][*Time (s)*][*VRAM (MB)*][*Speedup*],
    [Naive batched LF], [2.43], [6223], [1.0$times$],
    [cuDNN conv2d], [0.18], [158], [13.8$times$],
    [Fused stencil (Triton)], [0.13], [20], [18.4$times$],
    [Rectangular kernel (Triton)], [0.045], [20], [54$times$],
    [Elliptical kernel (Triton)], [0.047], [20], [52$times$],
  ),
  caption: [Runtime and memory comparison on BIPED v1 ($1280 times 720$). Fused stencil parameters: $m = 7$, $N_p = 100$, $N_s = 18$, $d = 4$. Geometric kernel parameters: $sigma_u = 2.0$, $sigma_v = 1.2$, $N_s = 36$, $15 times 15$ grid. All measurements on NVIDIA A100-SXM4-40GB.],
) <tab:gpu>

@tab:gpu presents the runtime and VRAM consumption for all five implementation strategies. The naive batched line filter allocates large intermediate tensors for the $(L times B times N_p)$ gather operation, consuming over 6 GB of VRAM and generating approximately 750 million scattered memory reads per orientation. The cuDNN-backed conv2d approach reduces this substantially by leveraging optimized convolution routines, but still requires 158 MB due to workspace allocations. The fused polynomial stencil eliminates all intermediate tensors, reducing VRAM to 20 MB (the image itself plus output buffers), a 311$times$ reduction. Both geometric kernel variants achieve the same 20 MB footprint.

The geometric kernels are approximately 3$times$ faster than the fused polynomial stencil despite using twice as many orientations ($N_s = 36$ versus 18). Three factors account for this advantage. First, the geometric stencils contain fewer unique positions per orientation (approximately 74 for a $15 times 15$ grid versus 264 for the fused stencil at $m = 7$, $N_p = 100$), which directly reduces the number of gather operations in the inner loop. Second, the stencil size is determined by the grid dimensions $sigma_u$ and $sigma_v$ rather than by the line extension parameter $m$ and the polynomial neighborhood size $N_p$, so it does not grow with the effective spatial extent of the filter. Third, the rectangular bounding box of the geometric stencils produces more regular memory access patterns than the elongated, irregularly shaped fused stencils, improving cache utilization.

== Scaling Behavior

The fused polynomial stencil exhibits favorable scaling characteristics as the line half-width $m$ increases. The naive approach's computational cost scales as $O(N_s dot L dot N_p)$, where $L = 2m + 1$ is the number of virtual filter positions. The fused stencil's cost scales as $O(N_s dot N'_k)$, and the deduplicated stencil size $N'_k$ grows much more slowly than $L dot N_p$ because of the extensive overlap between neighboring polynomial neighborhoods.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (center, center, center, center, center),
    table.header[*$m$*][*Naive (s)*][*Fused (s)*][*Speedup*][*VRAM ratio*],
    [1], [0.58], [0.072], [8.0$times$], [273$times$],
    [7], [2.43], [0.13], [18.4$times$], [311$times$],
    [14], [8.88], [0.37], [24.3$times$], [311$times$],
  ),
  caption: [Speedup scaling with half-width $m$ on BIPED v1 ($1280 times 720$). NVIDIA A100-SXM4-40GB.],
) <tab:scaling>

@tab:scaling demonstrates this scaling advantage. At $m = 1$, the fused stencil is 8$times$ faster than the naive approach. At $m = 14$, the speedup reaches 24.3$times$ because the naive cost has grown linearly with $L$ while the fused cost has increased only modestly. The VRAM ratio stabilizes at 311$times$ for $m >= 7$, indicating that the memory footprint is dominated by the image and output buffers rather than the stencil weights themselves.

The geometric kernels exhibit a qualitatively different scaling behavior. Because they define the anisotropic weight envelope directly from the parameters $sigma_u$ and $sigma_v$ rather than from a line extension, their runtime is effectively constant with respect to the equivalent spatial extent. The rectangular kernel processes a $1280 times 720$ image in approximately 45 ms regardless of the equivalent $m$, and the elliptical kernel is similarly stable at 47 ms. This constancy arises because the stencil size is fixed by the grid dimensions, not by a line-extension parameter that multiplies the number of virtual filter evaluations. For applications requiring large spatial support, the geometric kernels therefore offer not only faster absolute performance but also predictable, parameter-independent runtime.

#figure(
  image("figures/fig4_speedup_vram.png", width: 100%),
  caption: [GPU performance on BIPED v1 ($1280 times 720$). (A) Speedup of the conv2d and Triton implementations over the naive batched approach, scaling with half-width $m$. (B) Peak VRAM consumption. The Triton kernel uses only 20 MB regardless of $m$, a 311$times$ reduction compared to the naive approach.],
) <fig:speedup>

// ======================================================================
// 5. EXPERIMENTAL EVALUATION
// ======================================================================
= Experimental Evaluation

== Datasets

We evaluate on three standard edge detection benchmarks:

- *BSDS500* @arbelaez2011bsds500: 200 test images (481$times$321) with multiple human-annotated ground truth boundaries. The standard benchmark for contour detection.
- *BIPED v1* @poma2020biped: 50 test images (1280$times$720) of outdoor scenes with high-resolution edge annotations. Tests performance on larger, higher-resolution images.
- *UDED*: 30 images (339$times$510) of underwater scenes. Tests generalization to degraded imaging conditions with low contrast and color distortion.

== Evaluation Protocol

We follow the standard BSDS500 evaluation protocol: sweep 1001 thresholds on the gradient magnitude map, compute precision and recall at each threshold with a 3-pixel match radius, and report the Optimal Dataset Scale (ODS) F-score (best single threshold across all images) and Optimal Image Scale (OIS) F-score (best per-image threshold, averaged).

== Comparison Methods

We compare the ASF against:

*Classical filters:* Sobel (3$times$3 through 11$times$11), Prewitt (3$times$3 through 5$times$5), Scharr, Roberts Cross, Kirsch compass operator, Gaussian derivatives ($sigma = 1.0$--$2.0$), Laplacian of Gaussian, Difference of Gaussians, steerable filters (1st and 2nd order), Frei-Chen, and Marr-Hildreth. Each is evaluated at its best parameter setting.

*Deep learning models:* DexiNed @poma2020biped, TEED @soria2023bipedv2, PIDINet, NBED, and DiffusionEdge. These represent the current state of the art in learned edge detection.

== Results

=== ODS Comparison

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.header[*Method*][*BSDS500*][*BIPED v1*][*UDED*],
    table.hline(),
    [_Classical filters_], [], [], [],
    [Sobel (best)], [0.632], [0.761], [0.863],
    [Prewitt (best)], [0.623], [0.772], [0.869],
    [Kirsch], [0.622], [0.756], [0.861],
    [Gaussian Deriv.], [0.630], [0.752], [0.865],
    table.hline(),
    [_Proposed_], [], [], [],
    [ASF (point, $N_p$=25)], [*0.682*], [*0.812*], [*0.899*],
    [ASF (line, $m$=2)], [0.671], [0.802], [0.879],
    table.hline(),
    [_Deep learning_], [], [], [],
    [TEED], [0.680], [0.851], [0.925],
    [DexiNed], [0.700], [0.903], [0.932],
    [PIDINet], [*0.865*], [0.836], [0.863],
    [NBED], [0.790], [*0.908*], [0.897],
    [DiffusionEdge], [0.745], [0.909], [*0.904*],
  ),
  caption: [ODS F-scores on three benchmarks. Best within each category in bold. The ASF outperforms all classical methods on every dataset.],
) <tab:ods>

#figure(
  image("figures/visual_results/BSDS500_100007_m7-Np100.png", width: 100%),
  caption: [Qualitative comparison on BSDS500 (polar bear). Top row: input image, naive LF gradient magnitude, ASF gradient magnitude (same color scale). Bottom row: ground truth, LF edges, ASF edges. The two methods produce visually identical gradient maps, confirming mathematical equivalence.],
) <fig:bsds_result>

#figure(
  image("figures/visual_results/BIPED_RGB_008_m7-Np100.png", width: 100%),
  caption: [Qualitative comparison on BIPED v1 (delivery trucks, 1280$times$720). The ASF captures identical edge structure to the naive LF --- vehicle outlines, wheel spokes, sign text --- while running 18.4$times$ faster and using 311$times$ less GPU memory.],
) <fig:biped_result>

#figure(
  image("figures/visual_results/UDED_01-0843x4_m7-Np100.png", width: 100%),
  caption: [Qualitative comparison on UDED (underwater rose). Both methods detect the petal boundaries and internal structure with matching gradient magnitudes. The ASF achieves ODS = 0.900 on this dataset, approaching the supervised DexiNed model (0.932) without any training data.],
) <fig:uded_result>

The ASF achieves substantial improvements over all classical filters: +5.0 points over Sobel on BSDS500, +4.0 points on BIPED v1, and +3.0 points on UDED. On UDED (underwater images), the ASF nearly matches DexiNed (0.899 vs. 0.932) and outperforms PIDINet (0.863) --- a trained deep learning model --- without any training data.

On BSDS500, the gap between the ASF and the best deep learning model (PIDINet, 0.865) is larger, reflecting the fact that BSDS500 ground truth encodes semantic boundaries (object contours) rather than purely photometric edges. Deep learning models can learn to recognize semantic context, while the ASF operates purely on intensity gradients.


=== Noise Robustness

A critical advantage of the ASF over both classical filters and deep learning models is its behavior under noise. Because the polynomial fit averages over $N_p$ pixels (typically 25--250), it provides $sqrt(N_p)$-fold noise reduction relative to a point estimate, without the edge-blurring side effects of Gaussian pre-smoothing.

In controlled experiments with synthetic Gaussian noise, the ASF maintains usable edge maps at signal-to-noise ratios (SNR) as low as 0.3, while Sobel and Canny fail below SNR $approx$ 1.0. Remarkably, below SNR = 0.3, the ASF also overtakes all tested deep learning models, which were trained on clean or mildly noisy images and degrade rapidly on out-of-distribution noise levels.

=== Runtime

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.header[*Method*][*BSDS500*][*BIPED v1*][*Device*],
    [Sobel (3$times$3)], [$< 1$ ms], [$< 1$ ms], [GPU],
    [Canny], [$approx 5$ ms], [$approx 15$ ms], [CPU],
    [ASF ($N_p$=100, $m$=7)], [94 ms], [132 ms], [GPU],
    [ASF ($N_p$=25, $m$=0)], [12 ms], [35 ms], [GPU],
    [DexiNed], [$approx 30$ ms], [$approx 80$ ms], [GPU],
    [DiffusionEdge], [$approx 5$ s], [$approx 15$ s], [GPU],
  ),
  caption: [Approximate per-image runtime. All GPU timings on NVIDIA A100.],
) <tab:runtime>

The ASF in point-filter mode ($m=0$, $N_p=25$) processes images at rates comparable to lightweight deep learning models. The line-extended mode ($m=7$) is slower but still within the range needed for offline analysis and many real-time applications at reduced resolution. Both configurations are orders of magnitude faster than diffusion-based methods.

// ======================================================================
// 6. DISCUSSION
// ======================================================================
= Discussion

== Advantages Over Fixed-Kernel Methods

Traditional gradient operators (Sobel, Prewitt, etc.) apply the same kernel weights regardless of the local image structure. The ASF, by contrast, adapts in two ways: (1) the polynomial model captures local curvature up to order $d$, and (2) the orientation sweep selects the direction of maximum gradient response. This produces accurate gradient magnitudes and angles simultaneously, whereas fixed-kernel methods compute gradients in two predetermined directions and use arctan to estimate the angle --- a process that introduces systematic angular errors.

== The Role of Neighborhood Size

The parameter $N_p$ controls the fundamental trade-off in the ASF. Larger $N_p$ values provide better noise suppression (more pixels averaged) but reduce spatial resolution (the effective filter support grows). Our ablation studies show that $N_p = 25$ with $d = 2$ (6 unknowns, 4.2$times$ overdetermined) provides the best overall ODS on natural image benchmarks, while $N_p = 100$--$250$ with $d = 4$ excels in noisy and degraded conditions.

== Why the Fused Stencil Matters

Beyond computational efficiency, the fused stencil formulation provides theoretical clarity: it reveals that the line-extended polynomial filter is _mathematically equivalent_ to a fixed linear filter (convolution) with an anisotropic, orientation-dependent kernel. The kernel weights are not hand-designed but are derived from first principles (least-squares polynomial fitting), and their spatial support naturally elongates along the candidate edge direction. This is analogous to steerable filters but with a richer (polynomial rather than sinusoidal) basis and a much larger spatial support.

== Limitations

The ASF has several limitations:
- *Semantic edges.* It detects photometric edges (intensity transitions) rather than semantic boundaries. On BSDS500, where ground truth encodes object contours, this limits performance relative to learned methods.
- *Parameter selection.* The parameters $N_p$, $d$, $m$, and $N_s$ must be chosen for the application. Our analysis provides guidance, but automatic parameter selection remains future work.
- *Border effects.* The large stencil footprint creates a border region (up to 19 pixels for $m=14$, $N_p=100$) where the filter cannot be applied.

// ======================================================================
// 7. CONCLUSION
// ======================================================================
= Conclusion

We have presented the Anisotropic Stencil Filter, a non-learned edge detection operator that combines the mathematical rigor of polynomial surface fitting with the computational efficiency of modern GPU architectures. The fused stencil formulation --- which algebraically collapses a multi-step fitting pipeline into a single weighted sum --- achieves up to 24$times$ speedup and 311$times$ memory reduction over naive implementations, while preserving identical gradient estimates.

On standard benchmarks, the ASF outperforms all classical edge detectors by significant margins (up to +5 ODS points on BSDS500) and approaches deep learning accuracy on domain-specific datasets (0.899 vs. 0.932 ODS on UDED), all without training data. Its noise robustness exceeds both classical and learned methods at low SNR levels.

The ASF demonstrates that principled mathematical design, combined with careful GPU optimization, can yield non-learned methods that remain competitive with data-driven approaches --- with the added benefits of interpretability, guaranteed behavior, and zero training cost.

// ======================================================================
// REFERENCES
// ======================================================================
#pagebreak()

#bibliography("refs.bib", style: "ieee")
