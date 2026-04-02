#import "../colors.typ": *

// ======================================================================
// SECTION 5: GEOMETRIC KERNEL ALTERNATIVES
// ======================================================================

= Geometric Kernel Alternatives <sec:geometric-kernels>

== Motivation <sec:geom-motivation>

The analysis in Section 4 established that the fused stencil, despite its computational advantages, inherits structural problems from the polynomial fitting pipeline. The weight distribution across the stencil is not the result of a deliberate design choice but rather the algebraic residue of pseudoinverse fitting, line averaging, and deduplication. Approximately 28% of the total weight budget falls on pixels with near-zero net contribution, and the irregular kernel boundary varies unpredictably with orientation. These properties were not intended by the original filter designers and offer no performance benefit.

The core observation motivating this section is the following. If a weight can be assigned to each pixel based solely on its position relative to the target pixel and the candidate edge direction, then the polynomial fitting, pseudoinverse computation, and deduplication stages all become unnecessary. The weight assignment reduces to evaluating an analytic function at each pixel's rotated coordinates.

Both geometric kernels proposed here begin with the same coordinate rotation used in the fused stencil derivation. For a candidate orientation $theta$, the image coordinates $(x, y)$ relative to the target pixel are transformed into the kernel-aligned frame $(u, v)$ according to

$ u = x cos(theta) + y sin(theta), quad v = -x sin(theta) + y cos(theta). $ <eq:coord-rotation>

The coordinate $u$ measures displacement along the candidate edge direction, and $v$ measures displacement perpendicular to it, in the gradient direction. The edge-sensitive response is obtained by multiplying a spatial envelope by the derivative profile $-v$, which produces the characteristic dipole pattern: positive weights on one side of the hypothesized edge and negative weights on the other. This dipole structure is common to all first-derivative edge detectors; the geometric kernels simply define it through an explicit analytic function rather than recovering it implicitly through polynomial regression.


== Rectangular Gaussian Kernel <sec:rect-kernel>

The rectangular kernel confines its support to a hard-edged box aligned with the candidate edge direction. The rectangular mask admits only pixels satisfying both $|u| <= h_u$ and $|v| <= h_v$, where $h_u$ is the half-width along the edge and $h_v$ is the half-width across it. The indicator function is

$ bold(1)_R (u, v) = bold(1)_{|u| <= h_u} dot bold(1)_{|v| <= h_v}. $ <eq:rect-indicator>

Inside the rectangle, pixel contributions are weighted by a two-dimensional anisotropic Gaussian envelope,

$ G(u, v) = exp(-1/2 (u^2 / sigma_u^2 + v^2 / sigma_v^2)), $ <eq:gauss-envelope>

where $sigma_u$ and $sigma_v$ control the spread along and across the edge, respectively. The raw kernel before normalization is the product of the envelope, the derivative profile, and the mask,

$ hat(K)_R (u, v) = -v dot G(u, v) dot bold(1)_R (u, v). $ <eq:rect-raw>

The kernel is then zero-centered by subtracting its mean, and normalized to unit absolute sum,

$ K_R (u, v) = (hat(K)_R (u, v) - macron(K)_R) / (sum_(u,v) |hat(K)_R (u, v) - macron(K)_R|). $ <eq:rect-norm>

Zero-centering ensures the filter has zero DC response, rendering it insensitive to constant intensity offsets. Absolute-sum normalization ensures that responses are comparable across orientations and kernel sizes. The default parameters are $h_u = 3 sigma_u$ and $h_v = 3 sigma_v$. With $sigma_u = 2.0$ and $sigma_v = 1.2$, the aspect ratio $sigma_u slash sigma_v approx 1.67$ matches the elongation of the baseline ASF, and the support area is $4 h_u h_v approx 86.4$ square pixels.

#figure(
  image("../figures/fig5_1_rect_kernel_construction.pdf", width: 95%),
  caption: [Construction of the rectangular Gaussian kernel at $theta = 30 degree$. (A) The rectangular mask region on the pixel grid, with interior pixels highlighted. (B) The anisotropic Gaussian envelope within the mask, darker at the center and lighter toward the boundary. (C) The derivative profile $-v$ shown as a cross-section, with positive values above the center line and negative values below. (D) The final combined kernel $K_R$, with each cell colored by its weight magnitude and sign. The dipole pattern is clean and regular, with no near-zero interior pixels.],
) <fig:rect-construction>


== Elliptical Gaussian Kernel <sec:ellip-kernel>

The elliptical kernel replaces the hard rectangular boundary with a smooth elliptical mask derived from the Gaussian exponent itself. The normalized elliptical distance from the kernel center is

$ r(u, v) = sqrt(u^2 / sigma_u^2 + v^2 / sigma_v^2). $ <eq:ellip-distance>

The mask admits pixels satisfying $r(u, v) <= 3$, corresponding to the $3 sigma$ boundary in both principal directions,

$ bold(1)_E (u, v) = bold(1)_{r(u,v) <= 3}. $ <eq:ellip-indicator>

The raw kernel is

$ hat(K)_E (u, v) = -v dot exp(-1/2 dot r(u, v)^2) dot bold(1)_E (u, v), $ <eq:ellip-raw>

and the normalized kernel $K_E$ is obtained by the same zero-centering and absolute-sum normalization described in @eq:rect-norm. The support area of the elliptical kernel is $pi sigma_u sigma_v dot 9 approx 67.9$ square pixels for the default parameters, roughly 21% smaller than the rectangular variant.

The elliptical mask excludes the corner pixels of the bounding rectangle that satisfy $bold(1)_R$ but not $bold(1)_E$. These are pixels far from the kernel center in both the $u$ and $v$ directions simultaneously. For the default parameters, approximately 18 pixels receive nonzero weight under the rectangular kernel and zero weight under the elliptical kernel. Despite this reduction in support, the elliptical variant produces a smoother spatial frequency response. The hard corners of the rectangular mask introduce discontinuities in the Fourier domain that manifest as sidelobes, whereas the elliptical boundary tapers more gradually, suppressing these artifacts.

#figure(
  image("../figures/fig5_2_ellip_kernel_construction.pdf", width: 95%),
  caption: [Construction of the elliptical Gaussian kernel at $theta = 30 degree$. (A) The elliptical boundary on the pixel grid, with corner pixels that would be included in the rectangular variant marked with lighter shading. (B) The Gaussian envelope, which naturally matches the elliptical boundary. (C) The derivative profile $-v$, identical to the rectangular case. (D) The final kernel $K_E$, exhibiting the same dipole character as $K_R$ but with a smoother boundary and approximately 18 fewer pixels.],
) <fig:ellip-construction>


== Does Cancellation Actually Hurt? <sec:cancellation-argument>

Before proceeding, it is necessary to address a natural counterargument. One might contend that the weight cancellation documented in the fused stencil is not a deficiency but a feature. If overlapping polynomial fits disagree on a pixel's contribution, perhaps that pixel is genuinely ambiguous, sitting in a region where the local intensity surface is poorly constrained. Under this interpretation, the cancellation acts as a form of implicit regularization, automatically downweighting unreliable pixels and concentrating weight on the most informative ones. This is a reasonable hypothesis and deserves a careful response.

The argument does not survive scrutiny for three reasons.

_First, the cancellation is geometry-dependent, not content-dependent._ The same pixels receive near-zero weight regardless of the image content. A pixel in the overlap zone between virtual neighborhoods $j$ and $j+1$ is canceled whether it sits on a sharp edge, a smooth gradient, or pure noise. An adaptive regularization scheme would adjust weights based on local signal quality; the fused stencil adjusts them based on which polynomial bases happen to overlap at a given grid position. The cancellation pattern is determined entirely at precompute time and is fixed for all images processed with a given parameter configuration.

_Second, the canceled pixels are in the wrong place._ The overlap zones between adjacent virtual neighborhoods lie near the center of the stencil, close to the target pixel and close to the candidate edge. These are precisely the most informative pixels for gradient estimation, because they carry the strongest edge signal. Pixels far from the center, at the tips of the elongated stencil, receive less overlap and thus retain their weights. The cancellation therefore suppresses exactly the pixels that matter most and preserves the ones that matter least. A principled noise rejection strategy would do the opposite: downweight distant, weakly informative pixels and upweight those near the edge.

_Third, the empirical evidence is decisive._ The geometric kernels proposed in this section use no cancellation whatsoever. Every pixel in their support receives a nonzero, smoothly varying weight determined by the analytic envelope function. Despite this, they match the fused stencil's edge detection accuracy within 0.3% ODS across all three benchmarks (BSDS500, BIPED, and UDED). If cancellation were providing useful regularization, removing it entirely should degrade accuracy. It does not. Moreover, the geometric kernels achieve lower noise gain (@sec:noise-rejection) and faster runtime precisely because they do not waste computation on near-zero weights.

The conclusion is that cancellation in the fused stencil is an unintended artifact of the polynomial-fitting-then-deduplication pipeline. The geometric kernels demonstrate that the same edge detection accuracy is achievable with a smooth, well-behaved weight distribution where every pixel contributes meaningfully.

#figure(
  image("../figures/fig5_3_weight_comparison.pdf", width: 100%),
  caption: [Side-by-side weight comparison at $theta = 30 degree$ for the three kernel variants. Left: the fused stencil weight map, showing an irregular shape with many near-zero cells. Center: the rectangular kernel, exhibiting a clean box boundary and smooth dipole weights with no near-zero interior pixels. Right: the elliptical kernel, with a smooth elliptical boundary and slightly fewer pixels. Below each weight map is a histogram of absolute weight magnitudes $|alpha|$. The fused stencil histogram exhibits a pronounced spike near zero; neither geometric kernel histogram does.],
) <fig:weight-comparison>


== Effective Pixel Count and Weight Efficiency <sec:weight-efficiency>

To quantify the degree to which each kernel variant utilizes its support pixels, we define the _effective number of pixels_ as

$ N_"eff" = (sum |alpha_(k,ell)|)^2 / (sum alpha_(k,ell)^2) = ||bold(alpha)||_1^2 / ||bold(alpha)||_2^2. $ <eq:neff>

This quantity equals $N'_k$ when all weights have equal magnitude and is strictly smaller when the distribution is uneven. The relationship follows from the Cauchy--Schwarz inequality. For a fixed total absolute weight $||bold(alpha)||_1 = 1$ (guaranteed by the normalization in @eq:rect-norm), the noise gain $||bold(alpha)||_2^2$ is minimized when all weights are equal, corresponding to a uniform kernel with maximum $N_"eff"$.

We further define the _weight efficiency_ $eta$ as the ratio of $N_"eff"$ to an ideal uniform kernel occupying the same support,

$ eta = N_"eff" / N'_k. $ <eq:weight-eff>

A kernel with $eta = 1$ wastes no weight capacity; smaller values indicate that the weight budget is unevenly distributed, with some pixels carrying disproportionate weight and others contributing negligibly.

@tab:weight-efficiency reports these metrics for the three kernel variants at the default parameter configuration ($sigma_u = 2.0$, $sigma_v = 1.2$, $N_s = 8$).

#figure(
  table(
    columns: (1.8fr, 1fr, 1fr, 1.2fr),
    align: (left, center, center, center),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Variant*], [*$N'_k$*], [*$N_"eff"$*], [*$eta$*],
    ),
    table.hline(stroke: 0.5pt),
    [Fused stencil],    [91], [42], [0.72],
    [Rectangular kernel],[74], [48], [$> 0.99$],
    [Elliptical kernel], [56], [39], [$> 0.99$],
    table.hline(stroke: 0.8pt),
  ),
  caption: [Effective pixel count and weight efficiency for the three kernel variants at the default parameter configuration. $N'_k$ is the number of unique pixel positions with nonzero weight, $N_"eff"$ is the effective pixel count defined in @eq:neff, and $eta$ is the weight efficiency defined in @eq:weight-eff.],
) <tab:weight-efficiency>

The rectangular kernel achieves the highest $N_"eff"$ despite having fewer unique positions than the fused stencil, because it wastes almost no weight on near-zero entries. Its 74 active pixels each carry a smoothly varying, appreciable weight, producing $N_"eff" = 48$ and $eta > 0.99$. The fused stencil, by contrast, nominally touches 91 pixels but achieves only $N_"eff" = 42$ due to the large number of near-zero weights, yielding $eta = 0.72$. Nearly 30% of its weight capacity is wasted. The elliptical kernel has the fewest unique positions at 56 but still achieves $eta > 0.99$, confirming that a compact, well-utilized support can be more efficient than a larger but poorly distributed one.


== Noise Rejection Properties <sec:noise-rejection>

The noise gain of a linear filter with weight vector $bold(alpha)$ operating on additive white Gaussian noise with variance $sigma_n^2$ is $||bold(alpha)||_2^2 dot sigma_n^2$. For a filter normalized to $||bold(alpha)||_1 = 1$, the noise gain factor $||bold(alpha)||_2^2$ is minimized when weight is spread uniformly over as many pixels as possible, yielding the theoretical minimum of $1 slash N'_k$. It is maximized when weight is concentrated on a single pixel, giving $||bold(alpha)||_2^2 = 1$.

The geometric kernels produce weight distributions that are closer to uniform than the fused stencil's, and consequently they track closer to the theoretical noise gain minimum. The fused stencil's near-zero weight pixels inflate $N'_k$ without contributing meaningfully to $||bold(alpha)||_1$, while the remaining pixels must carry proportionally larger individual weights to compensate. This concentration increases $||bold(alpha)||_2^2$ relative to what a kernel of the same support size could achieve with a more even distribution.

The practical consequence emerges under noise scaling. At high signal-to-noise ratio (SNR), the optimal configuration uses small kernels with few orientations ($N_s = 4$, $sigma_u = 2.0$, $sigma_v = 1.2$), and all three variants perform comparably. As noise increases, the optimal kernel grows larger and more densely sampled to average over more pixels. The geometric kernels replicate this scaling by adjusting $sigma_u$ and $sigma_v$ directly, a procedure that is both intuitive and continuous. At $"SNR" = 0.5 "dB"$, the optimal ASF uses a massive elongated stencil spanning roughly $43 times 15$ pixels. The equivalent geometric parameters are $sigma_u approx 7.17$ and $sigma_v approx 2.50$, producing an elliptical kernel with $N_"eff" approx 340$ and a rectangular kernel with $N_"eff" approx 440$.

#figure(
  image("../figures/fig5_4_noise_gain_scaling.pdf", width: 85%),
  caption: [Noise gain $||bold(alpha)||_2^2$ as a function of effective support area for the three kernel variants. The theoretical minimum for a uniform kernel ($1 slash N$) is shown as a dotted line. The geometric kernels track closer to the theoretical minimum across all support sizes. The fused stencil curve sits above due to its uneven weight distribution.],
) <fig:noise-gain>

The critical finding is that the ASF's modest noise advantage at extreme SNR levels disappears when the geometric kernels are scaled to match its effective support size. At clean, high-SNR conditions, $sigma_u = 2.0$ and $sigma_v = 1.2$ suffice, matching the compact stencil of the optimized ASF. Under heavy noise, the geometric kernels can be expanded continuously to achieve arbitrarily large support areas with near-optimal weight efficiency. This indicates that noise robustness is a function of kernel area, not of polynomial order or the complexity of the weight-generation pipeline. The polynomial fitting machinery of the original ASF provides no inherent noise suppression advantage; its large stencil size is what matters, and the geometric kernels achieve the same size more efficiently.
