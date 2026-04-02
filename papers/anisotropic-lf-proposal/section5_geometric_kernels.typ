// ======================================================================
// 5. GEOMETRIC KERNEL DESIGN
// ======================================================================
= Geometric Kernel Design <sec:geometric>

The weight cancellation analysis in the preceding section demonstrated that the fused stencil's algebraic pipeline --- polynomial fitting, pseudoinverse extraction, neighborhood deduplication, and weight summation --- produces a weight distribution in which approximately 28% of the total budget is consumed by pixels whose net contributions are negligible. The root cause is structural. Overlapping circular neighborhoods contribute opposing signed weights to shared pixels, and the summation during fusion creates destructive interference that cannot be eliminated by tuning the filter parameters. This section presents two alternative kernel designs that resolve the cancellation problem by defining the anisotropic weight envelope directly from the geometry of the candidate edge, bypassing the polynomial fitting pipeline entirely.

== Design Principle

The fused stencil's weight at each pixel is the output of a multi-stage algebraic pipeline. Polynomial basis functions are evaluated at rotated coordinates, a pseudoinverse distributes fitting coefficients across neighbor pixels, line extension tiles these circular neighborhoods along the edge direction, and deduplication sums the overlapping contributions. The weight assigned to a given pixel is therefore the aggregate of a complex chain of operations, and the cancellation phenomenon is an emergent property of that aggregation.

The central observation motivating the geometric approach is that the _purpose_ of this pipeline is simple. It assigns a weight to each pixel based on two quantities: the pixel's displacement from the target along the candidate edge direction, and the pixel's displacement perpendicular to it (the gradient direction). If the weight is defined _directly_ as an analytic function of these two displacements, the entire intermediate pipeline becomes unnecessary. There are no overlapping neighborhoods to deduplicate, no pseudoinverse coefficients to sum, and no mechanism by which destructive interference can arise.

Both geometric kernels begin with the same coordinate rotation used in the polynomial model. For a candidate edge orientation $theta_k$, image coordinates $(x, y)$ relative to the target pixel are transformed to a rotated frame:

$ u = x cos theta_k + y sin theta_k $ <eq:u_rot>

$ v = -x sin theta_k + y cos theta_k $ <eq:v_rot>

Here $u$ measures displacement along the edge tangent direction and $v$ measures displacement perpendicular to the edge, aligned with the gradient. This decomposition is the natural coordinate system for any edge-aware filter, and both kernel variants operate entirely in the $(u, v)$ frame.

== Rectangular Gaussian Kernel

The first geometric variant confines the kernel support to a hard-edged rectangular region aligned with the candidate edge direction. The rectangle is defined by two half-widths: $h_u$ along the edge tangent and $h_v$ along the gradient direction. A pixel at rotated coordinates $(u, v)$ lies within the support if and only if both $|u| <= h_u$ and $|v| <= h_v$. The indicator function for this region is

$ bold(1)_R (u, v) = bold(1)_(|u| <= h_u) dot bold(1)_(|v| <= h_v) $ <eq:rect_indicator>

Within the rectangular support, the weight is determined by a two-dimensional anisotropic Gaussian envelope with independent bandwidths $sigma_u$ along the edge and $sigma_v$ across it:

$ G(u, v) = exp(-1/2 (u^2 / sigma_u^2 + v^2 / sigma_v^2)) $ <eq:gauss_envelope>

The Gaussian envelope serves the same role as the polynomial fit in the original ASF. It concentrates weight on pixels near the target (where the local edge model is most reliable) and tapers smoothly toward the boundary, reducing sensitivity to distant pixels that may belong to different structures. The anisotropic parameterization allows the kernel to extend further along the edge (large $sigma_u$) than across it (small $sigma_v$), matching the elongated support characteristic of the line-extended polynomial filter.

To produce a derivative-sensitive response, the envelope must be modulated by a function that is odd-symmetric in the gradient direction. The simplest such modulation is multiplication by $-v$, which creates a dipole pattern: pixels on one side of the edge receive positive weight, pixels on the other side receive negative weight, and the filter's response is proportional to the local intensity gradient perpendicular to the candidate edge. The raw (unnormalized) rectangular kernel is therefore

$ hat(K)_R (u, v) = -v dot G(u, v) dot bold(1)_R (u, v) $ <eq:rect_raw>

Two normalization steps are applied to the raw kernel. First, the mean weight over all pixels in the support is subtracted, ensuring that the kernel has zero DC response. A filter with zero DC response produces zero output when applied to a region of constant intensity, which is essential for insensitivity to absolute brightness offsets. Second, the kernel is divided by the sum of the absolute values of all weights, so that the total absolute weight equals unity. This absolute-sum normalization ensures that filter responses are comparable across orientations and kernel sizes.

The support half-widths are set proportional to the Gaussian bandwidths as $h_u = 3 sigma_u$ and $h_v = 3 sigma_v$, capturing 99.7% of the Gaussian mass along each axis. With $sigma_u = 2.0$ and $sigma_v = 1.2$, the aspect ratio $sigma_u / sigma_v approx 1.67$ matches the elongation characteristic of the polynomial ASF at comparable parameter settings. The support area of the rectangle is $4 h_u h_v approx 86.4$ square pixels. When sampled on a 15$times$15 integer grid, the rectangular kernel admits approximately $N'_k approx 74$ pixels that satisfy both half-width constraints in the rotated frame.

== Elliptical Gaussian Kernel

The second geometric variant replaces the hard rectangular boundary with a smooth elliptical mask derived from the Gaussian exponent itself. Rather than testing separate conditions on $u$ and $v$, the elliptical kernel defines a normalized distance that combines both axes into a single scalar:

$ r(u, v) = sqrt(u^2 / sigma_u^2 + v^2 / sigma_v^2) $ <eq:ell_distance>

The quantity $r(u, v)$ is the Mahalanobis distance from the origin under the anisotropic Gaussian, expressed in units of standard deviations. The elliptical mask admits all pixels satisfying $r(u, v) <= 3$, corresponding to the $3 sigma$ contour of the Gaussian in both principal directions simultaneously:

$ bold(1)_E (u, v) = bold(1)_(r(u, v) <= 3) $ <eq:ell_indicator>

The raw kernel follows the same construction as the rectangular variant, with the Gaussian envelope modulated by $-v$ and masked by the elliptical indicator:

$ hat(K)_E (u, v) = -v dot exp(-1/2 r(u, v)^2) dot bold(1)_E (u, v) $ <eq:ell_raw>

The same zero-centering and absolute-sum normalization are applied. The resulting kernel $K_E$ has zero DC response and unit absolute weight, identical in these properties to the rectangular kernel.

The elliptical support area is $pi sigma_u sigma_v dot 9 approx 67.9$ square pixels for $sigma_u = 2.0$ and $sigma_v = 1.2$, approximately 21% smaller than the rectangular support. The difference arises from the exclusion of corner pixels that lie within the bounding rectangle but outside the ellipse. Formally, the set of corner pixels excluded by the elliptical mask is

$ cal(C) = {(u, v) : bold(1)_R (u, v) = 1 "and" bold(1)_E (u, v) = 0} $ <eq:corner_set>

For the default parameters, $|cal(C)| approx 18$ pixels per orientation. These are pixels in the corners of the aligned rectangle where the combined displacement exceeds three standard deviations despite each individual displacement being within bounds. Under the rectangular kernel, these corner pixels receive nonzero weight; under the elliptical kernel, they receive exactly zero.

The practical consequence of excluding corner pixels is a smoother spatial frequency response. The sharp rectangular boundary introduces discontinuities in the Fourier domain that manifest as sidelobes in the filter's frequency response. These sidelobes can cause ringing artifacts in the filter output, particularly near high-contrast edges adjacent to textured regions. The elliptical mask, by tapering smoothly in all directions simultaneously, attenuates these sidelobes and produces a cleaner spectral profile. The trade-off is a modest reduction in the number of contributing pixels, which slightly increases noise gain as quantified in the analysis below.

== Cancellation Avoidance

The defining advantage of both geometric kernels over the polynomial fused stencil is the complete elimination of weight cancellation. The mechanism is straightforward. Each pixel in a geometric kernel receives exactly one weight, determined entirely by evaluating an analytic function at the pixel's rotated coordinates $(u, v)$. For the rectangular kernel, the weight at stencil position $ell$ is

$ alpha_(k, ell)^R = K_R (u_ell, v_ell) $ <eq:rect_weight>

This is a single evaluation of a smooth, well-defined function. The weight depends on the pixel's position relative to the target and on the kernel parameters ($sigma_u$, $sigma_v$, $h_u$, $h_v$), but it does not involve any summation over overlapping contributions from multiple source neighborhoods.

The contrast with the polynomial fused stencil is fundamental. The fused stencil weight at position $ell$ is the result of aggregating contributions from all virtual line positions whose circular neighborhoods include that pixel:

$ alpha_(k, ell) = sum_(j, i) w_j p_i^((k)) $ <eq:fused_weight_recall>

where the sum ranges over all $(j, i)$ pairs mapping to position $ell$ after rounding. Because the pseudoinverse weights $p_i^((k))$ alternate in sign across the neighborhood (they encode derivative information), and because the Gaussian line weights $w_j$ are always positive, the partial contributions to $alpha_(k, ell)$ can have opposing signs. Their sum is the net weight, which may be near zero despite each individual contribution being substantial. This is the destructive interference identified in Section 4.

In the geometric kernels, no such summation exists. The weight function $K_R$ or $K_E$ is evaluated once per pixel, and its value is determined by the smooth analytic envelope. The $-v$ modulation ensures that pixels on opposite sides of the edge axis receive opposite-sign weights (as required for derivative sensitivity), but no pixel's weight is the residual of a cancellation between large opposing terms. The weight distribution is smooth, predictable, and fully characterized by the parametric family ($sigma_u$, $sigma_v$) rather than being an emergent artifact of the algebraic pipeline.

== Noise Gain and Effective Pixel Count

The noise rejection capacity of a linear filter is characterized by its response to white Gaussian noise. If the input image $f(x, y)$ is replaced by independent identically distributed noise samples $n(x, y) tilde.op cal(N)(0, sigma_n^2)$, the variance of the filter response at orientation $k$ is

$ "Var"(R_k) = sigma_n^2 sum_(ell = 1)^(N'_k) alpha_(k, ell)^2 $ <eq:noise_var>

The sum $||bold(alpha)_k||_2^2 = sum_ell alpha_(k, ell)^2$ is the _noise gain_ of the filter. A lower noise gain indicates better noise suppression for a given signal response. For a filter with unit absolute weight ($||bold(alpha)_k||_1 = 1$, guaranteed by the normalization procedure), the Cauchy--Schwarz inequality establishes a lower bound on the noise gain:

$ ||bold(alpha)_k||_2^2 >= 1 / N'_k $ <eq:cauchy_schwarz>

Equality holds when all weights have identical magnitude $1 / N'_k$. Any deviation from uniform weighting increases the noise gain above this theoretical minimum. The effective pixel count, $N_"eff"$, provides an intuitive measure of how efficiently the filter utilizes its spatial support:

$ N_"eff" = (||bold(alpha)_k||_1^2) / (||bold(alpha)_k||_2^2) = 1 / (||bold(alpha)_k||_2^2) $ <eq:neff>

where the second equality follows from the unit-absolute-weight normalization. The effective pixel count equals $N'_k$ when all weights are uniform and is smaller when the weight distribution is concentrated on a subset of the support. It represents the number of equally weighted pixels that would produce the same noise gain as the actual weight distribution.

The weight cancellation problem in the fused stencil degrades $N_"eff"$ in a specific way. Pixels with near-zero net weight contribute negligibly to $||bold(alpha)_k||_1$ (the signal-carrying capacity) but the remaining nonzero-weight pixels must compensate by carrying larger individual weights. These larger weights disproportionately increase $||bold(alpha)_k||_2^2$ because the squared norm penalizes weight concentration quadratically.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.header[*Kernel*][*$N'_k$*][*$N_"eff"$*][*$N_"eff" slash N'_k$*],
    table.hline(),
    [Fused polynomial stencil], [91], [42], [0.46],
    [Rectangular Gaussian], [74], [48], [0.65],
    [Elliptical Gaussian], [56], [39], [0.70],
  ),
  caption: [Noise performance comparison. $N'_k$ is the number of pixels in the stencil support; $N_"eff"$ is the effective pixel count (@eq:neff). The efficiency ratio $N_"eff" slash N'_k$ measures how uniformly the filter distributes weight across its support.],
) <tab:neff>

@tab:neff summarizes the noise performance of the three kernel types under the default parameters ($sigma_u = 2.0$, $sigma_v = 1.2$, $m = 7$, $N_p = 100$, $d = 4$ for the polynomial stencil). The fused polynomial stencil nominally accesses 91 pixels but achieves an effective pixel count of only 42, yielding an efficiency ratio of 0.46. Nearly half of the stencil's spatial support is wasted, in the sense that the weight distribution is as noisy as a uniform filter over only 42 pixels. The rectangular Gaussian kernel accesses fewer pixels (74) but achieves a higher $N_"eff"$ of 48 and a substantially better efficiency ratio of 0.65. The elliptical Gaussian kernel has the smallest support (56 pixels) and the lowest $N_"eff"$ in absolute terms (39), but its efficiency ratio of 0.70 is the highest among the three variants.

The rectangular kernel achieves the highest absolute $N_"eff"$ and therefore the best noise rejection per orientation evaluation. Its larger support admits more pixels into the filter, and the smooth Gaussian taper distributes weight more uniformly than the polynomial stencil's irregular profile. The elliptical kernel trades a modest reduction in absolute noise rejection for the highest weight efficiency and a cleaner spectral response, making it preferable in applications where sidelobe suppression is important.

Both geometric kernels demonstrate that defining the weight envelope analytically, rather than deriving it through the polynomial fitting pipeline, produces a more efficient allocation of the filter's degrees of freedom. The weight budget is spent entirely on pixels that contribute meaningfully to the gradient estimate, with no portion lost to destructive interference between overlapping fits.

== Parameter Selection

The geometric kernels are parameterized by two intuitive quantities: the Gaussian bandwidths $sigma_u$ (along-edge smoothing) and $sigma_v$ (cross-edge resolution). These parameters have clear physical interpretations. Increasing $sigma_u$ extends the kernel along the edge tangent, incorporating more information from a longer segment of the edge and improving noise robustness at the cost of reduced sensitivity to edge curvature. Increasing $sigma_v$ widens the kernel perpendicular to the edge, providing more averaging across the gradient but reducing the filter's ability to resolve closely spaced parallel edges.

The aspect ratio $sigma_u / sigma_v$ controls the anisotropy of the kernel. A ratio of 1.0 produces an isotropic filter (circular for the elliptical variant, square for the rectangular variant). Larger ratios produce increasingly elongated kernels that favor straight edges over curved ones. The default ratio of $sigma_u / sigma_v approx 1.67$ provides a moderate degree of anisotropy that balances noise suppression along the edge with sensitivity to edge curvature and junction structures. This ratio was selected to match the effective elongation of the polynomial ASF at $m = 7$, $N_p = 100$, enabling direct comparison of the kernel geometries without confounding differences in spatial support shape.

The support truncation at $3 sigma$ in both kernel variants is a standard choice that captures 99.7% of the Gaussian mass along each principal axis. Extending to $4 sigma$ or beyond would add pixels with negligible weight, increasing computation without meaningfully improving noise rejection. Reducing to $2 sigma$ would sacrifice approximately 5% of the total weight, slightly degrading the zero-DC property after mean subtraction and producing a less smooth spatial taper.
