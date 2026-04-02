= Fused Stencil Formulation

The preceding section introduced the line-extended polynomial filter and its orientation-sweep response $R_k$ as a composition of $(2m+1)$ independent polynomial fits. This section demonstrates that the entire pipeline can be algebraically collapsed into a single precomputed stencil per orientation, analyzes the resulting deduplication statistics, and then reveals a structural deficiency --- extensive weight cancellation --- that motivates the geometric kernel alternatives developed in Section 5.

== Algebraic Collapse <sec:collapse>

The line response defined in @eq:lineresponse can be written in expanded form by substituting the polynomial gradient estimate @eq:fx at each virtual position $j$. At virtual position $j$ along orientation $theta_k$, the point filter gathers $N_p$ pixel intensities from a circular neighborhood centered at $(X_0 + j cos theta_k, Y_0 + j sin theta_k)$ and computes the normal derivative as a dot product with the precomputed pseudoinverse row $bold(p)_"fx"^((k))$. Let $(Delta x_i, Delta y_i)$ denote the offset of the $i$-th circular neighbor relative to the center of its local neighborhood. The global coordinates of this neighbor, when centered at virtual position $j$, are

$ (X_0 + j cos theta_k + Delta x_i, #h(0.5em) Y_0 + j sin theta_k + Delta y_i). $ <eq:global_coords>

Substituting into @eq:lineresponse yields

$ R_k = sum_(j=-m)^m w_j sum_(i=1)^(N_p) p_i^((k)) dot f(X_0 + j cos theta_k + Delta x_i, #h(0.5em) Y_0 + j sin theta_k + Delta y_i) $ <eq:full_expand>

where $p_i^((k)) = bold(p)_"fx"^((k))[i]$ is the $i$-th element of the pseudoinverse weight vector and $w_j = exp(-j^2 \/ (2 sigma^2))$ is the Gaussian line weight from @eq:lineresponse. The double summation in @eq:full_expand is entirely linear in the image values $f$. Every pixel accessed through the combined offset $delta_(j,i)^x = j cos theta_k + Delta x_i$, $delta_(j,i)^y = j sin theta_k + Delta y_i$ is rounded to the nearest integer coordinate for discrete sampling. Because many $(j,i)$ pairs share the same integer-rounded position --- particularly when the circular neighborhood radius exceeds the unit step between virtual positions --- the summation can be reorganized by grouping all contributions that address the same physical pixel.

Let $(tilde(delta)_ell^x, tilde(delta)_ell^y)$ for $ell = 1, dots, N'_k$ enumerate the _unique_ integer offsets that appear anywhere in the double summation. The fused weight for position $ell$ is obtained by accumulating all products $w_j dot p_i^((k))$ whose combined offset rounds to $(tilde(delta)_ell^x, tilde(delta)_ell^y)$.

$ alpha_(k,ell) = sum_({(j,i) : round(delta_(j,i)^x, delta_(j,i)^y) = (tilde(delta)_ell^x, tilde(delta)_ell^y)}) w_j dot p_i^((k)) $ <eq:fused_weight>

The line response then reduces to a single weighted sum over the deduplicated stencil.

$ R_k = sum_(ell=1)^(N'_k) alpha_(k,ell) dot f(X_0 + tilde(delta)_ell^x, #h(0.5em) Y_0 + tilde(delta)_ell^y) $ <eq:fused_stencil>

Equation @eq:fused_stencil is the _fused stencil_ representation. The entire LF pipeline --- coordinate rotation, Vandermonde matrix construction, least-squares inversion, Gaussian weighting, and summation along the line --- has been absorbed into the scalar coefficients $alpha_(k,ell)$. What remains is a single gather-and-dot-product operation per orientation per pixel.

The mathematical equivalence between @eq:full_expand and @eq:fused_stencil is exact; no approximation is introduced. The fusion is purely an algebraic rearrangement enabled by the linearity of the underlying operations. Any implementation that evaluates @eq:fused_stencil produces bit-identical results (up to floating-point reordering) to one that evaluates the original multi-step pipeline.

== Deduplication Statistics <sec:dedup>

The raw stencil for a given orientation contains $(2m+1) times N_p$ entries before deduplication. The number of unique integer positions $N'_k$ after grouping is always strictly less than this product, and the reduction factor grows with $m$ because increasing the line half-width causes progressively more overlap between adjacent virtual neighborhoods.

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
  caption: [Stencil deduplication statistics for $N_p = 100$, $d = 4$. "Unique" is the mean count of distinct integer pixel positions across all $N_s = 18$ orientations. Reduction $= 1 - "Unique" \/ "Raw size"$.],
) <tab:dedup_stats>

@tab:dedup_stats summarizes the compression for representative configurations. At $m = 1$, the three virtual positions ($j in {-1, 0, 1}$) share roughly half their circular neighborhoods, yielding a 57% reduction. At $m = 14$, the 29 virtual neighborhoods overlap so heavily that only 15% of the raw entries correspond to distinct pixels. The unique count $N'_k$ varies slightly across orientations because the rounding of continuous offsets to integer coordinates depends on $theta_k$; the values reported are averages over all 18 orientations.

The monotonic increase in reduction with $m$ can be understood geometrically. The circular neighborhood of each virtual position has radius $r approx ceil(sqrt(N_p \/ pi)) + 1$. When $m < r$, adjacent neighborhoods overlap in a band of width approximately $2(r - 1)$ pixels. As $m$ grows, more virtual positions fall within the radius of their neighbors, and the fraction of shared pixels approaches unity. In the limiting case $m >> r$, only the two outermost virtual positions contribute pixels not already covered by interior positions, and the unique count grows linearly as approximately $N_p + 2m$ rather than the quadratic $(2m+1) times N_p$.

== Computational Implications <sec:comp_implications>

The fused formulation transforms the filter from a multi-step algorithm --- rotate coordinates, build design matrix, compute pseudoinverse, gather pixels, multiply, weight, sum --- into a single vector operation per orientation.

$ R_k = bold(alpha)_k^top bold(g)_k $ <eq:dot_product>

Here $bold(alpha)_k in RR^(N'_k)$ is the precomputed fused weight vector and $bold(g)_k in RR^(N'_k)$ is the gathered intensity vector containing the $N'_k$ image values at the stencil offsets. The weights $bold(alpha)_k$ depend only on the filter parameters ($m$, $N_p$, $d$, $theta_k$) and not on the image content, so they are computed once during initialization and reused for every pixel in every frame.

Three consequences follow directly from @eq:dot_product. First, the filter requires a single gather operation of $N'_k$ reads per orientation, rather than $(2m+1)$ separate gathers of $N_p$ reads each. For $m = 7$ and $N_p = 100$, this reduces the memory traffic from 1500 reads to 264, an 82% reduction that translates directly to improved memory-bound throughput on GPU architectures. Second, no runtime matrix operations are needed. The pseudoinverse computation, coordinate rotation, Gaussian weighting, and summation along the line are all absorbed into $bold(alpha)_k$ at precompute time. The only per-pixel arithmetic is the gather and a multiply-accumulate loop. Third, the memory footprint is constant with respect to the image. The stencil weights occupy a fixed buffer (proportional to $N_s times max_k N'_k$), and the only per-pixel allocations are the output gradient magnitude and angle maps.

These properties make the fused stencil representation well suited to GPU execution, where memory bandwidth is the primary bottleneck and fixed-size, data-independent access patterns enable coalesced reads and predictable occupancy. The implementation details are discussed in Section 5.

== The Weight Cancellation Problem <sec:cancellation>

The algebraic collapse described above is exact, but the resulting fused weights $alpha_(k,ell)$ expose a structural deficiency that is invisible in the multi-step formulation. When contributions from multiple virtual positions are accumulated at a shared pixel, extensive cancellation occurs, leaving many stencil entries with negligibly small effective weights. This subsection characterizes the cancellation mechanism, quantifies its severity, and identifies three practical consequences.

=== Origin of Cancellation

Consider two adjacent virtual positions $j$ and $j+1$ whose circular neighborhoods overlap in a region $cal(N)_j sect cal(N)_(j+1)$. The polynomial fit at position $j$ produces a weight vector for the gradient coefficient via the pseudoinverse.

$ bold(w)_j = (bold(A)_j^top bold(A)_j)^(-1) bold(A)_j^top bold(e)_1 $ <eq:pinv_weights>

Here $bold(A)_j in RR^(N_p times M)$ is the Vandermonde design matrix constructed from the local monomial basis centered at virtual position $j$, and $bold(e)_1$ is the unit vector that selects the $f_x$ (normal derivative) coefficient from the solution vector. The fit at position $j+1$ produces an analogous weight vector $bold(w)_(j+1)$ from a _shifted_ basis, because the polynomial expansion is recentered one unit along the line direction.

For pixels in the overlap region $cal(N)_j sect cal(N)_(j+1)$, the local coordinates used to build the Vandermonde rows differ by a unit translation along the tangent direction. This translation reverses the parity of odd-power monomials in the tangent variable while leaving the normal-direction monomials largely unchanged. Because the gradient extraction targets the normal derivative (an odd-power term in the normal direction), the pseudoinverse weights for the gradient component tend to have _opposite signs_ at corresponding positions within the two overlapping neighborhoods. When the fused weight $alpha_(k,ell)$ sums these contributions according to @eq:fused_weight, the gradient-direction components cancel extensively, while higher-order fitting residuals --- which do not share this parity structure --- persist.

The result is a fused weight that is attributable primarily to the residual of the polynomial approximation rather than to genuine edge structure. Pixels carrying such weights contribute floating-point operations during the gather-dot-product but encode fitting artifacts rather than useful gradient information.

=== Quantification

To measure the severity of cancellation, define the _effective weight fraction_ of a fused stencil as the ratio of the sum of absolute fused weights to the total absolute weight that entered the accumulation before cancellation.

$ eta_k = (sum_(ell=1)^(N'_k) |alpha_(k,ell)|) / (sum_(j=-m)^m |w_j| sum_(i=1)^(N_p) |p_i^((k))|) $ <eq:efficiency>

A value of $eta_k = 1$ indicates zero cancellation (no shared pixels or all contributions reinforcing), while $eta_k << 1$ indicates severe cancellation. For typical configurations with $N_p = 100$ and $d = 4$, the efficiency ranges from $eta approx 0.85$ at $m = 1$ (modest overlap) to $eta approx 0.41$ at $m = 14$ (heavy overlap).

A complementary diagnostic examines the distribution of individual fused weights. Define a weight as _negligible_ if its magnitude falls below a fraction $epsilon$ of the maximum weight in the stencil.

$ |alpha_(k,ell)| < epsilon dot max_(ell') |alpha_(k,ell')| $ <eq:negligible>

For a representative configuration with $m = 7$, $N_p = 20$, and $d = 4$, the fused stencil contains 91 unique positions. With $epsilon = 0.05$, approximately 28% of these positions carry negligible weight. These pixels occupy locations in the interior of the stencil where the overlap between adjacent virtual neighborhoods is densest and the cancellation is most complete. The non-negligible weight concentrates along the periphery of the stencil footprint, where fewer virtual positions contribute and cancellation is less severe.

=== Practical Consequences

The weight cancellation problem has three consequences that collectively motivate the geometric kernel approach in Section 5.

_Wasted computation._ Pixels carrying negligible fused weights still participate in the gather and multiply-accumulate loop. For the $m = 7$, $N_p = 20$ configuration, 28% of the stencil entries contribute less than 5% of the peak weight, yet each requires a memory read and a floating-point multiply-add. Pruning these entries would reduce arithmetic cost and memory traffic without measurably affecting the output, but it would also introduce an irregular, data-dependent stencil structure that complicates GPU implementation.

_Degraded spatial frequency response._ The irregular distribution of fused weights --- with strong values at the periphery and near-zero values in the interior --- produces a frequency response that departs significantly from the smooth, monotonically decaying profiles of classical filters such as the Gaussian derivative or the Sobel operator. Fourier analysis of the fused stencil reveals sidelobes in the two-dimensional transfer function, particularly at spatial frequencies corresponding to the inter-virtual-position spacing. These sidelobes can amplify noise components in a frequency band that would be attenuated by a well-designed kernel, partially negating the noise suppression benefit of the large stencil support.

_Theoretical opacity._ Because the fused weight vector $bold(alpha)_k$ is constructed by the superposition of shifted polynomial pseudoinverses modulated by Gaussian line weights, its spatial profile does not belong to any standard parametric family (Gaussian, polynomial, spline, or otherwise). The kernel shape depends on $m$, $N_p$, $d$, $theta_k$, and the specific integer-rounding pattern, making it effectively opaque to classical filter design analysis. Formal characterization of noise rejection bandwidth, orientation selectivity, and edge localization accuracy requires numerical evaluation on a case-by-case basis rather than closed-form expressions. This stands in contrast to, for instance, Gaussian derivative filters @marr1980theory, whose spatial and frequency domain properties are fully characterized by the scale parameter $sigma$, or steerable filters @freeman1991steerable, whose angular response is a known trigonometric polynomial.

=== Structural Interpretation

The cancellation phenomenon admits a compact structural interpretation. The polynomial pseudoinverse at each virtual position acts as a matched filter for the local gradient, producing weights with a characteristic dipole pattern (positive on one side of the center, negative on the other). When dipoles from adjacent virtual positions are superimposed with their centers offset by one pixel, the positive lobe of one dipole overlaps the negative lobe of its neighbor. The summation partially annihilates both lobes, leaving a residual pattern that is neither a clean dipole nor a smooth envelope but an irregular mixture of the two.

This observation suggests a design principle for improved kernels. Rather than constructing the anisotropic stencil by superimposing many small polynomial fits and accepting the resulting cancellation, one should design the weight distribution directly in the fused domain, specifying a target spatial or frequency profile and solving for weights that realize it. The geometric kernel formulations presented in Section 5 follow precisely this strategy, replacing the polynomial pseudoinverse cascade with analytically defined weight functions whose properties --- support shape, frequency response, orientation selectivity --- are controllable by construction.

== Summary

The fused stencil formulation reduces the line-extended polynomial filter to a single precomputed weight vector per orientation, achieving up to 85% compression through deduplication of overlapping pixel accesses. This reformulation enables efficient GPU execution as a gather-dot-product and reveals the mathematical equivalence of the multi-step fitting pipeline to a fixed linear operator. However, the fusion also exposes the weight cancellation problem: adjacent polynomial fits contribute opposing weights to shared pixels, producing a fused stencil in which a substantial fraction of entries carry negligible effective weight. The resulting kernel shape is irregular, theoretically opaque, and suboptimal in its use of the available spatial support. These deficiencies motivate the direct-design approach of Section 5, where the anisotropic weight distribution is specified analytically rather than constructed as a byproduct of polynomial fitting.
