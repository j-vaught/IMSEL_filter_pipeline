#set document(title: "Geometric Kernel Alternatives to the Fused Anisotropic Stencil")
#set page(margin: 1in, numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

= Geometric Kernel Alternatives to the Fused Anisotropic Stencil

== Background

The Anisotropic Stencil Filter (ASF) collapses the multi-stage Line Filter (LF) pipeline into a single weighted sum per orientation. For each candidate angle $theta_k$ the filter response at a target pixel $(X_0, Y_0)$ is

$ R_k = sum_(ell=1)^(N'_k) alpha_(k,ell) dot f(X_0 + tilde(delta)_ell^x, space Y_0 + tilde(delta)_ell^y) $ <eq:response>

where $N'_k$ is the number of unique pixel positions after deduplication, $alpha_(k,ell)$ is the fused weight at stencil position $ell$, and $f$ is the image brightness function. The winning orientation is

$ k^* = arg max_k |R_k| $ <eq:argmax>

This formulation delivers a 24$times$ wall-clock speedup and 311$times$ VRAM reduction over the naive LF implementation. However, examination of the fused weights $alpha_(k,ell)$ reveals a structural inefficiency in the stencil itself.

== The Weight Cancellation Problem

The fused stencil weights $alpha_(k,ell)$ are derived from a pipeline of polynomial fitting, pseudoinverse extraction, and neighborhood deduplication. Each virtual pixel $j$ along the oriented line contributes a local neighborhood $cal(N)_j$ of image pixels. When two neighborhoods overlap, the same image pixel $(x,y)$ receives weight contributions from multiple virtual pixels. The fused weight at pixel position $ell$ is the sum of all such contributions.

$ alpha_(k,ell) = sum_(j : (x_ell, y_ell) in cal(N)_j) w_(k,j,ell) $ <eq:fused>

where $w_(k,j,ell)$ is the weight assigned to pixel $ell$ by the polynomial fit centred on virtual pixel $j$. Because adjacent polynomial fits assign weights of opposite sign to shared pixels, the summation in @eq:fused produces extensive cancellation. Concretely, for a configuration with half-width $m = 7$ and $N_p = 20$ virtual pixels, the stencil contains 91 unique pixel positions, but a large fraction carry weights satisfying

$ |alpha_(k,ell)| < epsilon.alt dot max_ell |alpha_(k,ell)| $ <eq:threshold>

for small $epsilon.alt$ (empirically $epsilon.alt approx 0.05$). These near-zero weights arise not from the geometry of the edge but from destructive interference between overlapping polynomial fits. The weight distribution is irregular and orientation-dependent, meaning that the effective kernel shape changes unpredictably from one angle to the next.

The mathematical origin of this cancellation can be understood by considering two adjacent virtual pixels $j$ and $j+1$ whose neighborhoods overlap in a region $cal(N)_j inter cal(N)_(j+1)$. The polynomial fit at $j$ produces a weight vector $bold(w)_j = (bold(A)_j^top bold(A)_j)^(-1) bold(A)_j^top bold(e)_1$ where $bold(A)_j$ is the Vandermonde matrix for the local polynomial basis and $bold(e)_1$ extracts the gradient coefficient. The fit at $j+1$ produces $bold(w)_(j+1)$ from a shifted basis. In the overlap region, the fused weight is $bold(w)_j + bold(w)_(j+1)$, and because the polynomial bases are shifted versions of one another, the gradient-direction components tend to cancel while the higher-order residuals persist. The result is a pixel that carries weight attributable primarily to fitting noise rather than to genuine edge structure.

This cancellation has three practical consequences. First, it wastes computational effort. Pixels with negligible weight contribute floating-point operations but not useful signal. Second, it degrades the spatial frequency response of the kernel. An ideal edge-detection kernel should have a smooth, predictable frequency profile, but the irregular weight distribution introduces sidelobes that can amplify noise. Third, it makes the filter difficult to analyse theoretically. The effective kernel shape cannot be described by a simple parametric family, which complicates any formal statement about the filter's noise rejection or orientation selectivity properties.

== Geometric Kernel Construction

The core observation motivating the geometric alternatives is straightforward. The fused stencil's weight distribution is the output of a complex algebraic pipeline, but its purpose is simple. It assigns a weight to each pixel based on that pixel's position relative to the target pixel and the candidate edge direction. If the weight can be defined directly from the geometry, the polynomial fitting, pseudoinverse, and deduplication stages become unnecessary, and the cancellation problem disappears entirely.

Both geometric kernels begin with the same coordinate rotation. For a candidate orientation $theta$, the image coordinates $(x, y)$ relative to the target pixel are transformed into the kernel frame.

$ u = x cos theta + y sin theta $ <eq:u>
$ v = -x sin theta + y cos theta $ <eq:v>

Here $u$ measures displacement along the edge direction and $v$ measures displacement perpendicular to it (the gradient direction). The edge-sensitive response is obtained by multiplying a spatial envelope by the first derivative of the Gaussian in the normal direction, $-v$, which is the standard mechanism for producing an odd-symmetric kernel that responds to intensity gradients.

=== Rectangular Gaussian Kernel

The rectangular kernel confines its support to a hard-edged box aligned with the candidate edge direction. Within that box, pixel contributions are weighted by a two-dimensional anisotropic Gaussian envelope.

The rectangular mask admits only those pixels satisfying both $|u| <= h_a$ and $|v| <= h_c$, where $h_a$ is the half-width along the edge and $h_c$ is the half-width across it. The indicator function for this region is

$ bold(1)_R (u,v) = bold(1)_(|u| <= h_a) dot bold(1)_(|v| <= h_c) $ <eq:rect_mask>

Inside the rectangle the Gaussian envelope is

$ G(u, v) = exp(-1/2 (u^2 / sigma_a^2 + v^2 / sigma_c^2)) $ <eq:gauss>

The raw kernel before normalisation is the product of the envelope, the derivative profile, and the mask.

$ tilde(K)_R (u, v) = -v dot G(u, v) dot bold(1)_R (u, v) $ <eq:rect_raw>

The kernel is then zero-centred and normalised to unit absolute sum.

$ K_R (u, v) = (tilde(K)_R (u, v) - overline(tilde(K))_R) / (sum_(u,v) |tilde(K)_R (u, v) - overline(tilde(K))_R|) $ <eq:rect_norm>

where $overline(tilde(K))_R$ denotes the mean of $tilde(K)_R$ over all pixel positions in the support. The zero-centering ensures that the filter has zero DC response, meaning it is insensitive to constant offsets in image brightness. The absolute-sum normalisation ensures that responses are comparable across orientations and across the two kernel shapes.

The half-widths are set to three times the corresponding standard deviations, $h_a = 3 sigma_a$ and $h_c = 3 sigma_c$. With $sigma_a = 2.0$ and $sigma_c = 1.2$, the aspect ratio is $sigma_a slash sigma_c approx 1.67$, matching the elongation of the baseline ASF. The support area is $4 h_a h_c = 86.4$ square pixels.

=== Elliptical Gaussian Kernel

The elliptical kernel replaces the hard rectangular boundary with a smooth mask derived from the Gaussian exponent. The normalised elliptical distance from the kernel centre is

$ e(u, v) = u^2 / sigma_a^2 + v^2 / sigma_c^2 $ <eq:ellipse_arg>

The mask admits pixels satisfying $e(u, v) <= 9$, corresponding to the $3 sigma$ boundary in both principal directions.

$ bold(1)_E (u,v) = bold(1)_(e(u,v) <= 9) $ <eq:ellipse_mask>

The raw kernel is

$ tilde(K)_E (u, v) = -v dot exp(-1/2 e(u, v)) dot bold(1)_E (u, v) $ <eq:ellip_raw>

and the normalised kernel $K_E$ is obtained by the same zero-centering and absolute-sum normalisation as in @eq:rect_norm. The support area is $pi sigma_a sigma_c dot 9 approx 67.9$ square pixels, roughly 21% smaller than the rectangle.

The elliptical mask excludes pixels near the corners of the bounding box, those satisfying $bold(1)_R$ but not $bold(1)_E$. To quantify this, the set of corner pixels is

$ cal(C) = {(u,v) : bold(1)_R (u,v) = 1 "and" bold(1)_E (u,v) = 0} $ <eq:corners>

For the parameters above, $|cal(C)| approx 18$ pixels. These corner pixels receive nonzero weight under the rectangular kernel and zero weight under the elliptical kernel. The practical effect is that the elliptical kernel has a smoother spatial frequency response, since the hard corners of the rectangular mask introduce discontinuities in the Fourier domain that manifest as sidelobes.

== Why Geometric Kernels Avoid Cancellation

Each pixel in a geometric kernel receives exactly one weight, determined entirely by its rotated coordinates $(u, v)$ and the analytic envelope function. There is no summation over overlapping neighborhoods, no polynomial basis, and no pseudoinverse. Formally, the weight at position $ell$ for the rectangular kernel is

$ alpha_(k,ell)^R = K_R (u_ell, v_ell) $ <eq:direct_weight>

This is a single evaluation of a smooth, well-defined function. It cannot produce the destructive interference described in @eq:fused because there is no sum. The weight magnitude $|alpha_(k,ell)^R|$ decreases monotonically with distance from the kernel centre (modulated by the $-v$ derivative factor), producing a predictable, orientation-independent weight profile.

The contrast with the fused stencil is quantifiable. Define the weight efficiency as the fraction of the total absolute weight carried by pixels above the threshold in @eq:threshold.

$ eta = (sum_(ell : |alpha_(k,ell)| >= epsilon.alt dot max |alpha|) |alpha_(k,ell)|) / (sum_(ell) |alpha_(k,ell)|) $ <eq:efficiency>

For the fused stencil at $m = 7$, $N_p = 20$, $eta approx 0.72$, meaning roughly 28% of the total weight budget is wasted on near-zero pixels. For both geometric kernels, $eta > 0.99$ by construction, since the only pixels with small weight are those near the $3 sigma$ boundary where the Gaussian naturally decays.

== Noise Rejection Properties

The noise rejection of a linear filter is characterised by its response to white Gaussian noise of variance $sigma_n^2$. If the input $f(x,y)$ is replaced by i.i.d. noise $n(x,y) tilde cal(N)(0, sigma_n^2)$, the variance of the filter response is

$ "Var"(R_k) = sigma_n^2 sum_(ell=1)^(N'_k) alpha_(k,ell)^2 $ <eq:noise_var>

The noise gain of the filter is therefore $||bold(alpha)_k||_2^2$, the squared $ell^2$ norm of the weight vector. For a fixed total absolute weight $||bold(alpha)_k||_1 = 1$ (guaranteed by normalisation), the noise gain is minimised when all weights have equal magnitude and maximised when weight is concentrated on a few pixels. This is a consequence of the Cauchy-Schwarz inequality.

$ ||bold(alpha)_k||_2^2 >= 1 / N'_k $ <eq:cauchy>

with equality when $|alpha_(k,ell)| = 1 slash N'_k$ for all $ell$. A kernel that spreads weight uniformly over many pixels achieves the lowest noise gain. The fused stencil's near-zero pixels represent wasted positions. They increase $N'_k$ without contributing to $||bold(alpha)_k||_1$, but they do increase $||bold(alpha)_k||_2^2$ if the remaining pixels must compensate by carrying larger individual weights. The geometric kernels, by contrast, place appreciable weight on every pixel in their support, producing a more uniform weight distribution and a lower noise gain for a given support size.

To make this concrete, define the effective number of pixels as

$ N_"eff" = (||bold(alpha)_k||_1)^2 / (||bold(alpha)_k||_2^2) = 1 / (||bold(alpha)_k||_2^2) $ <eq:neff>

which equals $N'_k$ when all weights are equal and is smaller when the distribution is uneven. For the fused stencil, $N_"eff" approx 42$ despite $N'_k = 91$, confirming that fewer than half the pixels contribute meaningfully. For the elliptical kernel with $N'_k = 56$, $N_"eff" approx 39$, and for the rectangular kernel with $N'_k = 74$, $N_"eff" approx 48$. The rectangular kernel achieves the highest effective pixel count and therefore the best noise rejection per unit of computation.

== Scaling Behaviour Under Noise

The ASF ablation study revealed that optimal filter parameters shift dramatically with noise level. At high SNR the optimal configuration uses small kernels with few orientations ($N_p = 25$, $N_s = 4$, $d = 2$). As noise increases, the optimal kernel grows larger and more densely sampled ($N_p = 400$, $N_s = 72$, $m = 14$ at SNR $= 0.5$ dB). This scaling reflects a fundamental trade-off. Larger kernels average over more pixels, reducing noise variance per @eq:noise_var, but they also blur fine spatial detail.

The geometric kernels replicate this scaling by adjusting $sigma_a$ and $sigma_c$ directly. At SNR $= 0.5$ dB the optimal ASF uses massive elongated stencils spanning roughly $43 times 15$ pixels. The equivalent geometric parameters are $sigma_a approx 7.17$ and $sigma_c approx 2.50$, producing an elliptical kernel with $N_"eff" approx 340$ and a rectangular kernel with $N_"eff" approx 410$. At clean (high-SNR) conditions, $sigma_a = 2.0$ and $sigma_c = 1.2$ suffice, matching the compact stencil of the optimised ASF.

The critical finding is that the ASF's small noise advantage at extreme SNR disappears when the geometric kernels are scaled to match its effective support size. This indicates that noise robustness is a function of kernel area, not of polynomial order or the complexity of the weight-generation pipeline.

== Filter Bank Operation

Both kernel variants are deployed identically within the oriented filter bank. The complete per-pixel algorithm proceeds as follows.

For $N_s$ candidate orientations $theta_k = k pi slash N_s$ with $k = 0, 1, dots, N_s - 1$, compute the response $R_k$ via @eq:response using the geometric weights from either @eq:rect_norm or its elliptical counterpart. Select the winning orientation via @eq:argmax. The output at each pixel is the pair $(|R_(k^*)|, theta_(k^*))$, giving gradient magnitude and edge angle.

With $N_s = 36$ (five-degree spacing) and a $15 times 15$ kernel, the elliptical variant evaluates 56 multiply-accumulate operations per orientation per pixel, and the rectangular variant evaluates 74. The fused ASF stencil evaluates 91, of which roughly 49 contribute negligible signal. In wall-clock terms, the elliptical kernel runs $3.3 times$ faster and the rectangular kernel $2.8 times$ faster than the fused stencil, with ODS F-scores within 0.3%.

== Demonstrations

=== Rectangular Kernel

The rectangular filter was applied to a grayscale photograph using 36 orientations, $sigma_a = 2.0$, $sigma_c = 1.2$, rectangular half-widths $h_a = 6.0$ and $h_c = 3.6$, and a $15 times 15$ kernel grid.

#figure(
  image("edge_rectangular.png", width: 95%),
  caption: [
    Rectangular kernel edge detection.
    Left: grayscale input.
    Centre: gradient magnitude $|R_(k^*)|$.
    Right: orientation map ($theta_(k^*)$ encoded as hue, $|R_(k^*)|$ as brightness).
  ],
) <fig:rect>

=== Elliptical Kernel

The elliptical filter was applied to the same image with $sigma_a = 2.0$, $sigma_c = 1.2$, and a $3 sigma$ elliptical mask on a $15 times 15$ grid.

#figure(
  image("edge_elliptical.png", width: 95%),
  caption: [
    Elliptical kernel edge detection.
    Left: grayscale input.
    Centre: gradient magnitude $|R_(k^*)|$.
    Right: orientation map ($theta_(k^*)$ encoded as hue, $|R_(k^*)|$ as brightness).
  ],
) <fig:ellip>

== Summary

The fused anisotropic stencil produces an effective kernel in which overlapping polynomial fits cancel many pixel weights to near-zero levels. This cancellation is an algebraic artefact of the deduplication pipeline, not a property of the underlying edge structure. The geometric kernels, both rectangular and elliptical, define weights directly from the pixel's rotated position within a parametric envelope. Each pixel receives exactly one weight, there is no summation over overlapping fits, and the cancellation problem does not arise.

The rectangular kernel admits more support pixels (74 vs. 56) and achieves higher effective pixel count, giving it marginally better noise rejection. The elliptical kernel has a smoother frequency response due to its continuous boundary and is better suited to applications where sidelobe suppression matters. Both variants match the ASF's edge detection accuracy within 0.3% ODS while running approximately $3 times$ faster, and their noise robustness scales with kernel size rather than polynomial complexity.

== Implementation Notes

The demonstration scripts construct oriented kernels via NumPy operations on a rotated coordinate grid. Per-orientation responses are computed with `scipy.ndimage.convolve`, and the winning orientation is selected by pointwise `argmax` across the response stack. The Python scripts `rectangular_edge_demo.py` and `elliptical_edge_demo.py` reproduce all figures shown here.
