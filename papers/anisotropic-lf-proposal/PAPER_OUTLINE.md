# Paper Outline

**Title:** GPU-Accelerated Anisotropic Edge Detection via Orientation-Selective Stencil Filters

**Author:** J. C. Vaught, Department of Mechanical Engineering, University of South Carolina

---

## Section 1: Introduction (~2.5 pages)

### The Problem
- Edge detection as a foundational CV operation; tension between noise robustness, spatial precision, and computational cost
- Three generations of methods: fixed-kernel operators (Sobel, Prewitt), multi-stage pipelines (Canny), deep learning (HED, DexiNed, DiffusionEdge)
- Each generation trades one problem for another: classical methods are fast but fragile; Canny blurs weak edges; DL requires training data and generalizes poorly outside its training domain

### The Gap
- The local-polynomial-fitting approach to edge detection has been independently reinvented multiple times across disconnected communities without cross-pollination
- Savitzky-Golay (1964) introduced local polynomial least-squares fitting for derivative estimation in spectroscopy. The technique was extended to 2D image grids by multiple authors (Gorry 1990, Meer et al. 1991, Luo et al. 2005)
- Freeman & Adelson (1991) solved orientation-selective filtering analytically with steerable filters
- Geusebroek et al. (2003) built anisotropic Gaussian derivative kernels elongated along edge directions
- Bagan & Wang (2021, 2023) independently rediscovered 2D Savitzky-Golay derivative filtering in a vision context (the WVF), added a brute-force orientation sweep (a discrete version of steerability), and extended it with virtual expansion points along a line (the LF). Their publications do not reference the Savitzky-Golay literature, steerable filters, or anisotropic Gaussian derivatives
- Nobody has connected these threads or asked the natural next question: if the LF is just an oriented, line-averaged 2D S-G filter, what is the simplest and fastest formulation that preserves its accuracy?

### Questions This Paper Addresses
- Can the multi-step LF pipeline be collapsed into a single linear operation suitable for GPU execution?
- What structural limitations does this algebraic fusion reveal about polynomial-fitting-based kernels?
- Can simpler geometric kernel definitions (rectangular and elliptical Gaussian envelopes) match or exceed polynomial fitting accuracy while avoiding those limitations?

### Contributions
1. **Fused stencil formulation.** We algebraically collapse the multi-step LF pipeline (coordinate rotation, design matrix construction, pseudoinverse, line averaging) into a single precomputed weight vector per orientation, revealing that the LF is mathematically equivalent to a fixed linear convolution with an anisotropic kernel. This achieves 24x speedup and 311x VRAM reduction on GPU.
2. **Weight cancellation analysis.** We identify destructive interference between overlapping polynomial neighborhoods as a structural artifact of the fusion process, with a formal weight efficiency metric showing approximately 28% of computational effort is wasted on near-zero weights.
3. **Geometric kernel alternatives.** We propose rectangular and elliptical Gaussian kernel variants that define anisotropic derivative weights directly from geometric envelopes, bypassing polynomial fitting entirely. These match accuracy within 0.3% ODS, run 3x faster than the fused stencil, and have provably uniform weight distributions.
4. **Comprehensive evaluation.** We evaluate all three variants across 1,200+ parameter configurations, 3 benchmarks, 54 classical baselines, and 5 deep learning models, including noise robustness analysis across 5 noise types and 7 SNR levels.


## Section 2: Related Work (~2.5 pages)

### 2.1 Local Polynomial Fitting for Derivative Estimation
- **Savitzky-Golay (1964):** The foundational work. Fit a polynomial of degree d to a moving window of 2m+1 data points via least squares; the fitted coefficients give smoothed derivatives of any order. Originally 1D for spectroscopy. The key insight: the convolution weights can be precomputed from the pseudoinverse of the Vandermonde matrix, making runtime application trivial.
- **Extensions to 2D:** Gorry (1990) generalized S-G convolution coefficients. Meer, Baugher, & Rosenfeld (1991) applied least-squares polynomial fitting to image neighborhoods for edge-preserving smoothing. Luo et al. (2005) derived 2D S-G filters on rectangular grids for image gradient estimation.
- **Local polynomial regression (LOESS/LOWESS):** Cleveland (1979) introduced distance-weighted local polynomial fitting in a statistical context. The WVF's circular neighborhood with uniform weighting is a special case (uniform kernel, no distance decay).
- **The WVF/LF as 2D Savitzky-Golay:** We show explicitly that Bagan & Wang's WVF is a 2D Savitzky-Golay derivative filter with (a) circular rather than rectangular support, (b) coordinate rotation into a candidate edge frame, and (c) max-response orientation selection. The LF adds a weighted average of S-G estimates along a line, analogous to a 1D moving average of 2D S-G outputs.

> **Fig 2.1a (CeTZ): 1D Savitzky-Golay.** A row of ~11 cells representing a 1D signal (pixel intensities as bar heights or shaded values). A sliding window of 7 cells highlighted in Garnet (#73000A). A smooth polynomial curve (degree 3) fitted through the windowed values, drawn in Atlantic (#466A9F). The derivative at the center point shown as a tangent line/arrow in Rose (#CC2E40). Label: "Savitzky & Golay, 1964." Annotate the window size (2m+1), polynomial degree d, and the derivative extraction step.

> **Fig 2.1b (CeTZ): 2D Savitzky-Golay on a rectangular grid.** A ~9x9 pixel grid. A 5x5 rectangular window highlighted in Garnet (#73000A), centered on a target pixel. Inside the window, show the fitted 2D polynomial as a color gradient (smooth surface implied by cell shading from light to dark). Two arrows at the center pixel: df/dx and df/dy in Atlantic (#466A9F). Label: "2D S-G (Meer et al., 1991)." Key point: rectangular support, axis-aligned, no rotation.

> **Fig 2.1c (CeTZ): LOESS/LOWESS.** A scatter of ~15 data points in 1D (dots on a curve). A target point highlighted. Distance-weighted kernel shown as a bell curve (tricube or Gaussian) centered on the target, drawn in Congaree (#1F414D). Points closer to the target are darker/larger (weighted more), far points are lighter/smaller. A locally fitted polynomial line through the weighted points in Atlantic (#466A9F). Label: "Cleveland, 1979." Annotate the distance-weighting function.

> **Fig 2.1d (CeTZ): WVF as rotated 2D S-G.** A ~15x15 pixel grid with a diagonal edge (upper-left dark, lower-right light). Center pixel marked. Circular neighborhood of ~25 pixels highlighted in Garnet (#73000A). Rotated local coordinate axes (x along edge normal, y along edge tangent) drawn in Atlantic (#466A9F) at angle theta. The Taylor polynomial fit is implied by the coordinate frame. Normal derivative arrow in Rose (#CC2E40) perpendicular to the edge. Label: "WVF (Bagan & Wang, 2021)." Annotate: circular support, rotated coordinates, same pseudoinverse math as S-G.

> **Fig 2.1e (CeTZ): LF as line-averaged S-G.** Same grid and edge as 2.1d. Now show (2m+1)=5 virtual evaluation points along the edge tangent direction, each with its own circular neighborhood drawn in progressively lighter shades of Garnet. The neighborhoods visibly overlap. Gaussian weight profile shown alongside the line (bell curve with w_j values). Arrow showing the weighted sum of derivative estimates. Label: "LF (Bagan & Wang, 2023)." Annotate: multiple S-G evaluations averaged along a line.

### 2.2 Orientation-Selective Filtering
- **Steerable filters (Freeman & Adelson, 1991):** Analytically interpolate filter responses across orientations using a finite basis. The WVF's discrete orientation sweep is a brute-force version of this. Steerable filters are limited to specific basis functions (derivatives of Gaussians); the polynomial approach is more general but loses analytic steerability.
- **Oriented energy (Perona & Malik, 1990; Morrone & Owens, 1987):** Phase-based and energy-based orientation estimation. Different mathematical framework but same goal: find the dominant edge direction.
- **Anisotropic Gaussian derivatives (Geusebroek et al., 2003):** Elongated Gaussian kernels whose principal axis aligns with the candidate edge direction. Our rectangular and elliptical kernel variants are closely related, differing primarily in the mask shape and the explicit derivative-profile modulation.

> **Fig 2.2a (CeTZ): Steerable filter basis and interpolation.** Left panel: show the basis kernels G1 and G2 (first derivatives of Gaussian in x and y directions) as small weight grids with red/blue coloring for positive/negative. Right panel: a polar plot or fan diagram showing how the response at any angle theta is obtained by R(theta) = cos(theta)*R1 + sin(theta)*R2. Draw the continuous response curve in Atlantic (#466A9F), mark a few specific angles. Label: "Freeman & Adelson, 1991." Key point: analytic interpolation from a finite basis, no need to evaluate every angle.

> **Fig 2.2b (CeTZ): Oriented energy.** A small image patch with a diagonal edge. Show the even-symmetric (cosine-phase) and odd-symmetric (sine-phase) filter pairs as small kernel weight grids. Below: the energy E(theta) = F_even^2 + F_odd^2 as a polar plot, with the peak indicating edge orientation. Use Garnet (#73000A) for the energy curve, Atlantic (#466A9F) for the individual filter responses. Label: "Morrone & Owens, 1987; Perona & Malik, 1990."

> **Fig 2.2c (CeTZ): Anisotropic Gaussian derivatives.** A pixel grid with a diagonal edge. Show an elongated Gaussian envelope aligned along the edge direction (elliptical contour lines in Congaree #1F414D). The derivative profile (-v * G) shown as a cross-section: positive on one side, negative on the other, colored red/blue. The combined kernel (envelope * derivative) shown as a small weight grid with dipole pattern. Label: "Geusebroek et al., 2003." Key point: this is essentially what our geometric kernels do, with different parameterization.

> **Fig 2.2d (CeTZ): WVF/ASF discrete orientation sweep.** Same pixel grid and edge. Show Ns=8 candidate orientations as evenly spaced arrows radiating from the center pixel, each in 50% Black (#A2A2A2). At each angle, a small bar indicating the response magnitude. The winning orientation (max response) highlighted in Garnet (#73000A) with its arrow thickened. Contrast with 2.2a: brute-force evaluation at discrete angles instead of analytic interpolation. Label: "WVF/ASF (this work)."

### 2.3 Classical Edge Detection
- Fixed-kernel operators: Sobel (1968), Prewitt (1970), Roberts (1963), Kirsch (1971), Frei-Chen (1977). All use small (3x3 or 5x5) hand-designed convolution masks.
- Gaussian derivatives and scale-space: Marr & Hildreth (1980), Canny (1986), Lindeberg (1998). Multi-scale edge detection via Gaussian smoothing followed by derivative estimation.
- Key limitation shared by all: fixed spatial support that cannot adapt to noise level or edge scale.

> **Fig 2.3a (CeTZ): Roberts Cross operator.** Two 2x2 kernel weight grids side by side, showing the diagonal difference masks: [[+1, 0], [0, -1]] and [[0, +1], [-1, 0]]. Cells colored by weight value (Garnet for +1, Atlantic for -1, white for 0). Below: a 2x2 region highlighted on a small pixel grid showing the operator applied. Label: "Roberts, 1963." Annotate: 4 pixels, 2 multiplications per kernel, diagonal gradient only.

> **Fig 2.3b (CeTZ): Sobel operator.** Two 3x3 kernel weight grids (Gx and Gy). Gx = [[-1,0,+1],[-2,0,+2],[-1,0,+1]], Gy transposed. Cells colored by weight magnitude (darker = larger weight, sign indicated by Garnet +/Atlantic -). Show the gradient magnitude formula G = sqrt(Gx^2 + Gy^2) and angle theta = arctan(Gy/Gx). A 3x3 region highlighted on a pixel grid. Label: "Sobel, 1968." Annotate: 9 pixels, fixed weights, two directions only.

> **Fig 2.3c (CeTZ): Prewitt operator.** Same layout as Sobel but with Prewitt weights [[-1,0,+1],[-1,0,+1],[-1,0,+1]]. Briefly contrast: uniform row weighting vs Sobel's center-weighted rows. Label: "Prewitt, 1970."

> **Fig 2.3d (CeTZ): Kirsch compass operator.** Show all 8 compass kernel masks arranged in a circle (or show 4 representative ones: 0, 45, 90, 135 degrees). Each is a 3x3 grid with weights colored. The center shows the max-response selection: "k* = argmax |R_k|." Label: "Kirsch, 1971." Key point: this is an early orientation sweep over 8 discrete angles, conceptually similar to the WVF's Ns-direction sweep.

> **Fig 2.3e (CeTZ): Frei-Chen operator.** Show the 9 basis masks (or a representative subset of 4-5) as small 3x3 weight grids. Group them by type: edge subspace (3 masks), line subspace (3 masks), average (1 mask). Indicate that the image patch is projected onto these subspaces. Label: "Frei & Chen, 1977." Key point: subspace decomposition, a precursor to learned basis ideas.

> **Fig 2.3f (CeTZ): Gaussian derivative (DoG / LoG).** Left: a 1D Gaussian curve, its first derivative (odd-symmetric), and its second derivative (Mexican hat / LoG). Right: the 2D versions as kernel weight grids (~7x7 or ~9x9), colored red/blue for positive/negative with smooth gradation. Show sigma as a parameter controlling the support size. Label: "Marr & Hildreth, 1980." Annotate: support size grows with sigma, but shape is fixed (isotropic).

> **Fig 2.3g (CeTZ): Canny detector pipeline.** A horizontal flowchart with 4-5 stages: (1) Gaussian smoothing (show a blurred grid), (2) gradient computation (show Gx, Gy arrows), (3) arctan angle estimation (show angle errors at non-cardinal directions), (4) non-maximum suppression (thin edges), (5) hysteresis thresholding (connected edges). Each stage is a small illustrative panel. Use Garnet (#73000A) for the final edge pixels. Label: "Canny, 1986." Annotate: the smoothing-before-gradient tradeoff — stronger smoothing kills weak edges.

### 2.4 Deep Learning Edge Detection
- HED (Xie & Tu, 2015): deeply supervised multi-scale feature fusion
- DexiNed (Poma et al., 2020): dense extreme inception blocks, no pre-training required
- TEED (Soria et al., 2023): tiny efficient encoder-decoder
- PIDINet (Su et al., 2021): pixel difference convolutions
- NBED (Chen et al., 2024): neural-network based edge detector
- DiffusionEdge (Ye et al., 2024): diffusion probabilistic model for edge detection
- Common limitations: training data dependency, domain shift, opacity of learned representations, no formal gradient accuracy guarantees

> **Fig 2.4a (CeTZ): HED architecture (simplified).** A schematic encoder showing an input image shrinking through 5 conv stages (boxes getting smaller and deeper). At each stage, a side output branch produces an edge map (shown as progressively coarser edge maps). All side outputs are fused into a final edge map. Use layered rectangles for feature maps, arrows for connections. Garnet (#73000A) for side output arrows, Atlantic (#466A9F) for the main backbone. Label: "HED (Xie & Tu, 2015)." Key point: multi-scale deeply supervised fusion.

> **Fig 2.4b (CeTZ): DexiNed architecture (simplified).** Similar encoder-decoder schematic but emphasize the dense inception blocks (show a small inset of parallel convolution paths with different kernel sizes merging). 6 side outputs fused. Label: "DexiNed (Poma et al., 2020)." Key point: no ImageNet pre-training, trained from scratch on edge data.

> **Fig 2.4c (CeTZ): PIDINet pixel difference convolutions.** Show a 3x3 pixel neighborhood. Instead of learned weights multiplied by pixel values, show pixel *differences* (arrows between center and neighbors, labeled with the difference values). A small learned weight is applied to each difference. Contrast with Sobel: PIDiNet's first layer is structurally similar to a classical gradient operator but with learned coefficients. Label: "PIDiNet (Su et al., 2021)." Key point: hybrid classical-learned approach, explicitly encodes gradient structure.

> **Fig 2.4d (CeTZ): DL generalization problem.** Two panels. Left ("In-domain"): clean natural image → sharp edge map, labeled "BSDS500 trained." Right ("Out-of-domain"): noisy/underwater image → broken, noisy edge map from the same model. Use Garnet (#73000A) for correct edges, 50% Black (#A2A2A2) for missed/spurious edges. Label: "Domain shift." Key point: learned models fail outside their training distribution.

### 2.5 GPU Acceleration for Image Filtering
- Standard convolution via cuDNN and cuBLAS
- Custom kernel approaches: Triton (Tillet et al., 2019), CUTLASS
- Relevance: the fused stencil and geometric kernels are both expressible as gather-dot-product operations, enabling a single variant-agnostic Triton kernel

> **Fig 2.5a (CeTZ): Convolution as gather-multiply-sum.** A pixel grid with a target pixel. Arrows from stencil neighbor positions gathering intensity values into a vector. A weight vector alongside. Element-wise multiply shown, then sum to produce a single output value. Two variants side by side: left shows a regular 3x3 convolution grid (cuDNN), right shows an irregular stencil pattern (the ASF's non-rectangular footprint). Both produce the same operation type: dot product of weights and gathered values. Label: "Gather-dot-product abstraction." Key point: all three proposed filter variants reduce to this same operation, enabling a single GPU kernel.


## Section 3: Mathematical Framework (~3 pages)

### 3.1 Local Polynomial Model
- Consider grayscale image f: Z^2 → R and target pixel at global coordinates (X_0, Y_0)
- Define a local coordinate system (x, y) at each candidate orientation theta_k, with x along the edge normal and y along the edge tangent
- Rotation equations: x_i = (X_i - X_0) cos(theta_k) + (Y_i - Y_0) sin(theta_k), etc.
- Model local intensity with a 2D Taylor expansion of order d
- f(x_i, y_i) ≈ f^0 + f_x x_i + f_y y_i + (f_xx / 2) x_i^2 + (f_yy / 2) y_i^2 + f_xy x_i y_i + ...
- M = (d+1)(d+2)/2 unknown coefficients (6 for d=2, 15 for d=4)
- Explicitly note the connection to Savitzky-Golay: this is the same polynomial basis, extended to 2D with rotated coordinates

### 3.2 Least-Squares Gradient Estimation
- Construct design matrix A ∈ R^(Np × M) from Taylor monomials evaluated at each neighbor's local coordinates
- Columns correspond to normalized monomials: 1, x, y, x^2/2, y^2/2, xy, ...
- With Np ≫ M (overdetermined system), solve via ordinary least squares
- ĉ^(k) = (A^T A)^{-1} A^T f = P_{theta_k} f
- P_{theta_k} is the Moore-Penrose pseudoinverse — precomputable, depends only on geometry and theta_k, not on the image
- The normal derivative estimate is the second component: f̂_x = e_1^T ĉ = p_fx^(k) · f, where p_fx^(k) = P_{theta_k}[1, :] ∈ R^{Np}
- Key insight for later sections: the entire gradient estimation at one orientation reduces to a single dot product of a precomputed weight vector with the gathered intensity vector

### 3.3 Orientation Sweep and Maximum-Response Selection
- Evaluate the normal derivative f̂_x^(k) at Ns orientations theta_k = k · 2π/Ns for k = 0, ..., Ns-1
- Select the orientation with maximum absolute response: k* = argmax_k |f̂_x^(k)|
- Gradient magnitude = |f̂_x^(k*)|, edge angle = theta_{k*}
- Contrast with Sobel/Canny: those compute gradients in two fixed directions and combine via arctan, introducing systematic angular error at non-cardinal orientations. The orientation sweep evaluates the normal derivative directly at each candidate angle, eliminating this error.

### 3.4 Neighbor Selection
- Select Np nearest integer-coordinate pixels to the origin (excluding origin itself), yielding an approximately circular support region
- Radius r ≈ ceil(sqrt(Np/π)) + 1
- Overdetermination ratio Np/M controls the noise-resolution tradeoff
  - For Np=100, d=4 (M=15): ratio is 6.7x, substantial noise averaging
  - For Np=25, d=2 (M=6): ratio is 4.2x, less averaging but smaller footprint
- Unlike Gaussian pre-smoothing (which blurs isotropically), the polynomial fit preserves edge structure because the Taylor model can represent sharp transitions
- The neighbor set is orientation-independent (same circular set for all theta_k); only the coordinate rotation changes

### 3.5 Line Extension
- The point filter (Sections 3.1-3.4) struggles with low-contrast boundaries where the transition is too weak for a single neighborhood
- Extend the support along the candidate edge direction by evaluating the polynomial fit at (2m+1) virtual positions along a line:
  - (X_j, Y_j) = (X_0 + j cos(theta_k), Y_0 + j sin(theta_k)), j ∈ {-m, ..., m}
- At each virtual position j, the full polynomial fit is applied to extract the normal derivative f̂_x^(j,k)
- Combine via Gaussian-weighted averaging: R_k = Σ_{j=-m}^{m} w_j · f̂_x^(j,k), where w_j = exp(-j^2 / (2σ^2)), σ = m/2
- Maximum-response orientation selected as before: k* = argmax_k |R_k|
- Two benefits: (1) incorporates wider spatial context for noise robustness; (2) enforces coherence along the edge tangent, bridging small gaps
- Computational cost: (2m+1) independent polynomial fits per orientation per pixel, each gathering Np intensities. For m=7, that is 15 separate gather operations per orientation, with heavily overlapping pixel reads. This redundancy motivates the fused formulation in Section 4.


## Section 4: Fused Stencil Formulation (~2.5 pages)

### 4.1 Algebraic Collapse
- Start from the line-extended response: R_k = Σ_{j=-m}^{m} w_j · f̂_x^(j,k)
- Expand each f̂_x^(j,k) as a dot product: f̂_x^(j,k) = Σ_{i=1}^{Np} p_i^(k) · f(X_0 + δ_{j,i}^x, Y_0 + δ_{j,i}^y)
- Where δ_{j,i}^x = j cos(theta_k) + Δx_i, δ_{j,i}^y = j sin(theta_k) + Δy_i are combined line-offset plus neighbor-offset positions (rounded to integer coordinates)
- Substituting: R_k = Σ_{j=-m}^{m} w_j Σ_{i=1}^{Np} p_i^(k) · f(X_0 + δ_{j,i}^x, Y_0 + δ_{j,i}^y)
- This is a double sum over (2m+1) × Np terms, but it is a linear operation on pixel intensities
- Therefore it can be rewritten as a single weighted sum: R_k = Σ_{ℓ=1}^{N'_k} α_{k,ℓ} · f(X_0 + δ̃_ℓ^x, Y_0 + δ̃_ℓ^y)
- Where N'_k is the number of unique pixel positions and α_{k,ℓ} is the sum of w_j · p_i^(k) for all (j, i) pairs mapping to the same integer position ℓ
- This is the central result: the entire LF pipeline collapses into a single gather-dot-product per orientation

### 4.2 Deduplication
- The raw stencil contains (2m+1) × Np entries
- Many map to the same pixel, especially when the neighbor radius is large relative to m (neighboring line positions share most of their circular neighborhoods)
- Deduplication process: group entries by integer offset, sum corresponding weights, produce compressed stencil
- Table: stencil compression statistics

| m  | Np  | Raw size | Unique (mean) | Reduction |
|----|-----|----------|---------------|-----------|
| 1  | 100 | 300      | 128           | 57%       |
| 2  | 100 | 500      | 152           | 70%       |
| 7  | 100 | 1500     | 264           | 82%       |
| 14 | 100 | 2900     | 431           | 85%       |

- Reduction increases with m because larger line extensions create more overlap between neighboring circular neighborhoods

### 4.3 The Weight Cancellation Problem
- Examination of the fused weights α_{k,ℓ} reveals a structural inefficiency: a large fraction of weights are near zero
- Origin: each virtual pixel j along the line contributes a local polynomial fit. Adjacent polynomial fits (at positions j and j+1) share most of their neighborhood pixels. The polynomial bases at j and j+1 are shifted versions of each other. In the overlap region, the gradient-direction components tend to cancel while higher-order residuals persist.
- Formally: the weight at shared pixel ℓ is α_{k,ℓ} = Σ_{j : ℓ ∈ N_j} w_j · p_{i(j,ℓ)}^(k). Because adjacent fits assign weights of opposite sign to shared pixels (the pseudoinverse row for f_x changes sign as the pixel moves from one side of the center to the other), the summation produces extensive cancellation.
- Quantification: define weight efficiency η = (Σ |α_{k,ℓ}| for |α_{k,ℓ}| > ε · max|α|) / (Σ |α_{k,ℓ}|). For m=7, Np=20, the stencil has 91 unique positions but η ≈ 0.72, meaning ~28% of total absolute weight is carried by near-zero entries.
- For small ε (empirically ε ≈ 0.05), a large fraction of pixels carry weights satisfying |α_{k,ℓ}| < ε · max(|α_{k,ℓ}|)

### 4.4 Consequences of Cancellation
- **Wasted computation.** Pixels with negligible weight contribute floating-point operations but no useful signal. The GPU gathers their intensity values and multiplies by near-zero weights.
- **Irregular frequency response.** The effective kernel shape cannot be described by a simple parametric family. The weight distribution is irregular and orientation-dependent, meaning the effective kernel changes shape unpredictably from angle to angle.
- **Difficult theoretical analysis.** Standard tools for analyzing filter properties (transfer functions, noise gain, spatial resolution) assume well-behaved kernel shapes. The fused stencil's irregular weights make formal statements about noise rejection or orientation selectivity difficult.
- **The key question this raises:** If the polynomial fitting pipeline produces an effective kernel with many wasted weights and irregular shape, could we define a well-behaved kernel directly from geometric principles and achieve the same (or better) edge detection accuracy?

> **Fig 4.1 (CeTZ): Fused stencil weight maps.** Show the fused stencil footprint at 3-4 orientations (0°, 45°, 90°, 135°) for m=7, Np=100. Each pixel position is a cell colored by its fused weight α (diverging colormap: Garnet #73000A for positive, Atlantic #466A9F for negative, white for near-zero). The dipole pattern (positive on one side, negative on the other) rotates with theta. Highlight the large number of near-white (near-zero) cells. Use existing fig3_stencil_orientations.png as reference or replace with CeTZ version.

> **Fig 4.2 (CeTZ): Weight cancellation illustration.** Zoom into a small region of the fused stencil where two adjacent virtual neighborhoods overlap. Show the contribution from virtual pixel j (weights in Garnet) and virtual pixel j+1 (weights in Atlantic) at each shared pixel position. Show the summed weight at each position: many are near-zero because the two contributions have opposite signs. A histogram inset showing the distribution of |α| values, with a long tail near zero.

> **Fig 4.3 (CeTZ): Weight efficiency comparison.** Bar chart or table. Three bars: fused stencil (η ≈ 0.72), rectangular kernel (η > 0.99), elliptical kernel (η > 0.99). Garnet for the fused stencil bar, Atlantic for the geometric kernels. Alternatively: show N_eff (effective number of pixels) for each variant. Fused: N_eff ≈ 42 out of N'=91 unique. Rectangular: N_eff ≈ 48 out of 74. Elliptical: N_eff ≈ 39 out of 56. The geometric kernels achieve higher effective pixel utilization despite fewer total positions.

## Section 5: Geometric Kernel Alternatives (~2.5 pages)

### 5.1 Motivation
- Section 4 showed that the fused stencil, despite its computational advantages, inherits structural problems from the polynomial fitting pipeline: weight cancellation, irregular kernel shape, wasted computation
- The fused stencil's weight distribution is the output of the original author's design decisions, and was idneitifed via the algebraic process I worked on (pseudoinverse → line averaging → deduplication). 
- The kernel shape seems to not have been an intentional design decision during the development of the filter, and thus why is seemingly un-optimal.
- The core observation is such as follows: Assign a weight to each pixel based on (a) its position relative to the target pixel and (b) the candidate edge direction. If we can define that weight directly from the geometry — from the pixel's rotated coordinates and an analytic envelope function — then the polynomial fitting, pseudoinverse, and deduplication stages all become unnecessary.
- Both proposed geometric kernels begin with the same coordinate rotation as the fused stencil (Eq. from Section 3.1). For a candidate orientation theta, image coordinates (x, y) relative to the target pixel are transformed into the kernel frame:
  - u = x cos(theta) + y sin(theta)  (displacement along the edge direction)
  - v = -x sin(theta) + y cos(theta)  (displacement perpendicular, i.e. the gradient direction)
- The edge-sensitive response is obtained by multiplying a spatial envelope by the first derivative of the Gaussian in the normal direction, -v, which produces the characteristic dipole pattern (positive on one side of the edge, negative on the other)

### 5.2 Rectangular Gaussian Kernel
- The rectangular kernel confines its support to a hard-edged box aligned with the candidate edge direction
- The rectangular mask admits only pixels satisfying both |u| ≤ h_u and |v| ≤ h_v, where h_u is the half-width along the edge and h_v is the half-width across it
- Indicator function: 1_R(u, v) = 1_{|u| ≤ h_u} · 1_{|v| ≤ h_v}
- Inside the rectangle, pixel contributions are weighted by a 2D anisotropic Gaussian envelope:
  - G(u, v) = exp(-1/2 (u²/σ_u² + v²/σ_v²))
- The raw kernel before normalization is the product of the envelope, the derivative profile, and the mask:
  - K̂_R(u, v) = -v · G(u, v) · 1_R(u, v)
- The kernel is zero-centered (subtract mean) and normalized to unit absolute sum:
  - K_R(u, v) = (K̂_R(u, v) - K̄_R) / Σ |K̂_R(u, v) - K̄_R|
- Zero-centering ensures the filter has zero DC response (insensitive to constant intensity offsets). Absolute-sum normalization ensures responses are comparable across orientations and kernel sizes.
- Default parameters: h_u = 3σ_u, h_v = 3σ_v. With σ_u = 2.0 and σ_v = 1.2, the aspect ratio σ_u/σ_v ≈ 1.67, matching the elongation of the baseline ASF. Support area = 4 h_u h_v ≈ 86.4 square pixels.

### 5.3 Elliptical Gaussian Kernel
- The elliptical kernel replaces the hard rectangular boundary with a smooth elliptical mask derived from the Gaussian exponent
- The normalized elliptical distance from the kernel center is:
  - r(u, v) = sqrt(u²/σ_u² + v²/σ_v²)
- The mask admits pixels satisfying r(u, v) ≤ 3, corresponding to the 3σ boundary in both principal directions:
  - 1_E(u, v) = 1_{r(u, v) ≤ 3}
- The raw kernel: K̂_E(u, v) = -v · exp(-1/2 · r(u, v)²) · 1_E(u, v)
- Normalized K_E is obtained by the same zero-centering and absolute-sum normalization as the rectangular variant
- Support area = πσ_uσ_v · 9 ≈ 67.9 square pixels, roughly 21% smaller than the rectangle
- The elliptical mask excludes corner pixels near the bounding box that satisfy 1_R but not 1_E. These are pixels far from the kernel center in both directions simultaneously. Quantify: for the default parameters, |C| ≈ 18 pixels receive nonzero weight under the rectangular kernel and zero weight under the elliptical kernel.
- Practical effect: the elliptical kernel has a smoother spatial frequency response because the hard corners of the rectangular mask introduce discontinuities in the Fourier domain that manifest as sidelobes

### Why Uniform Weights Are Preferable
- If overlapping polynomial fits disagree on a pixel's contribution, maybe that pixel is genuinely ambiguous — sitting in a region where the local intensity surface is poorly constrained. The cancellation could be acting as a form of regularization, automatically downweighting unreliable pixels and concentrating weight on the most informative ones.
- This argument does not survive scrutiny for three reasons.
- **First, the cancellation is geometry-dependent, not content-dependent.** The same pixels receive near-zero weight regardless of the image content. A pixel in the overlap zone between virtual neighborhoods j and j+1 gets canceled whether it sits on a sharp edge, a smooth gradient, or pure noise. An adaptive scheme would adjust weights based on local signal quality; the fused stencil adjusts them based on the accident of which polynomial bases happen to overlap at that position. The cancellation pattern is determined entirely at precompute time and is fixed for all images.
- **Second, the canceled pixels are in the wrong place.** The overlap zones between adjacent virtual neighborhoods lie near the center of the stencil, close to the target pixel and close to the candidate edge. These are the most informative pixels for gradient estimation — they carry the strongest edge signal. Pixels far from the center (at the tips of the elongated stencil) receive less overlap and thus retain their weights. The cancellation therefore suppresses exactly the pixels that matter most and preserves the ones that matter least. True noise rejection would do the opposite: downweight distant, weakly informative pixels and upweight those near the edge.
- **Third, the empirical evidence is decisive.** The geometric kernels use no cancellation — every pixel in their support receives a nonzero, smoothly varying weight — and they match the fused stencil's edge detection accuracy within 0.3% ODS across all three benchmarks. If cancellation were providing useful regularization, removing it entirely should degrade accuracy. It does not. Moreover, the geometric kernels achieve lower noise gain (Section 5.6) and faster runtime (Section 6.4) precisely because they do not waste computation on near-zero weights.
- Cancellation in the fused stencil is a unintended artifact of the polynomial-fitting-then-deduplication pipeline. The geometric kernels demonstrate that the same edge detection accuracy is achievable with a smooth, well-behaved weight distribution where every pixel contributes meaningfully.

### 5.5 Effective Pixel Count and Weight Efficiency
- Define the effective number of pixels as N_eff = (Σ |α_{k,ℓ}|)² / (Σ α_{k,ℓ}²) = 1 / (||α||₂² / ||α||₁²)
- N_eff equals N'_k when all weights have equal magnitude and is smaller when the distribution is uneven
- This is a consequence of the Cauchy-Schwarz inequality: for a fixed total absolute weight ||α||₁ = 1, the noise gain ||α||₂² is minimized when all weights are equal (uniform kernel), giving the lowest noise and highest N_eff

| Variant           | N'_k (unique) | N_eff | η (weight efficiency) |
|-------------------|---------------|-------|-----------------------|
| Fused stencil     | 91            | 42    | 0.72                  |
| Rectangular kernel| 74            | 48    | > 0.99                |
| Elliptical kernel | 56            | 39    | > 0.99                |

- The rectangular kernel achieves the highest N_eff despite having fewer unique positions than the fused stencil, because it wastes almost no weight on near-zero entries
- The elliptical kernel has the fewest unique positions but still achieves η > 0.99

### 5.6 Noise Rejection Properties
- The noise gain of a linear filter is ||α||₂², the squared ℓ² norm of the weight vector
- For a fixed total absolute weight ||α||₁ = 1 (guaranteed by normalization), the noise gain is minimized when weight is spread uniformly over many pixels and maximized when weight is concentrated on a few
- The fused stencil's near-zero weight pixels increase N'_k without contributing to ||α||₁, but they do increase ||α||₂² if the remaining weights must compensate by carrying larger individual values
- The geometric kernels place appreciable weight on every pixel in their support, producing a more uniform weight distribution and lower noise gain for a given support size
- Scaling behavior under noise: at high SNR, the optimal configuration uses small kernels with few orientations (Ns=4, σ_u=2.0, σ_v=1.2). As noise increases, the optimal kernel grows larger and more densely sampled. The geometric kernels replicate this scaling by adjusting σ_u and σ_v directly. At SNR = 0.5 dB, the optimal ASF uses a massive elongated stencil spanning roughly 43 × 15 pixels. The equivalent geometric parameters are σ_u ≈ 7.17 and σ_v ≈ 2.50, producing an elliptical kernel with N_eff ≈ 340 and a rectangular kernel with N_eff ≈ 440.
- At clean (high-SNR) conditions, σ_u = 2.0 and σ_v = 1.2 suffice, matching the compact stencil of the optimized ASF
- The critical finding: the ASF's small noise advantage at extreme SNR disappears when the geometric kernels are scaled to match its effective support size. This indicates that noise robustness is a function of kernel area, not of polynomial order or the complexity of the weight-generation pipeline.

> **Fig 5.1 (CeTZ): Rectangular kernel construction.** Step-by-step on a pixel grid (~15x15). Panel A: the rectangular mask region outlined (hard box aligned at ~30° to horizontal), pixels inside highlighted in 10% Black (#ECECEC). Panel B: the Gaussian envelope shown as a smooth color gradient within the mask (darker at center, lighter toward edges), using a Garnet-to-white gradient. Panel C: the derivative profile -v shown as a cross-section alongside the kernel (positive above the center line in Garnet #73000A, negative below in Atlantic #466A9F). Panel D: the final combined kernel K_R on the pixel grid, each cell colored by its weight (Garnet positive, Atlantic negative, intensity proportional to magnitude). The dipole pattern is clean and regular. Label each panel with the corresponding equation term.

> **Fig 5.2 (CeTZ): Elliptical kernel construction.** Same step-by-step layout as Fig 5.1 but with the elliptical mask. Panel A: elliptical boundary drawn (smooth curve, ~30° orientation), corner pixels that would be included in the rectangle but excluded here marked with X or lighter shading. Panel B: Gaussian envelope (now naturally matches the elliptical boundary). Panel C: derivative profile (same as rectangular). Panel D: final kernel K_E. Visually compare to Fig 5.1D — smoother boundary, fewer pixels, but same dipole character. Annotate the ~18 corner pixels that differ between rectangular and elliptical.

> **Fig 5.3 (CeTZ): Side-by-side weight comparison (all three variants).** Three panels at the same orientation (e.g., 30°). Left: fused stencil weight map (irregular shape, many near-white cells, elongated blob). Center: rectangular kernel weight map (clean box, smooth dipole, no near-zero cells). Right: elliptical kernel weight map (clean ellipse, smooth dipole, slightly smaller). All on the same pixel grid scale. Same diverging colormap (Garnet/white/Atlantic). Below each: a 1D histogram of |α| values. The fused stencil histogram has a spike near zero; the geometric kernel histograms do not.

> **Fig 5.4 (CeTZ): Noise gain scaling.** Plot of noise gain (||α||₂²) vs effective support area for the three kernel variants. X-axis: support area in pixels. Y-axis: noise gain (log scale). Three curves: fused stencil (Garnet #73000A, dashed), rectangular (Atlantic #466A9F, solid), elliptical (Congaree #1F414D, solid). The theoretical minimum (1/N for uniform kernel) shown as a dotted 90% Black line. The geometric kernels track closer to the theoretical minimum. The fused stencil curve sits above due to wasted weight on near-zero pixels.


## Section 6: Unified GPU Implementation (~2 pages)

### 6.1 Architecture Overview
- Two-phase execution shared by all three filter variants
- **Precompute phase (CPU, one-time):** constructs orientation-indexed stencils and transfers them to GPU memory. Runs once per parameter configuration, not per image.
  - For the fused polynomial stencil: select Np circular neighbors → build Taylor design matrix A_{theta_k} for each orientation → compute pseudoinverse P_{theta_k} → extract gradient row p_fx^(k) → enumerate all (2m+1) × Np stencil positions → round to integer coordinates → deduplicate and sum weights → pack into padded arrays
  - For the rectangular kernel: for each orientation theta_k, enumerate pixel offsets within the bounding box → rotate into local (u, v) frame → evaluate K_R(u, v) = -v · G(u, v) · 1_R(u, v) → zero-center → normalize → pack
  - For the elliptical kernel: same as rectangular but with 1_E mask and elliptical distance check
- **Compute phase (GPU, per-image):** transfers image to device memory, applies stencils, writes gradient magnitude and angle to output buffers
- Despite very different construction procedures, all three variants produce the same output format: each orientation theta_k is represented by a list of integer offsets (Δx_ℓ, Δy_ℓ) and corresponding scalar weights α_{k,ℓ} for ℓ = 1, ..., N'_k
- This format uniformity is the key design property: a single GPU kernel processes any variant without modification

### 6.2 The Variant-Agnostic Kernel
- The compute-phase kernel consumes only (offset, weight) pairs per orientation. It has no knowledge of how the weights were generated.
- For every pixel (X_0, Y_0) and every orientation theta_k:
  1. Gather N'_k intensity values from the image at precomputed stencil offsets
  2. Multiply each by its corresponding weight
  3. Accumulate the sum to obtain directional response R_k
  4. Track maximum |R_k| and its index k* across all orientations
- Upon completion, write gradient magnitude |R_{k*}| and edge angle theta_{k*} to output buffers
- Switching between fused stencil, rectangular kernel, and elliptical kernel requires only swapping the precomputed stencil arrays. No recompilation or kernel modification is necessary.

### 6.3 Custom Triton Kernel
- Implemented using Triton, a Python-embedded GPU programming language that compiles to optimized PTX through LLVM
- Block size: 128 pixels along each image row. For each block, the kernel iterates over all Ns orientations, performing the stencil gather-dot-product and maintaining a running maximum across orientations.
- Inner loop for a single orientation (simplified pseudocode):
  ```
  for i in range(N_max):
      active = i < n_unique[k]
      dy, dx = load(offsets[k, i])
      w = load(weights[k, i])
      vals = load(image[row + dy, cols + dx])
      response += w * vals * active
  ```
- N_max is the maximum stencil size across all orientations. The `active` mask handles padding for orientations with fewer unique positions.
- Each iteration loads one (offset, weight) pair and gathers 128 intensity values in parallel (one per pixel in the block)
- Column-aligned memory access pattern ensures coalesced reads from global memory, critical for throughput on modern GPU architectures
- Triton's JIT compiler handles warp-level scheduling, loop unrolling for the inner stencil loop, and shared memory allocation

### 6.4 Memory Efficiency

| Method                       | Time (s) | VRAM (MB) | Speedup |
|------------------------------|----------|-----------|---------|
| Naive batched LF             | 2.43     | 6223      | 1.0×    |
| cuDNN conv2d                 | 0.18     | 158       | 13.8×   |
| Fused stencil (Triton)       | 0.13     | 20        | 18.4×   |
| Rectangular kernel (Triton)  | 0.045    | 20        | 54×     |
| Elliptical kernel (Triton)   | 0.047    | 20        | 52×     |

- All measurements on NVIDIA A100-SXM4-40GB, BIPED v1 (1280×720)
- Fused stencil parameters: m=7, Np=100, Ns=18, d=4. Geometric kernel parameters: σ_u=2.0, σ_v=1.2, Ns=36, 15×15 grid.
- The naive batched LF allocates large intermediate tensors for the (L × B × Np) gather operation: over 6 GB of VRAM, ~750M scattered float32 reads per orientation
- cuDNN conv2d reduces this via optimized convolution routines but still requires 158 MB workspace
- The fused stencil eliminates all intermediate tensors: VRAM drops to 20 MB (image + output), a 311× reduction
- Both geometric kernels achieve the same 20 MB footprint
- The geometric kernels are ~3× faster than the fused stencil despite using twice as many orientations (Ns=36 vs 18). Three factors:
  1. Fewer unique positions per orientation (~74 for 15×15 grid vs ~264 for fused at m=7, Np=100), directly reducing gather operations
  2. Stencil size determined by grid dimensions σ_u, σ_v rather than m × Np, so it does not grow with spatial extent
  3. Rectangular bounding box produces more regular memory access patterns than the elongated irregular fused stencil, improving cache utilization

### 6.5 Scaling Behavior
- The fused stencil's cost scales as O(Ns · N'_k), where N'_k grows sublinearly with m (due to deduplication). The naive approach scales as O(Ns · L · Np), where L = 2m+1 grows linearly.

| m  | Naive (s) | Fused (s) | Speedup | VRAM ratio |
|----|-----------|-----------|---------|------------|
| 1  | 0.58      | 0.072     | 8.0×    | 273×       |
| 7  | 2.43      | 0.13      | 18.4×   | 311×       |
| 14 | 8.88      | 0.37      | 24.3×   | 311×       |

- At m=1, the fused stencil is 8× faster. At m=14, the speedup reaches 24.3× because naive cost grows linearly with L while fused cost increases modestly.
- VRAM ratio stabilizes at 311× for m ≥ 7 (dominated by image + output, not stencil weights)
- The geometric kernels exhibit qualitatively different scaling: runtime is effectively constant with respect to the equivalent spatial extent. The rectangular kernel processes 1280×720 in ~45 ms regardless of equivalent m; the elliptical kernel is similarly stable at ~47 ms.
- This constancy arises because stencil size is fixed by grid dimensions, not by a line-extension parameter. For applications requiring large spatial support (e.g., high noise conditions), the geometric kernels offer not just faster absolute performance but predictable, parameter-independent runtime.

> **Fig 6.1 (CeTZ): Precompute and compute phase diagram.** Two-column layout. Left column ("Precompute — CPU, once"): three parallel paths, one for each variant. Fused stencil path: boxes for "Select Np neighbors" → "Build A_{theta}" → "Pseudoinverse P_{theta}" → "Extract p_fx" → "Line offsets" → "Deduplicate." Rectangular path: "Define mask 1_R" → "Evaluate G(u,v)·(-v)" → "Normalize." Elliptical path: "Define mask 1_E" → "Evaluate G(u,v)·(-v)" → "Normalize." All three paths converge into a single output box: "(offsets, weights) per orientation." Right column ("Compute — GPU, per image"): single path for all variants: "Load image" → "For each pixel, for each theta: gather → dot product → track max" → "Output: magnitude, angle." Use Garnet (#73000A) for fused stencil path, Atlantic (#466A9F) for rectangular, Congaree (#1F414D) for elliptical, 90% Black (#363636) for the shared compute path. Arrows connecting precompute output to compute input.

> **Fig 6.2 (CeTZ): Runtime scaling plot.** X-axis: line half-width m (or equivalent spatial extent for geometric kernels). Y-axis: runtime in seconds (log scale). Four curves: naive batched LF (90% Black #363636, dashed), fused stencil (Garnet #73000A), rectangular kernel (Atlantic #466A9F), elliptical kernel (Congaree #1F414D). The naive curve rises steeply. The fused curve rises gently. The geometric kernel curves are flat horizontal lines. A shaded region between the geometric and fused curves labeled "3× gap." Annotate the VRAM at key points. Use existing fig4_speedup_vram.png data as reference.

> **Fig 6.3 (CeTZ): Memory access pattern comparison.** Two panels showing a small image region (~10×10 pixels). Left ("Fused stencil"): the irregular stencil footprint overlaid, with arrows showing scattered gather operations to non-contiguous memory locations. Some arrows cross, indicating cache-unfriendly access. Right ("Geometric kernel"): the rectangular footprint overlaid, with arrows showing a regular row-by-row scan pattern. Arrows are parallel and sequential, indicating coalesced, cache-friendly access. Label: "Coalesced vs scattered memory access."

## Section 7: Parameter Analysis (~2.5 pages)

### 7.1 Methodology and Scope
- All parameter ablations in this section were conducted using the naive (unfused) WVF and LF implementations, prior to the derivation of the fused stencil and geometric kernel formulations. Because the fused stencil is mathematically equivalent to the naive LF (Section 4.1), and the geometric kernels are evaluated separately in Section 8, the parameter findings transfer directly.
- The naive implementation's high computational cost (Section 6.4) was the primary constraint on ablation scope. Each full-dataset LF evaluation at Bagan & Wang's published parameters (Np=250, Ns=18, m=14, d=4) required ~2.4 seconds per image on an A100 GPU, making exhaustive grid search over the full parameter space infeasible. The fused stencil and geometric kernels, derived after these ablations were complete, would have reduced per-evaluation cost by 18–54× and enabled a much denser search. We note this as a limitation and identify priority ablations for future work.
- Total experimental scope: 1,206 WVF configurations and 168 LF configurations evaluated across 4 datasets (UDED 30 images, BIPED v1 50 images, BIPED v2 50 images, BSDS500 200 images), totaling 330 images and over 500,000 individual filter evaluations.

### 7.2 Support Size (Np)
- **Why we ablated this:** Np is the most computationally consequential parameter. It directly controls the number of memory reads per pixel per orientation (the dominant cost term) and determines the physical size of the filter footprint. Bagan & Wang's recommended Np=250 implies a circular neighborhood of radius ~9 pixels. If smaller values suffice, both accuracy and speed improve simultaneously, which is unusual — most parameters trade one for the other - However, this mystery wil be illumniated when doing SNR-based studies; higher noise requires larger filter.
- **What we tested:** Np ∈ {10, 15, 25, 50, 75, 100, 150, 200, 250, 300, 400, 500} for the WVF; a subset for the LF due to cost constraints
- **Finding:** On clean imagery, optimal Np lies in the range 25–100 depending on dataset. ODS improvements of 0.01–0.05 over Np=250 for the WVF and 0.06–0.16 for the LF. Larger neighborhoods average over too many pixels, smoothing out fine edge structure that the evaluation protocol rewards.
- **Under noise:** The optimal Np grows monotonically with noise severity. At high SNR, small neighborhoods (Np=25–50) are optimal because they preserve fine spatial detail. As SNR decreases, the noise averaging benefit of larger neighborhoods outweighs the loss of spatial resolution, and the optimal Np shifts to 250–500. This relationship between noise level and optimal filter size is consistent with classical results in statistical estimation: the bias-variance tradeoff shifts toward variance reduction (larger support) as observation noise increases. Bagan & Wang's large-support parameterization reflects this — their maritime application operates in a regime where noise dominates and large Np is appropriate.

### 7.3 Polynomial Order (d)
- **Why we ablated this:** d controls the expressiveness of the local intensity model. Higher d captures more complex local structure (curvature, inflection) but requires more unknowns (M = 15 for d=4 vs M = 6 for d=2), reducing the overdetermination ratio Np/M and thus the noise averaging. Bagan & Wang use d=4 universally. We hypothesized that d=2 would suffice for gradient estimation (first derivative) since higher-order terms do not contribute to the f_x coefficient — they only affect the conditioning of the least-squares system.
- **What we tested:** d ∈ {2, 3, 4} at each Np and Ns combination
- **Finding:** d=2 consistently outperforms d=4 on clean imagery across all four datasets. The margin is small (0.005–0.02 ODS) but consistent. At d=2 with Np=25, the system is 4.2× overdetermined; at d=4 with Np=25, it is only 1.67× overdetermined, providing far less noise averaging.
- **Why we did not test d=1:** A first-order polynomial (d=1, M=3) reduces the Taylor model to a plane fit, which is equivalent to a weighted average of finite differences. This loses the ability to distinguish edge curvature from noise and would collapse the method to something resembling a weighted Sobel operator. We considered this outside the interesting range.

### 7.4 Orientation Count (Ns)
- **Why we ablated this:** Ns determines the angular resolution of the orientation sweep. Bagan & Wang use Ns=18 (20° spacing). More orientations improve angular precision but multiply computational cost linearly. If performance saturates at low Ns, the savings are substantial.
- **What we tested:** Ns ∈ {2, 4, 6, 8, 12, 18, 24, 36}
- **Finding:** Performance saturates at Ns=4–6 on all datasets. Going from Ns=6 to Ns=18 yields < 0.002 ODS improvement while tripling computation. This is consistent with the angular precision of the polynomial fit: the least-squares derivative estimate is already smooth over angle, so the discrete maximum is well-resolved even with coarse sampling. 
- **Why we did not test Ns=1:** A single orientation reduces the filter to a fixed-direction gradient operator, losing the orientation-selective property entirely. This was tested implicitly by evaluating Sobel/Prewitt (which are effectively Ns=2 with arctan combination) in the classical baseline comparison.

### 7.5 Line Half-Width (m)
- **Why we ablated this:** m controls the spatial extent of the LF's line extension and is the parameter unique to the LF (the WVF is the m=0 special case). Bagan & Wang use m=14, creating a line of 29 virtual evaluation points. This is the most expensive parameter: it multiplies the number of polynomial fits per pixel per orientation by (2m+1). Kruskal-Wallis analysis from Report 4 showed m has the highest effect size of any parameter (η²=0.34).
- **What we tested:** m ∈ {0, 1, 2, 3, 5, 7, 10, 14} at selected (Np, Ns, d) combinations
- **Finding:** On clean data, the WVF (m=0) matches or exceeds the LF across all four datasets. The line extension provides no benefit on clean imagery — the additional spatial context it provides is unnecessary when noise is low and edges are well-defined. Under noise, moderate values (m=2–7) help; m=14 is excessive even at low SNR.
- **Why the LF ablation grid is sparser (168 vs 1,206 configs):** Each LF evaluation is (2m+1)× more expensive than the corresponding WVF evaluation. At m=14, a single full-dataset LF run takes ~36 seconds per image. Exhaustive search was computationally prohibitive, so we sampled a representative subset of (Np, Ns, d) combinations at each m value.

### 7.6 Geometric Kernel Parameters (σ_u, σ_v)
- **Why we ablated this:** σ_u and σ_v control the spatial extent of the geometric kernels along and across the edge direction, respectively. They play an analogous role to Np and m in the polynomial filter but with cleaner semantics: σ_u directly sets the edge-parallel extent, σ_v sets the edge-normal extent, and the aspect ratio σ_u/σ_v controls the anisotropy.
- **What we tested:** σ_u ∈ {1.0, 1.5, 2.0, 3.0, 5.0, 7.0}, σ_v ∈ {0.8, 1.0, 1.2, 1.5, 2.0, 2.5}, with the grid resolution fixed at ceil(3σ) in each direction
- **Finding:** On clean data, σ_u=2.0, σ_v=1.2 is optimal across all datasets, matching the compact stencil of the optimized polynomial filter. Under noise, both parameters grow: at SNR=0.5 dB, optimal values are σ_u ≈ 7.2, σ_v ≈ 2.5.
- **Equivalence to polynomial parameters:** σ_u ≈ m (line half-width in pixel units), σ_v ≈ sqrt(Np/π) (effective neighborhood radius). This mapping allows direct comparison of geometric and polynomial parameter sweeps.

### 7.7 Published vs Optimal Parameters
- Bagan & Wang's recommended settings (Np=250, Ns=18, d=4, m=14) are suboptimal on every clean-imagery benchmark tested, by margins of 0.01–0.16 ODS
- The optimal clean-data configuration (Np=25–50, Ns=4–6, d=2, m=0) is simultaneously more accurate and 112× cheaper to compute - AGAIN, i think would be best to make the claim that the filter is HIGHLY paramater sensitive, and will 
- Under noise, the published parameters are partially vindicated: large Np and nonzero m become beneficial below SNR ≈ 5–10 dB. However, even under noise, d=4 does not outperform d=2, and Ns=18 remains unnecessary.
- Interpretation: the published parameters were tuned for noise-dominated maritime conditions and are unnecessarily conservative for standard computer vision benchmarks.

### 7.8 Ablations Not Conducted and Future Work
- **Polynomial basis type:** We tested only the standard monomial basis (1, x, y, x²/2, ...). Orthogonal polynomial bases (Legendre, Chebyshev, Zernike) could improve the conditioning of the design matrix A and potentially change the optimal d. This is particularly relevant for large Np where the monomial Vandermonde matrix becomes ill-conditioned. We did not test alternative bases because Bagan & Wang's formulation specifies the monomial basis and our goal was to evaluate their method as published before proposing alternatives.
- **Weighted least squares:** The current formulation uses uniform (unweighted) least squares. Distance-weighted fitting (as in LOESS) would downweight distant neighbors, creating a smooth taper rather than the hard circular cutoff. This could improve accuracy at the edges of the support region. Not tested because it would introduce additional hyperparameters (kernel bandwidth, kernel shape) and the geometric kernel variants already achieve the desired smooth taper through the Gaussian envelope.
- **Adaptive parameter selection:** All results use a fixed parameter setting across all pixels in an image. Local adaptation (e.g., selecting Np based on estimated local SNR) could improve performance in images with spatially varying noise. Not tested due to the complexity of the estimation and the need for a reliable local SNR estimator.
- **Joint Np-m-d optimization:** Our ablation varied parameters semi-independently due to computational cost. A full joint grid search over (Np × Ns × d × m) at the full-dataset level would require ~50,000 LF evaluations per dataset. At ~2.4 seconds each with the naive implementation, this is ~33 GPU-hours per dataset. The fused stencil (0.13 s/eval) would reduce this to ~1.8 GPU-hours, and the geometric kernels (0.045 s/eval) to ~0.6 GPU-hours, making exhaustive joint optimization feasible. This is a high-priority future experiment enabled by the computational contributions of this paper. I would aso expect the gemoetric and ASF to reach the same speed as number of images increase since we only need to computer th e kernels/filters once per oritnation and can be re-used. 
- **Geometric kernel ablation on noise:** The σ_u, σ_v sweep under noise was conducted at a smaller scale than the polynomial parameter sweep. A full noise × geometric parameter ablation matching the scope of the polynomial noise study (5 noise types × 7 SNR levels) is a clear next step.
- **Post-processing interaction:** We tested NMS (non-maximum suppression) and hysteresis thresholding on the raw gradient maps and found both degrade ODS by 0.06–0.09. We did not exhaustively search post-processing parameters (NMS radius, hysteresis thresholds) because the degradation was consistent across all settings tested, suggesting a fundamental mismatch between the thick gradient maps produced by large-support filters and the thin-edge assumption of NMS.

## Section 8: Experimental Evaluation (~3.5 pages)

### 8.1 Datasets
- **BSDS500** (Arbelaez et al., 2011): 200 test images, 481×321 pixels. Multiple human-annotated ground truth boundaries per image. The standard benchmark for contour detection. Important caveat: BSDS500 ground truth encodes *semantic* boundaries (object contours, scene transitions), not purely photometric edges. This biases evaluation toward methods that can learn semantic context, which inherently disadvantages non-learned methods.
- **BIPED v1** (Poma et al., 2020): 50 test images, 1280×720 pixels. Outdoor urban scenes with high-resolution edge annotations. Single annotator. Edges are predominantly photometric (sharp intensity transitions at object boundaries, text, structural lines). More favorable to gradient-based methods than BSDS500.
- **UDED** (Soria, 2022): 30 images sampled from 15 diverse source datasets (BIPED, BSDS500, Cityscapes, ADE20K, NYUD, DIV2K, WIRE-FRAME, CID, MDBD, THANGKA, PASCAL-Context, SET14, URBAN10, and others). Images selected via intensity IQR to maximize diversity, capped at 720×720 pixels. Designed specifically as a cross-domain generalization benchmark for edge detection, not tied to any single imaging domain.
- **BIPED v2** (Soria et al., 2023): 50 test images, same resolution as v1 but with revised annotations and additional scenes. Used for cross-validation of BIPED v1 findings; results reported in supplementary where they differ.
- **Why these four datasets:** They span a range of image characteristics (resolution, annotation style, imaging domains) and difficulty levels. BSDS500 is included because it is the de facto standard despite its semantic bias. BIPED provides high-resolution photometric ground truth. UDED tests cross-domain generalization by sampling from 15 diverse source datasets. Together, they allow us to assess whether findings generalize across imaging domains or are dataset-specific. (Spoiler from Report 4: BSDS500 is a strong outlier — Spearman rank correlations between BSDS500 and the other three datasets are ρ=0.03–0.17, while correlations among the other three are ρ=0.55–0.78.)

### 8.2 Evaluation Protocol
- Standard BSDS500 evaluation protocol used for all datasets
- Sweep 1,001 thresholds on the gradient magnitude map (evenly spaced in [0, 1])
- At each threshold, binarize the gradient map and compute precision and recall against ground truth using a 3-pixel match radius (the standard BSDS tolerance)
- **ODS (Optimal Dataset Scale) F-score:** best single threshold across all images in the dataset. Measures how well a single global threshold serves the entire dataset.
- **OIS (Optimal Image Scale) F-score:** best per-image threshold, averaged across images. Upper bound on performance with oracle per-image thresholding.
- Both metrics reported; ODS is the primary comparison metric throughout the paper because it reflects practical deployment (where a single threshold must be chosen).
- All gradient magnitude maps are evaluated as raw continuous outputs. No non-maximum suppression or hysteresis thresholding applied (Section 7.8 showed these degrade ODS for large-support filters).

### 8.3 Comparison Methods

**Classical filters (54 configurations):**
- Sobel: kernel sizes 3×3, 5×5, 7×7, 9×9, 11×11
- Prewitt: kernel sizes 3×3, 5×5
- Scharr: 3×3 (optimized Sobel variant)
- Roberts Cross: 2×2
- Kirsch compass: 3×3, 8 orientations
- Gaussian derivatives: σ ∈ {0.5, 1.0, 1.5, 2.0}, 1st and 2nd order
- Laplacian of Gaussian: σ ∈ {1.0, 1.5, 2.0}
- Difference of Gaussians: multiple σ pairs
- Steerable filters: 1st and 2nd order (Freeman & Adelson basis)
- Frei-Chen: 3×3 subspace decomposition
- Marr-Hildreth: σ ∈ {1.0, 1.5, 2.0}
- Each configuration evaluated at its best parameter setting per dataset. The best classical result per dataset is reported for comparison.

**Deep learning models (5):**
- DexiNed (Poma et al., 2020): pre-trained on BIPED, no ImageNet initialization
- TEED (Soria et al., 2023): lightweight encoder-decoder, pre-trained on BIPED v2
- PIDINet (Su et al., 2021): pixel difference convolutions, pre-trained on BSDS500
- NBED (Chen et al., 2024): neural-network based, pre-trained on BSDS500
- DiffusionEdge (Ye et al., 2024): diffusion model, pre-trained on BSDS500
- All models evaluated using their published pre-trained weights. No fine-tuning or domain adaptation applied. This represents the standard "download and run" deployment scenario.

**Proposed methods (3 variants, optimized parameters):**
- Fused polynomial stencil (ASF-poly): best parameters from Section 7 ablation (Np=25, Ns=4, d=2, m=0 for clean; Np=100, Ns=8, d=2, m=7 for noise)
- Rectangular kernel (ASF-rect): σ_u=2.0, σ_v=1.2, Ns=36 for clean
- Elliptical kernel (ASF-ellip): σ_u=2.0, σ_v=1.2, Ns=36 for clean

### 8.4 Clean-Data ODS Results

Main comparison table:

| Method               | BSDS500 | BIPED v1 | UDED  |
|----------------------|---------|----------|-------|
| *Classical (best)*   |         |          |       |
| Sobel (best)         | 0.632   | 0.761    | 0.863 |
| Prewitt (best)       | 0.623   | 0.772    | 0.869 |
| Kirsch                | 0.622   | 0.756    | 0.861 |
| Gaussian Deriv.      | 0.630   | 0.752    | 0.865 |
| *Proposed*           |         |          |       |
| ASF-poly (point)     | 0.682   | 0.812    | 0.899 |
| ASF-rect             | ~0.680  | ~0.810   | ~0.897|
| ASF-ellip            | ~0.679  | ~0.809   | ~0.896|
| *Deep learning*      |         |          |       |
| TEED                 | 0.680   | 0.851    | 0.925 |
| DexiNed              | 0.700   | 0.903    | 0.932 |
| PIDINet              | 0.865   | 0.836    | 0.863 |
| NBED                 | 0.790   | 0.908    | 0.897 |
| DiffusionEdge        | 0.745   | 0.909    | 0.904 |

- All three ASF variants produce nearly identical ODS on clean data (within 0.3% of each other), confirming that the geometric kernels match polynomial fitting accuracy
- The ASF outperforms all 54 classical configurations on every dataset: +5.0 ODS points over Sobel on BSDS500, +4.0 on BIPED v1, +3.0 on UDED
- On UDED (cross-domain generalization benchmark), the ASF nearly matches DexiNed (0.899 vs 0.932) and outperforms PIDINet (0.863) — a trained model — without any training data
- On BSDS500, the gap to the best DL model (PIDINet, 0.865) is larger, reflecting the semantic boundary bias in BSDS500 ground truth. This is an inherent limitation of non-learned methods, not a failure of the filter design.
- Note: geometric kernel ODS values marked with ~ are preliminary and will be finalized from full evaluation runs

### 8.5 Noise Robustness

**8.5.1 Noise Models**
- Gaussian (additive white): the standard noise model. Controlled via standard deviation σ_n relative to image dynamic range. SNR = 20 log10(signal_std / σ_n).
- Salt-and-pepper (impulse): random pixels set to 0 or 255. Controlled via corruption probability p.
- Poisson (shot): signal-dependent noise, variance proportional to intensity. Common in low-light and photon-counting imaging.
- Speckle (multiplicative): noise proportional to signal intensity. Common in SAR, ultrasound, and coherent imaging systems.
- Uniform (quantization): additive uniform noise over [-a, a]. Models quantization error and low-bit digitization.
- All noise types parameterized to a common SNR scale for cross-comparison.

**8.5.2 Experimental Design**
- 5 representative images per dataset (20 total), selected to span difficulty levels
- 7 SNR levels: clean, 20 dB, 15 dB, 10 dB, 5 dB, 1 dB, 0.5 dB
- 5 noise types × 7 levels × 20 images = 700 noisy evaluation conditions
- Each condition evaluated with the ASF at multiple parameter settings (from Section 7 ablation) and all 5 DL models at their fixed pre-trained weights
- Limitation: noise ablation uses 5 images per dataset, not full datasets. Acknowledged as a constraint from the naive implementation's computational cost. The fused stencil and geometric kernels make full-dataset noise ablation feasible as future work (Section 7.8).

**8.5.3 DL Degradation Under Noise**
- All five DL models lose 50–60% of their clean-data ODS under severe noise (SNR ≤ 1 dB)
- The degradation is not graceful: performance drops sharply between SNR=10 dB and SNR=5 dB for most models, suggesting the models have a narrow noise tolerance band
- DiffusionEdge degrades most slowly (diffusion-based denoising provides implicit robustness) but still falls below the clean-data ASF at SNR ≈ 3 dB
- PIDINet degrades fastest, consistent with its shallow architecture and pixel-difference first layer (which amplifies noise similarly to classical gradient operators)
- None of the DL models were trained with noise augmentation. Models fine-tuned on noisy data would likely perform better, but this represents additional training effort and domain-specific knowledge that the ASF does not require.

**8.5.4 ASF Behavior Under Noise**
- The ASF's noise robustness is controlled by its parameter settings, not by training data
- At fixed (clean-optimal) parameters, the ASF also degrades under noise — but the degradation can be counteracted by increasing Np, m (or equivalently σ_u, σ_v for geometric kernels)
- The relationship is monotonic: larger support → more noise averaging → better performance under noise, at the cost of spatial resolution
- At SNR ≤ 1 dB with noise-optimal parameters (Np=250–500, m=7), the ASF overtakes all tested DL models
- This is the core practical advantage: the ASF can be tuned to the operating noise level, while DL models are fixed at their training-time noise assumptions

**8.5.5 Crossover Analysis**
- Define the crossover SNR as the noise level at which the optimally-tuned ASF first exceeds a given DL model's ODS
- Crossover SNR varies by DL model and noise type:
  - vs TEED: crossover at SNR ≈ 10–15 dB (TEED is weakest DL model)
  - vs PIDINet: crossover at SNR ≈ 8–12 dB
  - vs DexiNed: crossover at SNR ≈ 3–5 dB
  - vs DiffusionEdge: crossover at SNR ≈ 1–3 dB (most robust DL model)
- Gaussian noise produces the highest crossover SNR (easiest for DL to handle); speckle and salt-and-pepper produce the lowest (DL degrades fastest on multiplicative and impulse noise)
- Bootstrap 95% confidence intervals on the crossover SNR are reported (from Report 4 statistical analysis)

### 8.6 Runtime Comparison

| Method                      | BSDS500 (ms) | BIPED v1 (ms) | Device |
|-----------------------------|-------------- |----------------|--------|
| Sobel 3×3                   | < 1           | < 1            | GPU    |
| Canny                       | ~5            | ~15            | CPU    |
| ASF-rect (σ_u=2.0, Ns=36)  | ~18           | ~45            | GPU    |
| ASF-ellip (σ_u=2.0, Ns=36) | ~19           | ~47            | GPU    |
| ASF-poly (Np=25, m=0, Ns=4)| ~5            | ~12            | GPU    |
| ASF-poly (Np=100, m=7, Ns=18)| ~38         | ~132           | GPU    |
| DexiNed                     | ~12           | ~80            | GPU    |
| TEED                        | ~8            | ~30            | GPU    |
| PIDINet                     | ~10           | ~35            | GPU    |
| DiffusionEdge               | ~2000         | ~15000         | GPU    |

- All GPU timings on NVIDIA A100-SXM4-40GB
- The ASF in point-filter mode (Np=25, m=0, Ns=4) is comparable to lightweight DL models in speed while outperforming all classical methods in accuracy
- The geometric kernels at Ns=36 are slower than the point filter but faster than DexiNed on BIPED-sized images, while providing the anisotropic support that improves noise robustness
- DiffusionEdge is 2–3 orders of magnitude slower than all other methods due to the iterative diffusion process
- Key framing: the ASF occupies a previously empty region of the accuracy-speed Pareto frontier — faster than DL models of comparable accuracy, and more accurate than classical methods of comparable speed

### 8.7 Visual Results

> **Fig 8.1: Clean-data visual comparison (BSDS500).** 4-column layout repeated for 2–3 images. Columns: (A) input image, (B) ASF-poly gradient magnitude, (C) ASF-rect gradient magnitude, (D) best DL model (DexiNed) edge map. Use existing visual_results/BSDS500_*.png images. Caption emphasizes that ASF-poly and ASF-rect produce nearly identical gradients (confirming mathematical equivalence and geometric kernel accuracy match), while the DL model produces thinner, more semantic edges.

> **Fig 8.2: Clean-data visual comparison (BIPED v1).** Same layout for 2 BIPED images. Use existing visual_results/BIPED_RGB_*.png images. Caption highlights that the ASF captures fine structural details (sign text, wheel spokes, window frames) that small-kernel classical operators miss.

> **Fig 8.3: Clean-data visual comparison (UDED).** Same layout for 2 UDED images (cross-domain samples). Use existing visual_results/UDED_*.png images. Caption: the ASF generalizes across diverse imaging domains without any domain-specific training, approaching DexiNed's supervised performance on this cross-domain benchmark.

> **Fig 8.4: Geometric kernel edge detection demos.** 2-row layout. Row 1: rectangular kernel results (input, gradient magnitude, orientation map with theta encoded as hue and |R| as brightness). Row 2: elliptical kernel results on the same image. Extract from the rect_ellipse_filter.pdf demonstration images (ant photo). Caption compares the two kernel variants visually — very similar edge maps, slightly smoother orientation map for the elliptical kernel.

> **Fig 8.5: Noise robustness visual comparison.** A single image shown at 4 noise levels (clean, SNR=10 dB, SNR=5 dB, SNR=1 dB) with edge maps from: ASF-poly (noise-optimized params), ASF-rect (scaled σ), DexiNed, Sobel. At high SNR all methods perform similarly. At low SNR the DL and classical outputs become unusable while the ASF maintains coherent edge structure. Use Garnet (#73000A) tint for the ASF columns, Atlantic (#466A9F) for DL, 70% Black (#5C5C5C) for classical, to visually group the methods.

> **Fig 8.6 (CeTZ): Accuracy-speed Pareto plot.** X-axis: per-image runtime in ms (log scale). Y-axis: ODS F-score on BIPED v1. Each method is a labeled point. Classical methods cluster in the lower-left (fast, low accuracy). DL models in the upper-right quadrant (slower, high accuracy) except DiffusionEdge far off to the right. ASF variants in the upper-left (fast, high accuracy among non-learned). Draw the Pareto frontier. The ASF occupies the previously empty gap between classical and DL. Use brand colors: Garnet for ASF variants, Atlantic for DL, 70% Black for classical. Serif font labels.

## Section 9: Discussion (~2 pages)

### 9.1 What the Fused Stencil Reveals About Polynomial Fitting
- The fused stencil formulation was originally motivated as a computational optimization. But its most important contribution was that it reveals that the entire multi-step polynomial fitting pipeline (gather neighbors, rotate coordinates, build Vandermonde matrix, compute pseudoinverse, extract derivative row, repeat along line, weight, sum) is equivalent to a single linear operation on the image — a fixed convolution with an anisotropic, orientation-dependent kernel.
- This means the polynomial fitting machinery is not doing anything that a carefully designed convolution kernel cannot do. The Taylor expansion, the least-squares solve, the pseudoinverse — these are all mechanisms for *generating* kernel weights. Once the weights are generated, the polynomial model plays no further role.
- The weight cancellation problem (Section 4.3) is a direct consequence of this insight: the weights generated by the polynomial pipeline are not optimized for any edge detection objective. They are the algebraic byproduct of a fitting procedure designed for a different purpose (recovering smooth polynomial coefficients), repurposed for gradient estimation. The near-zero weights, irregular shape, and orientation-dependent support are symptoms of this mismatch.
- The geometric kernels represent the natural conclusion of this reasoning: if the polynomial machinery is only a weight-generation mechanism, and a suboptimal one at that, replace it with a weight-generation mechanism designed from the start for the task at hand.

### 9.2 The Noise-Resolution Tradeoff and Adaptive Filtering
- Every gradient estimator faces the same fundamental tradeoff: larger spatial support provides better noise averaging but worse spatial resolution. A 3×3 Sobel operator localizes edges to within 1 pixel but has almost no noise robustness. A WVF with Np=500 can detect edges in severe noise but localizes them only to within ~13 pixels.
- Classical methods (Sobel, Canny) address this by fixing the support size at design time and accepting the resulting tradeoff. DL methods learn an implicit tradeoff from training data that cannot be adjusted at deployment time.
- The ASF and geometric kernels offer a third option: explicit, continuous control over the tradeoff via parameters (Np or σ_u/σ_v) that can be set per-application or even per-image based on estimated noise conditions. This parametric adaptability is the ASF's primary practical advantage over both classical and learned methods.
- The results from Sections 7 and 8 quantify this tradeoff precisely. On clean imagery, small support is optimal (Np=25–50 or σ_u=2.0). Under severe noise, large support is needed (Np=250–500 or σ_u=7.0). The geometric kernels make this adjustment particularly natural because σ_u and σ_v have direct physical meaning (standard deviations of the spatial envelope in pixels).

### 9.3 Why Geometric Kernels Are the Preferred Variant
- All three filter variants (fused polynomial stencil, rectangular Gaussian kernel, elliptical Gaussian kernel) produce nearly identical edge detection accuracy on clean data (within 0.3% ODS). The choice between them is therefore driven by secondary criteria: computational cost, theoretical analyzability, parameter interpretability, and noise scaling behavior.
- On every secondary criterion, the geometric kernels are superior:
  - **Speed:** 3× faster than the fused stencil at matched accuracy (Section 6.4)
  - **Weight efficiency:** η > 0.99 vs η ≈ 0.72 for the fused stencil (Section 5.5)
  - **Analyzability:** the kernel shape is a known parametric function (Gaussian envelope × derivative profile), enabling closed-form analysis of frequency response, noise gain, and spatial resolution
  - **Parameter interpretability:** σ_u and σ_v have direct physical meaning; Np, d, and m interact in complex, non-obvious ways
  - **Noise scaling:** adjusting σ_u and σ_v produces predictable, monotonic changes in noise robustness and spatial resolution; adjusting Np and d has nonlinear interactions (Section 7)
  - **Constant-time execution:** runtime is independent of spatial extent, unlike the fused stencil whose cost grows with m (Section 6.5)
- Between the two geometric variants: the rectangular kernel has slightly higher N_eff (better noise rejection per unit computation) and is marginally faster. The elliptical kernel has a smoother frequency response and fewer sidelobe artifacts. For most applications, the difference is negligible (<0.1% ODS). We recommend the rectangular kernel as the default due to its computational advantages, with the elliptical kernel preferred when sidelobe suppression matters (e.g., texture-heavy images).

### 9.4 Semantic vs Photometric Edges
- The largest performance gap between the ASF and DL models occurs on BSDS500, where the best DL model (PIDINet, 0.865) leads the ASF (0.682) by 18.3 ODS points. On BIPED v1 and UDED, the gap narrows to 9–10 and 3 points respectively.
- This discrepancy is explained by the nature of the ground truth. BSDS500 annotations encode semantic boundaries — the contours that humans judge as meaningful object boundaries. These include boundaries defined by texture change, color change, or semantic context rather than by sharp intensity gradients. A non-learned gradient operator cannot detect a boundary between two regions of different texture but similar average intensity.
- BIPED and UDED annotations are more photometric — edges correspond to actual intensity transitions. On these datasets, the ASF is competitive with DL models because the task aligns with what gradient estimation measures.
- This is not a failure of the ASF; it is a fundamental limitation of any non-learned gradient operator. The ASF answers the question "where does intensity change sharply?" DL models answer the broader question "where do humans perceive boundaries?" These are different questions, and the gap between them defines the ceiling for non-learned edge detection on semantic benchmarks.

### 9.5 No Single Best Edge Detector
- The results from Sections 7 and 8 converge on a finding that the edge detection literature has been slow to acknowledge: there is no universally best edge detection method.
- On clean, high-resolution photometric imagery (BIPED), DL models dominate. On clean semantic benchmarks (BSDS500), DL models dominate even more. On cross-domain generalization (UDED), the gap narrows substantially. Under noise, the ranking inverts entirely: the ASF overtakes all DL models below dataset-dependent and noise-type-dependent SNR thresholds.
- Even within a single method family, the optimal configuration is dataset-dependent. The cross-dataset Spearman rank correlations (Section 8.1) confirm this: the ranking of ASF configurations on BSDS500 has almost no correlation (ρ=0.03–0.17) with their ranking on BIPED or UDED.
- The practical implication: method selection should be driven by the deployment conditions (noise level, image domain, whether semantic or photometric edges are needed), not by benchmark rankings on BSDS500 alone.

### 9.6 Practical Deployment Recommendations
- **Clean natural images, photometric edges needed:** ASF-rect with σ_u=2.0, σ_v=1.2, Ns=36. Matches DL accuracy on BIPED/UDED at lower computational cost and with no training data dependency. Runtime ~45 ms on A100 for 1280×720.
- **Clean natural images, semantic edges needed:** Use a DL model (DexiNed or PIDINet depending on target domain). The ASF cannot compete on semantic boundary detection.
- **Noisy or degraded images (SNR < 10 dB):** ASF-rect with increased σ_u, σ_v scaled to the estimated noise level. The ASF's parametric adaptability provides a principled advantage over fixed DL models in this regime.
- **Real-time applications (< 15 ms budget):** ASF-poly in point-filter mode (Np=25, Ns=4, d=2, m=0). Processes BIPED-sized images in ~12 ms with ODS exceeding all classical methods.
- **Resource-constrained environments (no GPU):** Classical Sobel or Prewitt remains appropriate. The ASF requires GPU execution for competitive throughput.


## Section 10: Conclusion (~0.75 pages)

### Summary
- We presented three orientation-selective edge detection filters — the fused polynomial stencil, the rectangular Gaussian kernel, and the elliptical Gaussian kernel — unified under a common GPU execution framework.
- The progression from polynomial fitting to geometric kernels was driven by a structural insight: algebraically collapsing the multi-step Line Filter pipeline into a single stencil revealed that approximately 28% of the effective kernel weights cancel to near-zero, an artifact of overlapping polynomial neighborhoods rather than a useful property. This motivated the geometric kernel formulations, which define anisotropic derivative weights directly from spatial envelopes and avoid cancellation entirely.
- We placed this work in the historical context of local polynomial fitting for derivative estimation, showing that the WVF and LF belong to a lineage extending from Savitzky-Golay (1964) through 2D extensions, steerable filters, and anisotropic Gaussian derivatives. The geometric kernels represent the natural endpoint of this lineage: the simplest formulation that preserves the benefits of orientation-selective, large-neighborhood gradient estimation.

### Key Results
- The fused stencil achieves 24× speedup and 311× VRAM reduction over the naive Line Filter implementation while producing identical output.
- The geometric kernels achieve a further 3× speedup over the fused stencil while matching its accuracy within 0.3% ODS, with provably uniform weight distributions (η > 0.99 vs η ≈ 0.72).
- All three variants outperform every classical edge detector tested (54 configurations across 7 operator families) on every benchmark.
- On clean imagery, the ASF approaches but does not match DL accuracy, with the gap attributable to semantic vs photometric edge definitions. Under noise (SNR ≤ 5 dB), the ASF overtakes all tested DL models through parametric adaptation.
- Comprehensive parameter ablation (1,200+ configurations, 330 images) reveals that published WVF/LF parameters are suboptimal on clean imagery but partially justified under noise, and that the optimal configuration is fundamentally condition-dependent.

### Future Work
- **Automatic parameter selection.** Developing a lightweight estimator that selects σ_u, σ_v, and Ns based on local or global image statistics (estimated SNR, texture density) would eliminate manual parameter tuning and make the ASF fully automatic.
- **Learned envelope functions.** The geometric kernels use a fixed Gaussian envelope. Learning the envelope shape from data — while keeping the orientation-sweep and gather-dot-product architecture — could narrow the gap to DL models on semantic benchmarks while retaining the ASF's noise adaptability and interpretability.
- **Real-time video pipeline.** The geometric kernel's constant-time execution (~45 ms per frame) is within range of real-time processing at reduced resolution. Temporal coherence constraints (edge tracking across frames) and adaptive per-frame parameter selection are natural extensions for video applications.
- **Alternative polynomial bases.** Orthogonal bases (Zernike, Legendre) in place of the monomial Taylor basis could improve conditioning of the design matrix for large Np and potentially change the optimal polynomial order (Section 7.8).

## References (~1 page)

---

**Estimated total: ~22 pages** (including figures and tables)

## Existing figures available
- `fig1_neighborhood_comparison.png` (Section 3)
- `fig1a_point_filter.pdf` (Section 3)
- `fig2_computation_flow.png` (Section 6)
- `fig3_stencil_orientations.png` (Section 4)
- `fig4_speedup_vram.png` (Section 6)
- `visual_results/` — BSDS500, BIPED, UDED panels (Section 8)
- Rectangular/elliptical kernel demo images from the rect_ellipse PDF (Section 8)

## Key narrative arc
Line Filter (prior work) → Fused Stencil (our optimization, reveals cancellation) → Geometric Kernels (our resolution, cleaner + faster) — all introduced in one paper, with the structural limitation as the intellectual bridge. The historical context (Savitzky-Golay, steerable filters, anisotropic Gaussians) frames the entire contribution as connecting decades of disconnected work rather than claiming novelty from scratch.
