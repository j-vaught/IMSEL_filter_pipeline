// Section 3: Mathematical Framework
// Pure math section — no figures

= Mathematical Framework <sec:math>

This section develops the mathematical framework underlying the Anisotropic Savitzky--Golay Filter (ASF). The central idea is to estimate image gradients by fitting a local polynomial surface at each pixel, with the coordinate system rotated to align with each candidate edge orientation. We begin with the polynomial model, derive the least-squares estimator, describe the orientation sweep, define the neighbor selection strategy, and introduce the line extension that provides anisotropic noise averaging along the edge tangent.


== Local Polynomial Model <sec:poly-model>

Consider a grayscale image $f: ZZ^2 -> RR$ and a target pixel at global coordinates $(X_0, Y_0)$. The goal is to estimate the gradient magnitude and orientation at this pixel by fitting a two-dimensional polynomial to its local neighborhood. The key distinction from conventional gradient operators is that the fitting coordinate system is rotated to align with each candidate edge orientation, so that the normal derivative (perpendicular to the putative edge) can be read off directly from the polynomial coefficients.

For a candidate orientation $theta_k$, a local coordinate system $(x, y)$ is defined with $x$ along the edge normal and $y$ along the edge tangent. The transformation from global to local coordinates for any neighbor pixel $(X_i, Y_i)$ is given by the standard rotation

$ x_i = (X_i - X_0) cos theta_k + (Y_i - Y_0) sin theta_k $ <eq:rot-x>

$ y_i = -(X_i - X_0) sin theta_k + (Y_i - Y_0) cos theta_k $ <eq:rot-y>

which can be written compactly in matrix form as

$ vec(x_i, y_i) = mat(cos theta_k, sin theta_k; -sin theta_k, cos theta_k) vec(X_i - X_0, Y_i - Y_0) $ <eq:rot-matrix>

The intensity surface in the neighborhood of the origin is then approximated by a two-dimensional Taylor expansion of order $d$. For $d = 2$, this takes the explicit form

$ f(x_i, y_i) approx c_0 + c_1 x_i + c_2 y_i + c_3 x_i^2 / 2 + c_4 y_i^2 / 2 + c_5 x_i y_i $ <eq:taylor-d2>

where $c_0 = f^0$ is the intensity at the origin, $c_1 = f_x^0$ and $c_2 = f_y^0$ are the first partial derivatives, and $c_3 = f_(x x)^0$, $c_4 = f_(y y)^0$, $c_5 = f_(x y)^0$ are the second partials. The general expansion for arbitrary order $d$ is

$ f(x_i, y_i) approx sum_(p+q <= d) (f^0_(underbrace(x dots.c x, p) underbrace(y dots.c y, q))) / (p! dot q!) dot x_i^p y_i^q $ <eq:taylor-general>

The number of unknown coefficients in this expansion is

$ M = ((d+1)(d+2)) / 2 $ <eq:num-coeffs>

For common polynomial orders, $M$ takes the values 3 ($d = 1$), 6 ($d = 2$), 10 ($d = 3$), and 15 ($d = 4$). Collecting these unknowns into a coefficient vector $bold(z) in RR^M$, the approximation at pixel $i$ becomes the inner product of $bold(z)$ with a monomial basis vector evaluated at $(x_i, y_i)$.

This local polynomial model has a direct connection to the classical Savitzky--Golay filter @savitzky1964smoothing. In the one-dimensional case, fitting a polynomial of degree $d$ to a uniformly-spaced window and evaluating the $n$-th derivative at the center is precisely the Savitzky--Golay procedure. The present formulation generalizes this to two dimensions, non-uniform (integer-grid) sampling, and orientation-dependent coordinate systems. The "Savitzky--Golay" designation in the filter name reflects this lineage.


== Least-Squares Gradient Estimation <sec:ls-gradient>

Given $N_p$ neighbor pixels surrounding $(X_0, Y_0)$, the Taylor expansion @eq:taylor-general evaluated at each neighbor yields an overdetermined linear system. Define the design matrix $bold(A)_(theta_k) in RR^(N_p times M)$ whose $i$-th row contains the monomial basis evaluated at the local coordinates $(x_i, y_i)$ of the $i$-th neighbor. For polynomial order $d = 2$, row $i$ of the design matrix is

$ bold(A)_(theta_k)[i, :] = vec(1, x_i, y_i, x_i^2 / 2, y_i^2 / 2, x_i y_i)^top $ <eq:design-row>

The system to be solved is then

$ bold(A)_(theta_k) bold(z) = bold(b) $ <eq:ls-system>

where $bold(b) = (f(X_1, Y_1), dots, f(X_(N_p), Y_(N_p)))^top in RR^(N_p)$ is the vector of observed pixel intensities at the neighbor locations. Because $N_p > M$, this system is overdetermined and generally has no exact solution.

The ordinary least-squares (OLS) solution minimizes the sum of squared residuals and is given by the normal equations

$ hat(bold(z)) = (bold(A)_(theta_k)^top bold(A)_(theta_k))^(-1) bold(A)_(theta_k)^top bold(b) $ <eq:ols>

This can be written compactly using the Moore--Penrose pseudoinverse $bold(P)_(theta_k) = bold(A)_(theta_k)^+$ as

$ hat(bold(z)) = bold(P)_(theta_k) bold(b), quad bold(P)_(theta_k) = (bold(A)_(theta_k)^top bold(A)_(theta_k))^(-1) bold(A)_(theta_k)^top in RR^(M times N_p) $ <eq:pinv>

A critical observation is that the pseudoinverse $bold(P)_(theta_k)$ depends only on the geometry of the neighborhood (the relative pixel positions and the orientation angle $theta_k$), not on the pixel intensities. Since the neighborhood geometry is the same for every pixel in the image (assuming a fixed $N_p$ and $theta_k$), the pseudoinverse can be _precomputed_ once for each orientation and reused across the entire image. This converts what would otherwise be a per-pixel matrix inversion into a single dot product.

The estimated normal derivative, which is the quantity of interest for edge detection, is the coefficient $hat(z)_1 = hat(f)_x$. This corresponds to row 1 (zero-indexed) of the pseudoinverse matrix, yielding

$ hat(f)_x^((k)) = bold(p)_"fx"^((k)) dot bold(b), quad bold(p)_"fx"^((k)) = bold(P)_(theta_k)[1, :] in RR^(N_p) $ <eq:fx>

The gradient estimate at orientation $theta_k$ thus reduces to a single inner product of the precomputed weight vector $bold(p)_"fx"^((k))$ with the local pixel intensities $bold(b)$. This structure is shared with all linear filter operations, but the key distinction is that the weights $bold(p)_"fx"^((k))$ are derived from the polynomial fitting framework rather than from a handcrafted kernel.


== Orientation Sweep and Maximum-Response Selection <sec:orient-sweep>

Classical gradient-based edge detectors such as the Sobel operator and the Canny detector @canny1986computational estimate the gradient by computing partial derivatives along two fixed axes ($x$ and $y$) and then combining them. The gradient magnitude is obtained as $G = sqrt(G_x^2 + G_y^2)$ and the orientation as $theta = arctan(G_y, G_x)$. This two-direction approach implicitly assumes that the gradient can be accurately decomposed into two orthogonal components, which is reasonable for smooth intensity fields but introduces angular quantization error when the edge orientation does not align with one of the two axes.

The ASF takes a fundamentally different approach. Rather than fixing two filter directions and computing the gradient as a derived quantity, it evaluates the normal derivative _directly_ at each of $N_s$ candidate orientations uniformly distributed over the half-circle

$ theta_k = (k dot pi) / N_s, quad k = 0, 1, dots, N_s - 1 $ <eq:orient-set>

At each orientation, the polynomial fitting procedure of @sec:ls-gradient produces a normal derivative estimate $hat(f)_x^((k))$ via @eq:fx. The filter response and estimated orientation are then selected by the maximum-response criterion

$ k^* = arg max_k |hat(f)_x^((k))| $ <eq:kstar>

$ G = |hat(f)_x^((k^*))| $ <eq:mag>

$ theta^* = theta_(k^*) $ <eq:orient-est>

This sweep-and-select strategy has several advantages. First, it avoids the angular bias of two-direction schemes because the polynomial fit is optimally aligned with each candidate orientation. At the true edge direction, the $x$-axis of the local coordinate system is perpendicular to the edge, and the Taylor expansion captures the intensity transition most accurately. Second, the angular resolution of the orientation estimate is $pi / N_s$ and can be made arbitrarily fine by increasing $N_s$, at a linear cost in computation. Third, the maximum response $G$ is always at least as large as the response that would be obtained at any fixed pair of directions, making the ASF inherently more sensitive to weak edges whose orientations fall between the cardinal axes.

The half-circle range $[0, pi)$ suffices because the normal derivative changes sign (but not magnitude) under a $pi$ rotation. The unsigned magnitude $|hat(f)_x^((k))|$ is invariant to this sign flip, so orientations $theta_k$ and $theta_k + pi$ produce identical responses.


== Neighbor Selection <sec:neighbor>

The choice of the $N_p$ neighbor pixels determines the spatial support of the filter. The ASF selects the $N_p$ integer-coordinate pixels closest to the target pixel $(X_0, Y_0)$ in Euclidean distance, yielding an approximately circular support region. The radius of this region scales as

$ r approx ceil(sqrt(N_p / pi)) + 1 $ <eq:radius>

For example, $N_p = 100$ produces a circular patch of radius approximately 7 pixels. The support is constant across all orientations; only the coordinate system rotates.

The _overdetermination ratio_ $N_p / M$ governs the balance between noise suppression and spatial fidelity. A larger ratio provides more averaging and hence stronger noise reduction, at the cost of potentially smoothing fine spatial detail. For the default parameters $N_p = 100$ and $d = 4$ (giving $M = 15$), the ratio is

$ N_p / M = 100 / 15 approx 6.7 $ <eq:overdet>

This means the system has approximately 6.7 times more equations than unknowns, providing substantial noise averaging while retaining the polynomial's ability to represent intensity variations up to fourth order.

An important distinction from Gaussian smoothing is that the polynomial fit does not blur edges isotropically. A Gaussian kernel attenuates all spatial frequencies uniformly within its support, including the step-like intensity transition that defines an edge. The polynomial model, by contrast, is _capable_ of representing such transitions. A step edge along the $y$-axis (tangent direction) manifests as a large $c_1 = f_x^0$ coefficient, which the least-squares fit preserves rather than suppresses. The smoothing occurs only in the sense that the polynomial cannot represent intensity variations of order higher than $d$ within the support. This property is fundamental to why the ASF achieves simultaneous noise suppression and edge preservation.


== Line Extension <sec:line-ext>

The polynomial fit described above operates on a compact, approximately circular neighborhood centered on a single pixel. While this point-filter formulation is effective for strong edges, it may produce unreliable responses at low-contrast boundaries where the intensity transition is too weak relative to the noise level. The line extension addresses this limitation by extending the support region anisotropically along the candidate edge direction.

For a given orientation $theta_k$, $(2m+1)$ virtual evaluation positions are placed along the edge tangent direction at unit-pixel spacing

$ (X_j, Y_j) = (X_0 + j cos(theta_k + pi / 2), Y_0 + j sin(theta_k + pi / 2)), quad j in {-m, dots, 0, dots, m} $ <eq:line-positions>

where the offset direction $theta_k + pi / 2$ corresponds to the edge tangent (perpendicular to the normal direction $theta_k$). At each position $j$, the full polynomial fitting procedure is applied independently. The $N_p$ nearest neighbors of $(X_j, Y_j)$ are gathered, the design matrix is constructed at orientation $theta_k$, and the normal derivative $hat(f)_x^((j,k))$ is extracted via the pseudoinverse weight vector. These per-position derivative estimates are then combined via Gaussian-weighted averaging

$ R_k = sum_(j=-m)^m w_j dot hat(f)_x^((j,k)) $ <eq:line-response>

where the weights follow a Gaussian profile

$ w_j = exp(-j^2 / (2 sigma_ell^2)), quad sigma_ell = m / 2 $ <eq:line-weights>

The maximum-response selection is applied to the line-extended responses rather than to the point-filter responses

$ k^* = arg max_k |R_k|, quad G = |R_(k^*)| $ <eq:line-max>

The Gaussian weighting ensures that positions near the center of the line segment contribute more than those at the periphery, providing a smooth taper that avoids boundary artifacts. The parameter $m$ controls the half-length of the line extension. For $m = 7$, the line segment spans 15 pixels along the edge tangent, giving the filter access to a substantially larger spatial context than the point filter alone.

The line extension provides two complementary benefits. First, it averages derivative estimates from multiple positions along the edge, reducing the variance of the gradient magnitude estimate by a factor proportional to the effective number of independent samples. Second, it enforces spatial coherence along the edge tangent. If a true edge runs through the target pixel, the derivative estimates at neighboring positions along the tangent direction will be consistently large, reinforcing the response. Conversely, isolated noise spikes, which are spatially uncorrelated, will not produce coherent responses along the line and will therefore be suppressed.

The computational cost of the line extension is significant. Each orientation requires $(2m+1)$ independent polynomial fits, and each fit gathers $N_p$ pixel intensities from the image. The total number of memory accesses per pixel per orientation is therefore $(2m+1) times N_p$. For $m = 7$ and $N_p = 100$, this amounts to 1500 gathers per orientation, and with $N_s = 36$ orientations the total reaches 54000 gathers per pixel. Many of these accesses are redundant, as neighboring evaluation positions share overlapping neighborhoods. This redundancy motivates the fused stencil formulation developed in the following section, which algebraically collapses the multi-step pipeline into a single weighted gather per pixel per orientation.
