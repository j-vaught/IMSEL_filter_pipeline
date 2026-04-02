#set document(title: "Fused Anisotropic Stencil Filter: Complete Specification")
#set page(margin: 1in, numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

= Fused Anisotropic Stencil Filter: Complete Specification

This document provides a self-contained specification of the fused anisotropic stencil filter for oriented edge detection. All equations required to construct and apply the filter are given below, starting from a grayscale image and ending with per-pixel gradient magnitude and edge angle. No prior knowledge of the filter's construction history is assumed.

== Inputs and Parameters

The filter requires a grayscale image and six user-chosen parameters.

#v(0.3em)
#table(
  columns: (auto, 1fr),
  stroke: none,
  row-gutter: 4pt,
  [$f(X, Y)$], [Grayscale image intensity at pixel $(X, Y)$, with $X = 1, dots, N_h$ and $Y = 1, dots, N_v$.],
  [$N_s$], [Number of candidate orientations (e.g. 36 for $5 degree$ spacing).],
  [$P$], [Polynomial order for the Taylor expansion (e.g. 4 or 5).],
  [$N_p$], [Number of neighbor pixels in each wide-view neighborhood (e.g. 80--250).],
  [$m$], [Half-length of the line in pixels. The full line contains $2m + 1$ virtual pixels.],
  [$sigma_w$], [Standard deviation of the Gaussian weighting along the line.],
  [$r$], [Radius of the circular wide-view neighborhood around each virtual pixel.],
)

The candidate orientations are uniformly spaced over the half-circle.

$ theta_k = k dot pi / N_s, quad k = 0, 1, dots, N_s - 1 $

== Step 1: Define the Line of Virtual Pixels

For each orientation $theta_k$, define $2m + 1$ virtual pixels equally spaced along a line through the target pixel $(X_0, Y_0)$. The $j$-th virtual pixel sits at global coordinates as follows.

$ X_j^((k)) = X_0 + j cos theta_k, quad Y_j^((k)) = Y_0 + j sin theta_k, quad j = -m, dots, 0, dots, m $

These positions are generally non-integer. Each virtual pixel serves as the center of a local polynomial fit.

== Step 2: Build the Wide-View Neighborhood

Around each virtual pixel $(X_j^((k)), Y_j^((k)))$, select the $N_p$ nearest integer-coordinate pixels within radius $r$. Denote these neighbor pixels as $(X_(j,i), Y_(j,i))$ for $i = 1, dots, N_p$. Compute local offsets relative to the virtual pixel.

$ x_(j,i) = X_(j,i) - X_j^((k)), quad y_(j,i) = Y_(j,i) - Y_j^((k)) $

These offsets define the local coordinate system for the polynomial fit at virtual pixel $j$.

== Step 3: Construct the Taylor Expansion Matrix

At each virtual pixel $j$, the image intensity at neighbor $i$ is modeled as a 2D Taylor expansion about the virtual pixel location. For polynomial order $P$, the expansion is written as follows.

$ f(X_(j,i), Y_(j,i)) approx sum_(p=0)^(P) sum_(q=0)^(P-p) frac(1, p! dot q!) dot frac(partial^(p+q) f, partial x^p partial y^q) dot x_(j,i)^p dot y_(j,i)^q $

This is assembled into a matrix equation. Each row of the matrix $bold(A)_j in bb(R)^(N_p times M)$ corresponds to one neighbor pixel, and each column corresponds to one monomial term. The number of monomial terms is $M = binom(P + 2, 2)$. For example, with $P = 4$ the monomials in order are as follows.

$ 1, quad x, quad y, quad x^2 / 2, quad x y, quad y^2 / 2, quad x^3 / 6, quad x^2 y / 2, quad x y^2 / 2, quad y^3 / 6, quad dots $

The entry in row $i$, column corresponding to monomial $x^p y^q slash (p! q!)$, is simply $x_(j,i)^p dot y_(j,i)^q$.

The unknown vector $bold(z)_j in bb(R)^M$ contains the derivative coefficients at virtual pixel $j$, organized in the same monomial order.

$ bold(z)_j = [f^0, space f_x^0, space f_y^0, space f_(x x)^0, space f_(x y)^0, space f_(y y)^0, space dots]^top $

The system of equations is then the following.

$ bold(A)_j bold(z)_j = bold(b)_j $

where $bold(b)_j in bb(R)^(N_p)$ is the vector of image intensities at the neighbor pixels: $b_(j,i) = f(X_(j,i), Y_(j,i))$.

== Step 4: Solve via Pseudoinverse

Since $N_p > M$ (the system is overdetermined), the least-squares solution is obtained through the Moore--Penrose pseudoinverse.

$ bold(z)_j = bold(A)_j^+ bold(b)_j, quad "where" quad bold(A)_j^+ = (bold(A)_j^top bold(A)_j)^(-1) bold(A)_j^top $

The pseudoinverse $bold(A)_j^+$ is an $M times N_p$ matrix. Row 2 of $bold(A)_j^+$ (0-indexed) extracts $f_x^0$ and row 3 extracts $f_y^0$. Denote the row of $bold(A)_j^+$ that extracts the normal derivative of interest as $bold(a)_j^top in bb(R)^(1 times N_p)$. Then the derivative at virtual pixel $j$ is the following.

$ d_j = bold(a)_j^top bold(b)_j = sum_(i=1)^(N_p) a_(j,i) dot f(X_(j,i), Y_(j,i)) $

Which derivative row to select depends on the orientation $theta_k$. The normal derivative (perpendicular to the line) is a linear combination of $f_x^0$ and $f_y^0$. In the rotated local frame of the line, the normal derivative is the following.

$ d_j = -sin theta_k dot f_x^0 + cos theta_k dot f_y^0 $

This is obtained by taking the corresponding linear combination of rows 2 and 3 of $bold(A)_j^+$.

$ bold(a)_j^top = -sin theta_k dot bold(A)_j^+ [2, :] + cos theta_k dot bold(A)_j^+ [3, :] $

== Step 5: Gaussian Weighting Along the Line

The derivative estimates from the $2m + 1$ virtual pixels are combined with Gaussian weights that decay with distance from the line center.

$ w_j = exp(-j^2 / (2 sigma_w^2)), quad j = -m, dots, m $

The aggregate derivative at the target pixel is the weighted sum of the individual estimates.

$ d = sum_(j=-m)^(m) w_j dot d_j = sum_(j=-m)^(m) w_j sum_(i=1)^(N_p) a_(j,i) dot f(X_(j,i), Y_(j,i)) $

== Step 6: Fuse into a Single Stencil

The double summation above touches every pixel in the union of all $2m + 1$ neighborhoods. Many of these pixel positions overlap between adjacent virtual pixels. Let $Omega_k$ be the set of unique integer-coordinate pixel positions appearing in the union of all neighborhoods for orientation $k$.

$ Omega_k = union.big_(j=-m)^(m) {(X_(j,i), Y_(j,i)) : i = 1, dots, N_p} $

Denote the elements of $Omega_k$ as $(X_0 + tilde(delta)_ell^x, Y_0 + tilde(delta)_ell^y)$ for $ell = 1, dots, N'_k$, where $N'_k = |Omega_k|$.

For each unique position $ell$, the fused weight is the sum of all contributions from every virtual pixel $j$ and neighbor index $i$ that map to that position.

$ alpha_(k, ell) = sum_(j, i "such that" (X_(j,i), Y_(j,i)) = (X_0 + tilde(delta)_ell^x, Y_0 + tilde(delta)_ell^y)) w_j dot a_(j,i) $

This is the core result. Each $alpha_(k,ell)$ is a single scalar that encodes the polynomial fitting, pseudoinverse extraction, derivative selection, and Gaussian line weighting for one pixel position at one orientation.

== Step 7: Apply the Filter

The fused stencil is applied to every pixel in the image as a convolution. For each orientation $k$, construct a sparse 2D kernel $h_k$ of size $(2S+1) times (2S+1)$, where $S$ is the maximum stencil radius, by placing the fused weights at their offset positions.

$ h_k [tilde(delta)_ell^x + S, space tilde(delta)_ell^y + S] = alpha_(k, ell), quad ell = 1, dots, N'_k $

All other entries of $h_k$ are zero. The filter response at every pixel is then obtained by convolution.

$ R_k (X_0, Y_0) = (h_k star.op f)(X_0, Y_0) = sum_(ell=1)^(N'_k) alpha_(k,ell) dot f(X_0 + tilde(delta)_ell^x, space Y_0 + tilde(delta)_ell^y) $

Apply all $N_s$ kernels to produce a response volume $G in bb(R)^(N_s times N_h times N_v)$.

$ G_(k, X_0, Y_0) = R_k (X_0, Y_0), quad k = 0, dots, N_s - 1 $

== Step 8: Extract Gradient Magnitude and Edge Angle

At each pixel, select the orientation with the largest absolute response.

$ k^* (X_0, Y_0) = arg max_k |G_(k, X_0, Y_0)| $

The outputs are the gradient magnitude and the edge angle.

$ "magnitude"(X_0, Y_0) = |G_(k^*, X_0, Y_0)| $

$ "angle"(X_0, Y_0) = theta_(k^*) = k^* dot pi / N_s $

== Summary of Precomputation

All orientation-dependent quantities ($bold(A)_j$, $bold(A)_j^+$, $bold(a)_j$, $w_j$, $alpha_(k,ell)$, and $h_k$) depend only on the six parameters and the geometry of the stencil, not on any image data. They are computed once and stored. At runtime, applying the filter to an image of any size requires only the following two operations.

$ G = cal(H) star.op f, quad k^* = arg max_k |G_k| $

Here $cal(H) in bb(R)^(N_s times (2S+1) times (2S+1))$ is the precomputed filter bank. The first operation is a batched 2D convolution. The second is a pointwise argmax over the orientation axis.
