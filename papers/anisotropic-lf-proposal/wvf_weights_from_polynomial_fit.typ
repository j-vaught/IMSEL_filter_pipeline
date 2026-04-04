#set page(margin: (top: 0.8in, bottom: 0.8in, left: 0.85in, right: 0.85in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1.1")
#set par(justify: true, leading: 0.6em)
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11.5pt)
#show heading.where(level: 3): set text(size: 11pt)


#align(center)[
  #text(size: 17pt, weight: "bold")[
    Creating the Wide View Filter's Weights from a Polynomial Fit
  ]
  #v(0.5em)
  #text(size: 11pt, style: "italic")[
   My interpretation of the mathematical construction of the Wide View Filter.
  ]
]

= Introduction: The Savitzky-Golay Principle <sec:intro>

At its core, this method is a Savitzky--Golay derivative estimator. Savitzky and Golay's original 1964 contribution was to show that one can fit a polynomial locally by least squares and recover derivatives directly from the fitted coefficients @savitzkygolay1964. Our derivation uses the same principle, but adapts it to two-dimensional image neighborhoods.

Although a polynomial fit is usually described as solving for unknown coefficients, the least-squares solution is linear in the sampled pixel values. As a result, any fitted coefficient, including a derivative coefficient, can be expressed as one fixed weighted sum of the input pixels. The resulting weighted sum for each pixel is effectively the filter weights, similar to how Sobel or Prewitt weights are defined.

= A Pedagogical Example: The 3x3 Square Neighborhood <sec:3x3-example>

== Defining the Image Patch and Polynomial Model <sec:3x3-model>

We begin with a `3 x 3` example because it makes the algebra visible. Conceptually, nothing new happens here since this is the same two-dimensional polynomial-fitting procedure developed in the Savitzky--Golay tradition and later extended to image neighborhoods by Gorry, Meer et al., and Luo et al. @gorry1990sg @meer1991sg2d @luo2005sg2d in the '90s and early '00s. The small patch is used primarily to save me a lot of algebra and to make the logic more transparent. The same principles apply to larger patches, and we will show how to generalize to those later.



=== The Local Patch and Polynomial Ansatz <sec:3x3-patch>

As a simplification, let us begin with a local 3x3 image patch. We denote this neighborhood by $bold(B)$:

$
  bold(B) =
  mat(delim: "[",
    I_(-1,1), I_(0,1), I_(1,1);
    I_(-1,0), I_(0,0), I_(1,0);
    I_(-1,-1), I_(0,-1), I_(1,-1)
  ).
$ <eq:patch>

This matrix contains the pixel intensities in a local 3x3 window, with the pixel of interest located at the center. Now suppose we aim to fit a degree-1 polynomial to the matrix $bold(B)$. One simple way to represent that local polynomial is

$
  p(x, y) = c_0 + c_1 x + c_2 y.
$ <eq:poly1>

Here $c_0$ represents the local baseline brightness, ideally close to the brightness of the center pixel. The coefficients $c_1$ and $c_2$ describe how the brightness changes in the $x$ and $y$ directions, respectively. Since an edge can be understood as a spatial change in brightness, these change terms are the quantities we care about most for edge detection. In this sense, $c_1$ and $c_2$ are the derivative coefficients, while $c_0$ mainly helps the polynomial match the local patch.

If we instead use a degree-2 polynomial, we could write

$
  p(x, y) = c_0 + c_1 x + c_2 y + c_3 x^2 + c_4 x y + c_5 y^2.
$ <eq:poly2>

In this case, $c_0$, $c_1$, and $c_2$ play the same roles as before, while the additional coefficients capture curvature in the local intensity surface. Intuitively, they describe how the rate of change itself varies across the patch. Even when the polynomial degree is increased, the main coefficients used for estimating the local directional change are still the first-derivative terms. Increasing the degree simply gives the fit more flexibility, which can improve accuracy in some cases when the local image structure is not well described by a purely linear model.

With all that mathematical kerfuffle out of the way, the only question that remains is how to solve for those unknown coefficients. To do that, we write one fitting equation for each pixel in the local patch and then collect them into a single linear system.

=== From Pixel Coordinates to Fitting Equations <sec:3x3-fit-equations>

For example, if we want to write the formula for the degree-1 polynomial, we begin with @eq:poly1 and substitute the pixel coordinates into it. The pixel at $(-1, -1)$ has coordinates $x = -1$ and $y = -1$, so we plug those into the polynomial and plug in the known coordinates of a pixel in the patch. 

For the pixel at $(-1, -1)$, this gives

$
  p(-1, -1) = c_0 + c_1(-1) + c_2(-1)
$ <eq:sub1>
$
  quad = c_0 - c_1 - c_2.
$ <eq:sub1-simplified>

Since this fitted value should match the observed pixel intensity at that location, and since the degree-1 polynomial is only an approximation to the local pixel intensities, we use $approx$ instead of exact equality in the fitting equations.

$
  c_0 - c_1 - c_2 approx I_(-1,-1).
$ <eq:fit11>

We then repeat this same substitution for the other pixel coordinates in the 3x3 patch. Doing so yields

$
  c_0 - c_1 - c_2 approx I_(-1,-1)
$ <eq:fit-grid-1>
$
  c_0 - c_2 approx I_(0,-1)
$ <eq:fit-grid-2>
$
  c_0 + c_1 - c_2 approx I_(1,-1)
$ <eq:fit-grid-3>
$
  c_0 - c_1 approx I_(-1,0)
$ <eq:fit-grid-4>
$
  c_0 approx I_(0,0)
$ <eq:fit-grid-5>
$
  c_0 + c_1 approx I_(1,0)
$ <eq:fit-grid-6>
$
  c_0 - c_1 + c_2 approx I_(-1,1)
$ <eq:fit-grid-7>
$
  c_0 + c_2 approx I_(0,1)
$ <eq:fit-grid-8>
$
  c_0 + c_1 + c_2 approx I_(1,1).
$ <eq:fit-grid-9>

The resulting set of equations is all one needs to solve for the system. However, as you may remember from primary school, a system of equations can often be written more efficiently in matrix form. Doing so makes the mathematics easier to organize and manipulate. To see how this becomes matrix form, consider the first equation:

$
  c_0 - c_1 - c_2 approx I_(-1,-1).
$

This can be rewritten in a matrix form as

$
  [1, -1, -1]
  mat(delim: "[",
    c_0;
    c_1;
    c_2
  )
  approx I_(-1,-1),
$ <eq:row-first>

where the first value of the first matrix remains 1, since it is the coefficient of $c_0$, and the second and third values are $-1$ because they are the coefficients of $c_1$ and $c_2$, respectively. The row vector $[1, -1, -1]$ is called a row of the design matrix, and the column vector of unknowns is called the coefficient vector. The right-hand side is just the observed pixel intensity at that location.

 The same idea applies to every other pixel in the patch. For example, the pixel at $(0,-1)$ gives the row equation $[1, 0, -1] bold(z) approx I_(0,-1)$, and the pixel at $(1,-1)$ gives $[1, 1, -1] bold(z) approx I_(1,-1)$. Once all of these row equations are stacked together, they form one compact matrix system 

$
  bold(A) bold(z) approx bold(b).
$ <eq:system>

where $bold(A)$ is the design matrix containing the coefficients of the unknowns for each pixel, $bold(z)$ is the vector of unknown polynomial coefficients, and $bold(b)$ is the vector of observed pixel intensities.

== Constructing the Matrix System <sec:3x3-matrix-system>

=== Vectorizing the Pixel Data <sec:3x3-data-vector>

To derive $bold(b)$ and $bold(A)$, we first need to decide how to stack the pixel values from the patch into a vector. One common way is to read the pixels in row-major order, starting from the top-left corner and moving left to right, then down to the next row. Doing so gives us

$
  bold(b) =
  [
    I_(-1,-1),
    I_(0,-1),
    I_(1,-1),
    I_(-1,0),
    I_(0,0),
    I_(1,0),
    I_(-1,1),
    I_(0,1),
    I_(1,1)
  ]^top.
$ <eq:bvec>

=== Building the Design Matrix <sec:3x3-design-matrix>

Then, the design matrix is just the basis $[1, x, y]$ evaluated at one sample location. Therefore

  $
    bold(A) =
    mat(
      delim: "[",
      1, 1, 1, 1, 1, 1, 1, 1;
      -1, 0, 1, -1, 0, 1, -1, 0, 1;
      -1, -1, -1, 0, 0, 0, 1, 1, 1
    )^top.
  $ <eq:design3>

Then, finally, to solve for the unknown coefficients, we use the least-squares solution to the system in @eq:system.

== The Least-Squares Solution <sec:3x3-lstsq>

=== The Least-Squares Objective <sec:3x3-objective>

At this point, we have nine approximate equations but only three unknown coefficients. In general, there is no reason to expect one vector $bold(z)$ to satisfy every equation exactly. So instead of solving $bold(A) bold(z) = bold(b)$ exactly, we choose the coefficient vector that makes the overall mismatch as small as possible in the least-squares sense:

$
  hat(bold(z)) = arg min_(bold(z)) ||bold(A) bold(z) - bold(b)||^2.
$ <eq:lstsq-objective>

To make that step explicit, let's write out the objective function that we are minimizing, with $E$ representing error

$
  E(bold(z)) = ||bold(A) bold(z) - bold(b)||^2.
$ <eq:error3>

Using the identity $||bold(v)||^2 = bold(v)^top bold(v)$, we can rewrite the objective as

$
  E(bold(z)) = (bold(A) bold(z) - bold(b))^top (bold(A) bold(z) - bold(b)).
$ <eq:error-expand3>

Expanding this expression gives

$
  E(bold(z)) = bold(z)^top bold(A)^top bold(A) bold(z) - 2 bold(b)^top bold(A) bold(z) + bold(b)^top bold(b).
$ <eq:error-quadratic3>

Now differentiate with respect to $bold(z)$ and set the result equal to zero:

$
  2 bold(A)^top bold(A) bold(z) - 2 bold(A)^top bold(b) = 0.
$ <eq:grad3>

Rearranging gives the normal equations

$
  bold(A)^top bold(A) hat(bold(z)) = bold(A)^top bold(b).
$ <eq:normal3>

If $bold(A)^top bold(A)$ is invertible, we multiply both sides by $(bold(A)^top bold(A))^(-1)$ to isolate the coefficient vector. The fitted coefficients are therefore

$
  hat(bold(z)) = (bold(A)^top bold(A))^(-1) bold(A)^top bold(b).
$ <eq:lstsq3>

=== Evaluating the Pseudoinverse <sec:3x3-pseudoinverse>

For this particular 3x3 geometry, one can compute

$
  bold(A)^top bold(A) =
  mat(delim: "[",
    9, 0, 0;
    0, 6, 0;
    0, 0, 6
  ),
$ <eq:ata3>

so

$
  (bold(A)^top bold(A))^(-1) =
  mat(delim: "[",
    1/9, 0, 0;
    0, 1/6, 0;
    0, 0, 1/6
  ).
$ <eq:ata3inv>

Thus the pseudoinverse is

$
  bold(P) = (bold(A)^top bold(A))^(-1) bold(A)^top.
$ <eq:pinv3>

And we evaluate it as:
$
  bold(P) =
  mat(delim: "[",
    1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9;
    -1/6, 0, 1/6, -1/6, 0, 1/6, -1/6, 0, 1/6;
    -1/6, -1/6, -1/6, 0, 0, 0, 1/6, 1/6, 1/6
  ). 
$<eq:pinv3-eval> 

The second row of $bold(P)$ gives the weights for $c_1$, the $x$-derivative coefficient, and the third row gives the weights for $c_2$, the $y$-derivative coefficient. For example, the weight vector for $c_1$ is

$
  bold(p)_(c_1)^top =
  [
    -1/6, 0, 1/6,
    -1/6, 0, 1/6,
    -1/6, 0, 1/6
  ].
$ <eq:p-c1>

And the weight vector for $c_2$ is 
$
  bold(p)_(c_2)^top =
  [
    -1/6, -1/6, -1/6,
    0, 0, 0,
    1/6, 1/6, 1/6
  ].
$ <eq:p-c2>

Then, we can find the fitted derivative coefficient for $c_1$ is 

$
  hat(c_1) = bold(p)_(c_1)^top bold(b).
$ <eq:c1-est>

and the fitted derivative coefficient for $c_2$ is
$
  hat(c_2) = bold(p)_(c_2)^top bold(b).
$ <eq:c2-est>
 
Since we do not have the exact vlaue for $c_1$ and $c_2$, we can only write the estimates as approximations, hence the usage of the symbols $hat(c_1)$ and $hat(c_2)$.

== Extracting the Convolution Filters <sec:3x3-filters>

=== Writing the Derivative Estimates as Weighted Sums <sec:3x3-weighted-sums>

Finally, converting the results from the polynomial fit into derivative filters, we write out the explicit expressions:

$
  hat(c_1) = (-1/6) I_(-1,-1) + 0 I_(0,-1) + (1/6) I_(1,-1)
$ <eq:c1-exp1>
$
  quad + (-1/6) I_(-1,0) + 0 I_(0,0) + (1/6) I_(1,0)
$ <eq:c1-exp2>
$
  quad + (-1/6) I_(-1,1) + 0 I_(0,1) + (1/6) I_(1,1).
$ <eq:c1-exp3>

=== Rearranging the Weights into 3x3 Stencils <sec:3x3-stencil-formation>

Because the entries of $bold(b)$ were stacked in row-major order, the weights in $bold(p)_(c_1)^top$ and $bold(p)_(c_2)^top$ can be placed back into that same `3 x 3` geometry. In other words, the weighted sum for each derivative estimate can be rewritten as a local stencil whose entries sit at the same pixel locations as the original patch. Symbolically, this rearrangement is

$
  bold(K)_x = "reshape"_(3,3)(bold(p)_(c_1)^top), quad bold(K)_y = "reshape"_(3,3)(bold(p)_(c_2)^top).
$ <eq:k-reshape3>

Once that reshaping is done, the least-squares derivative estimate is no longer just a vector expression. It is an ordinary `3 x 3` filter that can be applied directly to the image.

=== The Resulting 3x3 Kernels <sec:3x3-kernels>

And converitng into standard 3x3 derivative filter format, we have

$
  bold(K)_x =
  mat(delim: "[",
    -1/6, 0, 1/6;
    -1/6, 0, 1/6;
    -1/6, 0, 1/6
  ).
$ <eq:kx3>

For $c_2$, we have the same idea, but rotated by 90 degrees:
$
  hat(c_2) = (-1/6) I_(-1,-1) + (-1/6) I_(0,-1) + (-1/6) I_(1,-1)
$ <eq:c2-exp1>
$  quad + 0 I_(-1,0) + 0 I_(0,0) + 0 I_(1,0)
$ <eq:c2-exp2>
$  quad + (1/6) I_(-1,1) + (1/6) I_(0,1) + (1/6) I_(1,1).
$ <eq:c2-exp3>
And the corresponding 3x3 filter is
$
  bold(K)_y =
  mat(delim: "[",
    -1/6, -1/6, -1/6;
    0, 0, 0;
    1/6, 1/6, 1/6
  ).
$ <eq:ky3>

=== Combining the Horizontal and Vertical Responses <sec:3x3-pipeline>

Once $bold(K)_x$ and $bold(K)_y$ are known, they can be used exactly like any other pair of derivative filters. Applying them to the image gives the horizontal and vertical responses

$
  G_x = bold(K)_x * I, quad G_y = bold(K)_y * I.
$ <eq:gxy3>

From those two responses, we form the usual gradient magnitude

$
  G = sqrt(G_x^2 + G_y^2),
$ <eq:gmag3>

and the corresponding gradient orientation

$
  theta_g = "atan2"(G_y, G_x).
$ <eq:gtheta3>

In other words, after the polynomial fit has been collapsed into $bold(K)_x$ and $bold(K)_y$, the rest of the pipeline is just the standard gradient-based edge-detection workflow.

= Generalization To An N x N Square <sec:nxn-square>

This `N x N` formulation is probably best understood as the standard two-dimensional Savitzky--Golay least-squares construction written in a form convenient for later sections @luo2005sg2d. We restate it here so that the derivation remains self-contained. The same polynomial basis, data vector, and pseudoinverse logic from the `3 x 3` case carry over directly.


Now suppose that instead of a 3x3 patch, we use a general $N times N$ square patch centered at the pixel of interest. To make sure there really is a center pixel, we assume that $N$ is odd. Then we define the half-width

$
  h = (N - 1)/2.
$ <eq:halfwidth>

With this notation, the local coordinates in the square support are simply

$
  (x, y), quad x, y in {-h, -h + 1, dots, h - 1, h}.
$ <eq:square-grid>

This is just the centered $N times N$ window around the pixel of interest. Since there are $N$ possible $x$-values and $N$ possible $y$-values, the total number of pixels in the support is

$
  N_p = N^2.
$ <eq:square-count>

To write the least-squares system compactly, we now enumerate those same grid points as

$
  (x_i, y_i), quad i = 1, 2, dots, N_p.
$ <eq:square-coords>

== The Monomial Basis and Unknown Coefficients <sec:nxn-basis>

For a polynomial of degree $d$, we can define the monomial basis vector. 

$
  phi_d(x, y) =
  [
    1,
    x,
    y,
    x^2/2,
    x y,
    y^2/2,
    dots
  ]^top,
$ <eq:phi>

This vector is just a compact way to list all polynomial terms up to degree $d$. For example, from @eq:poly1, we see that a degree-1 polynomial has the basis $[1, x, y]^top$, where we have the coefficients $c_0$, $c_1$, and $c_2$ corresponding to these terms. Additionally, from @eq:poly2, the monomial basis for a degree-2 polynomial is $[1, x, y, x^2/2, x y, y^2/2]^top$.

In other words, the vector $phi_d(x, y)$ simply collects every monomial term that can appear in a 2D polynomial of total degree at most $d$. Each entry of this basis will have one corresponding unknown coefficient in the polynomial fit. So the length of $phi_d(x, y)$ is exactly the number of coefficients we must solve for. We denote that number by $M$.

=== Counting the Basis Terms <sec:nxn-basis-count>

To see where this count comes from, group the monomials by total degree. For total degree 0, there is only 1 term, namely $1$. For total degree 1, there are 2 terms, namely $x$ and $y$. For total degree 2, there are 3 terms, namely $x^2$, $x y$, and $y^2$. In general, for total degree $n$, the admissible monomials are $x^n, x^(n-1) y, x^(n-2) y^2, dots, y^n$, so there are $n+1$ of them.

Since the basis includes every total degree from 0 up to $d$, the total number of basis terms is

$
  M = sum_(n=0)^d (n+1) = 1 + 2 + 3 + dots + (d + 1).
$ <eq:mbasis-sum>

Using the standard formula for the sum of the first $d+1$ positive integers, this becomes

$
  M = (d+1)(d+2)/2.
$ <eq:mbasis>

=== Minimum Sample Requirement <sec:nxn-min-data>

Now that we have acquire M through such painful means, we can now define the minimum data requirement for the least-squares fit. Since we have $M$ unknown coefficients, we need at least $M$ equations to solve for them. If we recall from @sec:3x3-fit-equations, each pixel in the support gives us one equation, so we need at least $M$ pixels in the support. In more personable terms, the number of pixels in the support must be greater than (or equal to) the number of coefficients. 

$
  N_p >= M.
$ <eq:min-data>

== Building the General Linear System <sec:nxn-system>

=== The Coefficient Vector and Design Matrix <sec:nxn-design>

For the generalized $N times N$ case, the unknown coefficient vector is

$
  bold(z) = [c_0, c_1, c_2, dots, c_(M-1)]^top,
$ <eq:zvecn>

where these coefficients multiply the monomials collected in $phi_d(x, y)$. 


Anyway, getting back on track, we can now write the design matrix $bold(A)$ for the generalized $N times N$ case. Each row corresponds to one sampled pixel in the support, and each column corresponds to one monomial term in the basis. The $i$-th row is simply the basis vector evaluated at the coordinates of the $i$-th pixel.

$
  bold(A) =
  mat( delim: "[",
    phi_d(x_1, y_1)^top;
    phi_d(x_2, y_2)^top;
    dots.v;
    phi_d(x_(N_p), y_(N_p))^top
  ).
$ <eq:designn>

=== The Data Vector and System Form <sec:nxn-data-system>

To complete the linear system, we stack the observed pixel intensities from those same sample locations, in the same order used for the rows of $bold(A)$.

$
  bold(b) =
  [
    f(x_1, y_1),
    f(x_2, y_2),
    dots,
    f(x_(N_p), y_(N_p))
  ]^top.
$ <eq:bvecn>

Using the same ordering in both objects, row $i$ of $bold(A)$ is built from the coordinates $(x_i, y_i)$, and entry $i$ of $bold(b)$ is the measured intensity at that same location. With those definitions, the generalized polynomial-fitting problem is

$
  bold(A) bold(z) approx bold(b).
$ <eq:systemn>

== Solving for the Derivative Weights <sec:nxn-weights>

=== The Pseudoinverse Map <sec:nxn-pseudoinverse>

This system is traditionally overdetermined by design (and we will later discuss the reason for such and a proof will be provided), so we solve for the coefficient vector in the least-squares sense. The fitted coefficient vector is

$
  hat(bold(z)) = (bold(A)^top bold(A))^(-1) bold(A)^top bold(b) = bold(P) bold(b).
$ <eq:lstsqn>

Where $hat(bold(z))$ means the estimated coefficient vector. We also define

$
  bold(P) = (bold(A)^top bold(A))^(-1) bold(A)^top
$ <eq:pinvn>

=== Extracting the First-Derivative Rows <sec:nxn-derivative-row>

The matrix $bold(P)$ is the pseudoinverse that maps sampled intensities directly to fitted coefficients. If the basis is ordered as $[1, x, y, dots]^top$, then the coefficient of $x$ is the estimate of the first derivative in the $x$-direction, and the coefficient of $y$ is the estimate of the first derivative in the $y$-direction. Those coefficients are entries of $hat(bold(z))$, so the corresponding rows of $bold(P)$ are the two derivative weight vectors

$
  bold(p)_(f_x)^top,
$ <eq:px-row>

and

$
  bold(p)_(f_y)^top.
$ <eq:py-row>

Applying those rows to the data vector gives the derivative estimates

$
  hat(f)_x = bold(p)_(f_x)^top bold(b).
$ <eq:fx-est>

$
  hat(f)_y = bold(p)_(f_y)^top bold(b).
$ <eq:fy-est>

=== Arranging the Rows into Square Stencils <sec:nxn-square-stencils>

Because the support is a square and the samples can be ordered in row-major form, the derivative rows can be reshaped directly back into `N x N` filter kernels:

$
  bold(K)_x^("square") = "reshape"_(N,N)(bold(p)_(f_x)^top), quad bold(K)_y^("square") = "reshape"_(N,N)(bold(p)_(f_y)^top).
$ <eq:nxn-kernels>

This is the exact analogue of the `3 x 3` construction in @sec:3x3-stencil-formation, except now the support size is arbitrary. The polynomial fit still produces two ordinary stencils, one for the horizontal derivative and one for the vertical derivative.

=== Combining the Square-Support Responses <sec:nxn-combine>

Once those square-support kernels are formed, they are applied exactly as in the small example. The two filter outputs are

$
  G_x^("square") = bold(K)_x^("square") * I, quad G_y^("square") = bold(K)_y^("square") * I.
$ <eq:nxn-responses>

The corresponding edge-strength measure is

$
  G^("square") = sqrt((G_x^("square"))^2 + (G_y^("square"))^2),
$ <eq:nxn-mag>

and the associated orientation estimate is

$
  theta^("square") = "atan2"(G_y^("square"), G_x^("square")).
$ <eq:nxn-theta>

So the logic is exactly the same as in the `3 x 3` example. First choose sample locations. Then evaluate the monomial basis there to build $bold(A)$. Then stack the measured intensities into $bold(b)$ in the same order. Solving the least-squares problem gives $hat(bold(z))$, and the derivative rows of the pseudoinverse become square filter stencils whose responses can then be combined into a gradient magnitude and orientation.

= Visualizing the Square-Support Filters <sec:square-visualization>

For a concrete visual example, it is useful to inspect the actual square-support kernels produced by this construction. If we choose $N = 15$, then the corresponding half-width is

$
  h = (N - 1)/2 = 7.
$ <eq:square-h-15>

With that support fixed, Figures @fig:square-d1, @fig:square-d3, and @fig:square-d5 show the square-support derivative stencils for polynomial degrees $d = 1$, $d = 3$, and $d = 5$.

#figure(
  image("figures/fig_square_filter_d1.pdf", width: 80%),
  caption: [
    Square-support derivative kernels for a `15 x 15` neighborhood with polynomial degree $d = 1$. The left panel shows $bold(K)_x^("square")$ and the right panel shows $bold(K)_y^("square")$. Garnet denotes positive weights, Atlantic denotes negative weights, and white denotes weights near zero.
  ],
) <fig:square-d1>

#figure(
  image("figures/fig_square_filter_d3.pdf", width: 80%),
  caption: [
    Square-support derivative kernels for a `15 x 15` neighborhood with polynomial degree $d = 3$.
  ],
) <fig:square-d3>

#figure(
  image("figures/fig_square_filter_d5.pdf", width: 80%),
  caption: [
    Square-support derivative kernels for a `15 x 15` neighborhood with polynomial degree $d = 5$.
  ],
) <fig:square-d5>

Several points are worth noting. First, the degree-1 kernels have the simplest possible structure. They form a monotone left-to-right ramp for $bold(K)_x^("square")$ and the corresponding top-to-bottom ramp for $bold(K)_y^("square")$. Second, once the degree is increased to $d = 3$, the kernels become more concentrated near the middle of the support and develop small opposite-sign boundary lobes near the outer edges. That same trend becomes even more pronounced for $d = 5$, where the central positive and negative bands strengthen further and the outer oscillatory structure is easier to see. Finally, each panel retains the expected square-support symmetry. In every case, the $y$-derivative kernel is the transpose of the $x$-derivative kernel, so the two heatmaps contain the same structure rotated by ninety degrees.

This figure also helps explain why the support geometry matters so much. Even before the circular and rotated constructions are introduced, the square support already imposes a visible spatial pattern on where the derivative estimate places its positive and negative weight. The next section changes only the support geometry, allowing us to isolate how much of the filter shape comes from the neighborhood itself rather than from the least-squares logic.

= Adapting to a Circular Support Region <sec:circular-support>

== Restricting the Sample Locations <sec:circular-samples>

Classical Savitzky--Golay filters and most of their descendants are usually presented on rectangular, axis-aligned neighborhoods, largely because those supports make both the algebra and the implementation straightforward @savitzkygolay1964 @luo2005sg2d. Now, suppose one whose hubris is such that they believe the circular case is a new algorithm. This modification would hypothetically allows for a new filter to better match the orientation-dependent structure of edges while preserving the underlying least-squares polynomial framework.

We can derive this new algorithm by considering the circular support, as it is the same least-squares construction, just with a different set of sample locations. Instead of using every point in a square, keep only the points satisfying the circular support constraint(better known by laypersons as myself as the eqaution of a circle):

$
  x_i^2 + y_i^2 <= r^2.
$ <eq:circle-support>

== Reusing the Least-Squares Construction <sec:circular-lstsq>

Now, the simiplifcaiotn from the prior section of redfining the $N_p$ rahter than $N^2$ will sae us some time. Defining the data vector to only contain the pixel values inside that circular support, we derive $bold(b)$ as

$
  bold(b) = [I_1, I_2, dots, I_(N_p)]^top,
$ <eq:circle-bvec>

and the design matrix keeps only the corresponding rows

$
  bold(A) =
  mat( delim: "[",
    phi_d(x_1, y_1)^top;
    phi_d(x_2, y_2)^top;
    dots.v;
    phi_d(x_(N_p), y_(N_p))^top
  ),
$ <eq:circle-design>

where the coordinates now come from the circular neighborhood rather than the full square. Naturally, the least-squares solution remains unchanged

$
  hat(bold(z)) = (bold(A)^top bold(A))^(-1) bold(A)^top bold(b).
$ <eq:circle-lstsq>

and again, the derivative estimates are obtained from rows of the pseudoinverse

$
  hat(f)_x = bold(p)_(f_x)^top bold(b).
$ <eq:circle-fx>

$
  hat(f)_y = bold(p)_(f_y)^top bold(b).
$ <eq:circle-fy>

=== Embedding the Circular Weights into Stencils <sec:circular-stencils>

The vectors $bold(p)_(f_x)^top$ and $bold(p)_(f_y)^top$ still define discrete filters, but now only at the sample locations that lie inside the circle. To implement them as image stencils, we place those weights at the corresponding coordinates of the bounding square and assign zero weight to locations outside the circular support. That produces two circular-support masks, which we denote by $bold(K)_x^("circ")$ and $bold(K)_y^("circ")$.

=== Combining the Circular-Support Responses <sec:circular-combine>

Once those circular-support stencils are formed, the corresponding filter outputs are

$
  G_x^("circ") = bold(K)_x^("circ") * I, quad G_y^("circ") = bold(K)_y^("circ") * I.
$ <eq:circle-responses>

From them we define the circular-support gradient magnitude

$
  G^("circ") = sqrt((G_x^("circ"))^2 + (G_y^("circ"))^2),
$ <eq:circle-mag>

and the associated orientation estimate

$
  theta^("circ") = "atan2"(G_y^("circ"), G_x^("circ")).
$ <eq:circle-theta>

So the circle support is just a different choice of sample locations, but the underlying least-squares logic is exactly the same. The resulting filter weights are different because the sample locations are different, but the mathematical construction is identical. We will show a visual represetnation of the differnce of these two supports later, but for now, the main takeaway is that the circular support is just a different set of sample locations, and the least-squares logic applies in exactly the same way.

= The Wide View Filter Formulation <sec:wvf-formulation>

== Why Rotate the Coordinate System? <sec:wvf-motivation>

Now, looking at the existing work, perhaps one would think that the somewhat arbitrary x and y direction of the filters is a problem. After all, edges can be oriented in any direction, so why should we only look at changes in the horizontal and vertical directions? To address this issue, a few solutions have been introduced over the years, including orientable anisotropic filters as well as steerable filters. 

The Wide View Filter (WVF) builds on top of this long history and takes the same polynomial-fitting approach but applies it in a rotated coordinate system like those of the prior oriented anisotropic filters, although WVF is not intrinsically anisotropic (it can be however anisotropic if setup poorly by selecting a poor support or $N_p$). This way, the derivative is taken in the direction that is most likely to capture the edge's structure.

== Rotated Local Coordinates and the Final Stencil <sec:wvf-stencil>

Unlike the Savitzky--Golay derivative filters derived in the previous sections, the WVF does not use a paired horizontal and vertical response. In the square and circular constructions, we extracted both derivative directions and then combined them afterward. Here, the goal is different. Since the filter is explicitly rotated through a collection of candidate orientations, the WVF only uses the derivative aligned with the rotated $x'$ direction. The orthogonal $y'$ derivative may still exist within the fitted polynomial, but it is not used to form the WVF response stencil.

=== The Rotation Step <sec:wvf-rotation>

Now, the WVF's changes to the Golay-Savitsky derivative filters are actually quite minimal. First, it rotates the local coordinates so that the derivative is taken in the candidate normal direction:

$
  x_i' = Delta x_i cos theta + Delta y_i sin theta,
$ <eq:rotate-x>
$
  y_i' = -Delta x_i sin theta + Delta y_i cos theta.
$ <eq:rotate-y>

=== The Orientation-Dependent Stencil <sec:wvf-oriented-stencil>

Second, it uses a circular support instead of a square one. But once the rotated circular sample locations are fixed, the same least-squares logic from Golay-Savitzky applies. The only difference is that the design matrix is now built from the rotated coordinates, so the $i$-th row of $bold(A)$ is $phi_d(x_i', y_i')^top$ instead of $phi_d(x_i, y_i)^top$. The data vector $bold(b)$ is unchanged since it still contains the same pixel intensities. With those definitions, the least-squares solution is

$
  hat(bold(z)) = bold(P)_theta bold(b),
$ <eq:wvf-lstsq>

and the WVF weight vector is just the row of $bold(P)_theta$ corresponding to the normal derivative coefficient:

$
  hat(f)_(x') = bold(p)_(f_(x'))^top bold(b).
$ <eq:wvf-fx>

That row is the WVF stencil. In other words, once the angle $theta$, the polynomial degree, and the circular support are fixed, the directional derivative estimate becomes a single weighted sum of the pixel intensities in the local neighborhood. Those weights are exactly the coefficients in $bold(p)_(f_(x'))^top$, so they define the discrete filter that is applied to the image. Each candidate orientation therefore has its own stencil, and applying that stencil gives the WVF estimate of the derivative in the rotated $x'$ direction, which is the candidate edge-normal direction.



#pagebreak()

#bibliography("journal_paper/refs.bib")
