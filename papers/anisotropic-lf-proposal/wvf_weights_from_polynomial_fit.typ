#set page(margin: (top: 0.8in, bottom: 0.8in, left: 0.85in, right: 0.85in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1.1")
#set par(justify: true, leading: 0.6em)
#set math.equation(numbering: (..nums) => {
  let heading-num = counter(heading).get()
  let section-num = if heading-num.len() > 0 { heading-num.at(0) } else { 0 }
  numbering("(1.1)", section-num, nums.pos().at(0))
})
#show heading.where(level: 1): it => {
  counter(math.equation).update(0)
  set text(size: 13pt)
  it
}
#show ref: it => {
  let el = it.element
  if el != none and el.func() == math.equation {
    let eq-num = counter(math.equation).at(el.location())
    let heading-num = counter(heading).at(el.location())
    link(el.location(), [Equation #numbering("1.1", heading-num.at(0), eq-num.at(0))])
  } else {
    it
  }
}
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

For a visual example, we can inspect the actual square-support kernels produced by this construction. If we choose $N = 15$, then the corresponding half-width is

$
  h = (N - 1)/2 = 7.
$ <eq:square-h-15>

With that support fixed, @fig:square-d1, @fig:square-d3, and @fig:square-d5 show the square-support derivative stencils for polynomial degrees $d = 1$, $d = 3$, and $d = 5$.

#figure(
  image("figures/fig_square_filter_d1.pdf", width: 80%),
  caption: [
    Square-support derivative kernels for a `15 x 15` neighborhood with polynomial degree $d = 1$. The left panel shows $bold(K)_x^("square")$ and the right panel shows $bold(K)_y^("square")$. 
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

There is also a useful parity observation hiding in these figures. For a centered symmetric square support, the first-derivative stencil changes only when the polynomial basis gains new terms with the symmetry needed to contribute to $f_x$ or $f_y$. As a result, the square-support derivative kernels tend to appear in odd-even pairs. In practice, this means that $d = 1$ and $d = 2$ look the same, $d = 3$ and $d = 4$ look the same, and $d = 5$ and $d = 6$ look the same. This is precisely why the visual comparison above uses $d = 1$, $d = 3$, and $d = 5$ rather than the intervening even degrees. Those odd degrees are the ones at which the square-support first-derivative stencils visibly change.

This figure also helps explain why the support geometry matters so much. Even before the circular and rotated constructions are introduced, the square support already imposes a visible spatial pattern on where the derivative estimate places its positive and negative weight. The next section changes only the support geometry, allowing us to isolate how much of the filter shape comes from the neighborhood itself rather than from the least-squares logic.

= Adapting to a Circular Support Region <sec:circular-support>

== Restricting the Sample Locations <sec:circular-samples>

Classical Savitzky--Golay filters and most of their descendants are usually presented on rectangular, axis-aligned neighborhoods, largely because those supports make both the algebra and the implementation straightforward @savitzkygolay1964 @luo2005sg2d. The circular case uses the same least-squares construction, but changes the set of sample locations so that the neighborhood is defined by radius rather than by axis-aligned extent.

We can derive this circular version by restricting the samples to those that satisfy the standard circle constraint:

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

= Visualizing the Circular-Support Filters <sec:circular-visualization>

To make that geometric change concrete, it is useful to visualize the corresponding circular-support derivative stencils. For a fair comparison with the square case, we keep the same bounding window size $N = 15$ and therefore the same half-width $h = 7$. The circular support is then defined as the exact integer lattice disk

$
  x_i^2 + y_i^2 <= 7^2.
$ <eq:circle-r7>

With this choice, the support contains

$
  N_p = 149
$

sample locations. That count is not chosen arbitrarily. It is simply the number of integer grid points inside the radius-7 disk. This exact construction is important because it preserves isotropy in the discrete support itself. By contrast, if one selects an arbitrary odd $N_p$ by taking the nearest pixels in distance order, the final few accepted points can depend on tie-breaking among equally distant lattice sites, which can introduce a slight directional bias into what is supposed to be a circular neighborhood.

With that isotropic support fixed, Figures @fig:circle-d1, @fig:circle-d3, and @fig:circle-d5 show the circular-support derivative stencils for polynomial degrees $d = 1$, $d = 3$, and $d = 5$.

#figure(
  image("figures/fig_circle_filter_d1.pdf", width: 80%),
  caption: [
    Circular-support derivative kernels for the exact radius-7 disk inside a `15 x 15` bounding box, with polynomial degree $d = 1$. The left panel shows $bold(K)_x^("circ")$ and the right panel shows $bold(K)_y^("circ")$.
  ],
) <fig:circle-d1>

#figure(
  image("figures/fig_circle_filter_d3.pdf", width: 80%),
  caption: [
    Circular-support derivative kernels for the exact radius-7 disk with polynomial degree $d = 3$.
  ],
) <fig:circle-d3>

#figure(
  image("figures/fig_circle_filter_d5.pdf", width: 80%),
  caption: [
    Circular-support derivative kernels for the exact radius-7 disk with polynomial degree $d = 5$.
  ],
) <fig:circle-d5>

The same odd-even pairing seen in the square-support case appears here as well. For this centered isotropic support, the first-derivative stencils again change only when the polynomial basis gains new terms with the correct symmetry to contribute to $f_x$ or $f_y$. Consequently, the visually distinct cases occur at $d = 1$, $d = 3$, and $d = 5$, while the intervening even degrees produce the same first-derivative stencils as their preceding odd neighbors. This is why the circular visualizations use the same degree sequence as the square ones.

The circular figures also make the geometric difference from the square support easy to see. The kernels are no longer forced to occupy the corners of the `15 x 15` bounding box, so the weight distribution conforms more closely to a radially symmetric neighborhood. The $x$- and $y$-derivative stencils still remain transposes of one another, but the support itself is now isotropic rather than axis-aligned. This gives a cleaner baseline for the next section, where the coordinate system itself is rotated and only the derivative in the candidate normal direction is retained.

= Rotating the Circular Support <sec:circular-rotation-visualization>

The next step toward the WVF is not to change the support again, but to rotate the derivative direction within that same isotropic circular neighborhood. To illustrate this, we take the same support from @fig:circle-d3, but rotate it across some set of angles, as seen in @fig:circle-orientation-sweep where we fix the exact radius-7 disk inside the `15 x 15` bounding box and shows the directional stencil obtained when the local coordinate system is rotated to $theta = 0degree$, $30degree$, $60degree$, and $90degree$. For this illustration, we use the representative odd degree $d = 3$.

#figure(
  image("figures/fig_circle_orientation_sweep_d3.pdf", width: 100%),
  caption: [
    Orientation sweep for the exact circular-support stencil with `N = 15`, `h = 7`, and polynomial degree $d = 3$. Each panel shows the directional derivative stencil obtained from the same isotropic disk after rotating the local coordinate system to a new angle $theta$.
  ],
) <fig:circle-orientation-sweep>

 The underlying support stays circular, but the positive and negative lobes of the derivative stencil rotate with the chosen coordinate system. In this case, the neighborhood itself remains isotropic while the extracted derivative direction changes. That is precisely the idea used by the WVF. Rather than relying only on the fixed horizontal and vertical derivatives, the filter is evaluated in several candidate directions and the directional derivative _most_ aligned with the candidate normal is retained.

= The Wide View Filter Formulation <sec:wvf-formulation>

The WVF keeps the same least-squares construction introduced in @sec:circular-support, but changes the coordinate system in which the derivative is taken. Once that rotated directional stencil has been defined for one candidate angle, the filter is evaluated over a finite set of candidate angles and the strongest directional response is retained.

== Rotated Local Coordinates and the Directional Stencil <sec:wvf-stencil>

Unlike the Savitzky--Golay derivative filters derived in the previous sections, the WVF does not use a paired horizontal and vertical response. In the square and circular constructions, @eq:fx-est and @eq:fy-est showed that the fit naturally produces both $hat(f)_x$ and $hat(f)_y$. Here the objective is different. The WVF rotates the local coordinates first, and then keeps only the derivative aligned with the rotated $x'$ direction. The orthogonal $y'$ coefficient may still appear in the fitted coefficient vector, but it is not used when the WVF forms its final response.

=== The Rotation Step <sec:wvf-rotation>

The first change is to rotate the local coordinates so that the derivative is taken in a candidate normal direction:

$
  x_i' = Delta x_i cos theta + Delta y_i sin theta,
$ <eq:rotate-x>
$
  y_i' = -Delta x_i sin theta + Delta y_i cos theta.
$ <eq:rotate-y>

=== The Orientation-Dependent Stencil <sec:wvf-oriented-stencil>

The second change is that the WVF uses the circular support from @sec:circular-support rather than the square support from @sec:nxn-square. Once the rotated coordinates in @eq:rotate-x and @eq:rotate-y are fixed, the same least-squares logic used in @eq:circle-lstsq still applies. Using @eq:rotate-x and @eq:rotate-y, the $i$-th row of the design matrix becomes $phi_d(x_i', y_i')^top$ instead of $phi_d(x_i, y_i)^top$. The data vector $bold(b)$ is unchanged, because it still contains the same sampled pixel intensities as in @eq:circle-lstsq. With those definitions, the fitted coefficient vector is

$
  hat(bold(z)) = bold(P)_theta bold(b),
$ <eq:wvf-lstsq>

Using @eq:wvf-lstsq, the row of $bold(P)_theta$ corresponding to the $x'$ coefficient gives the directional derivative estimate

$
  hat(f)_(x') = bold(p)_(f_(x'))^top bold(b).
$ <eq:wvf-fx>

Equation @eq:wvf-fx shows that the directional derivative is again a weighted sum of the sampled intensities, just as in @eq:fx-est and @eq:circle-fx. Therefore the coefficients in $bold(p)_(f_(x'))^top$ are the WVF stencil for the candidate angle $theta$. Each candidate angle has its own stencil, and applying that stencil gives the WVF estimate of the derivative in the rotated $x'$ direction.

== Discrete Orientation Sampling <sec:wvf-sampling>

The WVF does not stop after constructing one stencil. Instead, it constructs one orientation-dependent stencil for each sampled angle. Since the rotated $x'$ axis at angle $theta + pi$ points in the opposite direction from the rotated $x'$ axis at angle $theta$, the directional derivative changes sign but not magnitude. Because of that sign symmetry, it is sufficient to sample candidate angles on the interval $[0, pi)$. We therefore define the sampled angle set by

$
  Theta_(N_s) = { theta_k = k pi / N_s : k = 0, 1, dots, N_s - 1 }.
$ <eq:wvf-angle-set>

Using @eq:wvf-fx, the directional response at the candidate angle $theta_k$ is

$
  R(theta_k) = bold(p)_(f_(x'))(theta_k)^top bold(b).
$ <eq:wvf-response>

Equation @eq:wvf-response makes the WVF comparison step explicit. For every sampled angle in @eq:wvf-angle-set, the filter evaluates one weighted sum of the sampled intensities. Because the WVF is built from the $x'$ row of $bold(P)_theta$ in @eq:wvf-fx, only the rotated $x'$ derivative is used when the candidate angles are compared.

== Selection of the Winning Direction <sec:wvf-selection>

Once the candidate responses in @eq:wvf-response have been computed, the WVF keeps the angle with the largest absolute directional response. In symbols,

$
  theta_* = arg max_(theta_k in Theta_(N_s)) |R(theta_k)|.
$ <eq:wvf-theta-star>

Using @eq:wvf-theta-star, the WVF magnitude returned at that pixel is

$
  G^("WVF") = |R(theta_*)|.
$ <eq:wvf-mag>

Equations @eq:wvf-angle-set, @eq:wvf-response, @eq:wvf-theta-star, and @eq:wvf-mag together complete the WVF definition. The filter first samples a discrete set of candidate angles, then evaluates the directional derivative in the rotated $x'$ direction for each angle, and finally returns the angle and magnitude associated with the strongest absolute response.

= Low-Degree Overdetermined Fits <sec:low-degree-overdetermined>

The WVF construction above is still built from a local polynomial fit, so the quality of the filter depends on the quality of that fit. For that reason, the next step is to explain why low-degree fits on well-sampled supports are preferable when the goal is stable first-derivative estimation.

== Exact Recovery for Affine Data <sec:affine-exact>

The cleanest case is an affine intensity field, which simply means the intensity is a linear function of the spatial coordinates. We can define the intensity function of such a field as

$
  f_"aff"(x, y) = a_0 + a_1 x + a_2 y.
$ <eq:affine-field>

If the sampled data are generated exactly from @eq:affine-field, then the sampled intensity vector can be written as

$
  bold(b)_0 = bold(A)_1 bold(z)_"aff",
$ <eq:affine-data>

where $bold(A)_1$ is the degree-1 design matrix and $bold(z)_"aff" = [a_0, a_1, a_2]^top$. Using the least-squares map from @eq:lstsqn and the pseudoinverse definition from @eq:pinvn, the fitted coefficient vector is

$
  hat(bold(z))
  = bold(P)_1 bold(b)_0
  = (bold(A)_1^top bold(A)_1)^(-1) bold(A)_1^top bold(A)_1 bold(z)_"aff"
  = bold(z)_"aff".
$ <eq:affine-recovery>

@eq:affine-recovery shows that the degree-1 fit recovers the affine coefficients exactly whenever the data are noiseless and truly affine. Since the coefficient of $x$ in @eq:affine-field is $a_1$ and the coefficient of $y$ is $a_2$, @eq:affine-recovery implies

$
  hat(f)_x = a_1, quad hat(f)_y = a_2.
$ <eq:affine-derivatives>

@eq:affine-derivatives is the exact-recovery statement we need. For affine data, the degree-1 polynomial already contains the entire local signal, so higher polynomial degree does not add any new first-derivative information in the noiseless case.

== The Affine Model as a Local Approximation <sec:affine-local>

The affine model in @eq:affine-field is not intended as a global model of the image, rather it is a local approximation. We assume local smoothness because the goal is to estimate local derivatives and derivatives and Taylor expansions are only meaningful where the intensity field is sufficiently regular. Under that assumption, the first-order Taylor expansion provides the affine approximation we use.  

If the underlying intensity field is smooth near the center pixel, then its first-order Taylor expansion about that center point is

$
  f(x, y) = f(0, 0) + f_x(0, 0) x + f_y(0, 0) y + r_2(x, y),
$ <eq:taylor-first>

where $r_2(x, y)$ collects every quadratic and higher-order term. Using Taylor's theorem, the magnitude of that remainder satisfies a bound of the form

$
  |r_2(x, y)| <= (1/2) sup_((u, v) in Omega_h) ||bold(H)_f(u, v)|| ||(x, y)||^2,
$ <eq:taylor-remainder>

where $Omega_h$ is the local support window and $bold(H)_f$ is the Hessian of the intensity field. Equation @eq:taylor-first shows that the affine term is the leading local approximation, while @eq:taylor-remainder shows that the approximation error grows quadratically with distance from the expansion point. Over a small enough support, the tangent line is the first-order approximation to the smooth profile, and @eq:taylor-remainder tells us how the neglected curvature enters.

== Noise Propagation Through the Derivative Stencil <sec:noise-propagation>

To discuss robustness, we now write the sampled data as the sum of a clean signal and additive noise, where $bold(b)_0$ is the clean signal and $bold(epsilon)$ is the noise vector

$
  bold(b) = bold(b)_0 + bold(epsilon),
$ <eq:noise-model>

with $E$ denoting the expectation operator. We assume that the noise is zero-mean and isotropic, so its mean and covariance are

$
  E[bold(epsilon)] = 0, quad "Cov"(bold(epsilon)) = sigma^2 bold(I).
$ <eq:noise-stats>

Using the derivative-row relation in @eq:fx-est, the noisy derivative estimate is

$
  hat(f)_x = bold(p)_(f_x)^top bold(b)
  = bold(p)_(f_x)^top bold(b)_0 + bold(p)_(f_x)^top bold(epsilon).
$ <eq:noise-fx>

Taking expectations in @eq:noise-fx and using @eq:noise-stats gives

$
  E[hat(f)_x] = bold(p)_(f_x)^top bold(b)_0.
$ <eq:noise-mean>

@eq:noise-mean shows that zero-mean noise does not change the average fitted derivative. The remaining effect of noise is therefore the random spread of the estimator around that average, which is measured by the variance. Using @eq:noise-fx and @eq:noise-stats,

$
  "Var"(hat(f)_x)
  = bold(p)_(f_x)^top "Cov"(bold(epsilon)) bold(p)_(f_x)
  = sigma^2 bold(p)_(f_x)^top bold(p)_(f_x)
  = sigma^2 ||bold(p)_(f_x)||^2.
$ <eq:noise-variance>

@eq:noise-variance gives a direct robustness criterion. If the image noise level sigma^2 is fixed, then the only factor controlling the variance of the derivative estimate is $||bold(p)_(f_x)||^2$. In this sense, the squared norm of the derivative row is the noise gain of the stencil. Larger derivative-row norms therefore mean stronger noise amplification and a less stable estimate.

== Higher-Degree Fits as Less Overdetermined Systems <sec:high-degree-overdetermined>

The basis-count result in @eq:mbasis shows that the number of unknown coefficients grows with degree, while the minimum-data requirement in @eq:min-data shows that the support size must remain at least as large as that coefficient count. If the support size $N_p$ is fixed, then increasing the degree must decrease the excess number of samples,

$
  E_d = N_p - M_d,
$ <eq:excess-samples>

where $M_d$ is the basis size from @eq:mbasis. @eq:excess-samples shows that higher degree makes the fit less overdetermined. That loss of redundancy can be written directly in terms of the derivative stencil. 

For a fixed support and a fixed degree $d$, let $bold(A)_d$ be the design matrix and let $bold(e)_(f_x)^(d)$ be the unit vector that selects the $x$ coefficient. If a degree-$d$ polynomial has coefficient vector $bold(z)_d$, then its sampled values are $bold(A)_d bold(z)_d$. If a stencil $bold(alpha)$ is to recover the $x$ coefficient from those samples for every possible $bold(z)_d$, then we must have

$
  bold(alpha)^top bold(A)_d bold(z)_d = bold(e)_(f_x)^(d top) bold(z)_d
$

for every coefficient vector $bold(z)_d$. Since that identity must hold for every $bold(z)_d$, it is equivalent to the linear constraint

$
  bold(A)_d^top bold(alpha) = bold(e)_(f_x)^(d).
$ <eq:exactness-constraint>

@eq:exactness-constraint states that the stencil reproduces the $x$ coefficient exactly for every polynomial in the degree-$d$ basis. Using the pseudoinverse definition in @eq:pinvn, the derivative row is the Moore-Penrose solution of those linear constraints, so among all vectors satisfying @eq:exactness-constraint it is the minimum-norm choice:

$
  bold(p)_(f_x)^(d) = arg min_(bold(alpha)) ||bold(alpha)||^2
  quad "subject to" quad
  bold(A)_d^top bold(alpha) = bold(e)_(f_x)^(d).
$ <eq:min-norm-stencil>

Now compare degree $d$ and degree $d+1$ on the same support. Since every degree-$d$ polynomial is also a degree-$(d+1)$ polynomial, the degree-$(d+1)$ exactness conditions include all of the degree-$d$ conditions and then add more. Therefore the feasible set defined by @eq:exactness-constraint for degree $d+1$ is a subset of the feasible set for degree $d$. Using @eq:min-norm-stencil, the minimum achievable stencil norm cannot decrease when the degree is raised:

$
  ||bold(p)_(f_x)^(d+1)|| >= ||bold(p)_(f_x)^(d)||.
$ <eq:norm-monotone>

Using @eq:noise-variance and @eq:norm-monotone, the derivative variance under the additive noise model cannot decrease when the degree is raised on the same support:

$
  "Var"(hat(f)_x^(d+1)) >= "Var"(hat(f)_x^(d)).
$ <eq:variance-monotone>

@eq:variance-monotone is the central tradeoff. Higher degree enforces exactness for a larger polynomial space, but it does so by imposing more cancellation constraints on the derivative stencil. Those extra cancellations are what make the higher-degree kernels in @fig:square-d3, @fig:square-d5, @fig:circle-d3, and @fig:circle-d5 look more oscillatory than their lower-degree counterparts. The higher-degree fit is therefore less overdetermined, more constrained, _*and*_  more noise sensitive, even before one discusses any empirical performance differences.

= Symmetry, Redundancy, and the Practical Limits of WVF <sec:wvf-redundancy>

The WVF is still built from the same local polynomial fit, so the next question is whether rotating the coordinate system creates any new first-order information. Thankfully, the answer can be determined by following the geometry of the support, the chain rule for directional derivatives, and the least-squares objective itself.

== Fixed Circular Support Under Rotation <sec:wvf-fixed-support>

Let the circular support be the fixed set of sample locations

$
  S_r = { (Delta x_i, Delta y_i) in ZZ^2 : Delta x_i^2 + Delta y_i^2 <= r^2 }.
$ <eq:wvf-support>

Using @eq:wvf-support, the support is chosen before any angle is considered. Using @eq:rotate-x and @eq:rotate-y, the WVF then rotates only the coordinate description of those same sample locations. This means that the sampled intensity vector $bold(b)$ does not change with angle. Only the coordinates used to build the design matrix change.

== Directional Derivatives from the Fitted Gradient <sec:wvf-directional-derivatives>

To express the rotated derivative in terms of the fitted gradient, we first invert the rotation in @eq:rotate-x and @eq:rotate-y. The inverse relations are

$
  x = x' cos theta - y' sin theta,
$ <eq:rotate-inverse-x>
$
  y = x' sin theta + y' cos theta.
$ <eq:rotate-inverse-y>

Using @eq:rotate-inverse-x and @eq:rotate-inverse-y, the chain rule gives

$
  partial / partial x'
  = cos theta partial / partial x + sin theta partial / partial y.
$ <eq:chain-xprime>

Applying @eq:chain-xprime to the fitted polynomial at the center point and using the derivative definitions from @eq:fx-est and @eq:fy-est yields

$
  hat(f)_(x')(theta) = cos theta hat(f)_x + sin theta hat(f)_y.
$ <eq:directional-gradient>

@eq:directional-gradient shows that once $hat(f)_x$ and $hat(f)_y$ are known, the directional derivative in any rotated direction is already determined.

== The Continuous Optimal Direction <sec:wvf-continuous-direction>

@eq:directional-gradient can be rewritten in amplitude-phase form. Define

$
  G = sqrt(hat(f)_x^2 + hat(f)_y^2), quad theta_g = "atan2"(hat(f)_y, hat(f)_x).
$ <eq:gradient-polar>

Using @eq:gradient-polar, the directional derivative in @eq:directional-gradient becomes

$
  hat(f)_(x')(theta) = G cos(theta - theta_g).
$ <eq:directional-phase>

Since the largest possible magnitude of $cos(theta - theta_g)$ is 1, @eq:directional-phase implies

$
  max_(theta in [0, pi)) |hat(f)_(x')(theta)| = G = sqrt(hat(f)_x^2 + hat(f)_y^2).
$ <eq:directional-max>

@eq:directional-max shows that a continuous search over direction returns exactly the gradient magnitude, while the maximizing direction is the gradient direction $theta_g$ from @eq:gradient-polar. So, in the case where we rotate the WVF in continuous setting, the directional sweep does not add new first-order information. It only re-expresses the fitted gradient already dervied from the standard horizontal and vertical fits.

== Rotational Equivariance of the Exact Circular Fit <sec:wvf-equivariance>

The previous subsection showed that directional derivatives of one fitted polynomial are already determined by $hat(f)_x$ and $hat(f)_y$. We now show that the rotated WVF fit is still that same fitted polynomial, only written in a rotated basis.

Let $Pi_d$ denote the space of all bivariate polynomials of total degree at most $d$. For any polynomial $q in Pi_d$, define the rotation map

$
  (T_theta q)(x', y') = q(x' cos theta - y' sin theta, x' sin theta + y' cos theta).
$ <eq:rotation-map>

Using @eq:rotation-map, every monomial in $q$ is evaluated after a linear substitution in $x'$ and $y'$. A linear substitution does not increase total degree, so $T_theta q$ is still in $Pi_d$. Therefore @eq:rotation-map defines a one-to-one correspondence from $Pi_d$ onto itself.

With that notation, the least-squares fit in the original coordinates is

$
  q^* = arg min_(q in Pi_d) sum_i (q(Delta x_i, Delta y_i) - I_i)^2.
$ <eq:lsq-original-poly>

Using the rotated coordinates from @eq:rotate-x and @eq:rotate-y, the WVF fit at angle $theta$ is

$
  q_theta^* = arg min_(tilde(q) in Pi_d) sum_i (tilde(q)(x_i', y_i') - I_i)^2.
$ <eq:lsq-rotated-poly>

Now take any candidate polynomial $q in Pi_d$ in @eq:lsq-original-poly and map it to $tilde(q) = T_theta q$. Using @eq:rotation-map together with @eq:rotate-x and @eq:rotate-y, the residual of $tilde(q)$ at the rotated sample $(x_i', y_i')$ is exactly the residual of $q$ at the original sample $(Delta x_i, Delta y_i)$. Therefore the objective value in @eq:lsq-rotated-poly is the same as the objective value in @eq:lsq-original-poly after the one-to-one change of variables induced by @eq:rotation-map.

Under the same full-rank assumption used in @eq:lstsqn and @eq:pinvn, the least-squares minimizer is unique. Since @eq:lsq-original-poly and @eq:lsq-rotated-poly are the same minimization problem written in two coordinate systems, their minimizers represent the same fitted polynomial. The only thing that changes is the basis in which that polynomial is written. Using that equivalence together with the derivative transformation in @eq:directional-gradient, the rotated $x'$ coefficient in the WVF fit is exactly the directional derivative already determined by $hat(f)_x$ and $hat(f)_y$.

The conclusion is therefore structural rather than numerical. For a fixed support and a complete polynomial basis, the rotated WVF fit does not create new first-order information. It rewrites the same fitted polynomial in rotated coordinates and then extracts the directional derivative already implied by the fitted gradient. If an implementation changes the sampled neighborhood with angle or introduces other anisotropic choices, then this exact equivariance argument no longer applies in the same form.

= From the Wide View Filter to the Line Filter <sec:lf-formulation>

The WVF defined in @sec:wvf-formulation evaluates one orientation-dependent directional stencil at one pixel. The Line Filter (LF) keeps the same WVF stencil at each candidate angle, but evaluates it at several shifted centers and combines the resulting directional responses @bagan2023lf. The next four subsections define those virtual centers, the WVF responses collected along the line, the Gaussian weighting applied to them, and the final orientation-selection rule.

== Virtual Evaluation Points Along the Candidate Direction <sec:lf-virtual-points>

Using the sampled angle set from @eq:wvf-angle-set, the LF first defines integer line offsets along the candidate tangent direction by

$
  (delta x_j(theta_k), delta y_j(theta_k))
  = ("round"(-j sin theta_k), "round"(j cos theta_k)),
  quad j in {-m, dots, m}.
$ <eq:lf-line-offsets>

Equation @eq:lf-line-offsets shows that the parameter $m$ controls the half-length of the virtual line. Since the rotated $x'$ axis in @eq:rotate-x and @eq:rotate-y is the candidate normal direction, the offset in @eq:lf-line-offsets is perpendicular to that normal and therefore follows the candidate tangent. Using @eq:lf-line-offsets, the $j$-th virtual evaluation point for the candidate angle $theta_k$ is

$
  (X_j, Y_j)
  = (X_0 + delta x_j(theta_k), Y_0 + delta y_j(theta_k)).
$ <eq:lf-virtual-points>

Using the circular support from @eq:wvf-support, the sampled intensity vector gathered around the virtual point in @eq:lf-virtual-points is

$
  bold(b)^(j)(theta_k)
  =
  [
    I(X_j + Delta x_1, Y_j + Delta y_1),
    dots,
    I(X_j + Delta x_(N_p), Y_j + Delta y_(N_p))
  ]^top.
$ <eq:lf-point-data>

Equation @eq:lf-point-data is the first part inherited from the WVF. The support offsets $(Delta x_i, Delta y_i)$ are the same offsets used in @eq:wvf-support. The new LF element is that the same support is recentered at every virtual point indexed by $j$.

== WVF Responses Along the Line <sec:lf-line-point-responses>

Using the WVF derivative row in @eq:wvf-fx, the directional response at the $j$-th virtual point is

$
  R_j(theta_k)
  = bold(p)_(f_(x'))(theta_k)^top bold(b)^(j)(theta_k).
$ <eq:lf-point-response>

Equation @eq:lf-point-response shows that each LF line sample is still a WVF response. The orientation-dependent stencil $bold(p)_(f_(x'))(theta_k)^top$ is inherited directly from @eq:wvf-fx, while the virtual-point data vector $bold(b)^(j)(theta_k)$ is the shifted sample vector from @eq:lf-point-data. Therefore the LF does not alter the underlying point-stencil construction. It evaluates that same WVF stencil repeatedly along the line indexed by $j$.

== Gaussian-Weighted Line Response <sec:lf-line-response>

The LF then combines the point responses from @eq:lf-point-response by a Gaussian-weighted average along the line. The line weights are

$
  w_j
  =
  exp(-j^2 / (2 sigma_m^2))
  /
  sum_(q=-m)^m exp(-q^2 / (2 sigma_m^2)),
  quad sigma_m = m / 2.
$ <eq:lf-line-weights>

Using @eq:lf-line-weights and @eq:lf-point-response, the LF response at the candidate angle $theta_k$ is

$
  L(theta_k)
  = sum_(j=-m)^m w_j R_j(theta_k).
$ <eq:lf-line-response>

Equation @eq:lf-line-response shows which part of the LF is new relative to the WVF. The WVF response in @eq:wvf-response is one weighted sum over one support. The LF response in @eq:lf-line-response is a second weighted sum taken over the collection of WVF responses defined by @eq:lf-point-response.

== Selection of the Winning LF Direction <sec:lf-selection>

The LF uses the same discrete angle set from @eq:wvf-angle-set and the same absolute-response selection rule used for the WVF in @eq:wvf-theta-star. The winning LF direction is therefore

$
  theta_*^("LF")
  = arg max_(theta_k in Theta_(N_s)) |L(theta_k)|.
$ <eq:lf-theta-star>

Using @eq:lf-theta-star, the LF magnitude returned at that pixel is

$
  G^("LF") = |L(theta_*^("LF"))|.
$ <eq:lf-mag>

Equations @eq:lf-line-offsets through @eq:lf-mag complete the LF definition. The elements inherited from the WVF are the angle set in @eq:wvf-angle-set and the directional point stencil in @eq:wvf-fx. The new LF elements are the shifted virtual points in @eq:lf-virtual-points, the Gaussian line weights in @eq:lf-line-weights, and the line aggregation in @eq:lf-line-response.

= Collapsing the Line Filter into a Fused Anisotropic Stencil <sec:lf-fused-stencil>

The LF response in @eq:lf-line-response is a weighted sum of the WVF responses from @eq:lf-point-response, and each WVF response is itself a weighted sum of sampled intensities. Because both stages are linear, the entire LF at a fixed angle can be rewritten as one weighted sum over image samples. The next subsections make that collapse explicit and define the fused anisotropic stencil that results from it.

== Expanding the Line Response Into Nested Sums <sec:fused-expand>

Substituting the point response in @eq:lf-point-response into the line response in @eq:lf-line-response gives

$
  L(theta_k)
  =
  sum_(j=-m)^m
  w_j bold(p)_(f_(x'))(theta_k)^top bold(b)^(j)(theta_k).
$ <eq:fused-expand-dot>

Using the sampled intensity vector in @eq:lf-point-data, Equation @eq:fused-expand-dot becomes

$
  L(theta_k)
  =
  sum_(j=-m)^m
  sum_(i=1)^(N_p)
  w_j p_i(theta_k)
  I(X_j + Delta x_i, Y_j + Delta y_i),
$ <eq:fused-expand-sum>

where $p_i(theta_k)$ denotes the $i$-th entry of the WVF derivative row $bold(p)_(f_(x'))(theta_k)^top$ from @eq:wvf-fx. Using the virtual-point coordinates in @eq:lf-virtual-points, the relative image offset addressed by the pair $(j, i)$ is

$
  bold(delta)_(j,i)(theta_k)
  =
  (
    delta x_j(theta_k) + Delta x_i,
    delta y_j(theta_k) + Delta y_i
  ).
$ <eq:fused-raw-offset>

Using @eq:fused-raw-offset, Equation @eq:fused-expand-sum can be written relative to the target pixel as

$
  L(theta_k)
  =
  sum_(j=-m)^m
  sum_(i=1)^(N_p)
  w_j p_i(theta_k)
  I(
    X_0 + delta_(j,i)^x(theta_k),
    Y_0 + delta_(j,i)^y(theta_k)
  ).
$ <eq:fused-expand-relative>

Equation @eq:fused-expand-relative is still the LF. It simply makes each raw pixel contribution explicit.

== Grouping Repeated Pixel Contributions <sec:fused-group>

Equation @eq:fused-expand-relative contains $(2m + 1) N_p$ index pairs $(j, i)$, but those pairs need not address distinct pixels. Using @eq:fused-raw-offset, let

$
  tilde(S)_k
  =
  {
    bold(tilde(delta))_ell(theta_k)
    : ell = 1, dots, N'_k
  }
$ <eq:fused-unique-support>

be an enumeration of the distinct offsets appearing in the raw collection ${bold(delta)_(j,i)(theta_k)}$.

Using @eq:fused-unique-support, Equation @eq:fused-expand-relative can be regrouped by unique pixel location:

$
  L(theta_k)
  =
  sum_(ell=1)^(N'_k)
  sum_({
    (j, i):
    bold(delta)_(j,i)(theta_k) = bold(tilde(delta))_ell(theta_k)
  })
  w_j p_i(theta_k)
  I(
    X_0 + tilde(delta)_ell^x(theta_k),
    Y_0 + tilde(delta)_ell^y(theta_k)
  ).
$ <eq:fused-grouped>

Equation @eq:fused-grouped changes only the order in which the terms from @eq:fused-expand-relative are collected. No approximation has been introduced.

== The Fused Weight Map <sec:fused-alpha>

Using the grouped form in @eq:fused-grouped, define the fused weight assigned to the unique offset $bold(tilde(delta))_ell(theta_k)$ by

$
  alpha_(k, ell)
  =
  sum_({
    (j, i):
    bold(delta)_(j,i)(theta_k) = bold(tilde(delta))_ell(theta_k)
  })
  w_j p_i(theta_k).
$ <eq:fused-alpha>

Substituting @eq:fused-alpha into @eq:fused-grouped yields the fused LF response

$
  L(theta_k)
  =
  sum_(ell=1)^(N'_k)
  alpha_(k, ell)
  I(
    X_0 + tilde(delta)_ell^x(theta_k),
    Y_0 + tilde(delta)_ell^y(theta_k)
  ).
$ <eq:fused-response>

If the sampled intensities at those unique offsets are collected into the vector

$
  bold(g)_k
  =
  [
    I(X_0 + tilde(delta)_1^x(theta_k), Y_0 + tilde(delta)_1^y(theta_k)),
    dots,
    I(X_0 + tilde(delta)_(N'_k)^x(theta_k), Y_0 + tilde(delta)_(N'_k)^y(theta_k))
  ]^top,
$ <eq:fused-data-vector>

then Equation @eq:fused-response can be written compactly as

$
  L(theta_k) = bold(alpha)_k^top bold(g)_k.
$ <eq:fused-dotprod>

Equation @eq:fused-dotprod is the fused anisotropic stencil form of the LF at angle $theta_k$.

== Exact Equivalence Between the LF and the Fused Stencil <sec:fused-equivalence>

The equivalence between the original LF and the fused stencil follows directly from the sequence @eq:lf-line-response, @eq:fused-expand-dot, @eq:fused-expand-sum, @eq:fused-expand-relative, @eq:fused-grouped, and @eq:fused-response. Equation @eq:fused-expand-dot substitutes the WVF point response into the LF line average. Equation @eq:fused-expand-sum expands the inner product over the circular support. Equation @eq:fused-expand-relative rewrites each sampled value as an offset from the target pixel. Equation @eq:fused-grouped then partitions those raw terms according to the unique pixel positions from @eq:fused-unique-support, and Equation @eq:fused-alpha records the total coefficient attached to each unique position. Because each raw term from @eq:fused-expand-relative appears in exactly one group in @eq:fused-grouped and with exactly the same coefficient, Equation @eq:fused-response returns exactly the same scalar value as Equation @eq:lf-line-response for every image patch and every candidate angle.

== Weight Cancellation and Support Overlap <sec:fused-cancellation>

Using @eq:fused-unique-support, the fused support size satisfies

$
  N'_k <= (2m + 1) N_p.
$ <eq:fused-support-bound>

Equation @eq:fused-support-bound reflects the overlap created by the translated supports in @eq:lf-virtual-points. When two or more pairs $(j, i)$ map to the same unique offset in @eq:fused-alpha, their weights are added before the image intensity is sampled. If those contributing terms have opposite signs, then the fused coefficient $alpha_(k, ell)$ in @eq:fused-alpha can be smaller in magnitude than the individual raw terms, and it can vanish when the signed contributions cancel exactly. This behavior follows from support overlap and coefficient aggregation in @eq:fused-alpha. It is therefore a structural property of the fused stencil representation in @eq:fused-response, not a change in the underlying LF defined by @eq:lf-line-response.

= Anisotropic Filter Families <sec:anisotropic-families>

The LF and the fused anisotropic stencil are two oriented constructions built from the WVF response. To compare them with other anisotropic families, it is helpful to place all of the filters in one common coordinate system and then state how each family chooses its support, its weighting law, and its derivative extraction rule.

== A Common Oriented Coordinate System <sec:aniso-common-coords>

Using the same rotation already introduced in @eq:rotate-x and @eq:rotate-y, define the tangent-normal coordinates

$
  u = -Delta x sin theta + Delta y cos theta,
  quad
  v = Delta x cos theta + Delta y sin theta.
$ <eq:aniso-uv>

Equation @eq:aniso-uv identifies $u$ as the tangent coordinate and $v$ as the normal coordinate. To keep the normal derivative coefficient explicit in the polynomial-fit families, define the ordered monomial basis

$
  psi_d(u, v)
  =
  [1, v, u, v^2 / 2, u v, u^2 / 2, dots]^top.
$ <eq:aniso-basis>

Using @eq:aniso-basis, the second coefficient in the fitted parameter vector is always the first derivative in the normal direction $v$.

== Rectangular Anisotropic Polynomial-Fit Filter <sec:rect-poly-fit>

Let $h_u > 0$ and $h_v > 0$ denote the tangent and normal half-widths of an oriented rectangle. Using the coordinates from @eq:aniso-uv, the rectangular support is

$
  S_R(theta)
  =
  {
    (Delta x_i, Delta y_i) in ZZ^2
    :
    |u_i(theta)| <= h_u
    " and "
    |v_i(theta)| <= h_v
  }.
$ <eq:rect-support>

Using @eq:rect-support and the basis in @eq:aniso-basis, the rectangular anisotropic least-squares fit is

$
  hat(bold(z))_R(theta)
  =
  bold(P)_R(theta) bold(b)_R(theta),
  quad
  bold(P)_R(theta)
  =
  (bold(A)_R(theta)^top bold(A)_R(theta))^(-1) bold(A)_R(theta)^top,
$ <eq:rect-lstsq>

where the rows of $bold(A)_R(theta)$ are $psi_d(u_i(theta), v_i(theta))^top$ and $bold(b)_R(theta)$ contains the sampled intensities on $S_R(theta)$. Since the second entry of @eq:aniso-basis is the normal derivative coefficient, the rectangular anisotropic derivative estimate is

$
  hat(f)_v^R(theta)
  =
  bold(p)_(f_v, R)(theta)^top bold(b)_R(theta).
$ <eq:rect-fv>

Equation @eq:rect-fv is the rectangular-support analogue of the circular derivative relation in @eq:circle-fx.

== Elliptical Anisotropic Polynomial-Fit Filter <sec:ellipse-poly-fit>

Now let $a_u > 0$ and $a_v > 0$ denote the semi-axis lengths of an oriented ellipse measured in the same coordinates from @eq:aniso-uv. The elliptical support is

$
  S_E(theta)
  =
  {
    (Delta x_i, Delta y_i) in ZZ^2
    :
    u_i(theta)^2 / a_u^2 + v_i(theta)^2 / a_v^2 <= 1
  }.
$ <eq:ellipse-support>

Using @eq:ellipse-support and the same basis in @eq:aniso-basis, the elliptical anisotropic fit is

$
  hat(bold(z))_E(theta)
  =
  bold(P)_E(theta) bold(b)_E(theta),
  quad
  bold(P)_E(theta)
  =
  (bold(A)_E(theta)^top bold(A)_E(theta))^(-1) bold(A)_E(theta)^top,
$ <eq:ellipse-lstsq>

where the rows of $bold(A)_E(theta)$ are $psi_d(u_i(theta), v_i(theta))^top$ for the samples in $S_E(theta)$. Extracting the normal derivative coefficient gives

$
  hat(f)_v^E(theta)
  =
  bold(p)_(f_v, E)(theta)^top bold(b)_E(theta).
$ <eq:ellipse-fv>

Equations @eq:rect-fv and @eq:ellipse-fv differ only through the support geometry in @eq:rect-support and @eq:ellipse-support.

== Anisotropic Gaussian Derivative Filter <sec:aniso-gaussian>

The anisotropic Gaussian derivative family uses the same coordinates from @eq:aniso-uv, but defines its weights analytically rather than through a polynomial fit. Its envelope is

$
  G_(sigma_u, sigma_v)(u, v)
  =
  exp(-1/2 (u^2 / sigma_u^2 + v^2 / sigma_v^2)).
$ <eq:aniso-gauss-envelope>

Differentiating @eq:aniso-gauss-envelope with respect to the normal coordinate $v$ gives

$
  K_G(u, v)
  =
  -v / sigma_v^2 dot G_(sigma_u, sigma_v)(u, v).
$ <eq:aniso-gauss-kernel>

Using @eq:aniso-gauss-kernel, a truncated anisotropic Gaussian derivative response at one sampled angle $theta_k$ can be written as

$
  R_G(theta_k)
  =
  sum_((Delta x_i, Delta y_i) in S_G(theta_k))
  K_G(u_i(theta_k), v_i(theta_k))
  I_(X_0 + Delta x_i, Y_0 + Delta y_i),
$ <eq:aniso-gauss-response>

where $S_G(theta_k)$ is the chosen finite truncation of the Gaussian support. Equations @eq:rect-fv, @eq:ellipse-fv, and @eq:aniso-gauss-response place the anisotropic polynomial-fit and anisotropic Gaussian families in one common response form.

= Anisotropic Filters Relative to Isotropic Filters <sec:aniso-vs-iso>

The manuscript has now defined both isotropic and anisotropic filter families. The next step is to compare them structurally. This comparison does not rely on empirical results. It follows from the support geometry, the way orientation enters the response, and the parameter sets required to specify each family.

== Support Symmetry <sec:support-symmetry>

For any reference support $S(0)$, let the rotated support family be

$
  S(theta) = T_theta S(0),
$ <eq:support-covariance>

where $T_theta$ denotes rotation by angle $theta$. Using @eq:support-covariance, a support is isotropic when it is invariant under that rotation:

$
  T_theta S = S
  quad
  "for every"
  theta.
$ <eq:support-isotropy>

Equation @eq:support-isotropy is satisfied by the exact circular support in @eq:wvf-support because its membership rule depends only on radius. By contrast, the families defined by @eq:rect-support, @eq:ellipse-support, and the LF construction from @eq:lf-virtual-points through @eq:fused-response are orientation-covariant families of the form @eq:support-covariance.

== Directional Dependence and Angle Sampling <sec:directional-dependence>

In the isotropic polynomial-fit case, the directional derivative at any continuous angle is already determined by the fitted gradient through @eq:directional-gradient, and the maximizing continuous direction follows from @eq:directional-max. The anisotropic families instead define one response for each sampled angle in $Theta_(N_s)$. Their generic selection rule can therefore be written as

$
  G^("aniso")
  =
  max_(theta_k in Theta_(N_s)) |R(theta_k)|.
$ <eq:aniso-sampled-max>

Equation @eq:aniso-sampled-max differs from @eq:directional-max only in the domain over which the maximum is taken. The isotropic fit provides a continuous directional relation through @eq:directional-gradient, whereas the anisotropic families are typically evaluated on the sampled set from @eq:wvf-angle-set.

== Locality and Line Aggregation <sec:locality-aggregation>

Using the oriented coordinates from @eq:aniso-uv, define the tangent and normal half-spans of a support by

$
  D_u(S) = max_((Delta x, Delta y) in S) |u|,
  quad
  D_v(S) = max_((Delta x, Delta y) in S) |v|.
$ <eq:support-spans>

The corresponding aspect ratio is

$
  rho(S) = D_u(S) / D_v(S).
$ <eq:support-aspect>

Equations @eq:support-spans and @eq:support-aspect describe the geometric distinction between the families. The exact circular support in @eq:wvf-support has $rho(S_r)$ close to 1 in every direction. The LF inherits the same circular point support locally, but @eq:lf-virtual-points and @eq:fused-response extend the aggregate support tangentially so that the resulting fused support generally satisfies $rho > 1$. The rectangular, elliptical, and anisotropic Gaussian families express the same geometric idea directly through the separate tangent and normal scales in @eq:rect-support, @eq:ellipse-support, and @eq:aniso-gauss-envelope.

== Parameter Structure <sec:parameter-structure>

The structural difference between isotropic and anisotropic families also appears in their parameter sets. Representative isotropic parameterizations are

$
  Pi_"circ" = {N_p, d},
  quad
  Pi_"sq" = {N, d},
  quad
  Pi_"WVF" = {N_p, d, N_s},
$ <eq:iso-params>

while representative anisotropic parameterizations are

$
  Pi_"LF" = {N_p, d, m, N_s},
  quad
  Pi_"rect" = {h_u, h_v, d, N_s},
  quad
  Pi_"ellip" = {a_u, a_v, d, N_s},
  quad
  Pi_"G" = {sigma_u, sigma_v, N_s}.
$ <eq:aniso-params>

Equations @eq:iso-params and @eq:aniso-params summarize the parameter structure used in the later computational and empirical comparisons. The isotropic polynomial-fit filters are determined primarily by support size and degree. The anisotropic families additionally encode orientation sampling and direction-dependent support shape or aggregation length.

= Isotropic Polynomial-Fit Filters and the Gaussian Reference <sec:iso-and-gaussian>

The previous section compared isotropic and anisotropic families structurally. This section narrows the focus to the isotropic side and then places the isotropic polynomial-fit constructions beside the isotropic Gaussian derivative. The comparison remains mathematical and definitional. It establishes the reference models used in the later sections.

== Circular Low-Degree Polynomial-Fit Filters <sec:circular-low-degree-ref>

The circular-support construction in @sec:circular-support gives the most rotation-symmetric polynomial-fit baseline in the manuscript. Its derivative rows are defined by @eq:circle-fx and @eq:circle-fy, and the corresponding gradient magnitude and orientation are defined by @eq:circle-mag and @eq:circle-theta. The low-degree results in @sec:low-degree-overdetermined further show, through @eq:affine-derivatives and @eq:variance-monotone, that low polynomial degree provides exact affine recovery while keeping the derivative-row norm as small as the degree constraints permit on the chosen support.

Taken together, @eq:circle-fx, @eq:circle-fy, @eq:circle-mag, and @eq:variance-monotone identify the circular low-degree polynomial-fit filter as the primary isotropic baseline within the Savitzky--Golay family. It uses an area support, preserves the rotational symmetry expressed by @eq:support-isotropy, and does not require a sampled-angle sweep once the paired derivative rows have been computed.

== Square Low-Degree Polynomial-Fit Filters <sec:square-low-degree-ref>

The square-support construction from @sec:nxn-square remains a useful secondary isotropic baseline because its geometry and implementation are simple. Its derivative rows are defined by @eq:fx-est and @eq:fy-est, and its combined responses are given by @eq:nxn-mag and @eq:nxn-theta. Although the square support does not satisfy @eq:support-isotropy exactly, the low-degree arguments from @sec:low-degree-overdetermined still apply because they follow from the exactness relation @eq:exactness-constraint and the minimum-norm characterization @eq:min-norm-stencil rather than from circular symmetry alone.

The square filter is therefore a discrete axis-aligned reference rather than the primary isotropic target, while the circular low-degree filter remains the more rotation-symmetric member of the polynomial-fit family.

== The Gaussian Derivative as a Reference Model <sec:gaussian-reference>

The isotropic Gaussian at scale $sigma$ is

$
  G_sigma(x, y)
  =
  1 / (2 pi sigma^2)
  exp(-(x^2 + y^2) / (2 sigma^2)).
$ <eq:gauss-isotropic>

Differentiating @eq:gauss-isotropic with respect to the coordinate axes gives the first-derivative kernels

$
  partial G_sigma / partial x
  =
  -x / sigma^2 dot G_sigma(x, y),
$ <eq:gauss-deriv-x>
$
  partial G_sigma / partial y
  =
  -y / sigma^2 dot G_sigma(x, y).
$ <eq:gauss-deriv-y>

Equations @eq:gauss-isotropic, @eq:gauss-deriv-x, and @eq:gauss-deriv-y define a smooth isotropic gradient model whose scale is controlled by the single parameter $sigma$. Like the isotropic polynomial-fit filters, the Gaussian derivative supplies paired first-derivative components and therefore supports the same continuous directional relation described by @eq:directional-gradient and @eq:directional-max.

== The Isotropic Polynomial-Fit Filter Relative to the Gaussian Reference <sec:iso-vs-gaussian>

The circular low-degree polynomial-fit filter and the isotropic Gaussian derivative address the same local first-derivative problem in different ways. The circular polynomial-fit filter derives its weights from the polynomial exactness conditions in @eq:exactness-constraint and the minimum-norm characterization in @eq:min-norm-stencil. The Gaussian derivative derives its weights from the analytic form of @eq:gauss-isotropic and its derivatives in @eq:gauss-deriv-x and @eq:gauss-deriv-y.

The two constructions nevertheless share several structural properties. Both are isotropic in the sense of @eq:support-isotropy. Both produce paired derivative components that can be combined into gradient magnitude and direction through relations of the form @eq:circle-mag and @eq:gradient-polar. Both also couple smoothing and differentiation into one linear operator. The difference is that the polynomial-fit filter is defined on a finite discrete support with degree parameter $d$, while the Gaussian reference is defined through the continuous scale parameter $sigma$ and then discretized for implementation.

= Computational Structure and Cost <sec:computational-structure>

The theory sections above define several filter families with different support geometries and orientation-handling rules. This section compares their computational structure symbolically. The comparison uses the support sizes, angle counts, and fused-stencil sizes already introduced in the manuscript, and it focuses on the dominant gather-dot-product work required to apply each family at one pixel.

== Operation Counts for the Filter Families <sec:op-counts>

Let $M_d$ denote the basis size from @eq:mbasis. Since the pseudoinverse maps in @eq:pinvn, @eq:wvf-lstsq, @eq:rect-lstsq, and @eq:ellipse-lstsq can be precomputed for a fixed geometry, the runtime cost of applying a filter is determined by how many support samples are multiplied by weights and summed at each pixel. Representative per-pixel counts are

$
  C_"circ" = 2 N_p,
  quad
  C_"sq" = 2 N^2,
  quad
  C_"WVF" = N_s N_p,
$ <eq:cost-isotropic-wvf>

$
  C_"LF" = N_s (2m + 1) N_p,
  quad
  C_"fused" = sum_(k=0)^(N_s - 1) N'_k,
$ <eq:cost-lf-fused>

and

$
  C_"rect" = N_s N_R,
  quad
  C_"ellip" = N_s N_E,
  quad
  C_"aG" = N_s N_G,
$ <eq:cost-aniso-families>

where $N_R$, $N_E$, and $N_G$ denote the active pixel counts of the rectangular, elliptical, and anisotropic Gaussian supports, respectively. Equations @eq:cost-isotropic-wvf through @eq:cost-aniso-families compare the dominant response-evaluation work only. Shared constant-time reductions, such as the final maximum over orientations, are lower-order terms relative to those counts.

== Angle Sweeps and Aggregation Cost <sec:sweep-aggregation-cost>

The first structural distinction is whether the family depends on the sampled angle set $Theta_(N_s)$. The circular and square polynomial-fit filters form paired derivatives directly through @eq:circle-fx, @eq:circle-fy, @eq:fx-est, and @eq:fy-est, so their dominant work in @eq:cost-isotropic-wvf is independent of $N_s$. The WVF, LF, fused stencil, rectangular anisotropic polynomial-fit, elliptical anisotropic polynomial-fit, and anisotropic Gaussian families all scale linearly with $N_s$, as shown by @eq:cost-isotropic-wvf, @eq:cost-lf-fused, and @eq:cost-aniso-families.

The second distinction is whether one candidate orientation requires one support evaluation or several. Equation @eq:lf-line-response shows that the LF evaluates $(2m + 1)$ point responses for each sampled angle, which is why @eq:cost-lf-fused contains the multiplicative factor $(2m + 1) N_p$. Equation @eq:fused-dotprod removes the explicit repetition over $j$ by replacing it with one fused stencil of size $N'_k$, but the angle sweep itself remains.

== Cost Relative to Available First-Order Information <sec:cost-vs-information>

The WVF redundancy result in @sec:wvf-redundancy is directly relevant to the costs above. Equation @eq:directional-gradient shows that for an isotropic polynomial fit, the directional derivative at any angle is already determined by the paired first-derivative estimates $hat(f)_x$ and $hat(f)_y$. Equation @eq:directional-max then shows that the maximizing continuous direction is obtained from those paired derivatives without a sampled-angle sweep. For that reason, the orientation-dependent families in @eq:cost-isotropic-wvf, @eq:cost-lf-fused, and @eq:cost-aniso-families carry an angle factor that is absent from the paired isotropic baselines.

The LF and its fused form add a second layer of cost beyond that orientation factor. Equation @eq:lf-line-response introduces the line half-width $m$, and Equation @eq:fused-alpha shows that the fused stencil weights are generated by summing over the enlarged set of line-position and support-sample terms. The fused representation removes repeated runtime evaluation of those terms, but it still stores and applies one orientation-indexed stencil for each sampled angle. The computational distinction between the families is therefore structural. Some filters encode first-order information through one paired isotropic fit, while others encode it through a bank of orientation-indexed anisotropic stencils.


#pagebreak()

#bibliography("journal_paper/refs.bib")
