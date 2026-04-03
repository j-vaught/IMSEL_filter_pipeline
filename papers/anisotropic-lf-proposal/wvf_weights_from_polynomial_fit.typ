#set page(margin: (top: 0.8in, bottom: 0.8in, left: 0.85in, right: 0.85in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set par(justify: true, leading: 0.6em)
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 11.5pt)

#align(center)[
  #text(size: 17pt, weight: "bold")[
    How A Polynomial Fit Turns Into Filter Weights
  ]
  #v(0.5em)
  #text(size: 11pt, style: "italic")[
    A 3x3 Example, Then the General Square, Then a Circle
  ]
]

= The Core Idea

The important fact is simple. Once the sample locations are fixed, the least-squares polynomial fit is a linear map from sampled pixel values to polynomial coefficients. If we only want one coefficient, for example the derivative coefficient, then that coefficient is just one fixed weighted sum of the input pixels. Those fixed numbers are the filter weights.

We can see this most clearly by starting with a tiny 3x3 example.

= A 3x3 Toy Example

Consider a 3x3 patch centered at the origin with local coordinates

$
  (x, y) in \{ -1, 0, 1 \} times \{ -1, 0, 1 \}.
$

== Start With The Actual 3x3 Patch

As a simplification, let us begin with a local 3x3 image patch. We denote this neighborhood by $bold(B)$:

$
  bold(B) =
  mat(
    I_(-1,1), I_(0,1), I_(1,1);
    I_(-1,0), I_(0,0), I_(1,0);
    I_(-1,-1), I_(0,-1), I_(1,-1)
  ).
$

This matrix contains the pixel intensities in a local 3x3 window, with the pixel of interest located at the center.

Suppose we fit a degree-1 polynomial

$
  p(x, y) = c_0 + c_1 x + c_2 y.
$

Here $c_0$ tells us the local baseline brightness, while $c_1$ and $c_2$ tell us how that brightness changes across space. Since an edge is a place where intensity changes, the derivative coefficients are the quantities we care about most. The constant term helps the polynomial fit the local patch, but the edge information is carried by the change terms.

== Stack The Patch Into A Vector

To solve the least-squares system, we flatten those same nine values into a column vector

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
$

The unknown polynomial coefficients are

$
  bold(z) = [c_0, c_1, c_2]^top.
$

== Build The Design Matrix

Each row of the design matrix is just the basis $[1, x, y]$ evaluated at one sample location. Therefore

$
  bold(A) =
  mat(
    1, -1, -1;
    1, 0, -1;
    1, 1, -1;
    1, -1, 0;
    1, 0, 0;
    1, 1, 0;
    1, -1, 1;
    1, 0, 1;
    1, 1, 1
  ).
$

The least-squares model is then

$
  bold(A) bold(z) approx bold(b).
$

== Solve The Least-Squares System

The fitted coefficients are

$
  hat(bold(z)) = (bold(A)^top bold(A))^(-1) bold(A)^top bold(b).
$

For this particular 3x3 geometry, one can compute

$
  bold(A)^top bold(A) =
  mat(
    9, 0, 0;
    0, 6, 0;
    0, 0, 6
  ),
$

so

$
  (bold(A)^top bold(A))^(-1) =
  mat(
    1/9, 0, 0;
    0, 1/6, 0;
    0, 0, 1/6
  ).
$

Thus the pseudoinverse is

$
  bold(P) = (bold(A)^top bold(A))^(-1) bold(A)^top.
$

The second row of $bold(P)$ gives the weights for $c_1$, the $x$-derivative coefficient:

$
  bold(p)_(c_1)^top =
  [
    -1/6, 0, 1/6,
    -1/6, 0, 1/6,
    -1/6, 0, 1/6
  ].
$

So the fitted derivative coefficient is

$
  hat(c_1) = bold(p)_(c_1)^top bold(b).
$

Written out explicitly,

$
  hat(c_1) = (-1/6) I_(-1,-1) + 0 I_(0,-1) + (1/6) I_(1,-1)
$
$
  quad + (-1/6) I_(-1,0) + 0 I_(0,0) + (1/6) I_(1,0)
$
$
  quad + (-1/6) I_(-1,1) + 0 I_(0,1) + (1/6) I_(1,1).
$

This is the crucial moment. The polynomial fit has turned into a 3x3 derivative filter:

$
  bold(K)_x =
  mat(
    -1/6, 0, 1/6;
    -1/6, 0, 1/6;
    -1/6, 0, 1/6
  ).
$

The fit sounds like solving for a polynomial, but because the fit is linear, the derivative we extract is just one fixed weighted sum of the nine pixels.

= Generalization To An N x N Square

Now suppose the support is an arbitrary square window with coordinates

$
  (x_i, y_i), quad i = 1, 2, dots, N_p,
$

where for a square window $N_p = N^2$.

For a polynomial of degree $d$, define the monomial basis vector

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
$

with total length

$
  M = (d+1)(d+2)/2.
$

Then the design matrix is built exactly the same way:

$
  bold(A) =
  mat(
    phi_d(x_1, y_1)^top;
    phi_d(x_2, y_2)^top;
    dots.v;
    phi_d(x_(N_p), y_(N_p))^top
  ).
$

The fitted coefficients are still

$
  hat(bold(z)) = (bold(A)^top bold(A))^(-1) bold(A)^top bold(b) = bold(P) bold(b).
$

If the derivative coefficient we want is the entry corresponding to $x$, then one row of $bold(P)$ gives a weight vector

$
  bold(p)_(f_x)^top,
$

and the derivative estimate is

$
  hat(f)_x = bold(p)_(f_x)^top bold(b).
$

So nothing changes conceptually from 3x3 to NxN. The matrix gets larger, but the derivative is still one row of the pseudoinverse dotted with the pixel vector.

= From A Square To A Circle

The circular case is not a new algorithm. It is the same least-squares construction with a different set of sample locations.

Instead of using every point in a square, keep only the points satisfying

$
  x_i^2 + y_i^2 <= r^2.
$

Now the data vector contains only the pixel values inside that circular support,

$
  bold(b) = [I_1, I_2, dots, I_(N_p)]^top,
$

and the design matrix keeps only the corresponding rows

$
  bold(A) =
  mat(
    phi_d(x_1, y_1)^top;
    phi_d(x_2, y_2)^top;
    dots.v;
    phi_d(x_(N_p), y_(N_p))^top
  ),
$

where the coordinates now come from the circular neighborhood rather than the full square.

The least-squares solution is unchanged:

$
  hat(bold(z)) = (bold(A)^top bold(A))^(-1) bold(A)^top bold(b).
$

And again, the derivative estimate is one row of the pseudoinverse:

$
  hat(f)_x = bold(p)_(f_x)^top bold(b).
$

So the circle does not change the logic at all. It only changes which sample coordinates appear as rows of $bold(A)$.

= Why This Matters For WVF

The WVF does two additional things.

First, it rotates the local coordinates so that the derivative is taken in the candidate normal direction:

$
  x_i' = Delta x_i cos theta + Delta y_i sin theta,
$
$
  y_i' = -Delta x_i sin theta + Delta y_i cos theta.
$

Second, it uses a circular support instead of a square one.

But once those rotated circular sample locations are fixed, the exact same least-squares logic applies:

$
  hat(bold(z)) = bold(P)_theta bold(b),
$

and the WVF weight vector is just the row of $bold(P)_theta$ corresponding to the normal derivative coefficient:

$
  hat(f)_(x') = bold(p)_(f_(x'))^top bold(b).
$

That row is the WVF stencil.

= The Short Version

The shortest correct explanation is this.

The polynomial fit is a linear map from sampled pixel values to polynomial coefficients. Because the derivative coefficient is one entry of the fitted coefficient vector, it is also a linear map of the sampled pixels. Any linear map from pixels to a scalar can be written as one fixed weighted sum. Those fixed numbers are the filter weights.

So the WVF weights are not something added after the polynomial fit. They are exactly the polynomial fit, collapsed into the one row needed to extract the derivative.
