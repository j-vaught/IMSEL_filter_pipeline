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

Although a polynomial fit is usually described as solving for unknown coefficients, the least-squares solution is linear in the sampled pixel values. As a result, any fitted coefficient, including a derivative coefficient, can be expressed as one fixed weighted sum of the input pixels. The resulting weighted sum for each pixel is effectively the filter weights, similar to how Sobel or Prewitt weights are defined.

= A 3x3 Toy Example

== Start With The Actual 3x3 Patch

As a simplification, let us begin with a local 3x3 image patch. We denote this neighborhood by $bold(B)$:

$
  bold(B) =
  mat(delim: "[",
    I_(-1,1), I_(0,1), I_(1,1);
    I_(-1,0), I_(0,0), I_(1,0);
    I_(-1,-1), I_(0,-1), I_(1,-1)
  ).
$ <eq:patch>

This matrix contains the pixel intensities in a local 3x3 window, with the pixel of interest located at the center.

Now suppose we aim to fit a degree-1 polynomial to the matrix $bold(B)$. One simple way to represent that local polynomial is

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

== Stack The Patch Into A Vector

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

== Solve The Least-Squares System

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

== Filter Derivations

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

Once $bold(K)_x$ and $bold(K)_y$ are known, they can be used exactly like any other pair of derivative filters. We convolve the image with $bold(K)_x$ to obtain a horizontal derivative estimate $G_x$, and with $bold(K)_y$ to obtain a vertical derivative estimate $G_y$. From those two responses, we can form the gradient magnitude $sqrt(G_x^2 + G_y^2)$ to measure edge strength, and the gradient orientation $"atan2"(G_y, G_x)$ to measure edge direction. In other words, after the polynomial fit has been collapsed into $bold(K)_x$ and $bold(K)_y$, the rest of the pipeline is just the standard gradient-based edge-detection workflow.


The fit sounds like solving for a polynomial, but because the fit is linear, the derivative we extract is just one fixed weighted sum of the nine pixels.

= Generalization To An N x N Square

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

To see where this count comes from, group the monomials by total degree. For total degree 0, there is only 1 term, namely $1$. For total degree 1, there are 2 terms, namely $x$ and $y$. For total degree 2, there are 3 terms, namely $x^2$, $x y$, and $y^2$. In general, for total degree $n$, the admissible monomials are $x^n, x^(n-1) y, x^(n-2) y^2, dots, y^n$, so there are $n+1$ of them.

Since the basis includes every total degree from 0 up to $d$, the total number of basis terms is

$
  M = sum_(n=0)^d (n+1) = 1 + 2 + 3 + dots + (d + 1).
$ <eq:mbasis-sum>

Using the standard formula for the sum of the first $d+1$ positive integers, this becomes

$
  M = (d+1)(d+2)/2.
$ <eq:mbasis>

Now we can state the minimum data requirement for the least-squares fit. The coefficient vector $bold(z)$ has $M$ unknown entries, one for each monomial in the basis. Each sampled pixel contributes one scalar equation, because evaluating the polynomial at one location gives one relation between the unknown coefficients and the observed intensity. Therefore, if the support contains $N_p$ sampled pixels, we need at least as many samples as unknowns:

$
  N_p >= M.
$ <eq:min-data>

For the generalized $N times N$ case, the unknown coefficient vector is

$
  bold(z) = [c_0, c_1, c_2, dots, c_(M-1)]^top,
$ <eq:zvecn>

where these coefficients multiply the monomials collected in $phi_d(x, y)$. Next we build the design matrix $bold(A)$. Each row corresponds to one sampled pixel in the support, and each column corresponds to one monomial term in the basis. The $i$-th row is simply the basis vector evaluated at the coordinates of the $i$-th pixel:

$
  bold(A) =
  mat( delim: "[",
    phi_d(x_1, y_1)^top;
    phi_d(x_2, y_2)^top;
    dots.v;
    phi_d(x_(N_p), y_(N_p))^top
  ).
$ <eq:designn>

To complete the linear system, we stack the observed pixel intensities from those same sample locations, in the same order used for the rows of $bold(A)$:

$
  bold(b) =
  [
    f(x_1, y_1),
    f(x_2, y_2),
    dots,
    f(x_(N_p), y_(N_p))
  ]^top.
$ <eq:bvecn>

Using the same ordering in both objects is the key point. Row $i$ of $bold(A)$ is built from the coordinates $(x_i, y_i)$, and entry $i$ of $bold(b)$ is the measured intensity at that same location. With those definitions, the generalized polynomial-fitting problem is

$
  bold(A) bold(z) approx bold(b).
$ <eq:systemn>

This system is usually overdetermined, so we solve for the coefficient vector in the least-squares sense. The fitted coefficient vector is

$
  hat(bold(z)) = (bold(A)^top bold(A))^(-1) bold(A)^top bold(b) = bold(P) bold(b).
$ <eq:lstsqn>

Here $hat(bold(z))$ means the estimated coefficient vector, and

$
  bold(P) = (bold(A)^top bold(A))^(-1) bold(A)^top
$

is the pseudoinverse matrix that maps sampled intensities directly to fitted coefficients. If the basis is ordered as $[1, x, y, dots]^top$, then the coefficient of $x$ is the estimate of the first derivative in the $x$-direction. That coefficient is one entry of $hat(bold(z))$, so the corresponding row of $bold(P)$ is a weight vector:

$
  bold(p)_(f_x)^top,
$ <eq:px-row>

Applying that row to the data vector gives the derivative estimate

$
  hat(f)_x = bold(p)_(f_x)^top bold(b).
$ <eq:fx-est>

So the logic is exactly the same as in the 3x3 example. First choose sample locations. Then evaluate the monomial basis there to build $bold(A)$. Then stack the measured intensities into $bold(b)$ in the same order. Solving the least-squares problem gives $hat(bold(z))$, and one row of the pseudoinverse extracts whichever derivative coefficient we want.

= From A Square To A Circle

The circular case is not a new algorithm. It is the same least-squares construction with a different set of sample locations.

Instead of using every point in a square, keep only the points satisfying

$
  x_i^2 + y_i^2 <= r^2.
$ <eq:circle-support>

Now the data vector contains only the pixel values inside that circular support,

$
  bold(b) = [I_1, I_2, dots, I_(N_p)]^top,
$ <eq:circle-bvec>

and the design matrix keeps only the corresponding rows

$
  bold(A) =
  mat(
    phi_d(x_1, y_1)^top;
    phi_d(x_2, y_2)^top;
    dots.v;
    phi_d(x_(N_p), y_(N_p))^top
  ),
$ <eq:circle-design>

where the coordinates now come from the circular neighborhood rather than the full square.

The least-squares solution is unchanged:

$
  hat(bold(z)) = (bold(A)^top bold(A))^(-1) bold(A)^top bold(b).
$ <eq:circle-lstsq>

And again, the derivative estimate is one row of the pseudoinverse:

$
  hat(f)_x = bold(p)_(f_x)^top bold(b).
$ <eq:circle-fx>

So the circle does not change the logic at all. It only changes which sample coordinates appear as rows of $bold(A)$.

= Why This Matters For WVF

The WVF does two additional things.

First, it rotates the local coordinates so that the derivative is taken in the candidate normal direction:

$
  x_i' = Delta x_i cos theta + Delta y_i sin theta,
$ <eq:rotate-x>
$
  y_i' = -Delta x_i sin theta + Delta y_i cos theta.
$ <eq:rotate-y>

Second, it uses a circular support instead of a square one.

But once those rotated circular sample locations are fixed, the exact same least-squares logic applies:

$
  hat(bold(z)) = bold(P)_theta bold(b),
$ <eq:wvf-lstsq>

and the WVF weight vector is just the row of $bold(P)_theta$ corresponding to the normal derivative coefficient:

$
  hat(f)_(x') = bold(p)_(f_(x'))^top bold(b).
$ <eq:wvf-fx>

That row is the WVF stencil.
