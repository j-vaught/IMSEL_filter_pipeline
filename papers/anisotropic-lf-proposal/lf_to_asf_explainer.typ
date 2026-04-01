// LF → ASF Derivation Explainer
// A step-by-step walkthrough of the math behind the fused stencil
// Compile: typst compile lf_to_asf_explainer.typ lf_to_asf_explainer.pdf

#set page(margin: (top: 0.9in, bottom: 0.9in, left: 0.85in, right: 0.85in), numbering: "1")
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set par(justify: true, leading: 0.6em)
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): set text(size: 13pt)
#show heading.where(level: 2): set text(size: 12pt)
#show heading.where(level: 3): set text(size: 11pt)

#align(center)[
  #text(size: 18pt, weight: "bold")[
    From the Line Filter to the Anisotropic Stencil Filter
  ]
  #v(0.4em)
  #text(size: 13pt)[
    A Step-by-Step Mathematical Walkthrough
  ]
  #v(0.6em)
  #text(size: 10pt, style: "italic")[
    J.C. Vaught
  ]
  #v(1.5em)
]

// =====================================================================
= The Big Picture (Before Any Math)
// =====================================================================

The whole point of this document is to answer one question: how did we go from the Line Filter (LF), which is slow and memory-hungry, to the Anisotropic Stencil Filter (ASF), which is fast and lean, without changing the answer at all?

The short version is this. The Line Filter does a bunch of steps (gather pixels, build a matrix, solve a system, repeat along a line, weight, sum). Every single one of those steps is a linear operation on the original pixel values. When you chain a bunch of linear operations together, the result is still just one big linear operation. That means the entire LF pipeline, no matter how complicated it looks, is secretly equivalent to "multiply each pixel in a neighborhood by some weight and add them up." The ASF simply precomputes those weights ahead of time so the GPU only has to do the easy part: gather, multiply, sum. One operation instead of many. Same answer. Much faster.

The rest of this document walks through every step of the math that makes this work, slowly and with lots of explanation.

// =====================================================================
= Step 1: What Are We Even Computing?
// =====================================================================

== The Goal

We have a grayscale image, which is really just a grid of numbers. Each number is a brightness value. We want to find edges, the places where the brightness changes sharply.

To find edges, we need to estimate how fast the brightness is changing at every pixel. In calculus terms, we want the derivative of the image intensity. The bigger the derivative, the stronger the edge.

== The Problem with Simple Approaches

The most obvious way to estimate a derivative is to look at two neighboring pixels and take the difference. This is essentially what Sobel and Prewitt filters do (with some slight averaging). The problem is that if the image is noisy, those two pixels might have random junk in their values, and the difference will be meaningless.

Our approach uses way more pixels (say, 100 instead of 9) and fits a smooth polynomial surface through all of them. The derivative of that smooth surface is our edge estimate. More pixels means more noise averaging, and the polynomial fit means we are not blurring edges the way Gaussian smoothing does.

== Why Direction Matters

An edge has a direction. A vertical edge has a strong horizontal derivative and a weak vertical one. A diagonal edge has strong derivatives in neither the horizontal nor vertical direction individually, but a strong derivative perpendicular to the edge.

Rather than computing derivatives in just two directions (horizontal and vertical) and combining them, we try many candidate directions (say, 18 angles evenly spaced around the circle) and pick the one that gives the biggest response. This is what "anisotropic" means in the name: the filter looks different in different directions.

// =====================================================================
= Step 2: The Point Filter (One Pixel, One Direction)
// =====================================================================

This is the foundation of everything. Before we worry about lines or stencils, we need to understand how to estimate a derivative at a single pixel in a single direction.

== Setting Up Local Coordinates

Pick a target pixel at position $(X_0, Y_0)$ in the image, where $X_0$ is its column (horizontal position) and $Y_0$ is its row (vertical position). Pick a candidate edge direction $theta_k$, which is just an angle measured in radians. If we are testing $N_s = 18$ directions, then $theta_k$ would be one of 18 evenly spaced angles around the circle (0°, 20°, 40°, ..., 340°). The subscript $k$ is just an index that says "which angle are we currently testing" ($k = 0, 1, 2, dots, 17$).

We set up a local coordinate system centered on our pixel, rotated so that $x$ points along the normal to the edge (the direction we want the derivative in) and $y$ points along the edge itself.

For any neighbor pixel at $(X_i, Y_i)$, where $i$ is just an index labeling which neighbor we are talking about ($i = 1, 2, dots, N_p$), its local coordinates are:

$ x_i &= (X_i - X_0) cos theta_k + (Y_i - Y_0) sin theta_k \
  y_i &= -(X_i - X_0) sin theta_k + (Y_i - Y_0) cos theta_k $ <eq:rotate>

Here $x_i$ is the position of neighbor $i$ in the rotated coordinate system along the normal direction, and $y_i$ is its position along the tangent direction. The terms $(X_i - X_0)$ and $(Y_i - Y_0)$ are just "how far is this neighbor from the target pixel in the original image coordinates." The $cos theta_k$ and $sin theta_k$ terms rotate those offsets into the edge-aligned frame.

This is just a standard 2D rotation matrix applied to each neighbor's offset. We are re-expressing each neighbor's position relative to our target pixel, in a coordinate system aligned with the candidate edge direction.

To give a concrete example, suppose our target pixel is at $(50, 50)$ and we are testing $theta_k = 0degree$ (a vertical edge). Then $cos 0 = 1$ and $sin 0 = 0$, so the rotation is trivial: $x_i = X_i - 50$ and $y_i = Y_i - 50$. For $theta_k = 45degree$, the axes are rotated 45 degrees, so the $x$ and $y$ coordinates become mixtures of the horizontal and vertical offsets.

== Fitting a Polynomial Surface

Now we model the image brightness near our target pixel as a polynomial in these local coordinates. Using a Taylor expansion of order $d$ (we typically use $d = 4$). Here $d$ is the polynomial degree, a design choice we make. Higher $d$ means the polynomial can represent more complex shapes, but also means more unknowns to solve for. The function $f$ is the image brightness: $f(x_i, y_i)$ means "the brightness at the position $(x_i, y_i)$."

$ f(x_i, y_i) approx underbrace(f^0, "constant") + underbrace(f_x^0 x_i, "linear in " x) + underbrace(f_y^0 y_i, "linear in " y) + underbrace((f_(x x)^0) / 2 x_i^2, "quadratic") + dots.c $ <eq:taylor>

Here is what every symbol means.

$f^0$ is the brightness value right at the target pixel (the constant term). $f_x^0$ is the first partial derivative of brightness with respect to $x$ at the target pixel, i.e. how fast the brightness changes in the $x$-direction. This is the number we ultimately want because it tells us the edge strength. $f_y^0$ is the same thing but in the $y$-direction (along the edge, which we care less about). $f_(x x)^0$ is the second derivative with respect to $x$ (how the rate of change itself is changing), and so on for all the higher-order terms. The superscript $0$ on all of these just means "evaluated at the origin of our local coordinate system," i.e. at the target pixel.

There are $M = (d+1)(d+2) / 2$ unknown coefficients in this expansion. This formula comes from counting how many distinct monomial terms exist in a 2D polynomial of degree $d$. For $d = 4$, that is $M = (5 times 6) / 2 = 15$ unknowns. For $d = 2$, it would be $M = (3 times 4) / 2 = 6$ unknowns.

The key point is that all of these unknowns ($f^0$, $f_x^0$, $f_y^0$, $f_(x x)^0$, ...) are what we are trying to solve for. We know $f(x_i, y_i)$ at each neighbor (we read it from the image), and we know $(x_i, y_i)$ (we computed it from the rotation). The unknowns are the derivatives. Once we solve for them, we grab $f_x^0$ and that is our edge strength at this pixel in this direction.

== Gathering Neighbors

We select $N_p$ neighbor pixels around our target. $N_p$ stands for "number of points" and is a design parameter we choose. These neighbors are chosen as the $N_p$ closest integer-coordinate pixels to the target, forming an approximately circular patch. For $N_p = 100$, this circle has a radius of about 6 pixels.

Each neighbor gives us one equation. For neighbor $i$, we know two things: its local coordinates $(x_i, y_i)$ (computed from @eq:rotate) and its brightness value $f(X_i, Y_i)$ (read directly from the image). We plug $(x_i, y_i)$ and $f(X_i, Y_i)$ into @eq:taylor. That gives us one equation where the left side is a known number (the brightness we read from the image) and the right side is an expression involving the 15 unknown derivatives multiplied by known coordinate values. One neighbor, one equation.

With $N_p = 100$ neighbors, we get 100 equations. With $M = 15$ unknowns, we have far more equations than unknowns. This is called an overdetermined system. There is no single solution that satisfies all 100 equations exactly (because of noise in the image), but we can find the "best fit" solution that comes closest to satisfying all of them. This best fit naturally averages out the noise, and the more equations we have relative to unknowns, the better the averaging.

== The Linear System

We can write all $N_p$ equations at once as a single matrix equation:

$ bold(A)_(theta_k) bold(z) = bold(b) $ <eq:system>

This is the compact way of writing "100 equations with 15 unknowns." Here is what each piece is.

$bold(A)_(theta_k)$ is the design matrix, with dimensions $N_p times M$ (e.g., $100 times 15$). The subscript $theta_k$ reminds us that this matrix depends on which angle we are testing, because the local coordinates $(x_i, y_i)$ change when we rotate. Each row of $bold(A)$ corresponds to one neighbor pixel $i$. The entries in that row are the monomial basis terms (the $1, x_i, y_i, x_i^2/2, dots$ from @eq:taylor) evaluated at that neighbor's local coordinates. For example, for a 2nd-order fit ($d=2$, $M=6$), row $i$ would be $[1, x_i, y_i, x_i^2 / 2, y_i^2 / 2, x_i y_i]$. For $d=4$ there are 15 such entries per row.

$bold(z)$ is the vector of unknowns, with dimensions $M times 1$ (e.g., $15 times 1$). It contains all the derivative coefficients we are solving for: $bold(z) = [f^0, f_x^0, f_y^0, f_(x x)^0 / 2, dots]^top$. These are the same unknowns from @eq:taylor. Once we solve for $bold(z)$, we will grab its second entry ($f_x^0$) as our edge strength.

$bold(b)$ is the data vector, with dimensions $N_p times 1$ (e.g., $100 times 1$). It contains the actual brightness values we read from the image at each neighbor location: $bold(b) = [f(X_1, Y_1), f(X_2, Y_2), dots, f(X_(N_p), Y_(N_p))]^top$. This is the only part that changes from pixel to pixel; $bold(A)$ is the same for every pixel at a given angle $theta_k$.

The equation $bold(A) bold(z) = bold(b)$ says: "if the polynomial model were perfect, then multiplying the design matrix by the unknown derivatives would reproduce the observed brightness values exactly." In practice it will not be exact because of noise, so we find the best approximation.

== Solving It: The Pseudoinverse

Since we have more equations than unknowns ($N_p > M$), there is no exact solution. We find the least-squares best fit, meaning the $bold(z)$ that minimizes the total squared error across all 100 equations:

$ hat(bold(z)) = (bold(A)_(theta_k)^top bold(A)_(theta_k))^(-1) bold(A)_(theta_k)^top bold(b) $ <eq:lstsq>

Here $hat(bold(z))$ is the estimated (best-fit) version of $bold(z)$, the hat symbol $hat(dot.c)$ is standard notation for "estimated." The expression $bold(A)_(theta_k)^top$ is the transpose of $bold(A)$ (rows and columns swapped), and $(dot.c)^(-1)$ means matrix inverse. This formula is the standard least-squares solution from linear algebra; it is the value of $bold(z)$ that minimizes $||bold(A) bold(z) - bold(b)||^2$.

We can split this formula into two pieces. Define the pseudoinverse matrix:

$ bold(P)_(theta_k) = (bold(A)_(theta_k)^top bold(A)_(theta_k))^(-1) bold(A)_(theta_k)^top $ <eq:pinv>

$bold(P)_(theta_k)$ is called the Moore-Penrose pseudoinverse of $bold(A)_(theta_k)$. It is an $M times N_p$ matrix (e.g., $15 times 100$). The critical property of $bold(P)$ is that it transforms a vector of $N_p$ brightness values directly into the $M$ fitted derivative coefficients. In other words, the solution is simply $hat(bold(z)) = bold(P)_(theta_k) bold(b)$: one matrix-vector multiply and we have all 15 derivatives.

The reason we give $bold(P)$ a name is that it depends only on the geometry (the neighbor positions and the angle $theta_k$), not on the image. So we can compute $bold(P)_(theta_k)$ once ahead of time and reuse it for every pixel.

== Extracting the Derivative We Want

We do not actually need all 15 derivatives. We only need $hat(f)_x^0$, the normal derivative (edge strength). This is the second entry of $hat(bold(z))$ (index 1 if we start counting from 0, because index 0 is the constant $f^0$). Since $hat(bold(z)) = bold(P)_(theta_k) bold(b)$, the second entry of $hat(bold(z))$ is just the dot product of row 1 of $bold(P)$ with $bold(b)$:

$ hat(f)_x^((k)) = bold(p)_"fx"^((k)) dot bold(b) $ <eq:fx>

Here $hat(f)_x^((k))$ is the estimated normal derivative at angle $theta_k$. The superscript $(k)$ reminds us which angle we are testing. The vector $bold(p)_"fx"^((k))$ is row 1 of the pseudoinverse matrix $bold(P)_(theta_k)$. It has $N_p$ entries (one per neighbor pixel). The subscript "fx" stands for "$f_x$-extraction," meaning this is the row of $bold(P)$ that extracts the $f_x$ derivative. The vector $bold(b)$ is the same brightness vector from before (the brightness values at all $N_p$ neighbors).

This is the key insight of the entire derivation. The estimated derivative at one pixel in one direction is just a dot product: take each neighbor's brightness, multiply it by a fixed weight from $bold(p)_"fx"^((k))$, and add them all up. No matter how fancy the polynomial fitting sounds, in the end it is just "multiply each neighbor's brightness by a weight and sum."

The weights $bold(p)_"fx"^((k))$ depend on the geometry (where the neighbors are and what angle $theta_k$ we are testing) but not on the image. They can be computed once in advance and reused for every pixel in the image.

== Picking the Best Direction

We repeat this for all $N_s$ candidate angles and pick the direction with the strongest response. The angles are evenly spaced: $theta_k = k dot 2 pi / N_s$ for $k = 0, 1, dots, N_s - 1$. Here $N_s$ is the number of orientations we test (a design choice; we use $N_s = 18$, giving angles every 20°).

$ k^* = arg max_k |hat(f)_x^((k))| $ <eq:maxorient>

Here $k^*$ is the index of the winning orientation, the one whose absolute derivative response $|hat(f)_x^((k))|$ is largest. The operator $arg max_k$ means "find the value of $k$ that maximizes the expression." The absolute value $|dot.c|$ is needed because an edge could produce a positive or negative derivative depending on which side is brighter; we care about the magnitude, not the sign.

The gradient magnitude (edge strength) at this pixel is $|hat(f)_x^((k^*))|$, and the edge angle is $theta_(k^*)$. This completes the point filter.


// =====================================================================
= Step 3: The Line Extension (Making It Better, But Slower)
// =====================================================================

The point filter works, but it only looks at one circular patch around the target pixel. For low-contrast edges or noisy images, one patch might not give a strong enough signal. The line filter fixes this by looking at multiple patches arranged along the candidate edge direction.

== Virtual Evaluation Points

Instead of fitting the polynomial only at the target pixel $(X_0, Y_0)$, we also fit it at $(2m+1)$ virtual positions spaced along the edge direction. Here $m$ is the half-width of the line, a design parameter we choose. The term $(2m+1)$ counts the total number of positions: $m$ on each side of center, plus the center itself.

$ (X_j, Y_j) = (X_0 + j cos theta_k, Y_0 + j sin theta_k), quad j in {-m, dots, 0, dots, m} $ <eq:vpoints>

Here $(X_j, Y_j)$ is the position of the $j$-th virtual evaluation point. The index $j$ runs from $-m$ to $+m$ and labels each virtual position along the line. When $j = 0$, we are at the original target pixel. When $j = 1$, we have moved one step in the direction $theta_k$. When $j = -1$, we have moved one step in the opposite direction. The $cos theta_k$ and $sin theta_k$ terms convert the step along the angle $theta_k$ into horizontal and vertical pixel offsets.

For $m = 7$, this gives $2(7) + 1 = 15$ positions (from $j = -7$ to $j = 7$). At each one, we run the entire point filter from Step 2: gather $N_p$ neighbors around $(X_j, Y_j)$, build the design matrix, solve via pseudoinverse, extract $hat(f)_x^((j,k))$. The notation $hat(f)_x^((j,k))$ now has two superscripts: $j$ tells us which virtual position, and $k$ tells us which angle.

Think of it this way. We are sliding a circular patch along the candidate edge direction and measuring the edge strength at each position. If there really is an edge running in that direction, all 15 patches should see it, and the combined evidence is much stronger than any single patch.

== Combining the Estimates

We combine the $(2m+1)$ derivative estimates using a Gaussian-weighted average:

$ R_k = sum_(j=-m)^m w_j dot hat(f)_x^((j,k)) $ <eq:lineresponse>

Here $R_k$ is the line filter response for orientation $k$, a single number that summarizes how strong the edge evidence is in direction $theta_k$ when we consider all the virtual positions together. The summation $sum_(j=-m)^m$ means "add up for every virtual position $j$ from $-m$ to $m$." Each term in the sum is the Gaussian weight $w_j$ times the derivative estimate $hat(f)_x^((j,k))$ at virtual position $j$.

The weights $w_j$ are defined as:

$ w_j = exp(-j^2 / (2 sigma^2)), quad sigma = m / 2 $

Here $w_j$ is the Gaussian weight for virtual position $j$. It is just a bell-curve value: the $exp$ function (exponential, $e$ raised to a power) produces a number between 0 and 1. When $j = 0$ (center position), $w_0 = exp(0) = 1$ (maximum weight). As $|j|$ increases (positions farther from center), $w_j$ decreases toward zero. The parameter $sigma$ controls how fast the weights decay; we set $sigma = m/2$, which means positions beyond about $m/2$ steps from center contribute very little.

The Gaussian weighting means positions near the center (small $|j|$) count more than positions far away (large $|j|$). This is sensible: the target pixel is at $j=0$, and positions farther away are less relevant.

For $m = 7$ ($sigma = 3.5$), the weights look like this: $w_0 = 1.0$ (center), $w_(plus.minus 1) approx 0.96$, $w_(plus.minus 3) approx 0.68$, $w_(plus.minus 7) approx 0.05$ (small but nonzero at the ends).

== Why This Is Expensive

Every one of those 15 virtual positions requires its own gather of $N_p$ pixels, its own matrix-vector product with the pseudoinverse, and its own extraction of the derivative. With $N_p = 100$ and $m = 7$:

- $15 times 100 = 1500$ pixel reads per direction
- 15 separate matrix-vector multiplies
- All repeated for $N_s = 18$ directions

That is $15 times 100 times 18 = 27,000$ pixel reads per target pixel, plus thousands of multiply-add operations. For a 1280 x 720 image, that is about 25 billion pixel reads. This is why the naive LF is slow and uses tons of memory.

// =====================================================================
= Step 4: The Algebraic Collapse (The Key Step)
// =====================================================================

Here is where the magic happens. We are going to show that the entire line filter computation, all 15 polynomial fits combined with Gaussian weighting, is equivalent to a single weighted sum over pixel values. No matrices, no pseudoinverses, no loops at runtime.

== Expanding the Line Response

Let us write out what $R_k$ actually computes, in full detail. We start with the line response from @eq:lineresponse:

$ R_k = sum_(j=-m)^m w_j dot hat(f)_x^((j,k)) $ <eq:expand1>

This says: "for each virtual position $j$, take the derivative estimate $hat(f)_x^((j,k))$, multiply it by the Gaussian weight $w_j$, and add them all up."

Now we substitute what $hat(f)_x^((j,k))$ actually is. From @eq:fx, we know that the derivative estimate at any single position is a dot product of the pseudoinverse row with the brightness values. At virtual position $j$:

$ R_k = sum_(j=-m)^m w_j dot (bold(p)_"fx"^((k)) dot bold(b)_j) $ <eq:expand2>

Here $bold(b)_j$ is the vector of $N_p$ brightness values gathered around virtual position $j$ (not around the original target pixel, but around the shifted position $(X_j, Y_j)$). The weight vector $bold(p)_"fx"^((k))$ is the same for every virtual position because the neighbor pattern and the polynomial model are the same; only the center of the patch moves.

Now we expand the dot product $bold(p)_"fx"^((k)) dot bold(b)_j$ into its individual terms:

$ R_k = sum_(j=-m)^m w_j sum_(i=1)^(N_p) p_i^((k)) dot f(X_j + Delta x_i, Y_j + Delta y_i) $ <eq:expand3>

Here $p_i^((k))$ is the $i$-th entry of $bold(p)_"fx"^((k))$, i.e., the pseudoinverse weight for the $i$-th neighbor. The term $(Delta x_i, Delta y_i)$ is the offset of neighbor $i$ relative to the center of its patch (these are the same offsets for every virtual position, because we always use the same circular neighbor pattern). The function $f(X_j + Delta x_i, Y_j + Delta y_i)$ is just "read the brightness from the image at the pixel located at the virtual position plus the neighbor offset."

The inner sum $sum_(i=1)^(N_p)$ loops over all $N_p$ neighbors. The outer sum $sum_(j=-m)^m$ loops over all $(2m+1)$ virtual positions. Together, they produce $(2m+1) times N_p$ terms.

Now substitute the definition of $(X_j, Y_j)$ from @eq:vpoints, which is $(X_0 + j cos theta_k, Y_0 + j sin theta_k)$:

$ R_k = sum_(j=-m)^m sum_(i=1)^(N_p) underbrace(w_j dot p_i^((k)), "a single number") dot f(underbrace(X_0 + j cos theta_k + Delta x_i, "x-coordinate of pixel"), underbrace(Y_0 + j sin theta_k + Delta y_i, "y-coordinate of pixel")) $ <eq:expanded_full>

Let us pause and stare at this equation. It says: $R_k$ is a sum of $(2m+1) times N_p$ terms. Each term has two parts. Part one is $w_j dot p_i^((k))$, which is a product of two known numbers: the Gaussian weight at position $j$ and the pseudoinverse weight for neighbor $i$. Both of these are determined by the filter geometry and can be computed before we ever look at the image. Part two is $f(dots)$, which is a single pixel brightness value read from the image at a specific location.

So each term is just "a precomputable number times a pixel value." And $R_k$ is the sum of all such terms.

That is it. The entire line filter, the polynomial fitting, the pseudoinverse, the Gaussian weighting, all of it, reduces to "multiply some pixels by some precomputed numbers and add up." This is the algebraic collapse.

== Why Does This Work?

It works because every operation in the pipeline is linear:

+ Gathering neighbor pixels is linear (just selecting values from the image).
+ The matrix-vector product $bold(P)_(theta_k) bold(b)$ is linear in $bold(b)$.
+ Extracting one row of the result is linear.
+ Multiplying by $w_j$ is linear.
+ Summing over $j$ is linear.

When you compose linear operations, the result is linear. A linear operation on pixel values is always expressible as "multiply each pixel by a weight and sum." There is no escaping this.

To make it even more concrete: imagine the entire image is a single vector $bold(I)$ of all pixel values. Then $R_k$ is some vector $bold(alpha)_k$ dotted with $bold(I)$. The vector $bold(alpha)_k$ has a nonzero entry for every pixel that participates in the computation, and the value of that entry is the total weight that pixel receives. Most entries of $bold(alpha)_k$ are zero (pixels far from the target do not participate), so $bold(alpha)_k$ is sparse. But it is still just a dot product.

// =====================================================================
= Step 5: Building the Fused Stencil
// =====================================================================

We have shown that $R_k$ is a weighted sum over pixel values. But @eq:expanded_full has $(2m+1) times N_p$ terms. Many of those terms access the same pixel, because the circular neighborhoods of adjacent virtual positions overlap. We need to merge those duplicates.

== The Overlap Problem (With Numbers)

Consider $m = 7$ and $N_p = 100$. Each of the 15 virtual positions gathers 100 neighbors from an approximately circular patch of radius $approx 6$ pixels. Adjacent virtual positions are only 1 pixel apart. So two adjacent patches share most of their pixels.

Concretely, virtual position $j = 0$ gathers the 100 pixels closest to $(X_0, Y_0)$. Virtual position $j = 1$ (at $theta_k = 0degree$, this is one pixel to the right) gathers the 100 pixels closest to $(X_0 + 1, Y_0)$. These two circles of radius 6, centered 1 pixel apart, overlap in about 90 of their 100 pixels.

The raw stencil has $15 times 100 = 1500$ entries, but many of those entries point to the same pixel.

== Deduplication

We group all $(j, i)$ pairs that map to the same integer pixel position and sum their weights:

$ alpha_(k, ell) = sum_({(j, i) : "round"(j cos theta_k + Delta x_i, j sin theta_k + Delta y_i) = "position" ell}) w_j dot p_i^((k)) $ <eq:dedup>

Here $alpha_(k, ell)$ is the fused weight for orientation $k$ at unique pixel position $ell$. The subscript $ell$ is a new index that labels unique pixel positions in the fused stencil ($ell = 1, 2, dots, N'_k$). For each unique position $ell$, we find all the $(j, i)$ pairs from the raw stencil that land on that same pixel (after rounding to integer coordinates), and we add up their individual weights $w_j dot p_i^((k))$. The "round" operation reflects the fact that the computed offsets $j cos theta_k + Delta x_i$ may not be exact integers, so we snap them to the nearest pixel.

After deduplication, the stencil shrinks dramatically. For $m = 7$, $N_p = 100$: from 1500 raw entries down to about 264 unique pixel positions. That is an 82% reduction.

The fused stencil for orientation $k$ is then just two arrays. The first array stores the pixel offsets $( tilde(delta)_ell^x, tilde(delta)_ell^y)$, which say "how far from the target pixel is stencil position $ell$, in the $x$ and $y$ directions." The tilde ($tilde(dot.c)$) distinguishes these from the raw offsets; these are the deduplicated, unique positions. The second array stores the corresponding fused weights $alpha_(k,ell)$. Together, these two arrays completely define the filter for orientation $k$. There are $N'_k$ entries, where $N'_k$ is the number of unique positions (e.g., 264 instead of 1500).

== The Final Formula

After deduplication, the line filter response is:

$ R_k = sum_(ell=1)^(N'_k) alpha_(k, ell) dot f(X_0 + tilde(delta)_ell^x, Y_0 + tilde(delta)_ell^y) $ <eq:fused>

This says: "for each of the $N'_k$ unique stencil positions, read the pixel brightness at that position ($f$ evaluated at the target pixel plus the offset), multiply by its fused weight $alpha_(k,ell)$, and sum everything up." That is the entire filter. One loop, one multiply-add per stencil position, done.

Or in vector notation:

$ R_k = bold(alpha)_k^top bold(g)_k $ <eq:dotprod>

Here $bold(alpha)_k in RR^(N'_k)$ is the precomputed weight vector (all $N'_k$ fused weights stacked into a column), and $bold(g)_k in RR^(N'_k)$ is the gathered intensity vector (the brightness values read from the image at the $N'_k$ stencil positions). The notation $bold(alpha)_k^top$ means the transpose of $bold(alpha)_k$ (turning a column into a row so the dot product works). The notation $RR^(N'_k)$ just means "a vector of $N'_k$ real numbers."

This is mathematically identical to @eq:expanded_full. The same pixels participate. The same total weights are applied. The same answer comes out. We have just reorganized the computation to avoid redundant work.


// =====================================================================
= Step 6: Why This Is So Much Faster
// =====================================================================

== The Naive LF (Before Fusion)

For each pixel, for each of $N_s$ orientations, the naive LF does:

+ Loop over $(2m+1)$ virtual positions.
+ At each virtual position, gather $N_p$ neighbors from the image (a scattered memory read).
+ Compute $bold(P)_(theta_k) bold(b)_j$ (a matrix-vector product: $M times N_p$ multiplies).
+ Extract the normal derivative (pick one element).
+ Multiply by $w_j$.
+ Sum the $(2m+1)$ weighted derivatives.

Total work per pixel per orientation: $(2m+1) times N_p$ pixel reads, plus $(2m+1)$ matrix-vector products of size $M times N_p$.

Total pixel reads per image pixel: $(2m+1) times N_p times N_s$. For $m=7, N_p = 100, N_s = 18$: $15 times 100 times 18 = 27,000$ reads. Plus storing all those intermediate values in GPU memory.

== The Fused ASF (After Fusion)

For each pixel, for each of $N_s$ orientations, the ASF does:

+ Gather $N'_k$ pixel values from the image (one set of scattered reads).
+ Multiply each by its precomputed weight and sum (one dot product).

Total pixel reads per image pixel: $N'_k times N_s$. For $m=7, N_p = 100, N_s = 18$: $264 times 18 = 4,752$ reads. That is 5.7 times fewer reads, and zero matrix operations.

But the speedup is even larger than 5.7 times because:

- No matrix-vector products at runtime (those are baked into the weights).
- No intermediate storage (the naive approach stores huge tensors; the ASF stores only the input image and the output).
- The GPU kernel is a simple loop with predictable memory access, which modern hardware loves.

In practice, we measured 18 to 24 times speedup and 311 times reduction in GPU memory.


// =====================================================================
= Step 7: A Worked Example
// =====================================================================

Let us trace through the entire process with tiny numbers to make it concrete. We will use $d = 1$ (linear polynomial, 3 unknowns), $N_p = 4$ (4 neighbors), $m = 1$ (3 virtual positions), $N_s = 1$ (one orientation, $theta = 0$, vertical edge).

== The Point Filter Weights

With $d = 1$ and $theta = 0degree$, the polynomial model is $f(x, y) approx f^0 + f_x^0 x + f_y^0 y$.

Suppose our 4 neighbors (relative to the center) are at offsets:

$ (Delta x, Delta y) in {(-1, 0), (1, 0), (0, -1), (0, 1)} $

The design matrix is:

$ bold(A) = mat(1, -1, 0; 1, 1, 0; 1, 0, -1; 1, 0, 1) $

The pseudoinverse (which you can verify by computing $(bold(A)^top bold(A))^(-1) bold(A)^top$) is:

$ bold(P) = mat(1/4, 1/4, 1/4, 1/4; -1/2, 1/2, 0, 0; 0, 0, -1/2, 1/2) $

Row 1 (the one we care about) is $bold(p)_"fx" = [-1\/2, 1\/2, 0, 0]$.

This makes perfect sense. To estimate the horizontal derivative, the polynomial fit says "take the right neighbor minus the left neighbor, divided by 2." The top and bottom neighbors get zero weight because they carry no information about the horizontal derivative. This is reassuringly intuitive.

== The Line Extension

With $m = 1$, we have 3 virtual positions at $j in {-1, 0, 1}$. Since $theta = 0degree$, these are at horizontal offsets $(-1, 0), (0, 0), (1, 0)$.

Gaussian weights with $sigma = m / 2 = 0.5$:

$ w_(-1) = exp(-1 / (2 dot 0.25)) = exp(-2) approx 0.135 $
$ w_0 = exp(0) = 1.0 $
$ w_1 = exp(-2) approx 0.135 $

== The Raw Stencil

For each virtual position $j$, the 4 neighbors (at offsets from that virtual position) map to these absolute offsets from the target pixel:

- $j = -1$ (virtual position at $(-1, 0)$): neighbors at $(-2, 0), (0, 0), (-1, -1), (-1, 1)$
- $j = 0$ (virtual position at $(0, 0)$): neighbors at $(-1, 0), (1, 0), (0, -1), (0, 1)$
- $j = 1$ (virtual position at $(1, 0)$): neighbors at $(0, 0), (2, 0), (1, -1), (1, 1)$

The raw weight for each entry is $w_j dot p_i$:

#text(size: 10pt)[
#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (center, center, center, center, center, center),
    table.header[$j$][$i$][Offset][$ w_j$][$p_i$][$ w_j dot p_i$],
    [$-1$], [1], [(-2, 0)], [0.135], [$-1/2$], [$-0.068$],
    [$-1$], [2], [(0, 0)], [0.135], [$1/2$], [$+0.068$],
    [$-1$], [3], [(-1, -1)], [0.135], [$0$], [$0$],
    [$-1$], [4], [(-1, 1)], [0.135], [$0$], [$0$],
    [$0$], [1], [(-1, 0)], [1.0], [$-1/2$], [$-0.500$],
    [$0$], [2], [(1, 0)], [1.0], [$1/2$], [$+0.500$],
    [$0$], [3], [(0, -1)], [1.0], [$0$], [$0$],
    [$0$], [4], [(0, 1)], [1.0], [$0$], [$0$],
    [$1$], [1], [(0, 0)], [0.135], [$-1/2$], [$-0.068$],
    [$1$], [2], [(2, 0)], [0.135], [$1/2$], [$+0.068$],
    [$1$], [3], [(1, -1)], [0.135], [$0$], [$0$],
    [$1$], [4], [(1, 1)], [0.135], [$0$], [$0$],
  ),
  caption: [Raw stencil entries. 12 total (3 virtual positions times 4 neighbors).],
)
]

== Deduplication

Some offsets appear more than once. Specifically, $(0, 0)$ appears at $(j=-1, i=2)$ with weight $+0.068$ and at $(j=1, i=1)$ with weight $-0.068$. These sum to zero.

Also, many entries have weight zero (the neighbors with $p_i = 0$), so we can drop them.

After deduplication, the fused stencil is:

#figure(
  table(
    columns: (auto, auto),
    align: (center, center),
    table.header[Offset][Fused weight $alpha$],
    [(-2, 0)], [$-0.068$],
    [(-1, 0)], [$-0.500$],
    [(0, 0)], [$0.000$ (drops out)],
    [(1, 0)], [$+0.500$],
    [(2, 0)], [$+0.068$],
  ),
  caption: [Fused stencil after deduplication. Reduced from 12 entries to 4 nonzero entries.],
)

The result: $R_0 = -0.068 dot f(-2,0) - 0.500 dot f(-1,0) + 0.500 dot f(1,0) + 0.068 dot f(2,0)$.

This is a weighted central difference that emphasizes the immediate neighbors (weight $0.5$) and lightly includes the next-nearest neighbors (weight $0.068$). It is like a slightly wider, smarter version of the Sobel filter, computed automatically from the polynomial fitting + line extension math. And we can compute this weighted sum directly without ever building a matrix or solving a system.


// =====================================================================
= Step 8: The General Case
// =====================================================================

The worked example used tiny numbers. In the real filter with $N_p = 100$, $m = 7$, and $d = 4$, the process is identical but the numbers are bigger.

The raw stencil has $(2m+1) times N_p = 15 times 100 = 1500$ entries. After deduplication to unique pixel positions, about 264 entries remain. Each entry has a precomputed weight $alpha_(k, ell)$ that encodes the combined effect of the polynomial fitting, the pseudoinverse extraction, and the Gaussian line weighting.

The stencil shape is an elongated, anisotropic pattern that stretches along the candidate edge direction. When you plot the weights, you see a dipole pattern: positive weights on one side of the expected edge, negative weights on the other. This makes sense because the derivative should respond to the difference in brightness across the edge.

At runtime, for each pixel, the GPU kernel loops over the $N_s$ orientations, and for each one, it gathers the pixel values at the stencil offsets, multiplies by the precomputed weights, and sums. The orientation with the largest absolute response wins.

// =====================================================================
= Summary
// =====================================================================

The derivation has four conceptual steps.

The first step is recognizing that the point filter (polynomial fit followed by derivative extraction) is a linear operation on pixel values. The derivative at a single point is just $bold(p)_"fx"^((k)) dot bold(b)$, a dot product of fixed weights with brightness values.

The second step is recognizing that the line extension (evaluating the point filter at multiple virtual positions and taking a Gaussian-weighted sum) is also linear. You are summing dot products, which is itself a dot product.

The third step is expanding the double sum (over virtual positions $j$ and neighbors $i$) to get a single sum over $(j, i)$ pairs, each pair contributing $w_j dot p_i^((k))$ times one pixel value.

The fourth step is deduplication: grouping all $(j, i)$ pairs that reference the same pixel, summing their weights, and storing only the unique positions and fused weights. This compressed stencil is the ASF.

The result is mathematically identical to the original line filter. Same pixels, same weights, same answer. But the computation goes from "build matrices, solve systems, loop over virtual positions" to "gather pixels, multiply by precomputed weights, sum." This is why the ASF is 18 to 24 times faster and uses 311 times less memory than the naive LF implementation.
