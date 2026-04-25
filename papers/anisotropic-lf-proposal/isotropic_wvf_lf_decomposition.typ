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
    Rewriting the Line Filter with an Isotropic WVF Pair
  ]
  #v(0.5em)
  #text(size: 11pt, style: "italic")[
    A derivation showing where the LF can be simplified without making the WVF anisotropic.
  ]
]

= The Constraint <sec:constraint>

The simplification has to keep the Wide View Filter isotropic. That means the polynomial fit used to estimate first derivatives must use one fixed neighborhood around each pixel, independent of the candidate LF orientation. The orientation-dependent part is allowed to live in the line averaging stage, but the WVF itself should not be rebuilt as a different rotated support for every $theta$.

This distinction matters because the standard LF is usually written as if a WVF is evaluated at many virtual points for every candidate angle @bagan2021wvf @bagan2023lf. If that statement is implemented literally, it looks like the algorithm needs a different WVF derivative row for every $theta$. The derivation below shows that this is unnecessary when the WVF neighborhood and polynomial basis are isotropic. The rotated WVF derivative row is exactly a cosine-sine combination of one canonical $G_x$ row and one canonical $G_y$ row.

= The Isotropic WVF Pair <sec:isotropic-pair>

Let $P = {p_i}_{i=1}^{N_p}$ be a fixed circular set of neighbor offsets around a pixel $x$. The key word is fixed. The same offsets are used no matter which LF angle is being tested. Let $bold(A)$ be the Taylor design matrix evaluated at those offsets, and let $bold(A)^+$ be its pseudoinverse. For a local vector of sampled image intensities $bold(b)_x$, the isotropic derivative estimates are

$
  G_x(x) = bold(e)_x^top bold(A)^+ bold(b)_x,
  quad
  G_y(x) = bold(e)_y^top bold(A)^+ bold(b)_x.
$ <eq:gx-gy>

Here $bold(e)_x^top bold(A)^+$ and $bold(e)_y^top bold(A)^+$ are the first-derivative rows of the same local polynomial fit. They are not orientation-indexed. They are just two fixed filters. In implementation terms, $G_x$ and $G_y$ are obtained by two image correlations with two canonical WVF derivative kernels.

Now introduce a candidate direction

$
  n_theta = (cos theta, sin theta)^top.
$ <eq:normal>

The directional derivative of the same fitted polynomial in direction $n_theta$ is

$
  D_theta I(x)
    =
  n_theta dot
  mat(delim: "[",
    G_x(x);
    G_y(x)
  )
    =
  cos theta G_x(x) + sin theta G_y(x).
$ <eq:directional-derivative>

This is the central identity. It follows from the chain rule. If the local coordinate $u$ is defined by $u = X cos theta + Y sin theta$, then

$
  partial / partial u
    =
  cos theta partial / partial X
  + sin theta partial / partial Y.
$ <eq:chain-rule>

Since the Taylor basis contains the complete polynomial terms up to the selected degree, rotating coordinates changes the parameterization of the same fitted polynomial space. It does not require a new anisotropic WVF support. The first derivative in the rotated local $x$ direction is therefore exactly the cosine-sine combination in @eq:directional-derivative.

#figure(
  image("figures/isotropic_wvf_lf/fig_derivative_identity.png", width: 100%),
  caption: [
    A numerical check of @eq:directional-derivative for the local Taylor/WVF derivative rows. The first row shows derivative kernels obtained by rotating the WVF coordinates. The second row shows the same kernels reconstructed from one isotropic $G_x$ and $G_y$ pair. The third row shows the difference. The maximum row-weight error over the 36-angle bank is $6.55 times 10^(-15)$.
  ],
) <fig:derivative-identity>

= The Standard LF Formulation <sec:standard-lf>

The standard LF response at pixel $x$ and angle $theta$ can be written as a weighted sum of directional WVF responses at virtual positions along a line. With the current implementation's rounded virtual positions, define

$
  q_j(theta)
    =
  op("round")(j v_theta),
  quad
  j in {-m, ..., m}.
$ <eq:rounded-offsets>

For the standard implementation used in this repository,

$
  v_theta = (cos theta, sin theta)^top.
$ <eq:standard-line-direction>

Then the standard LF response is

$
  R_"std"(x, theta)
    =
  sum_(j=-m)^m w_j D_theta I(x + q_j(theta)).
$ <eq:standard-response>

This is the formulation that appears expensive. It appears to call for an orientation-indexed WVF derivative estimator $D_theta$ at each virtual point $x + q_j(theta)$. The next step is simply to substitute the isotropic identity from @eq:directional-derivative into @eq:standard-response.

= The Isotropic-WVF LF Rewrite <sec:rewrite>

Substituting @eq:directional-derivative into @eq:standard-response gives

$
  R_"std"(x, theta)
    =
  sum_(j=-m)^m w_j
  (
    cos theta G_x(x + q_j(theta))
    +
    sin theta G_y(x + q_j(theta))
  ).
$ <eq:substituted-response>

The trigonometric factors do not depend on $j$, so they can be pulled outside the line sum.

$
  R_"std"(x, theta)
    =
  cos theta
  sum_(j=-m)^m w_j G_x(x + q_j(theta))
  +
  sin theta
  sum_(j=-m)^m w_j G_y(x + q_j(theta)).
$ <eq:separated-response>

Define the oriented line-smoothing operator

$
  S_theta[f](x)
    =
  sum_(j=-m)^m w_j f(x + q_j(theta)).
$ <eq:line-smoothing>

Then @eq:separated-response becomes

$
  R_"std"(x, theta)
    =
  cos theta S_theta[G_x](x)
  +
  sin theta S_theta[G_y](x).
$ <eq:isotropic-lf-response>

This is the clean simplified form. The WVF is now only two isotropic derivative images. The orientation-dependent part is only line smoothing of those two derivative images, followed by a scalar cosine-sine combination.

= Kernel-Level Equivalence <sec:kernels>

The same statement can be written as a kernel identity. Let $K_x$ and $K_y$ be the two isotropic WVF derivative kernels. Let $L_theta$ be the discrete line-smoothing kernel formed by the offsets $q_j(theta)$ and weights $w_j$. Then the standard LF kernel for angle $theta$ is

$
  K_"std"(theta)
    =
  L_theta ast (cos theta K_x + sin theta K_y).
$ <eq:kernel-identity>

Here $ast$ denotes the discrete composition produced by shifting the derivative kernel to each virtual line position and summing the shifted weights. This is exactly what the fused LF kernel builder does. It sums over line positions and WVF neighbor positions, but @eq:kernel-identity says those sums can be interpreted as line smoothing applied after one fixed isotropic derivative pair.

#figure(
  image("figures/isotropic_wvf_lf/fig_lf_kernel_identity.png", width: 100%),
  caption: [
    Standard LF kernels compared with kernels rebuilt from one isotropic $G_x/G_y$ pair plus oriented line smoothing. The maximum absolute kernel difference over the full 36-angle bank is $1.03 times 10^(-15)$.
  ],
) <fig:kernel-identity>

The practical interpretation is subtle. This derivation does not prove that the exact LF can be evaluated by only two image convolutions. It proves that the WVF portion can be evaluated by two isotropic convolutions. The line-smoothing operator $S_theta$ still depends on $theta$. Therefore an exact implementation still needs orientation-dependent line smoothing, unless that line smoothing is accelerated by another method.

= Comparison to the Standard LF <sec:comparison>

The standard and isotropic forms are summarized in @tab:standard-vs-isotropic. Both produce the same signed LF response when the WVF support is fixed and isotropic. The difference is where the orientation dependence is placed.

#figure(
  table(
    columns: (1.2fr, 2.4fr, 2.4fr),
    align: (left, left, left),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header([*View*], [*Expression*], [*Interpretation*]),
    table.hline(stroke: 0.5pt),
    [Standard LF], [$sum_j w_j D_theta I(x + q_j(theta))$], [Evaluate an orientation-indexed WVF derivative along an orientation-indexed line.],
    [Isotropic-WVF LF], [$cos theta S_theta[G_x](x) + sin theta S_theta[G_y](x)$], [Compute one fixed $G_x/G_y$ pair, then apply orientation-indexed line smoothing.],
    [Structure tensor], [$n_theta^top J(x) n_theta$], [Square the derivative fields and maximize local second-moment energy. This is related but not the same LF response.],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Comparison between the standard LF expression, the exact isotropic-WVF rewrite, and the structure-tensor energy formulation.
  ],
) <tab:standard-vs-isotropic>

The numerical comparison in @fig:response-identity uses BIPED RGB008 resized to a maximum side length of 128 pixels, with $m=7$, $N_p=15$, degree $4$, circular WVF support, and 36 sampled angles. The standard LF response stack and the isotropic-WVF response stack agree to floating-point precision. The maximum response difference is $2.30 times 10^(-15)$, the relative response RMSE is $3.32 times 10^(-14)$, and the gated argmax orientation agreement is exactly 1.0.

#figure(
  image("figures/isotropic_wvf_lf/fig_response_identity.png", width: 100%),
  caption: [
    Image-level comparison between the standard LF response stack and the isotropic-WVF rewrite. The magnitude, angle, and response differences are at numerical roundoff.
  ],
) <fig:response-identity>

= What This Simplifies <sec:simplifies>

The derivation moves the LF from an orientation-indexed WVF view to an isotropic derivative-plus-line-smoothing view. The exact response is unchanged, but the mathematical structure is cleaner.

The expensive object is no longer a WVF bank. It is an oriented line-smoothing bank applied to two derivative images. That is a better target for simplification. Line smoothing has more exploitable structure than repeated local polynomial fitting. It may be accelerated by separable line kernels, sparse stencil gathering, integral-image style sums for unweighted lines, or a low-rank angular basis. Those are implementation choices. The core mathematical point is that they should operate on $G_x$ and $G_y$, not on a re-estimated anisotropic WVF.

This also explains why the structure tensor did not reproduce LF + spline. The tensor replaces the signed line-smoothed response in @eq:isotropic-lf-response with squared local energy. It answers a different question. The isotropic-WVF LF rewrite keeps the signed linear evidence and therefore keeps the behavior of the standard LF.

= The Edge-Tangent Variant <sec:tangent-variant>

There is one geometric caveat. The current standard implementation uses $v_theta = (cos theta, sin theta)^top$ for both the derivative direction and the virtual line direction. If we want the line smoothing to run along the edge tangent while $theta$ remains the edge normal, then the line direction should be

$
  t_theta = (-sin theta, cos theta)^top.
$ <eq:tangent-direction>

The corresponding tangent-smoothed response would be

$
  R_"tan"(x, theta)
    =
  cos theta S_(t_theta)[G_x](x)
  +
  sin theta S_(t_theta)[G_y](x).
$ <eq:tangent-response>

This form still keeps the WVF isotropic. However, it is not the same as the current standard LF unless the standard LF is also changed to use tangent virtual positions. The exact equivalence demonstrated in @fig:kernel-identity and @fig:response-identity applies to the current standard LF direction convention in @eq:standard-line-direction.

= Recommendation <sec:recommendation>

The next simplification should be based on @eq:isotropic-lf-response. It is exact for the current LF under an isotropic WVF and it gives a much better computational target than the structure tensor. The immediate implementation experiment should compute $G_x$ and $G_y$ once, apply the current rounded line smoothing operator to each field for each angle, and verify equality against the existing fused LF response stack at full resolution. After that equality test is locked down, the remaining work is to approximate or accelerate only the oriented line-smoothing bank.

#pagebreak()

#bibliography("journal_paper/refs.bib")
