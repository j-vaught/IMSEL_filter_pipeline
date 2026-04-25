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
    Disk-Weighted Structure Tensors and LF + Spline Orientation
  ]
  #v(0.5em)
  #text(size: 11pt, style: "italic")[
    A mathematical comparison and pilot implementation report by J.C. Vaught.
  ]
]

= The Question <sec:question>

The implementation question is whether the existing line-filter plus spline orientation stage can be replaced by a disk-weighted structure tensor built from the same first-derivative information. The motivation is clear. The current line-filter pipeline evaluates a bank of oriented line responses, samples a periodic response curve, and fits a cubic spline to locate the peak. The structure tensor instead forms three quadratic images, smooths them spatially, and obtains orientation by a closed-form eigensystem. If both methods are computing the same orientation in different coordinates, then the tensor version should be simpler, faster, and easier to explain.

The short answer is that the two methods are not mathematically equivalent in general. They become very similar under a restricted single-edge model where the local gradients are rank-one and do not change sign across the support. Outside that model, the line-filter plus spline stage and the structure tensor solve different optimization problems. The structure tensor maximizes local squared directional energy over a fixed isotropic window. The LF + spline stage maximizes the absolute value of a signed, orientation-indexed line average whose support rotates with the candidate angle. Those distinctions are small for a clean isolated edge and large near corners, nearby parallel structures, low-SNR texture, and small fragmented objects.

This report develops that statement explicitly, then connects it to the pilot implementation in `src/edgecritic/structure_tensor.py` and the comparison run in `outputs/structure_tensor_comparison/summary.json`. The experimental result is consistent with the analysis. The tensor variant is much faster, but the measured angle differences are far larger than the replacement criterion. On the local test subset, the best mean orientation differences are about 32 to 41 degrees, most gated pixels differ by more than 5 degrees, and the final edge maps have weak overlap with the LF outputs.

= The Two Orientation Problems <sec:two-problems>

== The LF + Spline Objective <sec:lf-objective>

Let $I$ be the image, let $x$ be the pixel location, and let $theta in [0, pi)$ be a candidate axial orientation. In the implemented line filter, the polynomial derivative row estimates a directional derivative along the rotated local $x$ axis. Denote that derivative estimator by $D_theta I$. The line filter then averages this derivative over virtual positions along the same candidate direction. A simplified expression for the signed LF response is

$
  R_"LF"(x, theta)
    =
  sum_(j=-m)^m w_j D_theta I(x + j v_theta),
  quad
  v_theta = (cos theta, sin theta).
$ <eq:lf-response>

The current orientation stage samples @eq:lf-response at a finite set of angles, takes absolute values, fits a periodic cubic spline through those samples, and chooses

$
  hat(theta)_"LF"(x)
    =
  arg max_(theta in [0, pi)) abs(R_"LF"(x, theta)).
$ <eq:lf-max>

The important features of @eq:lf-max are that the response is signed before the absolute value is taken, and the sampled support changes with $theta$. The method is therefore not merely asking which direction has large derivative energy at a pixel. It is asking which oriented line support produces the strongest coherent signed response after the WVF derivative estimates have been averaged along that line.

== The Structure-Tensor Objective <sec:tensor-objective>

The proposed tensor variant starts with two derivative fields, written here as

$
  g(x) =
  mat(delim: "[",
    G_x(x);
    G_y(x)
  ).
$ <eq:gradient-vector>

It then forms the three quadratic products

$
  J_"xx,raw" = G_x^2,
  quad
  J_"yy,raw" = G_y^2,
  quad
  J_"xy,raw" = G_x G_y.
$ <eq:raw-products>

After smoothing these products with a disk or Gaussian weight $W$, the local second-moment matrix is

$
  J(x)
    =
  mat(delim: "[",
    W * G_x^2, W * G_x G_y;
    W * G_x G_y, W * G_y^2
  ).
$ <eq:tensor>

For any unit direction $n(theta) = (cos theta, sin theta)^top$, the squared directional energy is

$
  E_"ST"(x, theta)
    =
  n(theta)^top J(x) n(theta).
$ <eq:tensor-energy>

Expanding @eq:tensor-energy gives

$
  E_"ST"
    =
  J_"xx" cos^2 theta
  + 2 J_"xy" sin theta cos theta
  + J_"yy" sin^2 theta.
$ <eq:tensor-expanded>

The maximizing direction is the dominant eigenvector of $J$. Differentiating @eq:tensor-expanded and setting the derivative to zero gives the familiar half-angle formula

$
  hat(theta)_"ST"
    =
  1 / 2 op("atan2")(2 J_"xy", J_"xx" - J_"yy").
$ <eq:tensor-angle>

The eigenvalues are

$
  lambda_1
    =
  1 / 2 ((J_"xx" + J_"yy") + sqrt((J_"xx" - J_"yy")^2 + 4J_"xy"^2)),
$ <eq:lambda-one>

$
  lambda_2
    =
  1 / 2 ((J_"xx" + J_"yy") - sqrt((J_"xx" - J_"yy")^2 + 4J_"xy"^2)).
$ <eq:lambda-two>

The two tested magnitudes are therefore

$
  M_A = sqrt(lambda_1),
  quad
  M_B = sqrt(lambda_1 - lambda_2).
$ <eq:tensor-magnitudes>

The coherence measure is

$
  c =
  (lambda_1 - lambda_2) / (lambda_1 + lambda_2 + epsilon).
$ <eq:coherence>

This is the classical structure tensor construction used for local orientation and corner analysis @forstner1987fast @bigun1987optimal @knutsson1989representing @jahne2005digital. The tensor objective is elegant because it converts the orientation search into a closed-form eigensystem. Its elegance, however, also reveals the difference. @eq:tensor-energy is a quadratic energy. @eq:lf-max is an absolute signed line response.

= The Equivalence That Does Hold <sec:equivalence>

There is one important case where the two procedures should agree. Suppose that, within the relevant support, the image contains one locally straight edge with one dominant normal direction $theta_0$. The derivative field can then be idealized as

$
  g(u) = rho(u) n(theta_0),
$ <eq:rank-one-field>

where $rho(u)$ is the scalar contrast profile and $n(theta_0)$ is the unit normal. Substituting @eq:rank-one-field into @eq:tensor gives

$
  J
    =
  (integral W(u) rho(u)^2 dif u)
  n(theta_0) n(theta_0)^top.
$ <eq:rank-one-tensor>

This matrix has rank one when the edge has nonzero contrast. Its dominant eigenvector is exactly $n(theta_0)$, so the structure tensor returns the true normal orientation.

The LF response also favors $theta_0$ in the same idealized setting. If the signed derivative samples along the line support have the same sign and do not vary strongly, the response in @eq:lf-response behaves approximately like

$
  R_"LF"(theta)
    approx
  cos(theta - theta_0) sum_(j=-m)^m w_j rho(x + j v_theta).
$ <eq:lf-clean-approx>

When the line average changes slowly with $theta$, @eq:lf-clean-approx is maximized at $theta_0$ modulo $pi$. In this restricted rank-one model, the tensor orientation, direct arctangent orientation, and dense LF orientation are all estimating the same underlying normal direction. This is the mathematical reason the tensor idea is plausible.

The conditions are stricter than they first appear. The edge must remain locally single-directional. The derivative signs must not cancel under line averaging. The line support must not collect additional structures as it rotates. The spatial weighting used by the tensor must also represent the same physical scale as the LF support. When those assumptions fail, the two methods separate.

= Why the General Equivalence Fails <sec:nonequivalence>

The most compact way to see the failure is to compare a squared first moment with a second moment. Ignore, for a moment, the fact that the LF support rotates. On one fixed support with normalized weights, let

$
  a_theta(u) = g(u) dot n(theta).
$ <eq:directional-signal>

The LF-like signed average is

$
  A(theta) = integral w(u) a_theta(u) dif u.
$ <eq:first-moment>

The tensor-like directional energy is

$
  Q(theta) = integral w(u) a_theta(u)^2 dif u.
$ <eq:second-moment>

By Cauchy's inequality,

$
  abs(A(theta))^2 <= Q(theta).
$ <eq:cauchy>

Equality in @eq:cauchy is not generic. It requires $a_theta(u)$ to be essentially constant over the support, up to the normalization of $w$. If the derivative changes sign, the signed LF average can cancel while the tensor energy increases. If two orientations are present, the tensor accumulates both as positive energy and returns the dominant eigenvector of the summed second moments. The LF response can instead prefer the angle whose line support preserves coherent signed evidence.

The rotating LF support makes the difference stronger. The tensor window is fixed before the orientation is chosen. A disk or Gaussian average at $x$ uses the same nearby samples for every $theta$. The line filter changes its sample locations with $theta$, so it is also searching over a family of anisotropic spatial supports. This extra spatial selectivity is not represented by @eq:tensor-energy. Therefore the tensor cannot be a closed-form solution to @eq:lf-max except under additional assumptions that collapse the support dependence.

This distinction also explains the magnitude behavior. $M_A = sqrt(lambda_1)$ is a dominant-energy magnitude. $M_B = sqrt(lambda_1 - lambda_2)$ subtracts isotropic energy, so it is closer to an edge-confidence measure. The LF magnitude is neither one exactly. It is the maximum absolute signed line average. A high tensor magnitude can come from multiple nearby structures that would not survive LF line coherence, and a strong LF response can occur when a thin structure aligns well with one line support but contributes little to a broader disk average.

= Implementation Used for the Pilot Comparison <sec:implementation>

The tensor implementation is in `src/edgecritic/structure_tensor.py`. The exported function is `structure_tensor_orientation(image, radius, weight_type)`. It returns the dominant orientation, both magnitudes, coherence, total energy, eigenvalues, and the derivative fields. Disk weights are normalized over integer pixels inside the radius. Gaussian weights use $sigma = L / 2$ and are evaluated by separable Gaussian filtering. Negative eigenvalues from floating-point roundoff are clipped before square roots.

The comparison driver is `scripts/figures/compare_structure_tensor_lf.py`. It runs the existing LF response stack and spline peak estimator, then evaluates tensor variants for disk and Gaussian weights at radii $8$, $15$, and $30$. The LF half-width is $m = 7$, so the full LF line length is $2m + 1 = 15$. The radius sweep therefore tests approximately half scale, matched scale, and double scale. For each image and parameter setting, the script reports angle differences on LF-gated pixels, Pearson magnitude correlations for $M_A$ and $M_B$, a convention sanity check against a 90-degree tangent rotation, final NMS plus hysteresis edge maps, and wall-clock timing.

The pilot used five local cases. They are a noisy synthetic step edge, a noisy multi-line synthetic image, one BIPED image, one UDED image, and one aquatic-domain example image. Ground-truth edge labels for these exact local files were not available in this run, so ODS, OIS, and mAP were not computed. Instead, the reported edge metric is F1 agreement against the LF-derived final edge map. That is not a benchmark detection score. It is a direct replacement test. If the tensor is a drop-in replacement for the LF orientation and magnitude source, its edge map should closely match the LF edge map even before external labels are introduced.

= Pilot Results <sec:results>

The replacement criterion was not met. The best setting by mean angle error is shown in @tab:pilot-summary. In every case, most gated pixels differ from the LF orientation by more than 5 degrees. The best mean angle errors range from 31.95 degrees on the aquatic example to 41.44 degrees on the synthetic multi-line image. Magnitude option A, $sqrt(lambda_1)$, correlates better with the LF magnitude than option B in all five cases. The Gaussian weight is usually the best orientation match and is much faster than disk convolution, although the aquatic image is the one case where the disk at radius $8$ gives the lowest angle error.

#text(size: 9pt)[
#figure(
  table(
    columns: (1.55fr, 1.25fr, 0.85fr, 0.75fr, 0.8fr, 0.7fr, 0.7fr, 0.7fr, 0.9fr),
    align: (left, left, right, right, right, right, right, right, right),
    stroke: none,
    table.hline(stroke: 0.8pt),
    table.header(
      [*Case*], [*Best setting*], [*Mean deg*], [*P90 deg*], [*>5 deg*], [*$r_A$*], [*$r_B$*], [*Edge F1*], [*Speedup*],
    ),
    table.hline(stroke: 0.5pt),
    [synthetic step], [Gaussian, $L=8$], [39.93], [79.16], [0.922], [0.708], [0.482], [0.052], [1473.9x],
    [synthetic multi-line], [Gaussian, $L=8$], [41.44], [78.04], [0.932], [0.403], [0.267], [0.038], [1430.8x],
    [BIPED 008], [Gaussian, $L=8$], [40.36], [71.06], [0.965], [0.739], [0.641], [0.228], [1396.9x],
    [UDED 02], [Gaussian, $L=8$], [38.16], [77.56], [0.920], [0.777], [0.550], [0.082], [1397.1x],
    [aquatic wake], [Disk, $L=8$], [31.95], [61.38], [0.954], [0.512], [0.452], [0.000], [184.8x],
    table.hline(stroke: 0.8pt),
  ),
  caption: [
    Best tensor setting for each local test case, selected by mean axial angle error against the LF + spline output on pixels where the LF magnitude exceeds 10 percent of its image maximum. $r_A$ and $r_B$ are Pearson correlations between LF magnitude and tensor magnitudes $M_A$ and $M_B$.
  ],
) <tab:pilot-summary>
]

The tangent-convention sanity check did not rescue the comparison. For the best settings in @tab:pilot-summary, the direct gradient convention was better than rotating the tensor orientation by 90 degrees. That means the mismatch is not explained by comparing an edge normal to an edge tangent. It is a genuine difference between the objectives in @eq:lf-max and @eq:tensor-energy.

#figure(
  image("../../outputs/structure_tensor_comparison/synthetic_step_snr1p5/gaussian_r8_panel.png", width: 100%),
  caption: [
    Synthetic step-edge comparison at the best tensor setting for that case. The orientation maps are not simply 90-degree rotations of each other, and the final edge maps have weak overlap after NMS and hysteresis.
  ],
) <fig:synthetic-step>

#figure(
  image("../../outputs/structure_tensor_comparison/biped_rgb008/gaussian_r8_panel.png", width: 100%),
  caption: [
    BIPED comparison with Gaussian radius $8$. The tensor magnitude is visually smoother and faster to compute, but the orientation-difference map and edge-disagreement panel show that it is not reproducing the LF source.
  ],
) <fig:biped>

#figure(
  image("../../outputs/structure_tensor_comparison/aquatic_dark_wake/disk_r8_panel.png", width: 100%),
  caption: [
    Aquatic example with disk radius $8$, the best setting for this case. This image gives the smallest mean angle error among the tested cases, but the final edge agreement remains poor.
  ],
) <fig:aquatic>

The radius sweep is also informative. Radius $8$ is consistently better than the matched LF line length radius $15$ and double radius $30$. Larger tensor windows smooth the quadratic products over too much competing structure, which pushes the tensor farther away from the line-filter response. This is the expected behavior if the LF is exploiting anisotropic line support rather than merely estimating a dominant orientation in a disk.

Timing strongly favors the tensor. The Gaussian tensor is roughly three orders of magnitude faster than the current LF + spline implementation at these image sizes. The disk tensor is slower than the Gaussian tensor because it uses a dense two-dimensional convolution, but it is still much faster than the LF bank at small radius. Speed alone is not enough for replacement, though. The measured angle and edge-map differences are too large for a drop-in substitution.

= Interpretation <sec:interpretation>

The mathematical and empirical results agree. The structure tensor is the right closed-form solution to a local second-moment orientation problem. LF + spline is solving an orientation-indexed line-coherence problem. They share the same first-order derivative ancestry, so they can point in similar directions when the local image really contains one clean edge. They diverge when the image patch contains several orientations, sign reversals, curved boundaries, fragmented small objects, or low-SNR texture.

This also clarifies the role of the GMM stage. A tensor produces one dominant orientation and a coherence scalar at each pixel. The LF stack produces an orientation response profile, and the existing GMM fusion can reason about multiple orientation candidates across scales. A tensor coherence value can identify whether the local second-moment matrix is rank-one, but it cannot recover the missing signed line-response profile after the quadratic products have been formed.

The result should not be read as a failure of the structure tensor. It is a strong baseline and a useful diagnostic. It tells us what happens when the problem is reduced to local dominant gradient energy. The observed divergence suggests that the LF stage is doing something beyond that reduction, especially through its oriented support and signed averaging. That difference is exactly the kind of behavior worth understanding in the method section.

= Recommendation <sec:recommendation>

The structure tensor should not replace LF + spline as a drop-in orientation and magnitude source in the current pipeline. The replacement criteria were mean angle error below about 1 degree, magnitude correlation above 0.95, and stable final edge maps. The pilot comparison is far from those thresholds. The best magnitude correlations reach 0.777 for $M_A$ and 0.641 for $M_B$ on individual cases, but the orientation differences and edge-map agreement are not close enough.

The better use is hybrid. Keep LF + spline for the path that depends on oriented line evidence and GMM fusion. Use the tensor path as a fast auxiliary estimator with three practical roles. First, it can provide a cheap dominant-orientation baseline for ablation. Second, it can provide a coherence gate that identifies pixels where a one-orientation model is mathematically credible. Third, it can help explain which LF gains come from anisotropic line support rather than from first-order derivative estimation alone.

The method-section rewrite should therefore state the relationship carefully. The tensor formulation is mathematically equivalent to LF-style orientation only under a local rank-one, no-cancellation, support-invariant edge model. In the general case it is not equivalent. It is a related second-moment formulation that deliberately discards sign and support-selection information. The divergence observed in the pilot comparison is a contribution because it isolates what the LF + spline stage is preserving.

#pagebreak()

#bibliography("structure_tensor_refs.bib")
