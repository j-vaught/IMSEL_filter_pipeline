#import "../colors.typ": *

// ======================================================================
// 4. FUSED STENCIL FORMULATION
// ======================================================================
= Fused Stencil Formulation <sec:fused>

The line-extended anisotropic steerable filter described in the previous section computes the orientation response $R_k$ through a sequence of $(2m+1)$ independent polynomial fits, each requiring a separate gather of $N_p$ pixel intensities, coordinate rotation, and pseudoinverse multiplication. For practical line-extension lengths ($m = 7$ implies 15 polynomial fits per orientation), this cascade dominates the computational cost and exhibits significant redundancy, since adjacent fit positions share most of their circular neighborhoods. This section derives an algebraic reformulation that collapses the entire cascade into a single weighted sum over a deduplicated stencil, and then examines a structural limitation of the resulting weights that motivates the geometric kernel designs of subsequent sections.


== Algebraic Collapse <sec:collapse>

Recall from @eq:line-response that the line-extended response at orientation $theta_k$ is a Gaussian-weighted sum of normal-derivative estimates along the candidate edge direction.

$ R_k = sum_(j=-m)^(m) w_j dot hat(f)_x^((j,k)) $ <eq:rk-recall>

Each term $hat(f)_x^((j,k))$ is itself a dot product of the pseudoinverse row $bold(p)_"fx"^((k))$ with the intensity vector gathered from the circular neighborhood centered at line position $j$. Substituting @eq:fx into @eq:rk-recall and writing the gathered intensity explicitly yields a double sum over line positions and neighbor indices.

$ R_k = sum_(j=-m)^(m) w_j sum_(i=1)^(N_p) p_i^((k)) dot f(X_0 + delta_(j,i)^x, Y_0 + delta_(j,i)^y) $ <eq:double-sum>

Here $p_i^((k)) = bold(p)_"fx"^((k))[i]$ denotes the $i$-th entry of the pseudoinverse gradient row for orientation $theta_k$, and the combined offsets are

$ delta_(j,i)^x = j cos theta_k + Delta x_i, quad delta_(j,i)^y = j sin theta_k + Delta y_i $ <eq:offsets>

where $(Delta x_i, Delta y_i)$ are the integer coordinates of the $i$-th circular neighbor relative to the fit center. Because the line offset $(j cos theta_k, j sin theta_k)$ adds a non-integer displacement, the combined offset is rounded to the nearest integer pixel.

The critical observation is that @eq:double-sum is a linear functional of image intensities. Any linear combination of pixel values can be written as a single weighted sum over the set of distinct pixel positions that appear in the combination. Grouping all $(j, i)$ pairs whose rounded offsets coincide at the same integer pixel $ell$, the double sum collapses to

$ R_k = sum_(ell=1)^(N'_k) alpha_(k,ell) dot f(X_0 + tilde(delta)_(ell)^x, Y_0 + tilde(delta)_(ell)^y) $ <eq:fused-stencil>

where $N'_k$ is the number of unique pixel positions in the stencil for orientation $k$, and the _fused weight_ at position $ell$ aggregates all contributions that map to that pixel.

$ alpha_(k,ell) = sum_((j,i) in cal(S)_ell) w_j dot p_i^((k)) $ <eq:fused-weight>

The index set $cal(S)_ell = {(j, i) : op("round")(delta_(j,i)^x, delta_(j,i)^y) = (tilde(delta)_ell^x, tilde(delta)_ell^y)}$ collects all line-position and neighbor-index pairs that round to the same integer offset. The result is that the entire line-extended filter at orientation $theta_k$ reduces to a single gather-dot-product operation.

$ R_k = bold(alpha)_k^top bold(g)_k $ <eq:gather-dot>

The weight vector $bold(alpha)_k in RR^(N'_k)$ depends only on the filter parameters ($m$, $N_p$, $d$, $theta_k$) and can be precomputed once. The intensity vector $bold(g)_k in RR^(N'_k)$ is assembled at runtime by reading the image at the $N'_k$ stencil offsets. No coordinate rotations, no design matrices, and no pseudoinverse products appear at evaluation time.

#figure(
  image("../figures/fig_sec04_fused_stencil_weight_maps.pdf", width: 100%),
  caption: [Fused weight maps $alpha_(k,ell)$ computed from the pseudoinverse for four representative orientations ($m = 7$, $N_p = 100$, $d = 4$). Garnet indicates positive weights and blue indicates negative weights, with intensity proportional to magnitude. The dark outline traces the outer boundary of the active fused stencil support. Even after deduplication, the support remains highly elongated and sparse, with weight efficiency $eta approx 78$--$84%$ across these orientations.],
  placement: top,
  scope: "parent",
) <fig:weight-maps>


== Deduplication <sec:dedup>

The raw stencil for orientation $theta_k$ contains $(2m+1) times N_p$ entries before grouping. Because adjacent line positions share most of their circular neighborhoods, many of these entries map to the same integer pixel after rounding. The deduplication process iterates over all $(j, i)$ pairs, computes the rounded integer offset, inserts the result into a hash map keyed by offset, and accumulates the product $w_j dot p_i^((k))$ into the corresponding weight. The output is a list of unique (offset, weight) pairs that fully characterize the stencil.

The degree of compression depends on the ratio of line-extension length to neighborhood radius. When $m$ is small relative to the neighbor count $N_p$, the circular neighborhoods at adjacent line positions overlap extensively, producing high redundancy. As $m$ increases, the stencil footprint elongates and the overlap fraction decreases, but the absolute number of duplicates continues to grow because the total raw entry count scales linearly with $m$. @tab:dedup-stats summarizes the compression achieved for representative parameter combinations.

#figure(
  table(
    columns: (1fr, 1fr, 1fr, 1fr, 1fr),
    align: center,
    stroke: 0.5pt + black70,
    table.header(
      table.cell(fill: black10)[*$m$*],
      table.cell(fill: black10)[*$N_p$*],
      table.cell(fill: black10)[*Raw entries*],
      table.cell(fill: black10)[*Unique positions*],
      table.cell(fill: black10)[*Reduction*],
    ),
    [1],  [100], [300],  [128], [57%],
    [2],  [100], [500],  [152], [70%],
    [7],  [100], [1500], [264], [82%],
    [14], [100], [2900], [431], [85%],
  ),
  caption: [Stencil deduplication statistics for $N_p = 100$ circular neighbors at polynomial order $d = 4$. Raw entries denotes the count $(2m+1) times N_p$ before grouping. Unique positions is the mean count of distinct integer pixel offsets across all $N_s$ orientations. Reduction is defined as $1 - N'_k slash ((2m+1) N_p)$.],
) <tab:dedup-stats>

At $m = 1$, only three line positions contribute and the circular neighborhoods overlap almost completely, yielding a 57% reduction. At $m = 7$, the 15 line positions produce 1500 raw entries that collapse to roughly 264 unique pixels, an 82% reduction. At $m = 14$, the compression reaches 85%, meaning fewer than one in six raw entries corresponds to a distinct memory access. This compression is the primary mechanism by which the fused formulation reduces both arithmetic operations and memory bandwidth relative to the naive cascade.


The fused formulation raises a natural question. Given that we have collapsed the polynomial fit cascade into a single weighted sum, can we instead define the stencil weights directly from a geometric prescription that avoids the pseudoinverse altogether? The answer, developed in the following section, is to replace the polynomial gradient estimation with an analytically defined kernel whose weights are constructed from a smooth envelope function and a linear ramp.
