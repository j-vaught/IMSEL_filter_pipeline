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
  image("../figures/fig_sec4_weight_maps.pdf", width: 95%),
  caption: [Fused weight maps $alpha_(k,ell)$ for six representative orientations ($m = 7$, $N_p = 100$, $d = 4$). Red indicates positive weights and blue indicates negative weights, with intensity proportional to magnitude. The dipole pattern (positive on one side of the edge normal, negative on the other) rotates with $theta_k$, and the stencil elongates along the candidate edge tangent.],
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


== The Weight Cancellation Problem <sec:cancellation>

Although the fused stencil eliminates redundant computation, it exposes a structural inefficiency in the polynomial-derived weights. A large fraction of the fused weight magnitudes $|alpha_(k,ell)|$ are close to zero, meaning that many of the unique stencil positions contribute negligibly to the response $R_k$. This phenomenon is not a numerical artifact but a systematic consequence of the algebraic structure of the pseudoinverse weights.

The origin of the cancellation can be understood as follows. Each polynomial fit at line position $j$ produces pseudoinverse weights $p_i^((k))$ that encode the gradient extraction from a local Taylor expansion. These weights contain both positive and negative entries, reflecting the differential nature of the gradient operator. When adjacent fit positions $j$ and $j+1$ share a pixel at the same integer offset, the contributions $w_j dot p_i^((k))$ and $w_(j+1) dot p_(i')^((k))$ that are summed in @eq:fused-weight come from different rows of different pseudoinverse matrices. Because the polynomial basis is shifted by one pixel between adjacent fits, the pseudoinverse entries at the shared pixel typically have opposite signs. The Gaussian line weights $w_j$ and $w_(j+1)$ are both positive, so the products partially cancel when summed.

More formally, the fused weight at a position $ell$ that is shared by $n$ overlapping neighborhoods takes the form

$ alpha_(k,ell) = sum_(s=1)^(n) w_(j_s) dot p_(i_s)^((k)) $ <eq:cancel-formal>

where the indices $(j_s, i_s)$ enumerate the distinct (line-position, neighbor-index) pairs mapping to $ell$. The pseudoinverse entries $p_(i_s)^((k))$ are elements of the first row of $(bold(A)_(theta_k)^top bold(A)_(theta_k))^(-1) bold(A)_(theta_k)^top$, evaluated at different column positions corresponding to different local coordinates within each shifted neighborhood. The sign pattern of these entries is determined by the monomial basis and the geometry of the neighbor set, and for pixels near the center of the stencil (where overlap is highest), adjacent fits produce contributions of opposite polarity.

To quantify the severity of cancellation, we define a _weight efficiency_ metric.

$ eta = (sum_(ell=1)^(N'_k) |alpha_(k,ell)|) / (sum_(j=-m)^(m) |w_j| sum_(i=1)^(N_p) |p_i^((k))|) $ <eq:efficiency>

The numerator is the total absolute weight in the fused stencil, and the denominator is the total absolute weight before any cancellation occurs (i.e., the sum of absolute contributions across all raw entries). A value $eta = 1$ would indicate no cancellation, while $eta < 1$ indicates that a fraction $1 - eta$ of the raw weight magnitude has been lost to sign cancellation during deduplication. Empirically, $eta approx 0.72$ for $m = 7$ and $N_p = 20$ at order $d = 4$, meaning approximately 28% of the aggregate pseudoinverse weight magnitude is destroyed by cancellation. For larger neighborhoods ($N_p = 100$), the efficiency drops further because the increased overlap multiplies the number of canceling pairs.

#figure(
  image("../figures/fig_sec4_cancellation.pdf", width: 85%),
  caption: [Weight cancellation illustration for orientation $theta_0 = 0$ with $m = 7$ and $N_p = 100$. Left: histogram of fused weight magnitudes $|alpha_(k,ell)|$, showing a concentration near zero. Right: weight efficiency $eta$ as a function of line-extension length $m$ for several neighborhood sizes $N_p$. Efficiency decreases monotonically with both $m$ and $N_p$.],
) <fig:cancellation>


== Consequences of Cancellation <sec:cancel-consequences>

The weight cancellation phenomenon has three practical consequences that limit the effectiveness of the polynomial-derived fused stencil.

First, the cancelled weights represent wasted computation. Pixels with near-zero fused weights are still gathered from memory and multiplied by their weights during the dot product @eq:gather-dot. Pruning these entries (by zeroing weights below a threshold) reduces the effective stencil size but introduces an approximation error that depends on the threshold in a non-obvious way, since the cancelled weights are distributed throughout the stencil rather than concentrated at the periphery.

Second, the cancellation produces an irregular frequency response. The fused stencil is a discrete linear filter and can be analyzed via its discrete Fourier transform. Weights that arise from a smooth analytic design (such as a Gaussian derivative) produce a predictable, well-behaved frequency response. By contrast, the polynomial-derived fused weights inherit the oscillatory structure of the pseudoinverse entries, and the partial cancellations introduce high-frequency ripple in the transfer function. This ripple makes it difficult to characterize the filter's noise behavior analytically, since the effective bandwidth does not correspond to a simple parametric form.

Third, the algebraic origin of the weights makes the filter's properties opaque to theoretical analysis. The fused weights $alpha_(k,ell)$ are the end product of a chain of operations (neighbor selection, monomial basis construction, pseudoinverse computation, Gaussian weighting, rounding, and summation) that obscures the relationship between the filter parameters and the resulting spatial-frequency characteristics. Unlike a Gaussian derivative kernel, whose bandwidth, zero-crossings, and moment properties follow directly from its analytic definition, the polynomial stencil's properties can only be determined numerically for each parameter configuration.

#figure(
  image("../figures/fig_sec4_efficiency.pdf", width: 90%),
  caption: [Comparison of weight distributions for the polynomial-derived fused stencil (left) and a matched elliptical Gaussian derivative kernel (right) at the same effective support size. The polynomial weights exhibit a broad spread of magnitudes with many near-zero entries, while the Gaussian derivative weights decay smoothly from the center with no cancellation artifacts.],
) <fig:efficiency-comparison>

These observations raise a natural question. If the polynomial fit machinery produces weights that partially cancel and yield an irregular frequency response, can we instead define the stencil weights directly from a geometric prescription that avoids the pseudoinverse altogether? The answer, developed in the following section, is to replace the polynomial gradient estimation with an analytically defined kernel whose weights are constructed from a smooth envelope function and a linear ramp, eliminating the cancellation problem by construction.
