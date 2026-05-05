# Multi-scale WVF weight derivation

This note fixes the analytical design for the first-pass multi-scale WVF
extension before any synthetic or real-image compute is launched.

## Scope

The candidate scale set is the existing Pareto trace

- `(r, d) = (3, 5)`
- `(r, d) = (5, 9)`
- `(r, d) = (9, 11)`
- `(r, d) = (15, 11)`
- `(r, d) = (25, 11)`
- `(r, d) = (50, 11)`

with `normalize_coords = True` throughout and the existing conditioning gate
from the degree-radius interaction sweep.

The combination strategies in scope here are the linear families `L1` and `L2`.
The nonlinear `L3` max-response baseline is intentionally left out of the
derivation because it does not preserve linearity or steerability.

## Theory anchors

The current theory manuscript does not expose these results as numbered theorem
blocks. The relevant statements appear as labeled equations in
[wvf_weights_from_polynomial_fit.typ](/Users/user/Documents/edge-detection-filter-critique/papers/anisotropic-lf-proposal/wvf_weights_from_polynomial_fit.typ).
Those are the references used below.

The bias-side control comes from the Taylor remainder bound:

- `@eq:taylor-remainder`

The noise-side control comes from the derivative-row variance formula:

- `@eq:noise-variance`

The degree-sensitivity result needed for the conditioning discussion comes from:

- `@eq:norm-monotone`
- `@eq:variance-monotone`

So the Phase 0 note cites equation labels and section labels rather than theorem
numbers because the source manuscript currently has no theorem-numbering layer to
target.

## Per-scale model

For a fixed WVF scale `i`, let the estimated gradient pair be

`g_i = (g_{x,i}, g_{y,i})`.

For a sufficiently smooth local field, write the scale-specific estimator as

`g_i = grad f + b_i(f) + epsilon_i`

where `b_i(f)` is the deterministic truncation bias induced by the local
remainder term and `epsilon_i` is the noise term.

The Taylor-remainder bound implies that the local bias grows with support size
once the support reaches beyond the radius on which the first-order expansion is
accurate. The exact constant is scene-dependent because it depends on the local
Hessian and higher derivatives.

On the variance side, `@eq:noise-variance` gives

`Var(g_{x,i}) = sigma_n^2 ||p_{f_x}^{(i)}||^2`.

The theory-paper asymptotics reduce that derivative-row norm to the closed-form
radius-degree scaling

`||p_{f_x}^{(i)}||^2 approx c_{d_i} / r_i^4`

with

`c_d = (d + 1)^2 (d + 3)^2 / (16 pi)`.

This is the closed-form variance prefactor used in the rest of the design.

## L2. Variance-inverse linear weights

The simplest linear combiner is

`g_ms = sum_i alpha_i g_i`

with the affine constraint

`sum_i alpha_i = 1`.

If we ignore cross-scale noise covariance and keep only the per-scale variances,
the variance proxy is

`Var(g_ms) approx sigma_n^2 sum_i alpha_i^2 c_{d_i} / r_i^4`.

Minimizing that quadratic subject to `sum_i alpha_i = 1` gives the standard
inverse-variance solution

`alpha_i = v_i^(-1) / sum_j v_j^(-1)`

with

`v_i = c_{d_i} / r_i^4`.

Therefore

`alpha_i propto r_i^4 / c_{d_i}`.

This is the proposed `L2` rule.

## L1. Bias-variance linear weights

If a signed scalar bias surrogate `beta_i(f)` is available for the local edge
normal direction at each scale, then a first-order MSE proxy for the combined
gradient component is

`MSE(alpha) = (sum_i alpha_i beta_i)^2 + sum_i alpha_i^2 v_i`

again with `sum_i alpha_i = 1`.

Writing

- `alpha` for the weight vector
- `beta` for the bias-surrogate vector
- `D = diag(v_1, ..., v_m)`

the quadratic objective becomes

`MSE(alpha) = alpha^T (D + beta beta^T) alpha`

subject to `1^T alpha = 1`.

The constrained minimizer is

`alpha* = M^(-1) 1 / (1^T M^(-1) 1)`

with

`M = D + beta beta^T`.

This is the natural `L1` rule.

Two practical consequences matter immediately.

First, if no usable bias surrogate is available, then `beta = 0` and `L1`
reduces exactly to `L2`.

Second, the exact same derivation works with a full cross-scale covariance
matrix `C` replacing `D`. That is,

`M = C + beta beta^T`

with

`C_{ij} = sigma_n^2 <p_{f_x}^{(i)}, p_{f_x}^{(j)}>`

or equivalently the kernel inner-product form. The current Phase 0 design keeps
the diagonal approximation because the theory paper provides the closed-form
per-scale variance prefactor directly, while the off-diagonal terms would need a
numerical evaluation pass. This is a refinement target, not a structural
blocker.

## Steerability

For linear combinations of derivative pairs,

`K_x^ms = sum_i alpha_i K_{x,i}`

and

`K_y^ms = sum_i alpha_i K_{y,i}`.

Therefore

`K_theta^ms = K_x^ms cos theta + K_y^ms sin theta = sum_i alpha_i (K_{x,i} cos theta + K_{y,i} sin theta)`.

So any fixed linear combiner over `(K_x, K_y)` preserves the steerability
identity to machine precision, assuming each component scale already satisfies
the discrete steerability relation.

This is why `L1` and `L2` remain analytically attractive and why `L3` is only a
nonlinear baseline.

## Concentration check on the six-scale trace

Using the `L2` rule on the original six-scale stack gives the following
normalized weights.

| Scale | `c_d` | `r^4 / c_d` | weight |
| --- | ---: | ---: | ---: |
| `(3, 5)` | `45.8366` | `1.7671` | `0.000148` |
| `(5, 9)` | `286.4789` | `2.1817` | `0.000183` |
| `(9, 11)` | `561.4986` | `11.6848` | `0.000979` |
| `(15, 11)` | `561.4986` | `90.1605` | `0.007556` |
| `(25, 11)` | `561.4986` | `695.6829` | `0.058302` |
| `(50, 11)` | `561.4986` | `11130.9264` | `0.932832` |

The largest scale carries about `93.28%` of the total mass.

That crosses the explicit `>90%` concentration stop rule, so the largest scale
must be dropped from the active stack before Phase 1.

## Active stack after the concentration rule

Dropping `(50, 11)` and renormalizing the same `L2` weights gives:

| Scale | renormalized weight |
| --- | ---: |
| `(3, 5)` | `0.002205` |
| `(5, 9)` | `0.002722` |
| `(9, 11)` | `0.014579` |
| `(15, 11)` | `0.112493` |
| `(25, 11)` | `0.868001` |

This is still strongly concentrated on `(25, 11)`, but it is below the explicit
drop threshold and therefore admissible for Phase 1.

The operational interpretation is straightforward. A pure variance-only
combiner, even after dropping `(50, 11)`, is still almost a wide-scale filter.
So if `L2` succeeds later, it will likely be because the remaining scales repair
localized failure cases at low measure rather than because all scales contribute
equally.

## Runtime safety rule

Before any multi-scale combination is formed, every active `(r, d)` pair must
pass the existing conditioning gate.

If any scale fails:

1. exclude it from the active stack
2. renormalize the remaining weights
3. record the exclusion in the summary JSON

This check must be explicit at runtime. Silent inclusion of a rank-deficient
scale would contaminate the combined response with numerical noise and make the
later comparisons uninterpretable.

## Phase 0 decision

There is no analytical blocker that forces the multi-scale direction to stop
before Phase 1.

There is, however, one design change forced immediately by the concentration
check:

- do not use the original six-scale stack for the first synthetic validation
- use the five-scale active stack
  - `(3, 5)`
  - `(5, 9)`
  - `(9, 11)`
  - `(15, 11)`
  - `(25, 11)`

Phase 1 should therefore test:

- `L1`, implemented as bias-aware when a usable local bias surrogate is
  available and otherwise reduced to `L2`
- `L2`, with the five-scale variance-inverse weights above
- `L3`, as the nonlinear max-magnitude baseline

## Phase 1.5 addendum

The synthetic Phase 1 follow-up exposed a methodological problem with using the
`>90%` concentration rule as a hard exclusion rule.

When the single-scale optimum itself sits at the largest available support,
excluding that support handicaps the multi-scale stack relative to the
single-scale baseline on exactly the stimuli where the large scale is supposed
to win. That makes the concentration rule useful as a diagnostic, but not as a
universal stack-pruning rule.

So for Phase 1.5 the six-scale stack is restored:

- `(3, 5)`
- `(5, 9)`
- `(9, 11)`
- `(15, 11)`
- `(25, 11)`
- `(50, 11)`

The concentration calculation above remains valid and still matters
interpretively. It means the variance-inverse `L2` rule is expected to behave
almost like the largest-scale WVF whenever the data support that bias. But the
Phase 1.5 rerun uses the full six-scale stack so the multi-scale combiner is no
longer competing with a strictly weaker scale set than the single-scale
baseline.

If the later synthetic or real-image phases show that `(25, 11)` still
dominates too strongly, the next refinement should be covariance-aware linear
weighting rather than immediately jumping to learned combiners.
