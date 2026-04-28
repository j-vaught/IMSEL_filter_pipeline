# Brief: Metal-accelerated c-GMM K=3 fusion

You are implementing a Metal-GPU port of the **c-GMM** K=3 fusion stage,
the per-pixel mixture fit that the edge-detection pipeline runs on the
output of orientation recovery. The goal is a drop-in replacement for
the numpy reference in
`agent_workspaces/cgmm_metal/reference_impl.py` that runs at least
**20x faster** at 1M rows and ideally enables full-image
(16.7M rows, N=16 measurements per pixel) processing in **under 2 s**
hard, **under 500 ms** stretch.

This is the third of three Metal ports in this repo. The pattern,
crate, and Python binding style follow `agent_workspaces/lf_metal/` and
`agent_workspaces/orientation_recovery_metal/`.

## What the operator does

For each pixel the pipeline carries N independent `(theta, M, v)`
measurements, one per `(channel, radius, degree, lf_half_length)`
configuration. The c-GMM fits a K=3 mixture on the doubled-angle
representation `(phi = 2*theta) mod 2*pi` to fuse those N measurements
into a primary orientation and (optionally) a secondary orientation.

It runs **two independent K=3 hard-EM fits** per pixel.

1. The **primary fit** ingests `(phi_p, w_p)`, the per-input primary
   `(theta, M)` from orientation recovery.
2. The **secondary fit** ingests `(phi_s, w_s)`, the per-input
   secondary `(theta, M)` from orientation recovery (NaN/zeroed where
   the recovery suppressed the secondary).

The two fits share no state, no initialization, no responsibilities.
Their `argmax_pi` components become the c-GMM primary and secondary
slots. A suppression rule then zeroes the secondary slot if the two
slots are too close in mass or angle.

## Inputs

| name      | shape   | dtype                   | meaning                                           |
|-----------|---------|-------------------------|---------------------------------------------------|
| `phi_p`     | `(P, N)`  | float64 (cast inside)   | primary doubled-angles in `[0, 2*pi)` per pixel   |
| `w_p`       | `(P, N)`  | float64 (cast inside)   | primary per-input weights `w = v * M` (>= 0)      |
| `phi_s`     | `(P, N)`  | float64 (cast inside)   | secondary doubled-angles in `[0, 2*pi)`           |
| `w_s`       | `(P, N)`  | float64 (cast inside)   | secondary per-input weights (>= 0)                |

Hyperparameters (production defaults).

| name           | default | role                                                        |
|----------------|---------|-------------------------------------------------------------|
| `K`              | 3       | mixture component count (the kernel may hard-code K=3)      |
| `n_iters`        | 30      | fixed EM iterations (no early stopping)                     |
| `init_kappa`     | 4.0     | initial concentration scalar                                |
| `hard_em`        | True    | hard E-step (production); soft-EM path is out of scope      |
| `tau_M_rel`      | 0.05    | secondary mass floor                                        |
| `theta_min_deg`  | 10.0    | secondary geometric separation floor (degrees)              |

## Outputs

| name                | shape    | dtype   | meaning                                          |
|---------------------|----------|---------|--------------------------------------------------|
| `theta_primary`       | `(P,)`     | float64 | primary orientation in `[0, pi)` or NaN          |
| `M_primary`           | `(P,)`     | float64 | primary mass `W[k_primary]`                      |
| `theta_sec`           | `(P,)`     | float64 | secondary orientation in `[0, pi)` or NaN        |
| `M_sec`               | `(P,)`     | float64 | secondary mass, 0 if suppressed                  |
| `v_fused`             | `(P,)`     | uint8   | per-pixel validity (range rule below)            |
| `primary_pi`          | `(P, K)`   | float64 | primary fit mixing weights (diagnostics)         |
| `primary_mu`          | `(P, K)`   | float64 | primary fit means in `[0, 2*pi)` (diagnostics)   |
| `primary_kappa`       | `(P, K)`   | float64 | primary fit concentrations (diagnostics)         |
| `secondary_pi`        | `(P, K)`   | float64 |                                                  |
| `secondary_mu`        | `(P, K)`   | float64 |                                                  |
| `secondary_kappa`     | `(P, K)`   | float64 |                                                  |
| `keep_secondary_mask` | `(P,)`     | uint8   | 1 if secondary survived the suppression rule     |

## Algorithm (production: K=3, hard_em=True, n_iters=30)

**Per-pixel validity (degenerate guard).**

```
primary_valid    := sum_n w_p[n] > 1e-12  AND  count_n (w_p[n] > 1e-12) >= K
secondary_valid  := primary_valid
                    AND sum_n w_s[n] > 1e-12
                    AND count_n (w_s[n] > 1e-12) >= K
v_fused          := primary_valid as uint8
```

If `primary_valid == False`, write `theta_primary = NaN`,
`M_primary = 0`, `theta_sec = NaN`, `M_sec = 0`, `v_fused = 0`,
all diagnostic arrays = NaN.

**Initialization (deterministic K-init on the circle).**

```
mu[0] = phi[argmax_n w[n]]
mu[k] = phi[argmax_n  w[n] * min_{j<k} circ_dist(phi[n], mu[j])]   for k = 1..K-1
pi[k] = 1 / K   for all k
kappa[k] = init_kappa
```

`circ_dist(a, b) = abs( ((a - b + pi) mod 2*pi) - pi )`, returns
values in `[0, pi]`.

**EM, fixed n_iters iterations, hard E-step.**

```
for iter in 0..n_iters:
    # E-step (hard)
    d[k, n]      = circ_dist(phi[n], mu[k])
    gamma[k, n]  = 1 if k == argmin_k d[k, n] else 0

    # M-step
    W[k]   = sum_n w[n] * gamma[k, n]
    C[k]   = sum_n w[n] * gamma[k, n] * cos(phi[n])
    S[k]   = sum_n w[n] * gamma[k, n] * sin(phi[n])
    mu[k]  = atan2(S[k], C[k]) mod 2*pi
    pi[k]  = W[k] / max(sum_k W[k], eps)
    R[k]   = sqrt(C[k]^2 + S[k]^2) / max(W[k], eps)
    kappa[k] = clip( inv_A1_banerjee(R[k]), 0, 700 )

eps = 1e-12

inv_A1_banerjee(R):
    R_clipped = clip(R, 0, 1 - 1e-6)
    return R_clipped * (2 - R_clipped^2) / (1 - R_clipped^2)
```

**Primary slot.**
```
k_p = argmax_k pi_p[k]
theta_primary = (mu_p[k_p] mod 2*pi) / 2
M_primary     = W_p[k_p]
```

**Secondary slot (only computed where secondary_valid).**
```
k_s = argmax_k pi_s[k]
theta_sec_candidate = (mu_s[k_s] mod 2*pi) / 2
M_sec_candidate     = W_s[k_s]

mass_ok = M_sec_candidate / max(M_primary, 1e-30) > tau_M_rel
sep_ok  = degrees( circ_dist(mu_p[k_p], mu_s[k_s]) ) / 2 > theta_min_deg

if mass_ok AND sep_ok:
    theta_sec  = theta_sec_candidate
    M_sec      = M_sec_candidate
    keep_secondary_mask = 1
else:
    theta_sec  = NaN
    M_sec      = 0
    keep_secondary_mask = 0
```

## Source of truth

`agent_workspaces/cgmm_metal/reference_impl.py` defines
`cgmm_fuse_two_pass(phi_p, w_p, phi_s, w_s, K=3, n_iters=30,
init_kappa=4.0, hard_em=True, tau_M_rel=0.05, theta_min_deg=10.0)`
returning a dict with the twelve fields listed above. **Do not modify
reference_impl.py.** The Metal port must reproduce these outputs to
within the acceptance tolerances below.

## Required deliverables

1. **`src/edgecritic/cgmm/_metal.py`** with two top-level symbols.
   ```python
   def cgmm_fuse_two_pass_metal(
       phi_p, w_p, phi_s, w_s,
       K=3, n_iters=30,
       init_kappa=4.0, hard_em=True,
       tau_M_rel=0.05, theta_min_deg=10.0,
   ) -> dict:
       """Returns same dict as cgmm_fuse_two_pass."""
       ...

   def cgmm_backend_available() -> bool:
       ...
   ```

2. Native code under `native/edgecritic_metal/`. Extend the existing
   crate. Add a new `extern "C"` symbol such as
   `edgecritic_metal_cgmm_fuse_two_pass` and a Metal compute kernel.
   The recovery and pipeline ports already in this crate are good
   templates for the threadgroup layout.

## Constraints

- **macOS-only**. Mirror the gating in `src/edgecritic/recovery/_metal.py`.
- **Float32 inside, float64 outside**. Inputs cast to f32 before GPU
  upload. Outputs cast back to f64 (`uint8` stays as is).
- **No external Python deps** beyond `ctypes`, `numpy`, the Rust crate.
- **Per-pixel independence.** Each pixel's two EM fits are independent
  of all other pixels. The kernel maps one threadgroup per pixel.
- **K is fixed at 3 in production.** You may hard-code K=3 inside the
  kernel for register-count reasons. The Python wrapper still accepts
  the `K` argument and rejects values other than 3.
- **N is up to 16 in production** (`P x N = 16.7M x 16` at full image).
  The kernel should accept any `N <= 64` with a static inner loop.
- **No early stopping.** All 30 iterations always run. This keeps the
  kernel branch-free and makes timings deterministic.

## Acceptance gate

Run from repo root:
```bash
PYTHONPATH=src:agent_workspaces/cgmm_metal \
    python3 agent_workspaces/cgmm_metal/tests/test_cgmm_metal.py
```

Four tests must pass.

1. **`test_correctness_small`** -- on the 1M-row reference slab
   (`reference/inputs.npz`, P=1,048,576, N=4) the Metal output must
   match the frozen reference (`reference/expected.npz`) within these
   tolerances.
   - `v_fused`: <= 0.1% rows may disagree.
   - On rows where both are valid, primary `theta` within 1.0 deg per
     row and primary `M` within 5% per row, on at least 99.9% of rows.
   - Secondary kept-ness: <= 0.5% rows may disagree.
   - Where both kept secondary, same tolerances as primary on at least
     99.5% of rows.

2. **`test_correctness_full`** -- runs the fused front end on the
   noisy 4096^2 image at N=16 (4 channels x 4 lf_half_lengths), runs
   the Metal port on the full image, then compares against the Python
   reference on a 500K random subset. Same tolerances. This catches
   scaling bugs (memory layout, threadgroup tiling) that only appear
   at full image size.

3. **`test_speed_small`** -- on the 1M-row reference slab, the Metal
   port must run at least **20x faster** than the Python reference.
   The Python reference takes ~16 s at 1M, N=4 on Apple Silicon, so
   the Metal target is `<= 800 ms` at this size.

4. **`test_speed_full`** -- on the full 4096^2 image (16.7M rows,
   N=16), the Metal port must complete in **under 2 seconds** wall
   time after warmup. **Stretch target: under 500 ms.** The Python
   reference takes 17+ minutes at full scale, so the hard target is a
   ~500x speedup and the stretch is ~2000x.

## Reference data

`agent_workspaces/cgmm_metal/reference/inputs.npz`.

| key                      | shape       | dtype | meaning                                |
|--------------------------|-------------|-------|----------------------------------------|
| `phi_p`                    | `(1048576, 4)` | f32   | primary doubled-angles                |
| `w_p`                      | `(1048576, 4)` | f32   | primary weights `v * M`               |
| `phi_s`                    | `(1048576, 4)` | f32   | secondary doubled-angles              |
| `w_s`                      | `(1048576, 4)` | f32   | secondary weights                     |
| `config_*`                 | scalar      | --    | hyperparameters                       |

`agent_workspaces/cgmm_metal/reference/expected.npz`.

| key                  | shape       | dtype |
|----------------------|-------------|-------|
| `theta_primary`        | `(1048576,)`  | f32   |
| `M_primary`            | `(1048576,)`  | f32   |
| `theta_sec`            | `(1048576,)`  | f32   |
| `M_sec`                | `(1048576,)`  | f32   |
| `v_fused`              | `(1048576,)`  | u8    |
| `primary_pi`           | `(1048576, 3)`| f32   |
| `primary_mu`           | `(1048576, 3)`| f32   |
| `primary_kappa`        | `(1048576, 3)`| f32   |
| `secondary_pi`         | `(1048576, 3)`| f32   |
| `secondary_mu`         | `(1048576, 3)`| f32   |
| `secondary_kappa`      | `(1048576, 3)`| f32   |
| `keep_secondary_mask`  | `(1048576,)`  | u8    |

The reference image is `nested_star_square_oval_low_contrast_mixed_chroma_1024.png`
with AWGN sigma=13. Front end: `r=9, d=3, m in {40, 60, 80, 100},
n_orient=32`. To regenerate from scratch, run
`python3 agent_workspaces/cgmm_metal/_make_reference_data.py` from
the repo root.

## Why this matters

After the fused WVF + LF + recovery pipeline (`wvf_lf_recover_metal`)
landed, c-GMM is the only remaining CPU-bound stage in the production
fusion pipeline. At full 4096^2 with N=16 inputs the Python reference
takes about **17 minutes**. The hard target of 2 s drops the full
fusion dump from ~18 minutes to ~88 seconds. The stretch target of
500 ms drops it to ~86 s and makes the c-GMM stage smaller than the
front end. With this in place, the entire fusion pipeline runs in
about a minute and a half end to end.

## Things NOT to do

- Do not change the algorithm. Hard EM, fixed init, fixed iterations,
  identical suppression rule.
- Do not introduce new tunable parameters. The signature is fixed.
- Do not implement the soft-EM path. Production uses `hard_em=True`
  exclusively. If the caller passes `hard_em=False`, raise
  `NotImplementedError` -- soft EM is not in scope.
- Do not silently downcast `theta` or `M` to float32 in the output.
  The Python reference returns float64 and downstream stages assume
  float64.
- Do not pre-filter the input rows by validity in the wrapper. The
  kernel must handle the validity check internally so a single launch
  covers every pixel.
