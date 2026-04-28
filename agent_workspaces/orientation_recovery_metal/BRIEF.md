# Brief: Metal-accelerated orientation recovery

You are implementing a Metal-GPU port of the per-pixel orientation
recovery stage. The goal is a drop-in replacement for the numpy
reference in `agent_workspaces/orientation_recovery_metal/reference_impl.py`
that runs at least **20x faster** at 200K rows and ideally enables
full-image (16.7M rows) processing in under 500 ms.

## What the operator does

For each pixel, the upstream LF produces an orientation-tuned response
curve sampled at K equally spaced orientations on `[0, pi)`. The
recovery stage converts each row of `(K,) -> (theta, M, theta_sec, M_sec, v)`,
the per-pixel quintuple consumed by the downstream c-GMM fusion stage.

**Inputs.**

| name        | shape  | dtype   | meaning |
|-------------|--------|---------|---------|
| `angles_rad`  | `(K,)`   | float64 | sample orientations in `[0, pi)`, equally spaced |
| `response_2d` | `(N, K)` | float32 (cast inside) | LF response value at each pixel and angle |

**Hyperparameters** (default values after `=`).

| name              | default | role |
|-------------------|---------|------|
| `tau_sec_floor`   | 0.40    | secondary kept only if `M_sec / M_p >= tau_sec_floor` |
| `tau_validity`    | 0.10    | per-pixel range rule, `v = 1 if R(p) > tau * R_ref` |
| `dense_n`         | 500     | number of sample points for the dense grid peak search |
| `min_sep_frac`    | 0.125   | secondary candidate must be more than this fraction of the dense grid away from the primary |

**Outputs.** Five arrays, all length `N`.

| name            | shape | dtype   | semantics |
|-----------------|-------|---------|-----------|
| `theta_primary`   | `(N,)`  | float64 | primary peak location in `[0, pi)` |
| `M_primary`       | `(N,)`  | float64 | primary peak value (the spline value at `theta_primary`) |
| `theta_secondary` | `(N,)`  | float64 | secondary peak location, or `NaN` if suppressed |
| `M_secondary`     | `(N,)`  | float64 | secondary peak value, or `0.0` if suppressed |
| `v`               | `(N,)`  | uint8   | per-pixel validity flag |

## Algorithm (matches `reference_impl.find_two_peaks`)

1. Build a periodic cubic spline of each row through the K samples,
   wrap-point appended (`y[K] := y[0]`).
2. Evaluate the spline on a dense grid of `dense_n` orientations on
   `[0, pi)`. The reference uses `scipy.interpolate.CubicSpline(...,
   bc_type="periodic")` followed by densification. The Metal port may
   instead use the closed-form per-segment quadratic root search of the
   paper, **as long as the acceptance test passes**. Both methods yield
   the same answer up to ~0.36 deg at `dense_n = 500`.
3. Mark dense-grid local maxima (`>= ` both periodic neighbours).
4. `primary = argmax over local maxima`. Record `theta_primary` and
   `M_primary`.
5. Secondary candidate is the largest local maximum at periodic dense
   index distance `> min_sep_frac * dense_n` from `primary`. Suppress
   the secondary slot (set `theta_secondary = NaN`, `M_secondary = 0`)
   if no such candidate exists, OR if `M_sec_candidate / M_primary <
   tau_sec_floor`.
6. Per-pixel validity flag (range rule).
   ```
   R(p)   = max_k y_k - min_k y_k    (per-row range)
   R_ref  = max over rows of R(p)    (image-wide reference)
   v(p)   = 1 if R(p) > tau_validity * R_ref else 0
   ```
   `R_ref` is global across the input batch. The Metal kernel must do
   one reduction pass (or two-pass: range per row, then max of ranges)
   before applying the gate.

## Source of truth

`agent_workspaces/orientation_recovery_metal/reference_impl.py` defines
`find_two_peaks(angles_rad, response_2d, tau_sec_floor=0.40,
tau_validity=0.10, dense_n=500, min_sep_frac=0.125)` returning the
quintuple `(th_p, M_p, th_s, M_s, v)`. **Do not modify reference_impl.py.**
The Metal port must match this output to within the acceptance
tolerances below.

## Required deliverables

1. **`src/edgecritic/recovery/_metal.py`** with two top-level symbols.
   ```python
   def recover_two_peaks_metal(
       angles_rad,               # (K,) float64
       response_2d,              # (N, K) float32 or float64
       tau_sec_floor=0.40,
       tau_validity=0.10,
       dense_n=500,
       min_sep_frac=0.125,
   ) -> tuple[np.ndarray, ...]:  # (th_p, M_p, th_s, M_s, v)
       ...

   def recovery_backend_available() -> bool:
       ...
   ```
   Returns five numpy arrays: `(theta_primary, M_primary,
   theta_secondary, M_secondary, v)`. Same shapes and dtypes as the
   reference (float64 for the four floats, uint8 for `v`).

2. Native code under `native/edgecritic_metal/` (extend the existing
   crate, do not start a parallel one). Add a new `extern "C"` symbol
   like `edgecritic_metal_recover_two_peaks` and a Metal compute
   kernel modelled after the existing
   `edgecritic_metal_lf_orientation_stack_box` shader.

3. (Optional) Helper to operate directly on the `(n_orientations, H,
   W)` LF stack so callers do not need to reshape to `(N, K)` first.

## Constraints

- **macOS-only**. Mirror the gating in `src/edgecritic/wvf/_metal.py`.
- **Float32 inside, float64 outside**. Inputs cast to f32 before GPU
  upload; outputs cast back to f64 (uint8 stays as is).
- **No external Python deps** beyond `ctypes`, `numpy`, the Rust crate.
- **Per-row independence.** Each pixel's spline solve and peak search
  are fully independent. The only cross-row reduction is the global
  `R_ref` for the validity flag — handle that in a separate kernel
  pass or via a partial-reduction strategy.
- **No hidden assumptions about K.** Production callers pass K = 64,
  but the kernel should accept any positive K (use 32-bit indexing
  inside, dense_n indexing fits in 32 bit).

## Acceptance gate

Run from repo root:
```bash
PYTHONPATH=src:agent_workspaces/orientation_recovery_metal \
    python3 agent_workspaces/orientation_recovery_metal/tests/test_recovery_metal.py
```

Three tests must pass.

1. **`test_correctness_small`** -- on the 200K-row reference slab
   (`reference/inputs.npz`), the Metal output must match the frozen
   reference (`reference/expected.npz`) within these tolerances.
   - Primary `theta`: `<= 0.5 deg` per row, all rows.
   - Primary `M`: `|dM|/M <= 5e-3` per row, all rows.
   - Secondary kept-ness: `<= 0.1%` of rows may disagree with the
     reference on whether the secondary slot is `NaN`.
   - On rows where both agree the secondary is kept, `theta` and `M`
     must match the same tolerances as the primary.
   - Validity flag `v`: `<= 0.1%` of rows may disagree.

2. **`test_correctness_full`** -- runs the Metal port on a freshly
   generated full 4096x4096 slab (16.7M rows) and compares to scipy on
   a 500K random subset. Same tolerances. This catches scaling bugs
   that only appear at full image size (memory layout, threadgroup
   tiling, etc.).

3. **`test_speed`** -- on the 200K-row reference slab, the Metal port
   must run at least **20x faster** than `find_two_peaks` (numpy +
   scipy.CubicSpline). The numpy reference takes roughly 4 seconds at
   200K rows on Apple Silicon, so the Metal target is `<= 200 ms`. A
   stretch target of `< 500 ms` for the full 16.7M rows would mean
   the recovery stage stops being a bottleneck for the full pipeline.

## Reference data

`agent_workspaces/orientation_recovery_metal/reference/inputs.npz`.

| key                   | shape           | dtype | meaning |
|-----------------------|-----------------|-------|---------|
| `angles`                | `(64,)`           | f64   | sample orientations |
| `response`              | `(200000, 64)`    | f32   | LF response slab |
| `sample_xs`, `sample_ys`| `(200000,)`       | i32   | source pixel coords (debug) |
| `config_*`              | scalar          | --    | hyperparameters used to build expected |

`agent_workspaces/orientation_recovery_metal/reference/expected.npz`.

| key               | shape    | dtype | meaning |
|-------------------|----------|-------|---------|
| `theta_primary`     | `(200000,)`| f64   | reference primary |
| `M_primary`         | `(200000,)`| f64   | reference primary value |
| `theta_secondary`   | `(200000,)`| f64   | reference secondary or NaN |
| `M_secondary`       | `(200000,)`| f64   | reference secondary or 0 |
| `v`                 | `(200000,)`| u8    | reference validity flag |

The reference image is `nested_star_square_oval_low_contrast_mixed_chroma_4096.png`
with AWGN at sigma=13. WVF (r=9, d=3) and LF (m=60, n_orient=64) were
applied via the existing Metal front end. To regenerate from scratch,
run `python3 agent_workspaces/orientation_recovery_metal/_make_reference_data.py`
from repo root.

## Why this matters

The current production pipeline calls the orientation recovery
millions of times per fusion run. The numpy reference takes ~25
seconds per 1M rows on Apple Silicon, which scales to ~7 minutes for
a single full 4096x4096 channel. With 4 channels x 2 d x 2 r x 4 m =
**64 recovery calls** per fusion dump, the recovery stage alone takes
hours and exceeds available RAM (~17 GB peak per call, see
`scripts/eval/cgmm_image_wide_eval.evaluate`). A 20x speedup makes
each call sub-second and unlocks full-image fusion without dilation
masks.

## Things NOT to do

- Do not change the secondary-suppression rule, the validity range
  rule, or the cubic-spline boundary conditions.
- Do not introduce new tunable parameters. The signature is fixed.
- Do not require the caller to provide `R_ref` -- compute it from the
  input batch.
- Do not silently downcast `theta` to float32 in the output; numpy
  reference returns float64 and downstream stages assume float64.
- Do not pre-smooth the input `response_2d` -- LF already shapes the
  curve.
