# Brief: Metal-accelerated LF response operator

You are implementing a Metal-GPU port of the Line Filter (LF) response
operator. The goal is a drop-in replacement for the numpy reference
that runs at least 5× faster per-call (target: 20× batched).

## What the operator does

For a single edge orientation $\theta$ and half-width $m$, given two
WVF gradient images $g_x$ and $g_y$ of shape $(H, W)$, the LF response
at a target pixel $(p_x, p_y)$ is:

$$
\mathrm{LF}(p, \theta, m) \;=\;
\left|\,\frac{\sum_{j=-m}^{m} w_j \cdot g_\perp\!\bigl(p + j\cdot s\cdot \mathbf{d}\bigr)}
              {\sum_{j=-m}^{m} w_j}\,\right|
$$

with

- $g_\perp(q) = -\sin\theta\,g_x(q) + \cos\theta\,g_y(q)$ — the
  perpendicular-gradient component at a non-integer pixel $q$ (use
  **nearest-neighbour rounding**, NOT bilinear; see "exact rounding"
  below);
- $\mathbf{d} = (\cos\theta, \sin\theta)$ — the line direction;
- $s = 1 / \max(|\cos\theta|, |\sin\theta|)$ — step length so the line
  advances by exactly 1 pixel along its dominant axis at each $j$;
- $w_j = \exp\!\bigl(-\tfrac{1}{2}(j/\sigma)^2\bigr)$ with
  $\sigma = m/2$.

Out-of-bounds samples contribute zero to **both** numerator and
denominator (they're dropped, not zero-padded). If every sample is
out of bounds, return 0 for that pixel.

For $m = 0$ the operator collapses to $|g_\perp(p)|$ at the center
pixel only.

## Exact rounding

The numpy reference does:
```python
ix = np.round(j_offsets * step * cos_t).astype(np.int32)
iy = np.round(j_offsets * step * sin_t).astype(np.int32)
```
Use the **same** rounding mode (round half to even, IEEE 754) on the
Metal side. The acceptance test compares against numpy outputs at
1e-4 absolute tolerance — sub-pixel-grid drift will fail.

## Source of truth

`agent_workspaces/lf_metal/reference_impl.py` defines
`lf_response_at_pixels(g_x, g_y, px, py, theta, m)`. This is the
operator your Metal code must match bit-for-bit (well, within 1e-4).
**Do not modify reference_impl.py.**

## Required deliverables

1. `src/edgecritic/lf/_metal.py` — Python entry point with
   ```python
   def lf_response_metal(g_x, g_y, px, py, theta, m): ...
   ```
   that takes the same arguments as the numpy reference and returns
   `(P,)` float64. Must autoselect Metal when `metal_backend_available()`,
   raise an informative error otherwise.

2. **Optional but desired**: a batched entry point
   ```python
   def lf_response_metal_batch(g_x, g_y, px, py, thetas, ms): ...
   ```
   that processes a `(T, M, P)` grid in one GPU launch, returning
   `(T, M, P)` float64. This is what the production fusion harness
   needs — looping per `(theta, m)` is what's slow.

3. Native code under `native/edgecritic_metal/src/lib.rs` (or a sibling
   crate). The existing crate already has the WVF kernel pattern — you
   can model after `edgecritic_metal_wvf_convolve_pair`.

## Constraints

- **macOS-only**. Mirror the gating in `src/edgecritic/wvf/_metal.py`:
  raise `MetalBackendError` on non-Darwin or when cargo is missing.
- **Float32 inside, float64 outside**. The numpy reference accepts
  either dtype; cast inputs to f32 on entry, cast output to f64 on
  exit. Float64 internal computation is fine if Metal supports it on
  the target hardware.
- **No external Python deps** beyond what `wvf/_metal.py` already
  imports (`ctypes`, `numpy`, the Rust crate).
- **No bilinear interpolation**. The reference uses round-then-clip;
  do exactly that.

## Acceptance gate

Run from repo root:
```bash
PYTHONPATH=src:agent_workspaces/lf_metal python3 \
    agent_workspaces/lf_metal/tests/test_lf_metal.py
```

Three tests must pass:
1. `test_correctness_per_call` — for every (theta, m) in the 16×6 grid,
   per-pixel responses match the reference within 1e-4 absolute.
2. `test_correctness_batched` — optional but must pass IF you provide
   `lf_response_metal_batch`.
3. `test_speed` — per-call ≥5× faster than the numpy reference at
   P=755 pixels (target a much stronger speedup for the batched form).

Reference data in `agent_workspaces/lf_metal/reference/`:
- `inputs.npz` — `g_x`, `g_y` (256×256 f32), `test_xs/ys` (755 int32),
  `thetas` (16 f64), `m_values` (6 int32 in {0, 5, 10, 20, 40, 80}),
  `image` (the source image used to compute the gradients, for
  debugging).
- `expected.npz` — `response` (16, 6, 755) f64.

## How to run

After implementing `src/edgecritic/lf/_metal.py`:
```bash
cd /Users/user/Documents/edge-detection-filter-critique
PYTHONPATH=src:agent_workspaces/lf_metal python3 \
    agent_workspaces/lf_metal/tests/test_lf_metal.py
```
The first invocation will trigger `cargo build --release` of the Rust
crate; subsequent invocations cache the dylib. Total first run: a few
minutes. Test execution after build: <30 s.

## Reference: how the existing WVF Metal binding is structured

- Python side: `src/edgecritic/wvf/_metal.py` — ctypes-loads the dylib,
  defines `wvf_radius_gradients_metal(image, kernels)`, gates with
  `metal_backend_available()`.
- Rust side: `native/edgecritic_metal/src/lib.rs` — `extern "C"` entry
  point, builds a Metal compute pipeline, dispatches one threadgroup
  per output tile.
- Build: `cargo build --release --manifest-path
  native/edgecritic_metal/Cargo.toml --target-dir
  build/edgecritic_metal_target`. The Python side calls cargo
  automatically when needed.

You can either extend the existing crate with a new `extern` symbol
or create a parallel crate. The existing-crate route is simpler.

## Why this matters

The current production pipeline (`scripts/eval/cgmm_image_wide_eval.py`)
runs the LF operator millions of times: 64 orientations × 10 m values
× 4 channels per fusion call. On a dilated 4096×4096 audit (~280K
pixels), the LF stage takes ~10–25 minutes on CPU. A 20× speedup gets
that under a minute and unlocks fast iteration on NMS audits and
section 7/8 figure regeneration.

## What success looks like

- All three acceptance tests pass.
- The batched entry point returns a `(T, M, P)` array in one Metal
  command-buffer submission per `(theta, m)` (or fewer).
- `cgmm_image_wide_eval.py` opt-in flag `--lf-backend metal` (you add
  this) routes through `lf_response_metal_batch` and produces
  numerically identical fusion outputs to the CPU path.

If the per-call test passes but the batched form is missing,
that's still a useful win — the harness can call the per-call
function in a Python loop with the GPU latency hidden by the inner
work, and the batched form is a follow-up optimisation.

## Things NOT to do

- Don't change the rounding mode, weight formula, or boundary
  convention. The acceptance test is strict.
- Don't pre-convolve with a Gaussian filter on the input; the LF is
  already a Gaussian-weighted directional integral.
- Don't introduce new tunable parameters. The signature is fixed.
- Don't reach into the c-GMM fusion stage. This module is a pure LF
  acceleration; the rest of the pipeline calls it through the same
  Python signature as the numpy reference.
