# Brief: Reviewer pass 5 fixes for orientation recovery

This is a follow-up to the original `BRIEF.md`. The original Metal port
matches `reference_impl.find_two_peaks` and ships fine. The paper's
peer reviewer flagged two algorithmic gaps in §6 that need to land in
both the Python reference and all three Metal kernel variants.

The two changes are small and self-contained. The acceptance test
gate stays the same (`test_recovery_metal.py`), but the frozen
reference data (`reference/expected.npz`) must be regenerated after
the Python reference is updated, so the Metal port has a target to
match.

---

## What changes and why

### Change 1 — Sample-knot local maxima are valid secondary candidates

Reviewer comment 9. The current Metal kernel finds the secondary peak
by enumerating interior critical points of the spline derivative
$s'(\theta)$ on each cubic segment, then taking the largest
non-adjacent one that passes the local-max test
$s''(\theta) < 0$. A true secondary peak that sits at a sample knot
$\theta_k$ satisfies neither condition, because $s'$ does not need to
vanish at a knot, and the segment-interior test will not fire there.

The dense-grid Python reference catches sample-knot peaks implicitly,
because a sample-knot local max in the discrete sample sequence shows
up as a dense-grid local max under the `dy >= left & dy >= right`
test. The closed-form Metal kernel does not, so it can miss real
junction directions that happen to land on a knot.

The fix is to add an explicit sample-knot local-max candidate at the
start of each lane's secondary search, gated by the same non-adjacency
rule that already filters interior critical points.

### Change 2 — Magnitude clamp at the largest sample value at the pixel

Reviewer comment 10. The recovered magnitudes are currently
$\hat{M} = s(\hat{\theta})$ and $\hat{M}_\text{sec} = s(\hat{\theta}_\text{sec})$,
i.e. the spline values at the recovered orientations. The cubic
interpolant can overshoot above the largest sample value at the
pixel, which would feed an inflated weight $w_n = v_n \hat{M}_n$ into
the downstream c-GMM fusion stage and bias the consensus toward
configurations whose splines happen to overshoot more aggressively.

The fix is to clamp at the largest sample value at the pixel,

$$
\hat{M} = \min(s(\hat{\theta}), \max_k y_k), \quad
\hat{M}_\text{sec} = \min(s(\hat{\theta}_\text{sec}), \max_k y_k).
$$

The clamp is applied **after** the suppression test
$\hat{M}_\text{sec} / \hat{M} < \tau_\text{sec}$, so the suppression
ratio still uses the unclamped spline values.

---

## Required code changes

### A. Python reference (`agent_workspaces/orientation_recovery_metal/reference_impl.py`)

The reference is the source of truth. Update the docstring and
`find_two_peaks` body in this order.

1. **Update the algorithm comment block.** Replace the existing
   numbered list with one that calls out sample-knot local maxima as
   secondary candidates and adds the clamp step.

2. **Apply the magnitude clamp** at the end of the function, after
   the existing suppression test sets `M_sec` to zero on suppressed
   rows.
   ```python
   y_max = response_2d.max(axis=1)
   M_hat = np.minimum(M_hat, y_max)
   M_sec = np.minimum(M_sec, y_max)
   ```
   The dense-grid local-max already covers the sample-knot case for
   #9, so no algorithm change is needed there. Add a comment noting
   this. The closed-form Metal kernel does not get that for free, see
   below.

3. **Regenerate the frozen reference** so the test harness has the
   updated targets.
   ```bash
   PYTHONPATH=src:agent_workspaces/orientation_recovery_metal \
       python3 agent_workspaces/orientation_recovery_metal/_make_reference_data.py
   ```
   Verify the diff in `reference/expected.npz` is small and limited to
   `M_primary` and `M_secondary` clamping (sample-knot fix should be a
   no-op on the dense-grid reference).

### B. Eval-side reference (`scripts/eval/cgmm_orientation_recovery.py`)

Same algorithmic change as the Python reference. Apply the clamp at
the end. Update the docstring to match.

### C. Metal kernels in `native/edgecritic_metal/src/lib.rs`

Three kernel variants need the same surgery, all in
`native/edgecritic_metal/src/lib.rs`.

| variant | function | approx line | layout |
|---|---|---|---|
| 1 | `recovery_peaks` | 983 | threadgroup-resident `y`, `m` |
| 2 | `recovery_peaks_private` | 1288 | response in device memory |
| 3 | `recovery_peaks_private_stack` | 1530 | `m` kept in thread memory |

Each variant has the same three-block structure.

- **Per-lane primary search.** Already includes sample-point evaluations
  via `recovery_dense_floor_idx + offset`, so the primary slot is
  unaffected by change 1. No edit here.
- **Per-lane secondary search.** Currently iterates over interior
  critical points only. Add change 1 here.
- **Lane-zero finalize block.** Writes `theta_p`, `m_p`, `theta_s`,
  `m_s`. Add change 2 here.

#### Change 1 in the secondary search

Each lane $i$ owns segment $i$, which starts at sample $\theta_i$. At
the top of the lane's secondary block (right after `best_value` and
`best_idx` are initialised to `-INFINITY` and `0`), check whether
sample $i$ is a non-adjacent local maximum in the discrete sequence
and, if so, register it as a candidate.

```c
const uint prev_sample = (i == 0) ? k - 1 : i - 1;
const uint next_sample = (i + 1 == k) ? 0 : i + 1;
if (y[i] > y[prev_sample] && y[i] > y[next_sample]) {
    const uint sample_dense = (i * params.dense_n) / k;
    uint dist = sample_dense > primary_idx
        ? sample_dense - primary_idx
        : primary_idx - sample_dense;
    dist = min(dist, params.dense_n - dist);
    if (dist > params.sep && y[i] > best_value) {
        best_value = y[i];
        best_idx = sample_dense;
    }
}
```

In variants 2 and 3 the array `y[]` is read from a different memory
space (device pointer or thread-local stack). Match the existing
indexing pattern of that variant. The structural logic is identical.

#### Change 2 in the lane-zero finalize block

In each finalize block, compute `row_max` from the row's samples (the
finalize block already has access to `y[]`), then clamp before
writing.

```c
float row_max = y[0];
for (uint j = 1; j < k; ++j) {
    row_max = max(row_max, y[j]);
}

theta_p[row] = float(primary_idx) * params.pi_over_dense;
m_p[row] = min(primary_value, row_max);              // clamp

const float ratio_den = max(primary_value, 1.0e-30f);
const bool suppress = !has_secondary
                   || (secondary_value / ratio_den) < params.tau_sec_floor;
if (suppress) {
    theta_s[row] = as_type<float>(0x7fc00000u);      // NaN
    m_s[row] = 0.0f;
} else {
    theta_s[row] = float(secondary_idx) * params.pi_over_dense;
    m_s[row] = min(secondary_value, row_max);        // clamp
}
```

The suppression test still uses the unclamped `secondary_value /
ratio_den`. Only the value written to `m_p[row]` and `m_s[row]` is
clamped.

In variants 2 and 3 the row's `y[]` may live in a different memory
space. Use whichever array the existing finalize block already
reads.

### D. (No FFI signature change)

The output buffers and FFI symbols stay the same. Both
`edgecritic_metal_recover_two_peaks` and
`edgecritic_metal_wvf_lf_recover` already return the per-pixel
quintuple. The clamp is computed inside the kernel.

---

## Acceptance gate

Run the existing acceptance test from the repo root.
```bash
PYTHONPATH=src:agent_workspaces/orientation_recovery_metal \
    python3 agent_workspaces/orientation_recovery_metal/tests/test_recovery_metal.py
```

The four tests must all pass against the **regenerated**
`reference/expected.npz`.

Tolerances stay where they are.
- Primary `theta`: `<= 0.5 deg` per row, all rows.
- Primary `M`: `|dM|/M <= 5e-3` per row, all rows.
- Secondary kept-ness: `<= 0.1%` of rows may disagree.
- Secondary on agreed-kept rows: same `theta` and `M` tolerances as
  primary.
- Validity flag: `<= 0.1%` of rows may disagree.

If the regenerated reference and the Metal output disagree above
threshold on `M` only (primary or secondary), the most likely cause
is missing the clamp in one of the three kernel variants. If they
disagree on secondary kept-ness, the most likely cause is missing the
sample-knot candidate in one of the three.

---

## How to verify the change is doing what we think

Two quick spot checks before running the full acceptance test.

1. **Clamp.** On the regenerated reference, every row should satisfy
   `M_primary <= max(response_2d[row, :])` and
   `M_secondary <= max(response_2d[row, :])`. A one-liner check:
   ```python
   y_max = response.max(axis=1)
   assert (M_primary <= y_max + 1e-6).all()
   assert (M_secondary <= y_max + 1e-6).all()
   ```

2. **Sample-knot recovery.** Construct a synthetic LF row with a clear
   sample-knot secondary peak, e.g. K = 8 samples shaped like
   `[1.0, 0.2, 0.1, 0.05, 0.6, 0.05, 0.1, 0.2]`. The sample at index
   4 is a local max with value 0.6 and is far from the primary at
   index 0. Both implementations should retain index 4 as the
   secondary slot.

---

## Out of scope

Nothing else changes. The c-GMM fusion stage, the NMS stage, the
validity rule, and the WVF/LF front end stay as they are. The paper's
§6 text has already been updated to match the algorithm above, so no
further documentation work is needed on this side.

---

## Status when done

Ship a single commit (or one per repo if the Python reference and the
Metal crate are in different commits) titled along the lines of
"Reviewer pass 5, recovery: sample-knot secondary candidates and
magnitude clamp." Push to `main`. Reply with the commit hash and the
acceptance-test output.
