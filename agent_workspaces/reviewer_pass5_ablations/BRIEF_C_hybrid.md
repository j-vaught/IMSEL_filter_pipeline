# Brief: Hybrid orientation estimator (option C from A2)

This is a follow-up to the A2 ablation. The decision rule fired on
trig fit, but the primary-precision regression was too large to
swallow whole. The user picked the hybrid path. Use the cubic
interpolating spline for the primary peak and use an order-4 trig
fit for the secondary peak, gated by the same suppression rules.

The pass-5 fixes already in place stay. Magnitude clamp at
$\max_k y_k$, sample-knot local-max secondary candidates, the FFI
signatures, the c-GMM stage. All unchanged.

---

## What this buys

A2 numbers, copied for reference.

| SNR    | cubic primary err | trig primary err | cubic recall | trig recall |
|--------|------------------:|-----------------:|-------------:|------------:|
| clean  | 0.120°            | 0.600°           | 0.556        | 1.000       |
| 30     | 0.132°            | 0.600°           | 0.556        | 0.917       |
| 20     | 0.126°            | 0.599°           | 0.556        | 0.944       |
| 10     | 0.188°            | 0.699°           | 0.611        | 0.917       |

The hybrid keeps cubic's sub-degree primary precision and picks up
trig's 30–40 percentage point recall gain at corners. Cost roughly
doubles per-pixel recovery work, which is acceptable given the
~470 ms full-image recovery already passes the speed gate.

---

## What changes in the algorithm

The per-pixel quintuple
$(\hat{\theta}, \hat{M}, \hat{\theta}_\text{sec}, \hat{M}_\text{sec}, v)$
keeps the same shape and suppression contract. Only the source of
the secondary slot changes.

1. **Primary path. No change.** Build the periodic cubic interpolating
   spline through the $N_s$ samples, find the largest local maximum
   via the existing per-segment closed-form root search plus
   sample-knot evaluations from pass 5. Apply the magnitude clamp.

2. **Secondary path. Replace the spline-derived secondary with a
   trig-derived secondary.** Build an order-4 doubled-angle trig fit,
   evaluate on the same dense grid the cubic primary already uses,
   take the largest non-adjacent dense-grid local maximum.

3. **Suppression rules. No change.** The $\tau_\text{sec} = 0.40$
   relative-magnitude floor, the angular-separation floor, and the
   non-adjacency rule all still fire on the trig-derived secondary.
   Same dense-grid distance check against the cubic primary.

4. **Magnitude clamp. No change.** Both $\hat{M}$ and
   $\hat{M}_\text{sec}$ still clamp at $\max_k y_k$.

---

## Reference implementation pointer

The order-4 trig estimator already exists in
`scripts/eval/ablation_a2_orientation.py` at
`_trig_estimator(response, angles, order=4)`. Match that estimator
exactly. Order is 4 doubled-angle harmonics, design matrix has
$2 \cdot 4 + 1 = 9$ columns, dense grid is 500 evenly-spaced points
on $[0, \pi)$.

For uniform sample spacing $\theta_k = (k - 1)\pi / N_s$, the
least-squares fit reduces to a DFT-like closed-form rather than a
true `pinv`. For the Metal port, precompute the
`(9, N_s)` pseudo-inverse on the host once, upload as a constant
buffer (mirror the way the existing cubic-spline `solver_inv` is
handled), and per-pixel work is one $9 \times N_s$ matrix-vector
multiply.

---

## Required code changes

### A. Python reference (`agent_workspaces/orientation_recovery_metal/reference_impl.py`)

The reference is the source of truth. Update the algorithm comment
block and the `find_two_peaks` body in this order.

1. Add an order-4 trig fit helper alongside the cubic spline path.
2. In `find_two_peaks`, compute the primary $(\hat{\theta}, \hat{M})$
   from the cubic spline as today.
3. Compute the secondary candidate as the largest non-adjacent
   trig-fit dense-grid local maximum, where non-adjacent uses the
   same `min_sep_frac * dense_n` index-distance rule against the
   cubic primary.
4. Apply the existing $\tau_\text{sec}$ floor and the magnitude
   clamp on the trig-derived $\hat{M}_\text{sec}$.
5. Update the docstring and the algorithm-comment block. The
   "dense grid is a numerical stand-in for closed-form" note no
   longer applies to secondary.
6. Regenerate `reference/expected.npz`.
   ```bash
   PYTHONPATH=src:agent_workspaces/orientation_recovery_metal \
       python3 agent_workspaces/orientation_recovery_metal/_make_reference_data.py
   ```

### B. Eval-side reference (`scripts/eval/cgmm_orientation_recovery.py`)

Same algorithmic change. Add the order-4 trig fit and replace the
secondary path. Update the docstring.

### C. Metal kernels (`native/edgecritic_metal/src/lib.rs`)

Three kernel variants need the trig path added.

| variant | function | line | layout |
|---|---|---|---|
| 1 | `recovery_peaks` | 983 | threadgroup-resident `y`, `m` |
| 2 | `recovery_peaks_private` | 1288 | response in device memory |
| 3 | `recovery_peaks_private_stack` | 1530 | `m` kept in thread memory |

Per kernel, three blocks change.

1. **Inputs and constants.** Add a constant buffer for the trig
   pseudo-inverse `trig_solver` shaped `(9, N_s)` plus an
   integer for the harmonic order. Mirror the existing
   `solver_inv` plumbing for the cubic spline.
2. **Per-lane trig setup.** Accumulate the 9 trig coefficients
   from the row's $y$ samples via the constant `trig_solver`.
   Threadgroup-broadcast or stack-store the coefficients.
3. **Per-lane secondary search.** Replace the existing
   spline-derivative root search for secondary candidates with a
   dense-grid evaluation of the trig fit on the same dense grid
   already used by the cubic primary, plus the existing local-max
   test, non-adjacency rule, and $\tau_\text{sec}$ floor.

The lane-zero finalize block is unchanged. Same writes to
`theta_p`, `m_p`, `theta_s`, `m_s` with the same clamp at
$\max_k y_k$ from pass 5.

### D. FFI signatures

No FFI signature change. Both
`edgecritic_metal_recover_two_peaks` and
`edgecritic_metal_wvf_lf_recover` keep the same input and output
buffers. The trig path runs inside the kernel.

---

## Cost estimate

The trig fit doubles the spline-construction work and adds a
dense-grid evaluation pass. Approximate per-pixel work goes from
$O(N_s)$ for the cubic path to $O(N_s + 9 \cdot N_s)$ for cubic
plus trig coefficients, plus an extra dense-grid sweep. Net effect
on the full 4096² image is a roughly 2× recovery time, from ~470 ms
to ~900 ms. The speed gate is 500 ms warmup but the existing test
already runs over that, so the gate may need to be raised. Flag if
the new time exceeds 1.2 s on the full image.

---

## Acceptance gate

Run the existing acceptance test from the repo root.

```bash
PYTHONPATH=src:agent_workspaces/orientation_recovery_metal \
    python3 agent_workspaces/orientation_recovery_metal/tests/test_recovery_metal.py
```

Tolerances stay the same as pass 5.

- Primary `theta`: <= 0.5 deg per row, all rows.
- Primary `M`: |dM|/M <= 5e-3 per row, all rows.
- Secondary kept-ness: <= 0.1% of rows may disagree.
- Secondary `theta` and `M` on agreed-kept rows: same tolerances
  as primary.
- Validity flag: <= 0.1% of rows may disagree.

Two important notes.

1. **The frozen reference must be regenerated first.** The
   `expected.npz` from pass 5 was cubic-only secondary. The
   regenerated one will have trig-derived secondary in different
   places. Confirm the diff is concentrated in
   `theta_secondary` and `M_secondary`, not in the primaries or
   the validity flag.

2. **The speed test may need its threshold relaxed.** If the
   500 ms warmup gate fails because of the trig pass, raise it to
   1.2 s and note in the commit message. Do not weaken the
   correctness tolerances.

---

## Spot check

The pass-5 K=8 hand-crafted row was
`[1.0, 0.2, 0.1, 0.05, 0.6, 0.05, 0.1, 0.2]`. The cubic primary
should still be at sample index 0, theta = 0, magnitude 1.0.

The trig-derived secondary should still be at sample index 4,
theta = π/2. The magnitude may now differ slightly from 0.6
because the order-4 trig fit smooths the discrete sequence
before the dense-grid local-max test, so the recovered value can
be a hair below 0.6. A clamp at $\max_k y_k = 1.0$ does not
trigger here. Confirm the secondary is retained, not suppressed.

---

## Out of scope

- The c-GMM fusion stage. Two-pass with $K = 3$ stays per A4.
- The validity flag default. The user is updating the paper text
  for A5 separately.
- The §8 figures. The user will regenerate after this lands.
- The §6 paper text. The user will rewrite the orientation-recovery
  prose to describe the hybrid algorithm.

---

## Status when done

Single commit titled along the lines of "Hybrid orientation
estimator: cubic primary, trig secondary." Push to `main`. Reply
with the commit hash, the acceptance-test output, and the new
full-image recovery time so the user can update the paper text.
