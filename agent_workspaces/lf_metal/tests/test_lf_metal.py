"""Acceptance harness for the Metal LF implementation.

The agent's deliverable is a Python entry point with the signature
    lf_response_metal(g_x, g_y, px, py, theta, m) -> response (P,)
that matches the reference numpy implementation in
agent_workspaces/lf_metal/reference_impl.py within an absolute tolerance
of 1e-4.  An optional batched entry point
    lf_response_metal_batch(g_x, g_y, px, py, thetas, ms) -> response (T, M, P)
can be provided for higher throughput; if absent, the acceptance test
loops over (theta, m) calling the per-call function.

The intended deliverable location is
    src/edgecritic/lf/_metal.py
exposing both functions, plus any Rust/Metal kernel additions to
    native/edgecritic_metal/src/lib.rs

Run with:
    PYTHONPATH=src:agent_workspaces/lf_metal pytest -xvs \
        agent_workspaces/lf_metal/tests/test_lf_metal.py
or:
    PYTHONPATH=src:agent_workspaces/lf_metal python3 \
        agent_workspaces/lf_metal/tests/test_lf_metal.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

# The reference numpy implementation (source of truth).
from reference_impl import lf_response_at_pixels


REF_DIR = Path(__file__).resolve().parents[1] / "reference"
ABS_TOL = 1e-4


def _load_reference():
    inp = np.load(REF_DIR / "inputs.npz")
    exp = np.load(REF_DIR / "expected.npz")
    return inp, exp


def test_correctness_per_call():
    """For every (theta, m) in the reference grid, the Metal LF must
    produce per-pixel responses matching the numpy reference within
    ABS_TOL."""
    from edgecritic.lf._metal import lf_response_metal      # agent provides

    inp, exp = _load_reference()
    g_x = inp["g_x"]
    g_y = inp["g_y"]
    px = inp["test_xs"]
    py = inp["test_ys"]
    thetas = inp["thetas"]
    m_values = inp["m_values"]
    expected = exp["response"]                              # (T, M, P)

    max_abs_err = 0.0
    for i, th in enumerate(thetas):
        for j, m in enumerate(m_values):
            got = lf_response_metal(g_x, g_y, px, py, float(th), int(m))
            ref = expected[i, j]
            err = np.abs(got - ref).max()
            max_abs_err = max(max_abs_err, err)
            assert err < ABS_TOL, (
                f"mismatch at theta_idx={i}, m_idx={j} ({th=:.4f} rad, "
                f"{m=}): max |got - ref| = {err:.6f}  (tol {ABS_TOL})")
    print(f"OK: max abs err across {expected.shape} = {max_abs_err:.2e}")


def test_correctness_batched():
    """Optional batched entry point.  Skip if not provided."""
    try:
        from edgecritic.lf._metal import lf_response_metal_batch
    except ImportError:
        print("SKIP: lf_response_metal_batch not provided (optional)")
        return
    inp, exp = _load_reference()
    g_x = inp["g_x"]
    g_y = inp["g_y"]
    px = inp["test_xs"]
    py = inp["test_ys"]
    got = lf_response_metal_batch(g_x, g_y, px, py,
                                   inp["thetas"], inp["m_values"])
    expected = exp["response"]
    err = np.abs(got - expected).max()
    assert err < ABS_TOL, (
        f"batched mismatch: max |got - ref| = {err:.6f}  (tol {ABS_TOL})")
    print(f"OK: batched max abs err = {err:.2e}")


def test_speed():
    """Per-call must be at least 5x faster than the numpy reference at
    P >= 1024 pixels (loose target).  Stricter target: 20x for the
    batched form on the full grid."""
    from edgecritic.lf._metal import lf_response_metal

    inp, _ = _load_reference()
    g_x = inp["g_x"]
    g_y = inp["g_y"]
    px = inp["test_xs"]
    py = inp["test_ys"]
    theta = float(inp["thetas"][1])     # ~11 deg
    m = int(inp["m_values"][3])         # m=20

    # Warm up.
    lf_response_metal(g_x, g_y, px, py, theta, m)
    lf_response_at_pixels(g_x, g_y, px, py, theta, m)

    n_iter = 50
    t0 = time.perf_counter()
    for _ in range(n_iter):
        lf_response_metal(g_x, g_y, px, py, theta, m)
    t_metal = (time.perf_counter() - t0) / n_iter

    t0 = time.perf_counter()
    for _ in range(n_iter):
        lf_response_at_pixels(g_x, g_y, px, py, theta, m)
    t_ref = (time.perf_counter() - t0) / n_iter

    speedup = t_ref / max(t_metal, 1e-9)
    print(f"per-call: numpy {t_ref*1e6:.1f} us  metal {t_metal*1e6:.1f} us  "
          f"speedup x{speedup:.1f}")
    assert speedup >= 5.0, (
        f"target speedup x5 not met (got x{speedup:.1f})")


if __name__ == "__main__":
    test_correctness_per_call()
    test_correctness_batched()
    test_speed()
    print("\nAll tests passed.")
