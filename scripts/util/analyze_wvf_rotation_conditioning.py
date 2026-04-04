#!/usr/bin/env python3
"""Analyze whether WVF rotation changes conditioning or derivative stability.

This script isolates one question. For a fixed support, if we rotate only the
local coordinate system, how much do the following quantities change with
orientation?

1. cond(A_theta^T A_theta)
2. ||p_(f_x')(theta)||_2
3. ||p_(f_x')(theta) - (cos(theta) p_(f_x) + sin(theta) p_(f_y))||_2
4. Var[p_(f_x')(theta)^T epsilon] for iid Gaussian noise epsilon

The first quantity measures numerical conditioning of the normal equations.
The second and fourth measure derivative-noise amplification. The third checks
whether the rotated derivative row is just the expected directional derivative
of the unrotated fitted gradient.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from edgecritic.core.taylor import build_taylor_matrix, get_circular_neighbors, rotate_coordinates


def exact_disk(radius: int, include_center: bool = True) -> np.ndarray:
    """Return the exact integer lattice disk of radius ``radius``."""
    coords = []
    for y in range(-radius, radius + 1):
        for x in range(-radius, radius + 1):
            if x * x + y * y > radius * radius:
                continue
            if not include_center and x == 0 and y == 0:
                continue
            coords.append((x, y))
    return np.array(coords, dtype=np.float64)


def basis_term_count(order: int) -> int:
    """Number of total-degree monomials up to ``order`` in two variables."""
    return (order + 1) * (order + 2) // 2


def support_library(radius: int) -> dict[str, np.ndarray]:
    """Build the fixed supports used in the experiment."""
    exact_inc = exact_disk(radius, include_center=True)
    exact_exc = exact_disk(radius, include_center=False)
    return {
        f"exact_r{radius}_inc": exact_inc,
        f"exact_r{radius}_exc": exact_exc,
        f"nearest_np{exact_inc.shape[0]}": get_circular_neighbors(exact_inc.shape[0]),
        f"nearest_np{exact_exc.shape[0]}": get_circular_neighbors(exact_exc.shape[0]),
    }


def analyze_support(
    coords: np.ndarray,
    order: int,
    angles: np.ndarray,
    sigma: float,
    noise_trials: int,
    seed: int,
) -> dict[str, float]:
    """Compute conditioning and derivative-row statistics over rotation."""
    if coords.shape[0] < basis_term_count(order):
        raise ValueError(
            f"support has only {coords.shape[0]} samples, but order {order} "
            f"needs at least {basis_term_count(order)} basis terms"
        )

    A0 = build_taylor_matrix(coords, order=order)
    P0 = np.linalg.pinv(A0)
    p_fx = P0[1]
    p_fy = P0[2]

    if noise_trials > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, sigma, size=(noise_trials, coords.shape[0]))
    else:
        noise = None

    conds = []
    row_norms = []
    eq_errors = []
    theo_vars = []
    emp_vars = []

    for theta in angles:
        rotated = rotate_coordinates(coords, theta)
        A = build_taylor_matrix(rotated, order=order)
        P = np.linalg.pinv(A)
        p_theta = P[1]
        p_expected = math.cos(theta) * p_fx + math.sin(theta) * p_fy

        conds.append(np.linalg.cond(A.T @ A))
        row_norms.append(np.linalg.norm(p_theta))
        eq_errors.append(
            np.linalg.norm(p_theta - p_expected) / max(np.linalg.norm(p_theta), 1e-15)
        )

        theo_var = sigma * sigma * float(np.dot(p_theta, p_theta))
        theo_vars.append(theo_var)

        if noise is not None:
            emp_vars.append(float(np.var(noise @ p_theta, ddof=1)))

    conds = np.array(conds)
    row_norms = np.array(row_norms)
    eq_errors = np.array(eq_errors)
    theo_vars = np.array(theo_vars)
    emp_vars = np.array(emp_vars) if emp_vars else None

    max_idx = int(np.argmax(conds))
    min_idx = int(np.argmin(conds))

    result = {
        "cond_min": float(conds[min_idx]),
        "cond_max": float(conds[max_idx]),
        "cond_mean": float(conds.mean()),
        "cond_ratio": float(conds[max_idx] / conds[min_idx]),
        "cond_theta_min_deg": float(np.degrees(angles[min_idx])),
        "cond_theta_max_deg": float(np.degrees(angles[max_idx])),
        "row_norm_min": float(row_norms.min()),
        "row_norm_max": float(row_norms.max()),
        "row_norm_ratio": float(row_norms.max() / row_norms.min()),
        "theo_var_min": float(theo_vars.min()),
        "theo_var_max": float(theo_vars.max()),
        "theo_var_ratio": float(theo_vars.max() / theo_vars.min()),
        "equivariance_err_max": float(eq_errors.max()),
        "equivariance_err_mean": float(eq_errors.mean()),
    }

    if emp_vars is not None:
        result.update(
            {
                "emp_var_min": float(emp_vars.min()),
                "emp_var_max": float(emp_vars.max()),
                "emp_var_ratio": float(emp_vars.max() / emp_vars.min()),
            }
        )

    return result


def print_report(
    supports: dict[str, np.ndarray],
    orders: list[int],
    angles: np.ndarray,
    sigma: float,
    noise_trials: int,
    seed: int,
) -> None:
    """Run the experiment and print a compact table for each support."""
    print(f"Angles sampled: {angles.shape[0]} over [0, 180) degrees")
    print(f"Noise sigma: {sigma}")
    print(f"Monte Carlo trials: {noise_trials}")
    print()

    for name, coords in supports.items():
        print(f"Support: {name}  points={coords.shape[0]}")
        print(
            "order  cond_ratio   row_norm_ratio  theo_var_ratio  "
            "emp_var_ratio   max_eq_err"
        )
        for order in orders:
            result = analyze_support(coords, order, angles, sigma, noise_trials, seed)
            emp_ratio = result.get("emp_var_ratio", float("nan"))
            print(
                f"{order:>5d}  "
                f"{result['cond_ratio']:>10.6f}  "
                f"{result['row_norm_ratio']:>14.6f}  "
                f"{result['theo_var_ratio']:>14.6f}  "
                f"{emp_ratio:>13.6f}  "
                f"{result['equivariance_err_max']:.3e}"
            )
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze WVF rotation-dependent conditioning on fixed supports."
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=7,
        help="Radius for the exact lattice disk supports.",
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[1, 3, 5],
        help="Polynomial orders to evaluate.",
    )
    parser.add_argument(
        "--n-angles",
        type=int,
        default=181,
        help="Number of orientation samples over [0, pi].",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Noise standard deviation for the Monte Carlo variance check.",
    )
    parser.add_argument(
        "--noise-trials",
        type=int,
        default=10000,
        help="Number of iid Gaussian noise trials. Use 0 to disable Monte Carlo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for the Monte Carlo variance check.",
    )
    parser.add_argument(
        "--supports",
        nargs="*",
        default=None,
        help=(
            "Optional subset of support names to analyze. Available names are "
            "printed by running without this flag."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_supports = support_library(args.radius)
    if args.supports:
        supports = {name: all_supports[name] for name in args.supports}
    else:
        supports = all_supports

    angles = np.linspace(0.0, np.pi, args.n_angles, endpoint=False)
    print_report(
        supports=supports,
        orders=args.orders,
        angles=angles,
        sigma=args.sigma,
        noise_trials=args.noise_trials,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
