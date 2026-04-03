#!/usr/bin/env python3

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np


TARGET_NP = 500
M_LINE = 13
SIGMA_L = M_LINE / 2.0
THETA_DEG = 0.0
DEGREES = (2, 4, 6)
GAUSS_ORDERS = (1, 2, 3, 4)


def fact(n: int) -> int:
    return math.factorial(n)


def neighbor_data(target_np: int) -> Tuple[List[Tuple[int, int]], float]:
    max_r = math.ceil(math.sqrt(target_np / math.pi)) + 3
    cands = []
    for dx in range(-max_r, max_r + 1):
        for dy in range(-max_r, max_r + 1):
            d2 = dx * dx + dy * dy
            cands.append((dx, dy, d2))
    cands.sort(key=lambda t: t[2])
    pts = [(dx, dy) for dx, dy, _ in cands[:target_np]]
    radius = math.sqrt(cands[target_np - 1][2])
    return pts, radius


NEIGHBORS, NBR_RADIUS = neighbor_data(TARGET_NP)


def monomial_basis(d: int, x: float, y: float) -> List[float]:
    out = []
    for deg in range(d + 1):
        for p in range(deg + 1):
            q = deg - p
            out.append((x ** p) * (y ** q) / (fact(p) * fact(q)))
    return out


def line_weight(j: int) -> float:
    return math.exp(-(j * j) / (2.0 * SIGMA_L * SIGMA_L))


def gradient_row(d: int, theta_deg: float) -> np.ndarray:
    theta = math.radians(theta_deg)
    ct = math.cos(theta)
    st = math.sin(theta)
    A = []
    for dx, dy in NEIGHBORS:
        x = dx * ct + dy * st
        y = -dx * st + dy * ct
        A.append(monomial_basis(d, x, y))
    A = np.asarray(A, dtype=float)
    P = np.linalg.inv(A.T @ A) @ A.T
    return P[2].copy()


def fused_stencil(d: int, theta_deg: float) -> Dict[Tuple[int, int], float]:
    theta = math.radians(theta_deg)
    p = gradient_row(d, theta_deg)
    tx = -math.sin(theta)
    ty = math.cos(theta)
    stencil: Dict[Tuple[int, int], float] = {}
    for j in range(-M_LINE, M_LINE + 1):
        wj = line_weight(j)
        lx = j * tx
        ly = j * ty
        for coeff, (dx, dy) in zip(p, NEIGHBORS):
            ox = round(lx + dx)
            oy = round(ly + dy)
            stencil[(ox, oy)] = stencil.get((ox, oy), 0.0) + wj * coeff
    return stencil


def hermite_prob(n: int, t: float) -> float:
    if n == 0:
        return 1.0
    if n == 1:
        return t
    h_nm2 = 1.0
    h_nm1 = t
    for k in range(2, n + 1):
        h_n = t * h_nm1 - (k - 1) * h_nm2
        h_nm2, h_nm1 = h_nm1, h_n
    return h_nm1


def gaussian_derivative_stencil(order: int) -> Dict[Tuple[int, int], float]:
    sigma_x = NBR_RADIUS / 3.0
    sigma_y = (M_LINE + NBR_RADIUS) / 3.0
    limit_x = math.ceil(3.0 * sigma_x)
    limit_y = math.ceil(3.0 * sigma_y)
    stencil: Dict[Tuple[int, int], float] = {}
    for x in range(-limit_x, limit_x + 1):
        for y in range(-limit_y, limit_y + 1):
            tx = x / sigma_x
            ty = y / sigma_y
            g = math.exp(-0.5 * (tx * tx + ty * ty))
            w = ((-1) ** order) * hermite_prob(order, tx) * g / (sigma_x ** order)
            stencil[(x, y)] = w

    # Normalize to reproduce the derivative of x^order / order! at the origin.
    denom = moment_response(stencil, order, 0)
    return {k: v / denom for k, v in stencil.items()}


def moment_response(stencil: Dict[Tuple[int, int], float], p: int, q: int) -> float:
    total = 0.0
    for (x, y), w in stencil.items():
        total += w * (x ** p) * (y ** q) / (fact(p) * fact(q))
    return total


def print_table(name: str, stencil: Dict[Tuple[int, int], float], terms: Iterable[Tuple[int, int]]) -> None:
    print(name)
    for p, q in terms:
      print(f"  x^{p} y^{q} / ({p}! {q}!): {moment_response(stencil, p, q): .6e}")
    print()


def main() -> None:
    print(f"theta = {THETA_DEG:.0f} deg, N_p = {TARGET_NP}, m = {M_LINE}")
    print(f"matched sigma_x = {NBR_RADIUS / 3.0:.4f}, sigma_y = {(M_LINE + NBR_RADIUS) / 3.0:.4f}")
    print()

    terms = [
        (0, 0),
        (1, 0),
        (0, 1),
        (2, 0),
        (1, 1),
        (0, 2),
        (3, 0),
        (2, 1),
        (1, 2),
        (0, 3),
        (4, 0),
    ]

    for d in DEGREES:
        print_table(f"LF d={d}", fused_stencil(d, THETA_DEG), terms)

    for order in GAUSS_ORDERS:
        print_table(f"Gaussian derivative order={order}", gaussian_derivative_stencil(order), terms)


if __name__ == "__main__":
    main()
