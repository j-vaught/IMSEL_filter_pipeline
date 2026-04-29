"""A4 reviewer-pass-5 ablation: two-pass c-GMM vs joint K=4 vMM.

The script reuses the current three-regime c-GMM sample dump, builds a
deterministic noisy-junction copy from the corner regime, and compares
the production two-pass K=3 design against a joint K=4 fit over pooled
primary and secondary measurements.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))

from cgmm_vmm import theta_M_to_phi_w, vmm_em, vmm_fuse_two_pass


GT_THETA_DEG = {
    "edge": [138.2],
    "corner": [66.4, 149.5],
    "noisy_junction": [66.4, 149.5],
}


def _theta_distance_deg(a: float, b: float) -> float:
    return abs(((a - b + 90.0) % 180.0) - 90.0)


def _nearest_error(theta: float, gt: list[float]) -> float:
    return min(_theta_distance_deg(theta, g) for g in gt)


def _recalls_both(theta_primary: float, theta_secondary: float, gt: list[float]) -> bool:
    if not math.isfinite(theta_secondary):
        return False
    recovered = [theta_primary, theta_secondary]
    return all(min(_theta_distance_deg(r, g) for r in recovered) <= 5.0 for g in gt)


def _load_base_cases(paper_root: Path) -> dict[str, dict]:
    path = paper_root / "cetz_figures" / "data" / "cgmm_three_regimes.json"
    data = json.loads(path.read_text())
    return {case["label"]: case for case in data["cases"]}


def _extract_streams(case: dict) -> dict[str, np.ndarray]:
    theta_p = np.asarray([s["theta"] for s in case["primary_samples"]], dtype=np.float64)
    mag_p = np.asarray([s["M"] for s in case["primary_samples"]], dtype=np.float64)
    theta_s = np.asarray([s["theta"] for s in case["secondary_samples"]], dtype=np.float64)
    mag_s = np.asarray([s["M"] for s in case["secondary_samples"]], dtype=np.float64)
    return {
        "theta_p": theta_p,
        "mag_p": mag_p,
        "theta_s": theta_s,
        "mag_s": mag_s,
    }


def _make_noisy_junction(corner_streams: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    out = {}
    for name, values in corner_streams.items():
        arr = values.copy()
        if name.startswith("theta"):
            arr = (arr + rng.normal(0.0, 2.4, size=arr.shape)) % 180.0
        else:
            arr = arr * rng.lognormal(mean=-0.08, sigma=0.25, size=arr.shape)
        out[name] = arr
    return out


def _fit_two_pass(streams: dict[str, np.ndarray]) -> tuple[dict, dict]:
    phi_p, w_p, _ = theta_M_to_phi_w(streams["theta_p"][None, :], streams["mag_p"][None, :])
    phi_s, w_s, _ = theta_M_to_phi_w(streams["theta_s"][None, :], streams["mag_s"][None, :])
    out = vmm_fuse_two_pass(
        phi_p,
        w_p,
        phi_s,
        w_s,
        K=3,
        n_iters=30,
        hard_em=True,
        tau_M_rel=0.05,
        theta_min_deg=10.0,
    )
    theta_primary = float(np.degrees(out["theta_primary"][0]))
    theta_secondary = (
        float(np.degrees(out["theta_sec"][0]))
        if np.isfinite(out["theta_sec"][0])
        else float("nan")
    )
    result = {
        "variant": "two_pass",
        "label": "two-pass K=3",
        "theta_primary": theta_primary,
        "theta_secondary": theta_secondary,
        "M_primary": float(out["M_primary"][0]),
        "M_secondary": float(out["M_sec"][0]),
        "keep_secondary": bool(out["keep_secondary_mask"][0]),
    }
    diagnostics = {
        "primary_components": _components_from_arrays(
            out["primary_mu"][0], out["primary_pi"][0], out["primary_kappa"][0]
        ),
        "secondary_components": _components_from_arrays(
            out["secondary_mu"][0], out["secondary_pi"][0], out["secondary_kappa"][0]
        ),
    }
    return result, diagnostics


def _components_from_arrays(mu: np.ndarray, pi: np.ndarray, kappa: np.ndarray) -> list[dict]:
    comps = []
    for m, p, k in zip(mu, pi, kappa, strict=True):
        if np.isfinite(m) and np.isfinite(p):
            comps.append(
                {
                    "theta": float(np.degrees((m % (2.0 * np.pi)) / 2.0)),
                    "weight": float(p),
                    "kappa": float(k),
                }
            )
    comps.sort(key=lambda c: -c["weight"])
    return comps


def _fit_joint(streams: dict[str, np.ndarray]) -> tuple[dict, dict]:
    theta = np.concatenate([streams["theta_p"], streams["theta_s"]])[None, :]
    mag = np.concatenate([streams["mag_p"], streams["mag_s"]])[None, :]
    phi, w, _ = theta_M_to_phi_w(theta, mag)
    out = vmm_em(phi, w, K=4, n_iters=30, hard_em=True)
    weights = out.W[0]
    order = np.argsort(-weights)
    selected: list[tuple[int, float, float]] = []
    for idx in order:
        theta_k = float(np.degrees((out.mu[0, idx] % (2.0 * np.pi)) / 2.0))
        if all(_theta_distance_deg(theta_k, item[1]) > 10.0 for item in selected):
            selected.append((int(idx), theta_k, float(weights[idx])))
        if len(selected) == 2:
            break
    theta_primary = selected[0][1] if selected else float("nan")
    M_primary = selected[0][2] if selected else 0.0
    theta_secondary = selected[1][1] if len(selected) > 1 else float("nan")
    M_secondary = selected[1][2] if len(selected) > 1 else 0.0
    keep_secondary = (
        len(selected) > 1
        and M_secondary > 0.05 * max(M_primary, 1.0e-30)
        and _theta_distance_deg(theta_primary, theta_secondary) > 10.0
    )
    if not keep_secondary:
        theta_secondary = float("nan")
        M_secondary = 0.0
    result = {
        "variant": "joint_k4",
        "label": "joint K=4",
        "theta_primary": theta_primary,
        "theta_secondary": theta_secondary,
        "M_primary": M_primary,
        "M_secondary": M_secondary,
        "keep_secondary": keep_secondary,
    }
    diagnostics = {
        "components": _components_from_arrays(out.mu[0], out.pi[0], out.kappa[0]),
    }
    return result, diagnostics


def _benchmark(streams: dict[str, np.ndarray], repeats: int = 512) -> dict[str, float]:
    tiled = {
        key: np.tile(value[None, :], (repeats, 1))
        for key, value in streams.items()
    }
    phi_p, w_p, _ = theta_M_to_phi_w(tiled["theta_p"], tiled["mag_p"])
    phi_s, w_s, _ = theta_M_to_phi_w(tiled["theta_s"], tiled["mag_s"])
    t0 = time.perf_counter()
    _ = vmm_fuse_two_pass(phi_p, w_p, phi_s, w_s, K=3, n_iters=30, hard_em=True)
    t_two = time.perf_counter() - t0

    theta = np.concatenate([tiled["theta_p"], tiled["theta_s"]], axis=1)
    mag = np.concatenate([tiled["mag_p"], tiled["mag_s"]], axis=1)
    phi, w, _ = theta_M_to_phi_w(theta, mag)
    t0 = time.perf_counter()
    _ = vmm_em(phi, w, K=4, n_iters=30, hard_em=True)
    t_joint = time.perf_counter() - t0
    return {
        "two_pass_us_per_pixel": float(t_two / repeats * 1.0e6),
        "joint_k4_us_per_pixel": float(t_joint / repeats * 1.0e6),
    }


def run_ablation(output_path: Path, paper_root: Path) -> dict:
    base = _load_base_cases(paper_root)
    streams_by_case = {
        "edge": _extract_streams(base["edge"]),
        "corner": _extract_streams(base["corner"]),
    }
    streams_by_case["noisy_junction"] = _make_noisy_junction(streams_by_case["corner"])

    cases = []
    variant_rows = {"two_pass": [], "joint_k4": []}
    for label, streams in streams_by_case.items():
        two, two_diag = _fit_two_pass(streams)
        joint, joint_diag = _fit_joint(streams)
        for result in (two, joint):
            gt = GT_THETA_DEG[label]
            result["primary_error_deg"] = _nearest_error(result["theta_primary"], gt)
            result["secondary_recall"] = (
                _recalls_both(result["theta_primary"], result["theta_secondary"], gt)
                if len(gt) == 2
                else False
            )
            variant_rows[result["variant"]].append(result)
        cases.append(
            {
                "label": label,
                "title": {
                    "edge": "regular edge",
                    "corner": "clean L-junction",
                    "noisy_junction": "noisy junction",
                }[label],
                "gt_theta": GT_THETA_DEG[label],
                "samples": [
                    {
                        "theta": float(t),
                        "M": float(m),
                        "stream": "primary",
                    }
                    for t, m in zip(streams["theta_p"], streams["mag_p"], strict=True)
                ]
                + [
                    {
                        "theta": float(t),
                        "M": float(m),
                        "stream": "secondary",
                    }
                    for t, m in zip(streams["theta_s"], streams["mag_s"], strict=True)
                ],
                "two_pass": {**two, **two_diag},
                "joint_k4": {**joint, **joint_diag},
            }
        )

    bench = _benchmark(streams_by_case["corner"])
    summary_rows = []
    for variant, label in (("two_pass", "two-pass K=3"), ("joint_k4", "joint K=4")):
        rows = variant_rows[variant]
        junction_rows = [row for row in rows if row["secondary_recall"] is not False]
        summary_rows.append(
            {
                "variant": variant,
                "label": label,
                "mean_primary_error_deg": float(np.mean([row["primary_error_deg"] for row in rows])),
                "worst_primary_error_deg": float(np.max([row["primary_error_deg"] for row in rows])),
                "junction_secondary_recall": float(
                    np.mean([row["secondary_recall"] for row in junction_rows])
                ),
                "cost_us_per_pixel": bench[f"{variant}_us_per_pixel"],
            }
        )

    two_summary = next(row for row in summary_rows if row["variant"] == "two_pass")
    joint_summary = next(row for row in summary_rows if row["variant"] == "joint_k4")
    recall_gain = (
        joint_summary["junction_secondary_recall"]
        - two_summary["junction_secondary_recall"]
    )
    primary_loss = (
        joint_summary["mean_primary_error_deg"]
        - two_summary["mean_primary_error_deg"]
    )
    if recall_gain > 0.05 and primary_loss <= 0.2:
        decision = "switch_to_joint_k4"
        decision_text = (
            "Switch to joint K=4; secondary recall improves by more than 5 "
            "percentage points without a primary-error loss above 0.2 deg."
        )
    else:
        decision = "keep_two_pass"
        decision_text = (
            "Keep the two-pass design; joint K=4 does not clear the recall "
            "gain rule without primary-error risk."
        )

    output = {
        "ablation": "A4",
        "cases": cases,
        "summary_rows": summary_rows,
        "summary": {
            "decision": decision,
            "decision_text": decision_text,
            "recall_gain": float(recall_gain),
            "primary_loss_deg": float(primary_loss),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-root",
        type=Path,
        default=ROOT.parent / "New project",
        help="paper repository root",
    )
    args = parser.parse_args()
    out = (
        args.paper_root
        / "cetz_figures"
        / "data"
        / "ablation_a4"
        / "results.json"
    )
    result = run_ablation(out, args.paper_root)
    print(f"wrote {out}")
    print(
        "A4 decision: "
        f"{result['summary']['decision']} "
        f"(recall gain {result['summary']['recall_gain']:.3f}, "
        f"primary loss {result['summary']['primary_loss_deg']:.3f} deg)"
    )
    for row in result["summary_rows"]:
        print(
            f"  {row['variant']:>8}: mean primary error "
            f"{row['mean_primary_error_deg']:.3f} deg, secondary recall "
            f"{row['junction_secondary_recall']:.3f}, cost "
            f"{row['cost_us_per_pixel']:.2f} us/pixel"
        )


if __name__ == "__main__":
    main()
