"""Regression and warm-performance checks for the standalone WVF Metal FFT path."""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import numpy as np


BASELINE_REVISION = "f388742d6219e38295e217bf9260928d7574a596"
CORRECTNESS_SIZES = ((256, 256), (2048, 2048))
CORRECTNESS_RADII = (9, 20, 44, 50)
PERFORMANCE_RADII = (20, 44, 50)
PERFORMANCE_SIZE = (2048, 2048)
DEGREE = 3
MAGNITUDE_MASK = 1e-5
MAX_ABS_TOL = 1e-6
MAX_ANGLE_TOL = 1e-3


@dataclass(frozen=True)
class CorrectnessCase:
    size: tuple[int, int]
    radius: int
    max_abs_gx: float
    max_abs_gy: float
    max_abs_magnitude: float
    max_angle_error: float


@dataclass(frozen=True)
class PerformanceCase:
    radius: int
    baseline_median: float
    current_median: float
    split_median: float | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone WVF Metal FFT correctness and warm-performance checks."
    )
    parser.add_argument(
        "--baseline-rev",
        default=BASELINE_REVISION,
        help="Git revision used for the warm-performance baseline.",
    )
    parser.add_argument(
        "--baseline-worktree",
        type=Path,
        default=None,
        help="Existing worktree or target path for the baseline checkout.",
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=10,
        help="Warm benchmark iterations per radius.",
    )
    parser.add_argument(
        "--skip-performance",
        action="store_true",
        help="Run correctness checks only.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the raw regression results as JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    package_root = Path(__file__).resolve().parent
    repo_root = package_root.parent

    current = load_package(package_root, "wvf_metal_current")
    correctness = run_correctness(current)
    ok = check_correctness(correctness)

    performance_results: list[PerformanceCase] = []
    cleanup_path: Path | None = None
    if not args.skip_performance:
        baseline_root, cleanup_path = ensure_baseline_package_root(
            repo_root,
            args.baseline_rev,
            args.baseline_worktree,
        )
        try:
            baseline = load_package(baseline_root, "wvf_metal_baseline")
            performance_results = run_performance(current, baseline, args.warm_runs)
            ok = check_performance(performance_results) and ok
        finally:
            if cleanup_path is not None:
                subprocess.run(
                    ["git", "-C", str(repo_root), "worktree", "remove", "--force", str(cleanup_path)],
                    check=True,
                )

    payload = {
        "correctness": [case.__dict__ for case in correctness],
        "performance": [case.__dict__ for case in performance_results],
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))

    return 0 if ok else 1


def load_package(package_root: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_root / "__init__.py",
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load package from {package_root}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def ensure_baseline_package_root(
    repo_root: Path,
    revision: str,
    worktree_path: Path | None,
) -> tuple[Path, Path | None]:
    created_path: Path | None = None
    if worktree_path is None:
        worktree_path = Path(tempfile.mkdtemp(prefix="wvf_metal_baseline."))
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "add", str(worktree_path), revision],
            check=True,
        )
        created_path = worktree_path
    elif not (worktree_path / "wvf_metal" / "__init__.py").exists():
        subprocess.run(
            ["git", "-C", str(repo_root), "worktree", "add", str(worktree_path), revision],
            check=True,
        )
        created_path = worktree_path
    return worktree_path / "wvf_metal", created_path


def run_correctness(module: ModuleType) -> list[CorrectnessCase]:
    results: list[CorrectnessCase] = []
    for size_index, (height, width) in enumerate(CORRECTNESS_SIZES):
        rng = np.random.default_rng(1000 + size_index)
        image = rng.random((height, width), dtype=np.float32)
        for radius in CORRECTNESS_RADII:
            gx_ref, gy_ref, mag_ref, angle_ref = module.wvf_magnitude_angle_metal(
                image,
                radius=radius,
                degree=DEGREE,
                variant="split",
            )
            gx_fft, gy_fft, mag_fft, angle_fft = module.wvf_magnitude_angle_metal(
                image,
                radius=radius,
                degree=DEGREE,
                variant="fft",
            )
            mask = mag_ref > MAGNITUDE_MASK
            angle_error = wrapped_angle_error(angle_ref, angle_fft)
            masked_error = angle_error[mask]
            case = CorrectnessCase(
                size=(height, width),
                radius=radius,
                max_abs_gx=float(np.max(np.abs(gx_fft - gx_ref))),
                max_abs_gy=float(np.max(np.abs(gy_fft - gy_ref))),
                max_abs_magnitude=float(np.max(np.abs(mag_fft - mag_ref))),
                max_angle_error=float(masked_error.max()) if masked_error.size else 0.0,
            )
            results.append(case)
    return results


def check_correctness(results: list[CorrectnessCase]) -> bool:
    print("Correctness")
    ok = True
    for case in results:
        passed = (
            case.max_abs_gx <= MAX_ABS_TOL
            and case.max_abs_gy <= MAX_ABS_TOL
            and case.max_abs_magnitude <= MAX_ABS_TOL
            and case.max_angle_error <= MAX_ANGLE_TOL
        )
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"{status} size={case.size[0]}x{case.size[1]} radius={case.radius} "
            f"gx={case.max_abs_gx:.3e} gy={case.max_abs_gy:.3e} "
            f"mag={case.max_abs_magnitude:.3e} angle={case.max_angle_error:.3e}"
        )
    return ok


def run_performance(
    current: ModuleType,
    baseline: ModuleType,
    warm_runs: int,
) -> list[PerformanceCase]:
    height, width = PERFORMANCE_SIZE
    rng = np.random.default_rng(2024)
    image = rng.random((height, width), dtype=np.float32)

    results: list[PerformanceCase] = []
    split_times = benchmark_variant(current, image, radius=9, variant="split", warm_runs=warm_runs)
    baseline_small = benchmark_variant(
        baseline,
        image,
        radius=9,
        variant="vkfft",
        warm_runs=warm_runs,
    )
    current_small = benchmark_variant(
        current,
        image,
        radius=9,
        variant="fft",
        warm_runs=warm_runs,
    )
    results.append(
        PerformanceCase(
            radius=9,
            baseline_median=statistics.median(baseline_small),
            current_median=statistics.median(current_small),
            split_median=statistics.median(split_times),
        )
    )

    for radius in PERFORMANCE_RADII:
        baseline_times = benchmark_variant(
            baseline,
            image,
            radius=radius,
            variant="vkfft",
            warm_runs=warm_runs,
        )
        current_times = benchmark_variant(
            current,
            image,
            radius=radius,
            variant="fft",
            warm_runs=warm_runs,
        )
        results.append(
            PerformanceCase(
                radius=radius,
                baseline_median=statistics.median(baseline_times),
                current_median=statistics.median(current_times),
            )
        )
    return results


def benchmark_variant(
    module: ModuleType,
    image: np.ndarray,
    radius: int,
    variant: str,
    warm_runs: int,
) -> list[float]:
    module.wvf_magnitude_angle_metal(image, radius=radius, degree=DEGREE, variant=variant)
    times: list[float] = []
    for _ in range(warm_runs):
        start = time.perf_counter()
        module.wvf_magnitude_angle_metal(image, radius=radius, degree=DEGREE, variant=variant)
        times.append(time.perf_counter() - start)
    return times


def check_performance(results: list[PerformanceCase]) -> bool:
    print("Performance")
    ok = True
    for case in results:
        if case.radius == 9:
            assert case.split_median is not None
            passed = (
                case.current_median <= case.baseline_median
                and case.current_median <= case.split_median
            )
            threshold = f"baseline={case.baseline_median:.6f}s split={case.split_median:.6f}s"
        else:
            passed = case.current_median <= 1.05 * case.baseline_median
            threshold = f"1.05x baseline={1.05 * case.baseline_median:.6f}s"
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        extra = (
            f" split={case.split_median:.6f}s"
            if case.split_median is not None
            else ""
        )
        print(
            f"{status} radius={case.radius} baseline={case.baseline_median:.6f}s "
            f"current={case.current_median:.6f}s{extra} gate={threshold}"
        )
    return ok


def wrapped_angle_error(reference: np.ndarray, test: np.ndarray) -> np.ndarray:
    difference = np.abs(reference - test)
    return np.minimum(difference, np.pi - difference)


if __name__ == "__main__":
    raise SystemExit(main())
