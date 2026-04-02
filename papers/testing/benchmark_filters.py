"""
Benchmark the three filter approaches on the same image.
Reports median wall-clock time over multiple runs.
"""

import time
import numpy as np
from PIL import Image

from aniso_edge_demo import anisotropic_edge_detect as detect_original
from elliptical_edge_demo import anisotropic_edge_detect as detect_elliptical
from rectangular_edge_demo import anisotropic_edge_detect as detect_rectangular


def bench(fn, gray, kwargs, n_runs=10, warmup=2):
    for _ in range(warmup):
        fn(gray, **kwargs)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn(gray, **kwargs)
        times.append(time.perf_counter() - t0)
    return np.median(times), np.min(times), np.max(times)


def main():
    img = Image.open("test_image.jpg").convert("L").resize((640, 427))
    gray = np.asarray(img, dtype=np.float64) / 255.0

    n_runs = 10
    print(f"Benchmarking {n_runs} runs each (+ 2 warmup), image {gray.shape}")
    print("-" * 65)

    med, lo, hi = bench(detect_original, gray,
                        dict(n_orientations=36, length=15, sigma=2.0),
                        n_runs=n_runs)
    print(f"Original (fused stencil):  {med*1000:7.1f} ms  "
          f"(min {lo*1000:.1f}, max {hi*1000:.1f})")
    t_orig = med

    med, lo, hi = bench(detect_elliptical, gray,
                        dict(n_orientations=36, sigma_along=2.0,
                             sigma_across=1.2, length=15),
                        n_runs=n_runs)
    print(f"Elliptical Gaussian:       {med*1000:7.1f} ms  "
          f"(min {lo*1000:.1f}, max {hi*1000:.1f})")
    t_ell = med

    med, lo, hi = bench(detect_rectangular, gray,
                        dict(n_orientations=36, half_along=6.0,
                             half_across=3.6, sigma_along=2.0,
                             sigma_across=1.2, length=15),
                        n_runs=n_runs)
    print(f"Rectangular Gaussian:      {med*1000:7.1f} ms  "
          f"(min {lo*1000:.1f}, max {hi*1000:.1f})")
    t_rect = med

    print("-" * 65)
    print(f"Elliptical vs Original:    {t_ell/t_orig:.3f}x  "
          f"({'faster' if t_ell < t_orig else 'slower'} by "
          f"{abs(1 - t_ell/t_orig)*100:.1f}%)")
    print(f"Rectangular vs Original:   {t_rect/t_orig:.3f}x  "
          f"({'faster' if t_rect < t_orig else 'slower'} by "
          f"{abs(1 - t_rect/t_orig)*100:.1f}%)")
    print(f"Rectangular vs Elliptical: {t_rect/t_ell:.3f}x  "
          f"({'faster' if t_rect < t_ell else 'slower'} by "
          f"{abs(1 - t_rect/t_ell)*100:.1f}%)")


if __name__ == "__main__":
    main()
