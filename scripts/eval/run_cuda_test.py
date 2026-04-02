"""Correctness and speed test for GPU-accelerated WVF.

Compares GPU output against CPU reference on a small crop,
then benchmarks full-resolution performance.
"""

import time

import numpy as np

from edgecritic.core.taylor import get_circular_neighbors
from edgecritic.synthetic import create_step_edge_image
from edgecritic.wvf._cpu import wvf_image as wvf_image_cpu

try:
    import torch
    from edgecritic.wvf._cuda import wvf_image_cuda
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


def main():
    print("=== Single-image correctness and speed test ===")
    print(f"CUDA available: {HAS_CUDA}")
    if HAS_CUDA:
        print(f"Device: {torch.cuda.get_device_name(0)}")

    # Create test image (481x321 to match BSDS500)
    image, _, _ = create_step_edge_image(size=321, edge_angle_deg=35.0, snr=0.0,
                                          high_val=220, low_val=40)
    image = np.pad(image, ((0, 0), (0, 160)), mode="edge")
    print(f"Test image shape: {image.shape}")

    np_count = 15
    n_orientations = 18
    order = 4

    # CPU reference (on small crop)
    crop = image[:64, :64]
    print(f"\nCPU reference on 64x64 crop...")
    t0 = time.perf_counter()
    mag_cpu, angle_cpu, _ = wvf_image_cpu(crop, np_count=np_count, order=order,
                                           n_orientations=n_orientations)
    cpu_time = time.perf_counter() - t0
    print(f"  CPU time: {cpu_time:.2f}s")

    if HAS_CUDA:
        # GPU on same crop for correctness check
        print(f"\nGPU on 64x64 crop (correctness check)...")
        mag_gpu, angle_gpu, _ = wvf_image_cuda(crop, np_count=np_count, order=order,
                                                 n_orientations=n_orientations)

        border = int(np.max(np.abs(get_circular_neighbors(np_count).astype(int)))) + 1
        interior = slice(border, -border)
        cpu_interior = mag_cpu[interior, interior]
        gpu_interior = mag_gpu[interior, interior]
        max_diff = np.max(np.abs(cpu_interior - gpu_interior))
        mean_diff = np.mean(np.abs(cpu_interior - gpu_interior))
        rel_err = mean_diff / (np.mean(np.abs(cpu_interior)) + 1e-12)
        print(f"  Max abs diff: {max_diff:.6f}")
        print(f"  Mean abs diff: {mean_diff:.6f}")
        print(f"  Relative error: {rel_err:.6e}")

        # GPU on full image for speed test
        print(f"\nGPU on full {image.shape[0]}x{image.shape[1]} image...")
        # Warmup
        _ = wvf_image_cuda(image, np_count=np_count, order=order,
                           n_orientations=n_orientations)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        mag_full, angle_full, _ = wvf_image_cuda(image, np_count=np_count, order=order,
                                                   n_orientations=n_orientations)
        torch.cuda.synchronize()
        gpu_time = time.perf_counter() - t0
        print(f"  GPU time: {gpu_time:.3f}s")

        pixel_ratio = (image.shape[0] * image.shape[1]) / (64 * 64)
        est_cpu = cpu_time * pixel_ratio
        print(f"\n  Estimated CPU time for full image: {est_cpu:.0f}s ({est_cpu/3600:.1f}h)")
        print(f"  GPU time: {gpu_time:.3f}s")
        print(f"  Speedup: {est_cpu / gpu_time:.0f}x")

        # Test larger Np
        for np_test in [50, 100, 250]:
            print(f"\nGPU on full image with Np={np_test}...")
            t0 = time.perf_counter()
            _ = wvf_image_cuda(image, np_count=np_test, order=order,
                               n_orientations=n_orientations)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            print(f"  GPU time: {t1-t0:.3f}s")
    else:
        print("\nNo CUDA device available.")


if __name__ == "__main__":
    main()
