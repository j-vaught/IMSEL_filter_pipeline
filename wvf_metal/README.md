# Standalone WVF Metal

This folder is a copyable WVF package with a macOS Metal/VkFFT path and a
portable CPU FFT fallback for backend `3`. WVF weights are built and cached in
Rust for the native path, convolution runs in Metal, and magnitude/angle
recovery stays on the GPU. The spatial variants use the `metal` crate
directly. Backend `3`, exposed as `fft` and the compatibility alias `vkfft`,
can run through the bundled VkFFT bridge or through the CPU fallback. Python
only loads inputs, allocates output arrays, and calls the Rust dynamic library
when the native backend is selected.

## Requirements

- Python 3.10 or newer.
- NumPy.
- For Metal or VkFFT GPU execution:
  - macOS with a Metal-capable GPU.
  - Xcode command line tools.
  - Rust with Cargo.

Image-file input for the CLI also needs `imageio`. Array input through `.npy`
or `.npz` does not.

## Install From This Folder

```bash
cd wvf_metal
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The Rust/Metal dynamic library builds automatically on first use of a native
Metal or VkFFT backend and is cached under `wvf_metal/build/target`.

## Python API

```python
import numpy as np
from wvf_metal import wvf_gradients_metal, wvf_magnitude_angle_metal

image = np.random.default_rng(0).random((1024, 1024), dtype=np.float32)
gx, gy = wvf_gradients_metal(image, radius=9, degree=3)
gx, gy, magnitude, angle = wvf_magnitude_angle_metal(image, radius=9, degree=3)
gx, gy, magnitude, angle = wvf_magnitude_angle_metal(
    image,
    radius=44,
    degree=3,
    variant="fft",
    fft_backend="cpu",
)
```

The default Metal variant is `split`. For comparisons, pass `variant="direct"`,
`variant="antipodal"`, or `variant="fft"`. `variant="vkfft"` remains as a
compatibility alias for backend `3`.

The package can be installed and used on non-macOS systems through
`variant="fft"` with `fft_backend="cpu"`. The spatial Metal variants and the
VkFFT GPU path still require macOS with Metal support.

## Kernel Variants

`direct` uses the full WVF support. Each Metal thread computes one output
pixel and loops over every disk offset, sampling with reflected boundaries and
accumulating both `Gx` and `Gy`.

`antipodal` uses the odd symmetry of derivative weights. It stores each
`(+dx,+dy)` and `(-dx,-dy)` pair once, samples both pixels, forms their
difference, and applies one paired weight. This cuts the loop count roughly in
half while producing the same derivative up to float32 roundoff.

`split` uses the same antipodal pairs but dispatches two kernels. Interior
pixels use a fast path with direct indexing and no reflected-boundary checks.
Only the border band uses reflected indexing. This is the default because most
pixels in normal images are interior pixels.

`fft` is the preferred public name for backend `3`. By default it uses the
bundled VkFFT bridge on macOS when the native backend is available, and falls
back to the portable CPU FFT path elsewhere. `vkfft` is kept as a compatibility
alias for the same backend id and behavior.

For `variant="fft"` or `variant="vkfft"`, pass `fft_backend="auto"`,
`fft_backend="cpu"`, or `fft_backend="vkfft"`. `auto` is the default.

For Metal or VkFFT GPU execution on macOS, you can choose the GPU with
`device_index=` in Python or `WVF_METAL_DEVICE_INDEX` in the environment.
Index `0` is the first device returned by Metal's device enumeration.

## CLI

```bash
wvf-metal input.npy output.npz --radius 9 --degree 3
wvf-metal input.npz output.npz --key image --radius 15 --degree 3 --variant direct
wvf-metal input.npy output.npz --radius 44 --degree 3 --variant fft
wvf-metal input.npy output.npz --radius 44 --degree 3 --variant fft --fft-backend cpu
wvf-metal-regression
```

The output archive contains `gx`, `gy`, `magnitude`, `angle`, `radius`,
`degree`, `variant`, and `input_path`.

## File Map

- `metal.py` selects the CPU FFT fallback or the Rust/Metal backend and exposes
  the Python API.
- `cli.py` provides the command line wrapper.
- `fft_regression.py` compares `split` against the VkFFT backend for
  correctness and warm performance.
- `rust/src/lib.rs` builds WVF weights, owns the Metal dispatch code, and
  exposes the C ABI.
- `rust/src/wvf.metal` contains the Metal compute kernels.
- `rust/src/vkfft_bridge.cpp` contains the active VkFFT bridge.
- `rust/third_party/` contains the vendored headers needed by that bridge.
