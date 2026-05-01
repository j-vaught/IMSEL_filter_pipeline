# WVF Metal

This folder is a standalone WVF package with native backends. On macOS, the
spatial variants use Metal directly. On macOS and Linux, the FFT path uses the
bundled VkFFT bridge when available and falls back to the Rust CPU FFT backend
when needed. Python only loads inputs, allocates output arrays, and calls the
native library.

## Requirements

- Python 3.10 or newer.
- NumPy.
- macOS or Linux.
- Rust with Cargo.
- For macOS spatial variants:
  - Xcode command line tools.
  - A Metal-capable GPU.
- For Linux VkFFT GPU execution:
  - An NVIDIA GPU with a working CUDA driver.
  - A CUDA toolkit discoverable through `WVF_CUDA_HOME`, `CUDA_HOME`,
    `CUDA_PATH`, `/usr/local/cuda`, or a supported MATLAB CUDA bundle.

Image-file input for the CLI also needs `imageio`. Array input through `.npy`
or `.npz` does not.

## Install

### Quick Start

```bash
cd wvf_metal
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install .
```

If you want image-file CLI input, install the `image` extra:

```bash
python -m pip install .[image]
```

The Rust dynamic library builds automatically on first use and is cached under
`wvf_metal/build/target`.

### Platform Notes

- macOS:
  - Install Xcode command line tools first.
  - `variant="split"`, `variant="antipodal"`, `variant="direct"`, and `variant="fft"` are available.
- Linux:
  - `variant="fft"` is available.
  - If CUDA and a compatible host compiler are present, `fft_backend="vkfft"` uses the GPU.
  - Otherwise the package still installs and `fft_backend="cpu"` remains available.

### Installation Check

Run this after install:

```bash
wvf-metal-doctor
```

Then run a minimal smoke test:

```bash
python - <<'PY'
import numpy as np
from wvf_metal import components

image = np.random.default_rng(0).random((128, 128), dtype=np.float32)
result = components(image, radius=9, degree=3, variant="fft", fft_backend="cpu")
print(result.gx.shape, result.magnitude.shape)
PY
```

## Python API

```python
import numpy as np
from wvf_metal import (
    gradients,
    magnitude,
    magnitude_orientation,
    components,
)

image = np.random.default_rng(0).random((1024, 1024), dtype=np.float32)
gx, gy = gradients(image, radius=9, degree=3)
mag = magnitude(image, radius=9, degree=3)
mag, angle = magnitude_orientation(image, radius=9, degree=3)
gx, gy, mag, angle = components(image, radius=9, degree=3)
gx, gy, mag, angle = components(
    image,
    radius=44,
    degree=3,
    variant="fft",
    fft_backend="cpu",
)
```

The default variant is `split` on macOS. For comparisons, pass
`variant="direct"`, `variant="antipodal"`, or `variant="fft"`.

On Linux, only `variant="fft"` is available. The spatial variants require
macOS Metal. `fft_backend="cpu"` selects the restored Rust `rustfft` backend
inside the native extension instead of the VkFFT GPU path.

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

`fft` is backend `3`. By default it uses the bundled VkFFT bridge or the
restored Rust CPU FFT backend, depending on which one benchmarks faster for the
current workload on the current device.

For `variant="fft"`, pass `fft_backend="auto"`, `fft_backend="cpu"`, or
`fft_backend="vkfft"`. `auto` is the default.
The first `auto` call for a given image shape, radius, degree, GPU device, and
requested output shape warms both FFT backends once, times the next call, and
caches the faster choice under the user cache directory. Later calls reuse that
choice until the native build fingerprint or workload key changes.

For Metal or VkFFT GPU execution, you can choose the GPU with `device_index=`
in Python or `WVF_GPU_DEVICE_INDEX` in the environment. Index `0` is the first
GPU returned by the native backend enumeration.

Linux notes:
- The Python wrapper auto-discovers common CUDA runtime library locations and
  preloads `cudart`, `nvrtc`, and `nvrtc-builtins` when needed.
- If no compatible CUDA host C++ compiler is found, the native extension builds
  in CPU-only mode and `fft_backend="auto"` chooses the Rust CPU FFT backend.
- If `nvcc` needs an older host compiler than the system default, set
  `WVF_CUDA_HOST_CXX` or `CUDAHOSTCXX` to a compatible `g++` wrapper or binary
  before first use.
- `WVF_CUDA_HOST_IO_MODE` can be set to `pageable`, `register`, or `pinned`
  to experiment with Linux CUDA host-buffer transfer modes. The default `auto`
  keeps the pageable path because the alternatives are device-dependent and are
  not yet a universal win.

## CLI

```bash
wvf-metal input.npy output.npz --radius 9 --degree 3
wvf-metal input.npy output.npz --radius 9 --degree 3 --mode gradients
wvf-metal input.npy output.npz --radius 9 --degree 3 --mode magnitude
wvf-metal input.npy output.npz --radius 9 --degree 3 --mode magnitude-angle
wvf-metal input.npz output.npz --key image --radius 15 --degree 3 --variant direct
wvf-metal input.npy output.npz --radius 44 --degree 3 --variant fft
wvf-metal input.npy output.npz --radius 44 --degree 3 --variant fft --fft-backend cpu
wvf-metal-doctor
wvf-metal-regression
```

The output archive always contains `radius`, `degree`, `variant`, `mode`, and
`input_path`, plus whichever arrays match the requested `--mode`.

## File Map

- `api.py` contains the short public API.
- `metal.py` contains the native binding, backend selection, and install logic.
- `cli.py` provides the command line wrapper.
- `fft_regression.py` compares `split` against the VkFFT backend for
  correctness and warm performance.
- `rust/src/lib.rs` builds WVF weights, owns the Metal dispatch code, and
  exposes the C ABI.
- `rust/src/fft_backend/` contains the restored Rust CPU FFT backend.
- `rust/src/wvf.metal` contains the Metal compute kernels.
- `rust/src/platform_metal.rs` contains the macOS Metal spatial path.
- `rust/src/vkfft_bridge.cpp` contains the macOS VkFFT Metal bridge.
- `rust/src/vkfft_cuda_bridge.cu` contains the Linux VkFFT CUDA bridge.
- `rust/third_party/` contains the vendored headers needed by that bridge.
