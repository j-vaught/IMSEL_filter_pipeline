# Fast WVF

`fast_wvf` is the standalone Wide View Filter package from this repository.
It gives you one public API for three practical cases:

1. Fast GPU FFT execution on Linux with CUDA/VkFFT.
2. Fast CPU FFT execution on Linux or macOS when a GPU is not available.
3. A pure Python reference path from the full repository when you cannot or do
   not want to build the native backend.

If you are sending WVF to a colleague, this is the directory to send.

## Platform Support

| Platform | Status | Notes |
| --- | --- | --- |
| Linux | supported | CPU FFT and CUDA/VkFFT GPU FFT |
| macOS | supported | CPU FFT and Metal spatial variants |
| Windows | not supported natively | use WSL2 for `fast_wvf`, or use the pure Python fallback from the full repository |

## What You Install

- Standalone package: `fast_wvf/`
- Pure Python reference implementation in the full repository: `src/wvf/`

`fast_wvf` is the production path. The pure Python code is slower but easier to
inspect and does not require the native extension.

## Requirements

- Python `3.10+`
- `numpy`
- `rust` and `cargo`
- Linux or macOS

Platform-specific setup:

- macOS:
  - Xcode command line tools
- Linux GPU FFT:
  - an NVIDIA GPU
  - a working NVIDIA driver
  - a CUDA toolkit
  - a CUDA-compatible host C++ compiler if `nvcc` does not accept the system default

For Linux GPU FFT execution you also need:

- an NVIDIA GPU
- a working CUDA driver
- a CUDA toolkit visible through one of:
  - `WVF_CUDA_HOME`
  - `CUDA_HOME`
  - `CUDA_PATH`
  - `/usr/local/cuda`

For CLI image-file input you also need `imageio`.

For the pure Python reference fallback from the full repository, install the
repository root with `python -m pip install .`. That path depends on `scipy` in
addition to `numpy`.

## Install

### Option 1. Install From A Repository Clone

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ./fast_wvf
```

This is the standard install flow on Linux and macOS.

If you want CLI image-file support:

```bash
python -m pip install './fast_wvf[image]'
```

### Option 2. Install Directly From GitHub

```bash
python -m pip install "git+https://github.com/j-vaught/edge-detection-filter-critique.git#subdirectory=fast_wvf"
```

### Linux Notes

For Linux CPU-only use, the install above is enough as long as `cargo` works.

For Linux CUDA/VkFFT use, make sure these are available before first import:

```bash
python --version
cargo --version
rustc --version
nvcc --version
echo $WVF_CUDA_HOME
echo $WVF_CUDA_HOST_CXX
```

If `nvcc` does not accept the system compiler, set:

```bash
export WVF_CUDA_HOME=/usr/local/cuda
export WVF_CUDA_HOST_CXX=/path/to/g++
```

### Windows Notes

The native `fast_wvf` extension is not supported on Windows.

If someone needs Windows, the practical options are:

1. Use WSL2 with Ubuntu and follow the Linux install path.
2. Use the pure Python reference path from the full repository:

```bash
python -m pip install .
```

That fallback is slower, but it does not require the native `fast_wvf`
extension.

### Sanity Check

```bash
fast-wvf-doctor
```

The first real call builds the native library and caches it under
`fast_wvf/build/target`.

Useful preflight checks:

```bash
python --version
cargo --version
rustc --version
```

For Linux CUDA builds:

```bash
nvcc --version
echo $WVF_CUDA_HOME
echo $WVF_CUDA_HOST_CXX
```

## Backend Choice

Use `variant="fft"` on Linux. That is the only variant available there.

| Situation | Recommended call |
| --- | --- |
| Linux with CUDA GPU | `variant="fft", fft_backend="vkfft"` |
| Linux or macOS CPU-only | `variant="fft", fft_backend="cpu"` |
| macOS Metal spatial path | default `variant="split"` or explicit `variant="split"` |
| No native build support | use the pure Python reference path from `src/wvf` |

For larger radii and higher polynomial degrees, use `normalize_coords=True`.

## Python Usage

### 1. Fast CPU FFT

```python
import numpy as np
from fast_wvf import components

image = np.random.default_rng(0).random((1024, 1024), dtype=np.float32)
result = components(
    image,
    radius=15,
    degree=11,
    normalize_coords=True,
    variant="fft",
    fft_backend="cpu",
)

print(result.gx.shape, result.magnitude.dtype)
```

### 2. Fast GPU FFT On Linux

```python
import numpy as np
from fast_wvf import backend_info, components

image = np.random.default_rng(0).random((1024, 1024), dtype=np.float32)
info = backend_info()
print(info)

result = components(
    image,
    radius=15,
    degree=11,
    normalize_coords=True,
    variant="fft",
    fft_backend="vkfft",
)
```

### 3. Pure Python Reference Fallback

This path lives in the full repository, not in the standalone `fast_wvf`
package.

```bash
python -m pip install .
```

```python
import numpy as np
from wvf import wvf_radius_gradients_cpu

image = np.random.default_rng(0).random((512, 512), dtype=np.float32)
gx, gy = wvf_radius_gradients_cpu(
    image,
    radius=15,
    order=11,
    normalize_coords=True,
)
```

## CLI Usage

```bash
fast-wvf input.npy output.npz --radius 15 --degree 11 --normalize-coords
fast-wvf input.npy output.npz --radius 15 --degree 11 --variant fft --fft-backend cpu
fast-wvf input.npy output.npz --radius 15 --degree 11 --variant fft --fft-backend vkfft
```

The output archive contains the requested arrays plus metadata such as
`radius`, `degree`, `variant`, and `normalize_coords`.

## Notebook

A minimal notebook lives at:

- `fast_wvf/examples/wvf_quickstart.ipynb`

It shows:

- CPU FFT usage
- GPU FFT usage
- the pure Python reference fallback

## Notes For Linux CUDA Builds

If CUDA is installed but the VkFFT backend does not build, check these first:

```bash
echo $WVF_CUDA_HOME
echo $WVF_CUDA_HOST_CXX
which nvcc
which cargo
```

Useful overrides:

```bash
export WVF_CUDA_HOME=/usr/local/cuda
export WVF_CUDA_HOST_CXX=/path/to/g++
```

If the CUDA build still does not work, `fft_backend="cpu"` remains available
without changing your Python code beyond the backend selector.

## Public API

- `gradients(image, radius, degree=4, normalize_coords=False, variant="split", fft_backend="auto")`
- `magnitude(...)`
- `magnitude_orientation(...)`
- `components(...)`
- `backend_info()`
- `metal_backend_available()`

## Tested Handoff Paths

This package was checked in three modes:

- `fast_wvf` with `fft_backend="cpu"`
- `fast_wvf` with `fft_backend="vkfft"`
- full-repository pure Python reference via `wvf_radius_gradients_cpu`
