# WVF Metal

`wvf_metal` is the standalone Wide View Filter package from this repository.
It gives you one public API for three practical cases:

1. Fast GPU FFT execution on Linux with CUDA/VkFFT.
2. Fast CPU FFT execution on Linux or macOS when a GPU is not available.
3. A pure Python reference path from the full repository when you cannot or do
   not want to build the native backend.

If you are sending WVF to a colleague, this is the directory to send.

## What You Install

- Standalone package: `wvf_metal/`
- Pure Python reference implementation in the full repository: `src/wvf/`

`wvf_metal` is the production path. The pure Python code is slower but easier to
inspect and does not require the native extension.

## Requirements

- Python `3.10+`
- `numpy`
- `rust` and `cargo`
- Linux or macOS

For Linux GPU FFT execution you also need:

- an NVIDIA GPU
- a working CUDA driver
- a CUDA toolkit visible through one of:
  - `WVF_CUDA_HOME`
  - `CUDA_HOME`
  - `CUDA_PATH`
  - `/usr/local/cuda`

For CLI image-file input you also need `imageio`.

## Install

### Option 1. Install From A Repository Clone

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install ./wvf_metal
```

If you want CLI image-file support:

```bash
python -m pip install './wvf_metal[image]'
```

### Option 2. Install Directly From GitHub

```bash
python -m pip install "git+https://github.com/j-vaught/edge-detection-filter-critique.git#subdirectory=wvf_metal"
```

### Sanity Check

```bash
wvf-metal-doctor
```

The first real call builds the native library and caches it under
`wvf_metal/build/target`.

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
from wvf_metal import components

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
from wvf_metal import backend_info, components

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

This path lives in the full repository, not in the standalone `wvf_metal`
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
wvf-metal input.npy output.npz --radius 15 --degree 11 --normalize-coords
wvf-metal input.npy output.npz --radius 15 --degree 11 --variant fft --fft-backend cpu
wvf-metal input.npy output.npz --radius 15 --degree 11 --variant fft --fft-backend vkfft
```

The output archive contains the requested arrays plus metadata such as
`radius`, `degree`, `variant`, and `normalize_coords`.

## Notebook

A minimal notebook lives at:

- `wvf_metal/examples/wvf_quickstart.ipynb`

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

- `wvf_metal` with `fft_backend="cpu"`
- `wvf_metal` with `fft_backend="vkfft"`
- full-repository pure Python reference via `wvf_radius_gradients_cpu`
