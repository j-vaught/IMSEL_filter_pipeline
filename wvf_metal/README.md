# Standalone WVF Metal

This folder is a copyable Apple Silicon implementation of radius-defined Wide
View Filter gradients. It contains only the WVF kernel builder, the Metal
compute kernels, Python bindings, and a small command line interface.

## Requirements

- macOS on an Apple Silicon machine.
- Xcode command line tools.
- Rust with Cargo.
- Python 3.10 or newer.
- NumPy.

Image-file input for the CLI also needs `imageio`. Array input through `.npy`
or `.npz` does not.

## Install From This Folder

```bash
cd wvf_metal
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The Rust/Metal dynamic library builds automatically on first use and is cached
under `wvf_metal/build/target`.

## Python API

```python
import numpy as np
from wvf_metal import wvf_gradients_metal, wvf_magnitude_angle_metal

image = np.random.default_rng(0).random((1024, 1024), dtype=np.float32)
gx, gy = wvf_gradients_metal(image, radius=9, degree=3)
gx, gy, magnitude, angle = wvf_magnitude_angle_metal(image, radius=9, degree=3)
```

The default Metal variant is `split`, which uses antipodal WVF pairs and a
separate interior kernel that avoids reflected-boundary math away from the
image border. For comparisons, pass `variant="direct"` or
`variant="antipodal"`.

## CLI

```bash
wvf-metal input.npy output.npz --radius 9 --degree 3
wvf-metal input.npz output.npz --key image --radius 15 --degree 3 --variant direct
```

The output archive contains `gx`, `gy`, `magnitude`, `angle`, `radius`,
`degree`, `variant`, and `input_path`.

## File Map

- `radius.py` builds the Taylor design matrix, dense reference kernels, and
  antipodal Metal weights.
- `metal.py` builds and loads the Rust/Metal backend and exposes the Python API.
- `cli.py` provides the command line wrapper.
- `rust/src/lib.rs` is the minimal Rust host layer and C ABI.
- `rust/src/wvf.metal` contains the Metal compute kernels.
