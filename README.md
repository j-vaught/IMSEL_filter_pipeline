# IMSEL Filter Pipeline

Research code for a multi-stage edge and orientation detector. The pipeline
runs two custom local filters (WVF and LF) over color channels at multiple
scales, recovers up to two dominant edge orientations per pixel, and fuses
those measurements with a circular Gaussian mixture model (c-GMM). It also
ships a paper-style enhanced non-maximum suppression (NMS) stage with
hysteresis thresholding so the gradient stack can be turned into a binary
edge map.

If you only want the headline filter (WVF) for your own project, skip
straight to [`fast_wvf/`](fast_wvf/README.md).

## What WVF, LF, and c-GMM Are

### WVF — Wide View Filter

WVF is a local Taylor-expansion gradient estimator. At each pixel and a
chosen orientation `theta`, it samples a small disk of `np_count` neighbor
pixels, rotates that neighborhood into a frame aligned with `theta`, and
fits a 2-D Taylor polynomial of `order` to the intensities. The first-order
coefficients of that fit give the normal derivative `f_x` (across the edge)
and the tangential derivative `f_y` (along the edge). Sweeping `theta`
across `n_orientations` and keeping the angle that maximizes `|f_x|`
recovers a per-pixel gradient magnitude and angle.

Compared to a plain Sobel or Gaussian-derivative filter, WVF is wider,
orientation-aware, and uses the full disk's worth of evidence at once.
The trade-off is cost — the per-pixel pseudoinverse of the Taylor matrix
has to be computed for every orientation. The Metal and CUDA/VkFFT backends
in this repo exist mostly to make that cost tractable on real images.

Reference implementation: `src/wvf/`. Standalone production package:
`fast_wvf/`.

### LF — Line Filter

LF chains WVF along a line. For a chosen orientation `theta` and a
half-width `m`, LF evaluates WVF at `2m + 1` pixels stepping along the line
through the target pixel, then averages the per-pixel WVF gradients with
optional Gaussian distance weighting. The output is again a normal and
tangential derivative, but now integrated over a line segment rather than a
single disk.

The point of LF is robustness on long, low-contrast edges. A single-pixel
WVF response can be dominated by texture or noise, while a line-integrated
response only stays large when the underlying edge is locally straight and
consistent over `2m + 1` pixels. Different `m` values trade locality for
stability, which is why the pipeline runs a small bank of LF half-lengths
in parallel.

Reference implementation: `src/lf/`.

### c-GMM — Circular Gaussian Mixture Fusion

A single pixel can have more than one real edge through it (a corner, a
T-junction, or two textures meeting). The pipeline produces many
`(theta, magnitude, validity)` measurements per pixel — one for each
combination of channel (`L, R, G, B`), WVF radius, Taylor order, and LF
half-length — and c-GMM fuses them.

Each measurement is mapped onto the unit circle using the standard
"double the angle" trick (`phi = 2 * theta`), which turns the
pi-periodic orientation into a 2pi-periodic angle. A weighted circular
EM with `K` mixture components is then fit per pixel. Component means
(halved back to `[0, pi)`) become candidate edge orientations, and the
mixture weights become their relative strengths. The pipeline keeps the
strongest component as the primary orientation and, when separation and
mass thresholds are satisfied, a secondary orientation as well.

Reference implementation: `src/cgmm/reference.py` (treated as the source
of truth — the Metal port has to reproduce its outputs bit-for-bit on
the relevant fields).

### Putting It Together

```
   image
     │
     ▼
  per channel (L, R, G, B)  ──►  WVF at radii × orders          ──┐
                                                                  │
                                 LF over half-lengths over WVF  ──┤── stack of (theta, M, v)
                                                                  │       per-pixel measurements
                                 orientation recovery             ──┘
                                 (top-2 peaks per response)
                                                                  │
                                                                  ▼
                                                           c-GMM K=3 fusion
                                                                  │
                                                                  ▼
                                              primary + secondary edge orientations
                                                                  │
                                                                  ▼
                                              enhanced NMS + hysteresis (`src/nms/`)
                                                                  │
                                                                  ▼
                                                          binary edge map
```

## Repository Layout

```text
fast_wvf/             standalone WVF package; CPU FFT and GPU FFT (CUDA/VkFFT) backends
metal_full_pipeline/  Rust + Metal implementation of the fused front end (macOS)
src/
  core/               Taylor-matrix construction, neighborhood sampling, shared types
  wvf/                Wide View Filter — Python reference + Metal helpers
  lf/                 Line Filter — Python reference + Metal helpers
  orientation/        top-2 orientation peak recovery from response stacks
  cgmm/               circular Gaussian mixture fusion (K=3 hard-EM)
  nms/                paper pipeline: spline orientation, GMM fusion, enhanced NMS, hysteresis
  pipeline/           full-image dumps, fused front end, synthetic eval, verification
```

## Backends

| Backend | Where | Use for |
| --- | --- | --- |
| Pure Python (NumPy/SciPy) | `src/wvf/reference.py`, `src/lf/reference.py`, `src/cgmm/reference.py` | correctness reference, easy to read |
| Metal | `src/{wvf,lf,orientation,cgmm,pipeline}/metal.py`, `metal_full_pipeline/` | macOS Apple Silicon |
| CPU FFT and CUDA/VkFFT | `fast_wvf/` | Linux fast paths and CUDA GPUs |

The Python reference is intentionally not the fast path — it exists so the
faster Metal and CUDA implementations have a fixed target to match.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the standalone WVF package only, install `fast_wvf/` directly. See
[`fast_wvf/README.md`](fast_wvf/README.md) for platform notes and the
GPU FFT setup.

## Installed Commands

```text
fast-wvf                              fast WVF CLI (CPU FFT or CUDA/VkFFT)
fast-wvf-doctor                       environment diagnosis for fast_wvf
fast-wvf-regression                   regression check against the reference
wvf                                   pure Python WVF CLI
edgecritic-nms                        run the NMS pipeline on an image
edgecritic-pipeline-full-dump         full-image c-GMM K=3 fusion dump
edgecritic-pipeline-fusion-dump       fused front-end dump (WVF + LF + recovery)
edgecritic-pipeline-synthetic-eval    synthetic-image evaluation harness
edgecritic-pipeline-verify            cross-backend verification
```

## License

See [LICENSE](LICENSE).
