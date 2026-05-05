# Paper Scripts

This directory contains the experiment harnesses used to generate the Section 7,
Section 8, and Section 9 tables and figures.

## Remote Execution

The heavier runs are intended to execute on the remote CUDA servers rather than
on a local workstation.

- `comech-2422`
- `comech-2080`

For long HD-scale runs, use `bash -lc` over SSH so the expected environment and
toolchain paths are loaded before Python imports the native backend.

## HRF Harness Notes

The HD retinal harness is:

- `run_sec09_real_image_hrf.py`

Operational notes from the matched-protocol HRF rerun:

- The full protocol uses `100` noisy draws and `201` ODS thresholds.
- The harness supports shard merges through `--merge-shard-jsons`.
- For lighter figure assets, use `--asset-max-width-px 800`. This writes
  downsampled preview PNGs under `assets_w800/` while leaving the metrics
  unchanged.
- If the summary JSON already exists, rerunning with `--asset-max-width-px 800`
  refreshes only the clean preview assets and rewrites the summary asset paths.
  It does not recompute the noisy metrics.

### comech-2080 CUDA Host Compiler

`comech-2080` has CUDA installed, but the default host compiler path is not the
one the WVF CUDA/VkFFT build should use.

Use:

```bash
export WVF_CUDA_HOME=/usr/local/MATLAB/R2024b/sys/cuda/glnxa64/cuda
export WVF_CUDA_HOST_CXX=/home/jvaught2/toolchains/gcc12/g++-12-wrapper
```

That wrapper is the known-good configuration for the HRF full-protocol rerun on
the `2080`.

## Zoom-Stack Viewer

Run `make viewer` to open the Streamlit HD tuning inspector backed by `papers/journal_paper/figures/data/sec09_real_image_hd_viewer/sec09_real_image_hd_viewer_summary.json` after generating assets with `run_sec09_real_image_hd_viewer.py`.
