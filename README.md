# Edge Detection Filter Critique

Implementation-only repository for the WVF/LF edge pipeline.

Kept source areas:

```text
metal/edgecritic_metal/ Rust/Metal kernels and FFI
src/                    Python implementations and bindings
papers/                 Paper sources and figures
```

Python package layout:

```text
src/wvf/          WVF reference code, radius kernels, Metal binding
src/lf/           LF reference code, LF response reference, Metal binding
src/orientation/  orientation recovery reference and Metal binding
src/cgmm/         two-pass c-GMM reference and Metal binding
src/nms/          enhanced NMS and related reference helpers
src/pipeline/     fused pipeline binding and runnable pipeline helpers
```

Metal layout:

```text
metal/edgecritic_metal/src/lib.rs                  Rust FFI and Metal host orchestration
metal/edgecritic_metal/src/shaders/common.metal    shared Metal structs/helpers
metal/edgecritic_metal/src/shaders/wvf.metal       WVF kernels
metal/edgecritic_metal/src/shaders/lf.metal        LF kernels
metal/edgecritic_metal/src/shaders/orientation.metal orientation recovery kernels
metal/edgecritic_metal/src/shaders/cgmm.metal      c-GMM kernels
```

Installed command-line helpers:

```text
edgecritic-nms
edgecritic-pipeline-full-dump
edgecritic-pipeline-fusion-dump
edgecritic-pipeline-synthetic-eval
edgecritic-pipeline-verify
wvf-metal
```

The retained implementation covers WVF, LF, hybrid orientation recovery,
multi-channel c-GMM fusion, fused Metal front-end execution, and enhanced
NMS.
