# Edge Detection Filter Critique

Implementation-only repository for the WVF/LF edge pipeline.

Kept source areas:

```text
native/edgecritic_metal/ Rust/Metal kernels and FFI
src/edgecritic/          Python implementations and bindings
papers/                  Paper sources and figures
```

Python package layout:

```text
src/edgecritic/wvf/          WVF reference code, radius kernels, Metal binding
src/edgecritic/lf/           LF reference code, LF response reference, Metal binding
src/edgecritic/orientation/  orientation recovery reference and Metal binding
src/edgecritic/cgmm/         two-pass c-GMM reference and Metal binding
src/edgecritic/nms/          enhanced NMS and related reference helpers
src/edgecritic/pipeline/     fused pipeline binding and runnable pipeline helpers
```

Installed command-line helpers:

```text
edgecritic-nms
edgecritic-pipeline-full-dump
edgecritic-pipeline-fusion-dump
edgecritic-pipeline-synthetic-eval
edgecritic-pipeline-verify
```

The retained implementation covers WVF, LF, hybrid orientation recovery,
multi-channel c-GMM fusion, fused Metal front-end execution, and enhanced
NMS.
