# Edge Detection Filter Critique

Implementation-only repository for the WVF/LF edge pipeline.

Kept source areas:

```text
agent_workspaces/        Python reference implementations
native/edgecritic_metal/ Rust/Metal kernels and FFI
scripts/eval/            Raw pipeline and c-GMM/NMS scripts
src/edgecritic/          Reusable Python package
papers/                  Paper sources and figures
```

The retained implementation covers WVF, LF, hybrid orientation recovery,
multi-channel c-GMM fusion, fused Metal front-end execution, and enhanced
NMS.
