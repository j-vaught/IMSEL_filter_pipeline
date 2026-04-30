= Unified GPU Implementation <sec:gpu>

All three filter variants described in preceding sections, the fused polynomial stencil, the rectangular kernel, and the elliptical Gaussian kernel, share a common execution architecture that cleanly separates filter construction from filter application. This separation enables a single GPU kernel to serve all variants without modification, reducing engineering complexity while achieving substantial speedups over conventional implementations.

== Architecture Overview

The implementation follows a two-phase design. The _precompute phase_ runs once on the CPU and constructs a set of orientation-indexed stencils. The _compute phase_ runs on the GPU for every input image and applies these stencils to produce gradient magnitude and angle maps.

The precompute phase differs substantially across variants. For the fused polynomial stencil, the procedure constructs the Taylor design matrix $bold(A)_(theta_k)$ for each orientation $theta_k$, computes the pseudoinverse, extracts the gradient row, and enumerates all $(2m+1) times N_p$ stencil positions along the line extension. Positions that map to the same pixel after rounding are deduplicated and their weights summed. For the rectangular kernel, a grid of pixel offsets within a bounding box is rotated into the local $(u, v)$ frame and the kernel function $w(u,v) = -v$ is evaluated within a hard mask. For the elliptical Gaussian kernel, the same rotated grid is used, but the weight function is the product of a Gaussian envelope and $(-v)$. In both geometric cases, the resulting weights are zero-centered and normalized to unit energy.

Despite these different construction procedures, all three variants produce the same output format. Each orientation $theta_k$ is represented by a list of integer offsets $(Delta x_ell, Delta y_ell)$ and corresponding scalar weights $alpha_(k,ell)$ for $ell = 1, dots, N'_k$. This format uniformity is the key architectural property. It enables a single, variant-agnostic GPU kernel to process any stencil without knowledge of its origin.


== The Variant-Agnostic Kernel

The compute-phase kernel consumes only (offset, weight) pairs and has no knowledge of how those weights were generated. For each pixel $(X_0, Y_0)$ and each orientation $theta_k$, the kernel performs three operations. First, it _gathers_ $N'_k$ intensity values from the input image at the precomputed offsets relative to $(X_0, Y_0)$. Second, it _multiplies_ each gathered value by the corresponding weight $alpha_(k,ell)$. Third, it _accumulates_ the weighted sum to obtain the directional response $R_k$. As the kernel iterates over all $N_s$ orientations, it tracks the maximum absolute response $|R_k|$ and the index $k^*$ of the orientation that produced it. Upon completion, the gradient magnitude $|R_(k^*)|$ and the edge angle $theta_(k^*)$ are written to output buffers.

This design means that switching between the fused polynomial stencil, the rectangular kernel, and the elliptical Gaussian kernel requires only swapping the precomputed stencil arrays in device memory. No recompilation, kernel modification, or conditional branching is necessary. The same compiled kernel binary serves all variants, which simplifies deployment and eliminates a common source of implementation errors.

== Custom Triton Kernel

We implement the compute phase as a custom kernel using Triton @tillet2019triton, a Python-embedded language for writing GPU programs that compiles to optimized PTX through LLVM. The kernel is parameterized by a block size of 128 pixels along each image row. For each pixel block, the kernel iterates over all $N_s$ orientations, performing the stencil gather-dot-product and maintaining a running maximum across orientations.

Each iteration of the inner loop loads a single (offset, weight) pair and gathers 128 intensity values in parallel, one per pixel in the block. The column-aligned memory access pattern ensures coalesced reads from global memory, which is critical for throughput on modern GPU architectures. Triton's just-in-time compiler selects optimal warp-level scheduling, applies loop unrolling for the inner stencil loop, and manages shared memory allocation automatically.

== Memory Efficiency

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    table.header[*Method*][*Time (s)*][*VRAM (MB)*][*Speedup*],
    [Naive batched LF], [2.43], [6223], [1.0$times$],
    [cuDNN conv2d], [0.18], [158], [13.8$times$],
    [Fused stencil (Triton)], [0.13], [20], [18.4$times$],
    [Rectangular kernel (Triton)], [0.045], [20], [54$times$],
    [Elliptical kernel (Triton)], [0.047], [20], [52$times$],
  ),
  caption: [Runtime and memory comparison on BIPED v1 ($1280 times 720$). Fused stencil parameters: $m = 7$, $N_p = 100$, $N_s = 18$, $d = 4$. Geometric kernel parameters: $sigma_u = 2.0$, $sigma_v = 1.2$, $N_s = 36$, $15 times 15$ grid. All measurements on NVIDIA A100-SXM4-40GB.],
) <tab:gpu-memory>

@tab:gpu-memory presents the runtime and VRAM consumption for all five implementation strategies applied to a full BIPED v1 image. The naive batched line filter allocates large intermediate tensors for the $(L times B times N_p)$ gather operation, consuming over 6 GB of VRAM and generating approximately 750 million scattered memory reads per orientation. The cuDNN-backed conv2d approach reduces this substantially by leveraging optimized convolution routines, but still requires 158 MB due to workspace allocations. The fused polynomial stencil eliminates all intermediate tensors, reducing VRAM to 20 MB, the image itself plus output buffers, a 311$times$ reduction from the naive approach. Both geometric kernel variants achieve the same 20 MB footprint.

The geometric kernels are approximately 3$times$ faster than the fused polynomial stencil despite using twice as many orientations ($N_s = 36$ versus 18). Three factors account for this advantage. First, the geometric stencils contain fewer unique positions per orientation, approximately 74 for a $15 times 15$ grid versus 264 for the fused stencil at $m = 7$ with $N_p = 100$, which directly reduces the number of gather operations in the inner loop. Second, the stencil size is determined by the grid dimensions $sigma_u$ and $sigma_v$ rather than by the line extension parameter $m$ and the polynomial neighborhood size $N_p$, so it does not grow with the effective spatial extent of the filter. Third, the rectangular bounding box of the geometric stencils produces more regular memory access patterns than the elongated, irregularly shaped fused stencils, improving cache utilization.

== Scaling Behavior

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (center, center, center, center, center),
    table.header[*$m$*][*Naive (s)*][*Fused (s)*][*Speedup*][*VRAM ratio*],
    [1], [0.58], [0.072], [8.0$times$], [273$times$],
    [7], [2.43], [0.13], [18.4$times$], [311$times$],
    [14], [8.88], [0.37], [24.3$times$], [311$times$],
  ),
  caption: [Speedup scaling with half-width $m$ on BIPED v1 ($1280 times 720$). The fused stencil speedup increases with $m$ because the naive cost grows linearly with $L = 2m+1$ while the fused stencil cost grows sublinearly due to pixel deduplication. NVIDIA A100-SXM4-40GB.],
) <tab:gpu-scaling>

@tab:gpu-scaling demonstrates the scaling advantage of the fused stencil approach. The naive implementation's computational cost scales as $O(N_s dot L dot N_p)$, where $L = 2m + 1$ is the number of virtual filter positions along the line. The fused stencil's cost scales as $O(N_s dot N'_k)$, and the deduplicated stencil size $N'_k$ grows much more slowly than $L dot N_p$ because neighboring polynomial neighborhoods overlap extensively. At $m = 1$, the fused stencil is 8$times$ faster than the naive approach. At $m = 14$, the speedup reaches 24.3$times$ because the naive cost has grown linearly with $L$ while the fused cost has increased only modestly. The VRAM ratio stabilizes at 311$times$ for $m >= 7$, indicating that the memory footprint is dominated by the image and output buffers rather than the stencil weights.

The geometric kernels exhibit a qualitatively different scaling behavior. Because they define the anisotropic weight envelope directly from the parameters $sigma_u$ and $sigma_v$ rather than from a line extension, their runtime is effectively constant with respect to the equivalent spatial extent. The rectangular kernel processes a $1280 times 720$ image in approximately 45 ms regardless of the equivalent $m$, and the elliptical kernel is similarly stable at 47 ms. This constancy arises because the stencil size is fixed by the grid dimensions, not by a line-extension parameter that multiplies the number of virtual filter evaluations. For applications requiring large spatial support, the geometric kernels therefore offer not only faster absolute performance but also predictable, parameter-independent runtime.
