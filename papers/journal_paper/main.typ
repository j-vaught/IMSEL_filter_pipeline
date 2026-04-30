// GPU-Accelerated Anisotropic Edge Detection via Orientation-Selective Stencil Filters
// Compile: typst compile main.typ main.pdf

#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [GPU-Accelerated Anisotropic Edge Detection via Orientation-Selective Stencil Filters],
  abstract: [
    We present three orientation-selective edge detection filters, the fused polynomial stencil, the rectangular Gaussian kernel, and the elliptical Gaussian kernel, unified under a common GPU execution framework. The filters compute image gradients by sweeping candidate edge orientations and selecting the direction of maximum normal-derivative response, incorporating hundreds of pixels into each estimate for inherent noise suppression without pre-smoothing. Our first contribution is a _fused stencil formulation_ that algebraically collapses the multi-step Line Filter pipeline of Bagan and Wang into a single precomputed weight vector per orientation, achieving 24$times$ speedup and 311$times$ reduction in GPU memory consumption while producing identical output. Analysis of the fused weights reveals a structural limitation: approximately 28% of the effective kernel weight cancels to near-zero due to destructive interference between overlapping polynomial neighborhoods. This motivates our second contribution, two _geometric kernel alternatives_ (rectangular and elliptical Gaussian envelopes) that define anisotropic derivative weights directly from spatial geometry, bypassing polynomial fitting entirely. The geometric kernels match the fused stencil's accuracy within 0.3% ODS F-score while running 3$times$ faster, with provably uniform weight distributions. We evaluate all three variants across 1,200+ parameter configurations, three benchmarks (BSDS500, BIPED, UDED), 54 classical baselines, and five state-of-the-art deep learning models. On clean imagery, the proposed filters outperform every classical edge detector tested and approach deep learning accuracy without any training data. Under noise, the filters overtake all tested deep learning models below dataset-dependent signal-to-noise ratio thresholds through parametric adaptation. We place this work in the historical context of Savitzky--Golay filtering, steerable filters, and anisotropic Gaussian derivatives, connecting sixty years of local polynomial fitting literature to modern GPU-accelerated edge detection.
  ],
  authors: (
    (
      name: "J. C. Vaught",
      department: [Department of Mechanical Engineering],
      organization: [University of South Carolina],
      location: [Columbia, SC 29208],
      email: "jvaught@sc.edu"
    ),
  ),
  index-terms: ("Edge detection", "Anisotropic filtering", "GPU acceleration", "Savitzky-Golay", "Polynomial fitting", "Noise robustness", "Deep learning", "Triton"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

// ============================================================
// SECTIONS
// ============================================================

#include "sections/sec1_introduction.typ"

#include "sections/sec2_related_work.typ"

#include "sections/sec3_math_framework.typ"

#include "sections/sec4_fused_stencil.typ"

#include "sections/sec5_geometric_kernels.typ"

#include "sections/sec6_gpu_implementation.typ"

#include "sections/sec7_parameter_analysis.typ"

#include "sections/sec8_experiments.typ"

#include "sections/sec9_discussion.typ"

#include "sections/sec10_conclusion.typ"

#include "sections/appendix.typ"
