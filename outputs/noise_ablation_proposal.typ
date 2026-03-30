#set page(paper: "us-letter", margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 16pt, weight: "bold")[Multi-Noise-Type Parameter Sensitivity Study]
  #v(0.3em)
  #text(size: 12pt)[How Noise Characteristics Affect Optimal WVF/LF Parameters]
  #v(0.5em)
  J. C. Vaught #sym.dot.c Edge-Detection Filter Critique Project #sym.dot.c March 2026
]

#v(1em)

= Motivation

Prior ablation studies on clean datasets (BIPED v1, BSDS500, UDED) consistently showed that small support sizes ($N_p approx 25$--$50$), low polynomial orders ($d = 2$), and short line filter half-widths ($m = 1$--$3$) outperform the published WVF/LF parameters ($N_p = 250$, $d = 4$, $m = 14$). However, a follow-up experiment adding Gaussian noise at controlled SNR levels to a single BIPED image revealed a crossover: below SNR $approx 2$, the published large-support, long-line parameters become optimal. At SNR $= 1$, the LF with $m = 14$ was the best configuration tested, with Bagan's exact parameters achieving ODS $= 0.568$ compared to the best clean-optimal WVF's ODS $= 0.554$.

This finding has a critical limitation: only Gaussian noise was tested on a single image. Real imaging environments---particularly the underwater and aquatic domains targeted by the WVF/LF papers---exhibit noise with fundamentally different statistical properties. Sonar and underwater cameras produce multiplicative speckle noise, not additive Gaussian noise. Sensor defects create salt-and-pepper impulse noise. Low-light conditions introduce Poisson shot noise. Each noise type degrades edge structure differently, and filters that excel under Gaussian noise may fail under speckle or impulse noise.

This study systematically evaluates how five noise types interact with WVF/LF parameter choices across four standard edge detection datasets.

= Experimental Design

== Image Selection

Five images are selected from each of four datasets, as summarized in @tab:datasets. Selection is performed by running a single WVF configuration ($N_p = 50$, $d = 2$, $N_s = 18$) on every image in each dataset, scoring each against its ground truth, and retaining the five highest-scoring images. This biases toward images with well-defined edges where parameter differences are most measurable, avoiding degenerate cases where all configurations score poorly regardless of parameters.

#figure(
  align(center,
    table(
      columns: 5,
      align: (left, right, left, left, left),
      stroke: none,
      table.hline(),
      table.header([*Dataset*], [*Images*], [*Resolution*], [*GT Format*], [*Domain*]),
      table.hline(),
      [UDED], [30], [321$times$250 to 720$times$720], [Binary PNG], [Underwater],
      [BIPED v1], [50], [720$times$1280], [Binary PNG], [Urban/outdoor],
      [BIPED v2], [50], [720$times$1280], [Binary PNG (re-annotated)], [Urban/outdoor],
      [BSDS500], [200], [481$times$321 or 321$times$481], [Multi-annotator .mat], [Natural scenes],
      table.hline(),
    )
  ),
  caption: [Datasets used in the study. Five images are selected from each.],
) <tab:datasets>

== Noise Models

Five noise types are applied, each parameterized to produce a consistent signal-to-noise ratio despite different statistical properties. @fig:noise-types shows all five noise types applied to a BIPED image crop at SNR $= 1.0$, and @fig:noise-grid shows how each noise type degrades the image across three SNR levels.

#figure(
  image("fig_noise_types_snr1.png", width: 100%),
  caption: [Visual comparison of five noise types at SNR $= 1.0$ applied to a BIPED v1 crop. Gaussian and uniform noise produce diffuse degradation; salt-and-pepper creates isolated extreme pixels; Poisson noise is signal-dependent (stronger in bright regions); speckle noise is multiplicative, preserving relative contrast better than additive models.],
) <fig:noise-types>

The mathematical formulation of each noise model is given in @tab:noise-models.

#figure(
  align(center,
    table(
      columns: 3,
      align: (left, left, left),
      stroke: none,
      table.hline(),
      table.header([*Noise Type*], [*Model*], [*Physical Origin*]),
      table.hline(),
      [Gaussian], [$I' = I + cal(N)(0, (Delta slash "SNR")^2)$], [Sensor thermal noise],
      [Salt & pepper], [$I' in {0, 255}$ with density $1 slash (1 + "SNR")$], [Dead/hot pixels, bit errors],
      [Poisson], [$I' = "Pois"(I dot "SNR"^2) slash "SNR"^2$], [Photon shot noise],
      [Speckle], [$I' = I dot (1 + cal(N)(0, 1 slash "SNR"^2))$], [Sonar, radar, underwater],
      [Uniform], [$I' = I + cal(U)(-Delta slash "SNR", Delta slash "SNR")$], [ADC quantization],
      table.hline(),
    )
  ),
  caption: [Noise models and their parameterization. $Delta$ denotes the image intensity range.],
) <tab:noise-models>

*Gaussian noise* adds independent, identically distributed samples from $cal(N)(0, sigma^2)$ to each pixel, where $sigma = Delta slash "SNR"$ and $Delta$ is the image intensity range. This is the standard additive noise model and serves as the baseline.

*Salt-and-pepper noise* randomly replaces pixels with either 0 or 255 at a density proportional to $1 slash (1 + "SNR")$. It is fundamentally different from continuous noise because it destroys local structure entirely at affected pixels rather than adding small perturbations. The WVF's least-squares fitting is sensitive to such outliers.

*Poisson noise* generates signal-dependent noise where variance is proportional to pixel intensity. This is the physically correct noise model for photon-limited imaging and produces stronger corruption in bright regions than dark regions.

*Speckle noise* applies multiplicative corruption: $I'(x,y) = I(x,y) dot (1 + cal(N)(0, 1 slash "SNR"^2))$. This is the dominant noise model in coherent imaging systems including sonar, radar, and certain underwater cameras---directly relevant to Bagan's target domain. Because it is multiplicative, edge contrast is preserved proportionally rather than corrupted by a fixed additive floor.

*Uniform noise* adds samples from $cal(U)(-a, a)$ where $a = Delta slash "SNR"$. It has heavier tails than Gaussian noise at the same variance, providing a contrast to the light-tailed Gaussian model.

#figure(
  image("fig_noise_snr_grid.png", width: 95%),
  caption: [Each noise type at three SNR levels (0.5, 1.0, 2.0) with clean reference. The character of degradation varies substantially: speckle preserves relative contrast while adding texture; salt-and-pepper creates sparse but extreme outliers; Poisson is barely visible in dark regions but prominent in bright areas.],
) <fig:noise-grid>

The noise residual distributions at SNR $= 1.0$ are shown in @fig:noise-dist, illustrating the distinct statistical profiles of each noise type.

#figure(
  image("fig_noise_distributions.png", width: 100%),
  caption: [Distribution of pixel-level noise (noisy $minus$ clean) at SNR $= 1.0$. Gaussian and uniform are symmetric and zero-mean; salt-and-pepper produces a bimodal distribution with extreme tails; Poisson is right-skewed due to signal dependence; speckle has heavier tails than Gaussian.],
) <fig:noise-dist>

== SNR Levels

Eight levels are tested, as shown in @tab:snr. The lowest levels (0.3--0.5) represent extreme degradation where edge structure is barely visible. The mid-range (1.0--2.0) spans the crossover region identified in the Gaussian-only study. Higher levels (3.0--4.0) represent moderate noise where clean-optimal parameters may still hold. All noise is generated with a fixed random seed (42) for reproducibility.

#figure(
  align(center,
    table(
      columns: 3,
      align: (center, left, left),
      stroke: none,
      table.hline(),
      table.header([*SNR*], [*Regime*], [*Gaussian $sigma$ (for $Delta = 245$)*]),
      table.hline(),
      [0.3], [Extreme --- edges barely visible], [$sigma approx 817$],
      [0.5], [Very heavy noise], [$sigma approx 490$],
      [1.0], [Heavy --- crossover region], [$sigma approx 245$],
      [1.5], [Moderate--heavy], [$sigma approx 163$],
      [2.0], [Moderate --- near crossover], [$sigma approx 123$],
      [3.0], [Mild], [$sigma approx 82$],
      [4.0], [Light noise], [$sigma approx 61$],
      [$infinity$], [Clean (no noise added)], [$sigma = 0$],
      table.hline(),
    )
  ),
  caption: [SNR levels tested. The Gaussian $sigma$ column shows the equivalent noise standard deviation for a typical image with intensity range $Delta = 245$.],
) <tab:snr>

== Parameter Grid

The grid is designed to capture all key phenomena identified in prior ablation studies while remaining computationally tractable. @tab:grid summarizes the parameter ranges.

#figure(
  align(center,
    table(
      columns: 4,
      align: (left, left, right, left),
      stroke: none,
      table.hline(),
      table.header([*Filter*], [*Parameter*], [*Count*], [*Values*]),
      table.hline(),
      [WVF], [$N_p$ (support size)], [20], [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500],
      [WVF], [$N_s$ (orientations)], [5], [3, 9, 18, 36, 72],
      [WVF], [$d$ (polynomial order)], [4], [2, 3, 4, 5],
      table.hline(),
      [LF], [$N_p$ (support size)], [7], [15, 25, 50, 75, 100, 150, 250],
      [LF], [$N_s$ (orientations)], [2], [18, 36],
      [LF], [$m$ (line half-width)], [8], [1, 2, 3, 5, 7, 10, 14, 20],
      table.hline(),
    )
  ),
  caption: [Parameter grid. WVF yields $approx 350$ valid configurations (after excluding underdetermined systems where $N_p < (d+1)(d+2) slash 2$). LF yields 112 configurations. Total: $approx 462$ configurations per image per noise condition.],
) <tab:grid>

The $N_s$ values for WVF are chosen to include the critical transition at $N_s = 3$ (below which performance collapses due to insufficient angular coverage) and sufficient resolution to confirm saturation. The $d$ range covers the published default ($d = 4$), the empirically superior $d = 2$, and the higher-order $d = 5$ that shows advantages at large $N_p$. Orientation count is reduced for LF because prior results showed it has even less effect on LF than on WVF.

== Evaluation Protocol

Each configuration is evaluated using the standard BSDS500 protocol, as summarized in @tab:eval. No non-maximum suppression or other post-processing is applied; the raw gradient magnitude map is evaluated directly. This matches the protocol used by DexiNed, TEED, and other published methods, and differs from Bagan's approach of applying Canny post-processing (which was shown in our earlier tests to uniformly degrade ODS scores by 0.03--0.14).

#figure(
  align(center,
    table(
      columns: 2,
      align: (left, left),
      stroke: none,
      table.hline(),
      table.header([*Setting*], [*Value*]),
      table.hline(),
      [Thresholds], [1001 uniformly spaced in $[0, max("magnitude")]$],
      [Match radius], [3 pixels (Euclidean distance transform)],
      [Metric], [ODS (optimal dataset scale F-score)],
      [Post-processing], [None (raw gradient magnitude)],
      [Evaluation method], [Vectorized: distance transform + maximum filter + searchsorted],
      table.hline(),
    )
  ),
  caption: [Evaluation protocol, matching the standard BSDS500 benchmark.],
) <tab:eval>

== Computational Budget

#figure(
  align(center,
    table(
      columns: 4,
      align: (left, right, right, right),
      stroke: none,
      table.hline(),
      table.header([*Dataset*], [*Images $times$ conditions*], [*Est. per condition*], [*Subtotal*]),
      table.hline(),
      [UDED (5 imgs)], [$5 times 36 = 180$], [$approx 15$ s], [$approx 0.8$ h],
      [BIPED v1 (5 imgs)], [$5 times 36 = 180$], [$approx 95$ s], [$approx 4.8$ h],
      [BIPED v2 (5 imgs)], [$5 times 36 = 180$], [$approx 95$ s], [$approx 4.8$ h],
      [BSDS500 (5 imgs)], [$5 times 36 = 180$], [$approx 20$ s], [$approx 1.0$ h],
      table.hline(),
      [*Total*], [*720*], [], [*$approx 11$ h*],
      table.hline(),
    )
  ),
  caption: [Estimated runtime on NVIDIA H200 NVL (141 GB VRAM). Each "condition" is one noise type at one SNR level (36 total: 5 types $times$ 7 SNR levels + 1 clean). SLURM time limit: 24 hours.],
)

= Expected Outcomes

Based on the Gaussian-only SNR study and the properties of each noise model, three hypotheses are tested.

*Hypothesis 1: Speckle noise shifts the crossover point.* Because speckle is multiplicative, it degrades high-contrast edges more than low-contrast ones. The WVF's local polynomial fit may be more sensitive to multiplicative corruption than to additive corruption, potentially shifting the SNR crossover between small and large support sizes. If confirmed, this would validate Bagan's parameter choices specifically for underwater sonar/camera applications.

*Hypothesis 2: Salt-and-pepper noise favors large supports.* The WVF uses least-squares fitting, which is sensitive to outliers. Salt-and-pepper noise creates extreme outliers (pixels at 0 or 255) that may catastrophically degrade small-support WVF fits while being partially averaged out by large-support or long-line LF configurations. This would suggest the WVF needs a robust fitting alternative (e.g., L1 regression) for impulse noise environments.

*Hypothesis 3: The optimal parameter trajectory is noise-type-dependent.* If the optimal $N_p$ and $m$ at each SNR depend on the noise type, then no single parameter choice is universally optimal, and the WVF/LF should be parameterized adaptively based on the noise regime. If the trajectories are similar across noise types, then the Gaussian-only analysis generalizes and a single parameter recommendation suffices.

= Relationship to Prior Work

Bagan and Wang's published WVF/LF parameters ($N_p = 250$, $d = 4$, $m = 14$) were originally developed for underwater and aquatic edge detection, where speckle noise and low contrast are dominant. Their evaluation datasets (UDED, BIPED) contain relatively clean images, creating a mismatch between the noise regime the parameters were tuned for and the conditions under which they were evaluated. @tab:prior summarizes the gap between published and optimal parameters on clean data.

#figure(
  align(center,
    table(
      columns: 5,
      align: (left, right, right, right, right),
      stroke: none,
      table.hline(),
      table.header([*Dataset*], [*Best WVF*], [*Bagan WVF*], [*Best LF*], [*Bagan LF*]),
      table.hline(),
      [UDED (30 imgs)], [0.899], [0.847], [0.888], [0.729],
      [BIPED v1 (50 imgs)], [0.812], [0.767], [0.802], [0.643],
      [BIPED v2 (50 imgs)], [0.830], [0.783], [0.819], [0.658],
      [BSDS500 (200 imgs)], [0.682], [0.671], [0.671], [0.615],
      table.hline(),
    )
  ),
  caption: [Dataset-wide ODS from the full dataset ablation (clean images, no noise). Bagan's parameters underperform across all datasets, with the largest gap on LF ($m = 14$ consistently ranks near the bottom).],
) <tab:prior>

This study closes the gap by explicitly testing whether Bagan's parameter choices are justified under the noise conditions his method was designed to address. If large $N_p$ and $m$ become optimal under speckle noise at realistic SNR levels, the published parameters represent a legitimate engineering tradeoff---optimizing for the target deployment environment at the cost of clean-image performance.
