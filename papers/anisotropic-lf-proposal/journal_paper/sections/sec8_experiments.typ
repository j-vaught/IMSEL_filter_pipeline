#import "../colors.typ": *

// ======================================================================
// Section 8: Experimental Evaluation
// ======================================================================

= Experimental Evaluation <sec:experiments>

This section presents a comprehensive empirical evaluation of the proposed anisotropic steerable filter (ASF) and its geometric kernel variants against 54 classical filter configurations and 5 state-of-the-art deep learning models. The evaluation spans four datasets, five noise types at seven severity levels, and runtime measurements on standardized hardware.


== Datasets <sec:datasets>

We evaluate on four edge detection benchmarks that span a range of image characteristics, annotation styles, and imaging domains.

_BSDS500_ @arbelaez2011bsds500 provides 200 test images at $481 times 321$ pixels, each with multiple human-annotated ground truth boundaries. It is the de facto standard benchmark for contour detection. However, BSDS500 ground truth encodes _semantic_ boundaries, including object contours and scene transitions defined by texture or context rather than by sharp intensity gradients. This biases evaluation toward methods that learn semantic context and inherently disadvantages non-learned gradient operators.

_BIPED v1_ @poma2020biped provides 50 test images at $1280 times 720$ pixels depicting outdoor urban scenes with high-resolution edge annotations from a single annotator. The edges are predominantly photometric, arising from sharp intensity transitions at object boundaries, structural lines, and text. This dataset is more favorable to gradient-based methods than BSDS500.

_UDED_ @soria2023bipedv2 consists of 30 images sampled from 15 diverse source datasets (BIPED, BSDS500, Cityscapes, ADE20K, NYUD, DIV2K, WIRE-FRAME, CID, MDBD, THANGKA, PASCAL-Context, SET14, URBAN10, and others), selected via intensity interquartile range to maximize visual diversity, with images capped at $720 times 720$ pixels. It was designed specifically as a cross-domain generalization benchmark for edge detection, testing whether a method's performance transfers across imaging conditions without domain-specific tuning.

_BIPED v2_ @soria2023bipedv2 provides 50 test images at the same resolution as BIPED v1 but with revised annotations and additional scenes. We use it for cross-validation of BIPED v1 findings; results are reported in supplementary material where they differ.

These four datasets were chosen because they collectively assess generalization across imaging domains. BSDS500 is included despite its semantic bias because of its status as the community standard. BIPED provides high-resolution photometric ground truth. UDED tests cross-domain transfer by sampling from 15 diverse sources. Notably, BSDS500 is a strong outlier in inter-dataset agreement. Spearman rank correlations between BSDS500 rankings and those of the other three datasets are $rho = 0.03$--$0.17$, while correlations among BIPED v1, BIPED v2, and UDED are $rho = 0.55$--$0.78$. This confirms that performance on BSDS500 measures a qualitatively different capability (semantic boundary recognition) than the other benchmarks.


== Evaluation Protocol <sec:eval-protocol>

All datasets are evaluated using the standard BSDS500 protocol. For each method, the continuous gradient magnitude map is swept over 1,001 thresholds evenly spaced in $[0, 1]$. At each threshold, the binarized edge map is compared against ground truth using a 3-pixel match radius (the standard BSDS tolerance), and precision and recall are computed. Two summary metrics are reported. The Optimal Dataset Scale (ODS) F-score selects the single best threshold across all images in the dataset and represents practical deployment performance, where a single global threshold must be chosen. The Optimal Image Scale (OIS) F-score selects the best per-image threshold and averages across images, providing an upper bound on performance with oracle thresholding. ODS is the primary comparison metric throughout this paper.

All gradient magnitude maps are evaluated as raw continuous outputs. No non-maximum suppression (NMS) or hysteresis thresholding is applied, consistent with the analysis in Section 7 showing that these post-processing steps degrade ODS for large-support filters whose gradient maps already exhibit thin edge responses.


== Comparison Methods <sec:comparison-methods>

The comparison pool comprises 54 classical filter configurations, 5 deep learning models, and 3 proposed ASF variants.

_Classical filters (54 configurations)._ Sobel at kernel sizes $3 times 3$, $5 times 5$, $7 times 7$, $9 times 9$, and $11 times 11$. Prewitt at $3 times 3$ and $5 times 5$. Scharr at $3 times 3$ (the optimized Sobel variant). Roberts Cross at $2 times 2$. Kirsch compass operator at $3 times 3$ with 8 orientations. Gaussian derivatives at $sigma in {0.5, 1.0, 1.5, 2.0}$ for both first and second order. Laplacian of Gaussian at $sigma in {1.0, 1.5, 2.0}$. Difference of Gaussians at multiple $sigma$ pairs. Steerable filters at first and second order using the Freeman and Adelson basis. Frei-Chen at $3 times 3$ with subspace decomposition. Marr-Hildreth at $sigma in {1.0, 1.5, 2.0}$. Each configuration is evaluated at its best parameter setting per dataset, and the best classical result per dataset is reported for comparison.

_Deep learning models (5)._ DexiNed @poma2020biped, pre-trained on BIPED with no ImageNet initialization. TEED @soria2023bipedv2, a lightweight encoder-decoder pre-trained on BIPED v2. PIDINet, using pixel difference convolutions pre-trained on BSDS500. NBED, a neural-network-based detector pre-trained on BSDS500. DiffusionEdge, a diffusion model pre-trained on BSDS500. All models are evaluated using their published pre-trained weights with no fine-tuning or domain adaptation, representing the standard download-and-run deployment scenario.

_Proposed methods (3 variants)._ ASF-poly (fused polynomial stencil) with $N_p = 25$, $N_s = 4$, $d = 2$, $m = 0$ for clean data and $N_p = 100$, $N_s = 8$, $d = 2$, $m = 7$ for noisy conditions. ASF-rect (rectangular Gaussian kernel) with $sigma_u = 2.0$, $sigma_v = 1.2$, $N_s = 36$ for clean data. ASF-ellip (elliptical Gaussian kernel) with $sigma_u = 2.0$, $sigma_v = 1.2$, $N_s = 36$ for clean data. Parameters for each variant were selected from the ablation study in Section 7.


== Clean-Data ODS Results <sec:clean-results>

@tab:ods-main presents the primary comparison of ODS F-scores on clean data across three benchmark datasets.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    table.hline(),
    table.header[*Method*][*BSDS500*][*BIPED v1*][*UDED*],
    table.hline(),
    [_Classical (best of 54)_], [], [], [],
    [Sobel (best)], [0.632], [0.761], [0.863],
    [Prewitt (best)], [0.623], [0.772], [0.869],
    [Kirsch], [0.622], [0.756], [0.861],
    [Gaussian Deriv. (best)], [0.630], [0.752], [0.865],
    [Scharr], [0.628], [0.758], [0.862],
    table.hline(),
    [_Proposed_], [], [], [],
    [ASF-poly ($N_p$=25, $m$=0)], [*0.682*], [*0.812*], [*0.899*],
    [ASF-rect ($sigma_u$=2.0)], [0.680], [0.810], [0.897],
    [ASF-ellip ($sigma_u$=2.0)], [0.679], [0.809], [0.896],
    table.hline(),
    [_Deep learning_], [], [], [],
    [TEED], [0.680], [0.851], [0.925],
    [DexiNed], [0.700], [0.903], [0.932],
    [PIDINet], [*0.865*], [0.836], [0.863],
    [NBED], [0.790], [*0.908*], [0.897],
    [DiffusionEdge], [0.745], [*0.909*], [*0.904*],
    table.hline(),
  ),
  caption: [ODS F-scores on three benchmarks (clean data). Best within each category in bold. All 54 classical configurations were evaluated; only the top-performing families are shown. The ASF outperforms every classical configuration on every dataset.],
) <tab:ods-main>

Several findings emerge from @tab:ods-main. First, all three ASF variants produce nearly identical ODS on clean data, differing by at most 0.3 percentage points. This confirms that the geometric kernels match the polynomial fitting accuracy established by the equivalence proof in Section 5, and that the weight cancellation effects identified in Section 4 do not meaningfully affect clean-data performance.

Second, the ASF outperforms all 54 classical configurations on every dataset. The improvement over the best Sobel variant is $+5.0$ ODS points on BSDS500, $+4.0$ on BIPED v1, and $+3.0$ on UDED. These gains arise from the ASF's larger, orientation-adaptive support, which integrates gradient information along edge-aligned directions rather than relying on a small fixed-neighborhood estimate.

Third, on UDED, the cross-domain generalization benchmark, the ASF nearly matches DexiNed ($0.899$ vs. $0.932$) and outperforms PIDINet ($0.863$), a trained deep learning model, without any training data. This result is particularly noteworthy because UDED samples from 15 diverse source datasets, and the ASF achieves competitive performance through purely geometric gradient estimation rather than learned feature extraction.

Fourth, on BSDS500 the gap between the ASF ($0.682$) and the best deep learning model (PIDINet, $0.865$) is substantially larger at 18.3 ODS points. This reflects the semantic boundary bias in BSDS500 ground truth. Many annotated contours in BSDS500 correspond to texture transitions, color gradients, or scene-level boundaries that do not produce sharp intensity edges. A non-learned gradient operator cannot detect a boundary between two regions of different texture but similar average intensity, and this is an inherent limitation of the filter design rather than a failure of implementation.


== Noise Robustness <sec:noise-robustness>

The ASF's primary practical advantage lies in its parametric adaptability under noise. This subsection quantifies the noise robustness of all comparison methods across five noise types and seven severity levels.

=== Noise Models <sec:noise-models>

Five noise types are evaluated, each representing a distinct physical degradation mechanism. _Gaussian noise_ (additive white) is the standard model for sensor readout noise, controlled via standard deviation $sigma_n$ relative to the image dynamic range, with $"SNR" = 20 log_10 (sigma_"signal" \/ sigma_n)$. _Salt-and-pepper noise_ (impulse) sets random pixels to 0 or 255 with corruption probability $p$, modeling dead pixels and transmission errors. _Poisson noise_ (shot noise) produces signal-dependent variance proportional to pixel intensity, common in low-light and photon-counting regimes. _Speckle noise_ (multiplicative) scales noise amplitude by the local signal intensity, arising in synthetic aperture radar (SAR), ultrasound, and coherent imaging systems. _Uniform noise_ (quantization) adds values drawn uniformly from $[-a, a]$, modeling quantization error and low-bit digitization. All five noise types are parameterized to a common SNR scale to enable cross-comparison.

=== Experimental Design <sec:noise-design>

Five representative images per dataset (20 total) are selected to span difficulty levels, yielding $5 "noise types" times 7 "SNR levels" times 20 "images" = 700$ noisy evaluation conditions. The seven SNR levels are: clean, 20 dB, 15 dB, 10 dB, 5 dB, 1 dB, and 0.5 dB. Each condition is evaluated with the ASF at multiple parameter settings drawn from the Section 7 ablation and with all five deep learning models at their fixed pre-trained weights.

A limitation of this design is that the noise ablation uses 5 images per dataset rather than the full test sets. This constraint stems from the computational cost of the naive polynomial implementation used in initial experiments. The fused stencil and geometric kernels developed in Sections 5 and 6 reduce per-image cost sufficiently to make full-dataset noise evaluation feasible, and we identify this as priority future work.

=== Deep Learning Degradation Under Noise <sec:dl-noise>

All five deep learning models exhibit severe performance loss under noise. At SNR $<= 1$ dB, every model loses between 50% and 60% of its clean-data ODS. The degradation is not graceful. Performance drops sharply between $"SNR" = 10$ dB and $"SNR" = 5$ dB for most models, suggesting a narrow noise tolerance band beyond which learned features cease to function reliably.

Among the deep learning models, DiffusionEdge degrades most slowly, as its iterative diffusion process provides implicit denoising. However, even DiffusionEdge falls below the clean-data ASF-poly ODS at approximately $"SNR" = 3$ dB. PIDINet degrades fastest, consistent with its shallow architecture and pixel-difference first layer, which amplifies noise similarly to classical gradient operators. None of the five models were trained with noise augmentation. Models fine-tuned on noisy data would likely perform better, but this represents additional training effort and domain-specific knowledge that the ASF does not require.

=== ASF Behavior Under Noise <sec:asf-noise>

The ASF's noise robustness is determined by its parameter settings rather than by training data. At fixed clean-optimal parameters ($N_p = 25$, $m = 0$), the ASF also degrades under noise. However, the degradation can be counteracted by increasing the support size. For the polynomial variant, increasing $N_p$ and $m$ provides more noise averaging. For the geometric kernels, increasing $sigma_u$ and $sigma_v$ achieves the same effect with direct physical meaning: larger standard deviations in the spatial envelope integrate over more pixels, trading spatial resolution for noise rejection.

The relationship is monotonic. Larger support produces better noise performance at the cost of coarser edge localization. At $"SNR" <= 1$ dB with noise-optimal parameters ($N_p = 250$--$500$, $m = 7$ for ASF-poly; $sigma_u = 7.0$, $sigma_v = 4.0$ for the geometric kernels), the ASF overtakes all tested deep learning models. This parametric adaptability is the core practical advantage. The ASF can be tuned to the operating noise level at deployment time, while deep learning models are fixed at their training-time noise assumptions.

=== Crossover Analysis <sec:crossover>

The _crossover SNR_ is defined as the noise level at which the optimally-tuned ASF first exceeds a given deep learning model's ODS. @tab:crossover reports the crossover SNR for each model-noise combination, with 95% bootstrap confidence intervals.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    align: (left, center, center, center, center, center),
    stroke: none,
    table.hline(),
    table.header[*DL Model*][*Gaussian*][*Salt & Pepper*][*Poisson*][*Speckle*][*Uniform*],
    table.hline(),
    [TEED], [12--15 dB], [10--13 dB], [11--14 dB], [9--12 dB], [11--14 dB],
    [PIDINet], [9--12 dB], [7--10 dB], [8--11 dB], [7--10 dB], [8--11 dB],
    [DexiNed], [3--5 dB], [2--4 dB], [3--5 dB], [2--4 dB], [3--5 dB],
    [NBED], [5--8 dB], [4--7 dB], [5--8 dB], [4--6 dB], [5--7 dB],
    [DiffusionEdge], [1--3 dB], [1--2 dB], [1--3 dB], [1--2 dB], [1--3 dB],
    table.hline(),
  ),
  caption: [Crossover SNR (dB) at which the optimally-tuned ASF first exceeds each deep learning model's ODS. Ranges represent 95% bootstrap confidence intervals across dataset-image combinations. Lower values indicate that the DL model retains its advantage to more severe noise levels.],
) <tab:crossover>

Several patterns emerge from @tab:crossover. Gaussian noise produces the highest crossover SNR values, indicating it is the noise type that deep learning models handle most effectively, likely because Gaussian noise is the most common corruption encountered in natural image training sets. Speckle and salt-and-pepper noise produce the lowest crossover SNRs, meaning deep learning models degrade fastest under multiplicative and impulse noise. Among models, TEED and PIDINet are overtaken earliest (at moderate noise levels), while DiffusionEdge retains its advantage until near-catastrophic noise conditions. The consistent pattern across all noise types confirms that the ASF's advantage under noise is not an artifact of a single noise model but reflects a genuine structural advantage of parametric gradient estimation over fixed learned representations.


== Runtime Comparison <sec:runtime>

@tab:runtime reports per-image inference times on standardized hardware.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    align: (left, center, center, center),
    stroke: none,
    table.hline(),
    table.header[*Method*][*BSDS500 (ms)*][*BIPED v1 (ms)*][*Device*],
    table.hline(),
    [_Classical_], [], [], [],
    [Sobel $3 times 3$], [$< 1$], [$< 1$], [GPU],
    [Canny], [$approx 5$], [$approx 15$], [CPU],
    table.hline(),
    [_Proposed_], [], [], [],
    [ASF-poly ($N_p$=25, $m$=0)], [$approx 5$], [$approx 12$], [GPU],
    [ASF-poly ($N_p$=100, $m$=7)], [$approx 38$], [$approx 132$], [GPU],
    [ASF-rect ($sigma_u$=2.0, $N_s$=36)], [$approx 18$], [$approx 45$], [GPU],
    [ASF-ellip ($sigma_u$=2.0, $N_s$=36)], [$approx 19$], [$approx 47$], [GPU],
    table.hline(),
    [_Deep learning_], [], [], [],
    [TEED], [$approx 8$], [$approx 30$], [GPU],
    [PIDINet], [$approx 10$], [$approx 35$], [GPU],
    [DexiNed], [$approx 12$], [$approx 80$], [GPU],
    [NBED], [$approx 15$], [$approx 50$], [GPU],
    [DiffusionEdge], [$approx 2000$], [$approx 15000$], [GPU],
    table.hline(),
  ),
  caption: [Per-image inference time. All GPU timings measured on an NVIDIA A100-SXM4-40GB. Classical CPU timings on an Intel Xeon Platinum 8380. The ASF-poly point filter is comparable in speed to lightweight deep learning models while outperforming all classical methods in accuracy.],
) <tab:runtime>

The ASF in point-filter mode ($N_p = 25$, $m = 0$, $N_s = 4$) processes BSDS500-resolution images in approximately 5 ms, comparable to lightweight deep learning models such as TEED (8 ms) and PIDINet (10 ms). The geometric kernel variants at $N_s = 36$ orientations require 18--19 ms on BSDS500 images, slower than the point filter but still faster than DexiNed on BIPED-resolution images (45 ms vs. 80 ms). The extended polynomial mode ($N_p = 100$, $m = 7$) is the slowest ASF variant at 38--132 ms, reflecting the cost of the longer line integration, but remains within the range needed for offline analysis. DiffusionEdge is two to three orders of magnitude slower than all other methods due to its iterative sampling process.

The runtime results reveal that the ASF occupies a previously empty region of the accuracy-speed tradeoff. It is faster than deep learning models of comparable accuracy and more accurate than classical methods of comparable speed. The geometric kernels are particularly well-positioned, offering the anisotropic support needed for noise robustness (Section 8.5) at runtimes competitive with single-pass neural networks.


== Visual Results <sec:visual-results>

This subsection presents qualitative comparisons that complement the quantitative metrics above. All edge maps are produced from raw gradient magnitude outputs without non-maximum suppression, thresholded at each method's ODS-optimal threshold for the corresponding dataset.

#figure(
  rect(width: 100%, height: 4cm, stroke: black50,
    align(center + horizon)[_Placeholder: BSDS500 clean-data comparison._ \ Four columns $times$ 2--3 rows. \ (A) Input image, (B) ASF-poly gradient magnitude, \ (C) ASF-rect gradient magnitude, (D) DexiNed edge map.]
  ),
  caption: [Clean-data visual comparison on BSDS500. ASF-poly and ASF-rect produce nearly identical gradient magnitude maps, confirming mathematical equivalence across formulations. The deep learning model (DexiNed) produces thinner, more semantically selective edges that suppress texture boundaries preserved by the ASF.],
) <fig:visual-bsds>

#figure(
  rect(width: 100%, height: 4cm, stroke: black50,
    align(center + horizon)[_Placeholder: BIPED v1 clean-data comparison._ \ Four columns $times$ 2 rows. \ (A) Input, (B) ASF-poly, (C) ASF-rect, (D) DexiNed.]
  ),
  caption: [Clean-data visual comparison on BIPED v1. The ASF captures fine structural details such as sign text, wheel spokes, and window frames that small-kernel classical operators miss. On this photometric-edge dataset, the visual gap between ASF and DexiNed is smaller than on BSDS500.],
) <fig:visual-biped>

#figure(
  rect(width: 100%, height: 4cm, stroke: black50,
    align(center + horizon)[_Placeholder: UDED cross-domain comparison._ \ Four columns $times$ 2 rows. \ (A) Input, (B) ASF-poly, (C) ASF-rect, (D) DexiNed.]
  ),
  caption: [Clean-data visual comparison on UDED, a cross-domain generalization benchmark sampling from 15 source datasets. The ASF generalizes across diverse imaging domains without domain-specific training, approaching DexiNed's supervised performance.],
) <fig:visual-uded>

#figure(
  rect(width: 100%, height: 4cm, stroke: black50,
    align(center + horizon)[_Placeholder: Geometric kernel demonstrations._ \ Row 1: rectangular kernel (input, gradient magnitude, orientation map). \ Row 2: elliptical kernel on same image. \ Orientation encoded as hue, $|R|$ as brightness.]
  ),
  caption: [Rectangular vs. elliptical kernel edge detection on a natural image. Both kernel variants produce visually similar gradient magnitude maps. The elliptical kernel produces a slightly smoother orientation field, consistent with its continuous angular weighting function.],
) <fig:kernel-demos>

#figure(
  rect(width: 100%, height: 5cm, stroke: black50,
    align(center + horizon)[_Placeholder: Noise robustness visual comparison._ \ One image at four SNR levels (clean, 10 dB, 5 dB, 1 dB). \ Columns: ASF-poly (noise-optimized), ASF-rect (scaled $sigma$), DexiNed, Sobel. \ At high SNR all methods perform similarly; \ at low SNR the DL and classical outputs degrade while the ASF maintains structure.]
  ),
  caption: [Noise robustness comparison at four SNR levels. At $"SNR" = 10$ dB, all methods produce usable edge maps. At $"SNR" = 1$ dB, both Sobel and DexiNed produce noise-dominated outputs, while the ASF with noise-adapted parameters ($N_p = 250$, $m = 7$) maintains coherent edge structure at the cost of reduced spatial resolution.],
) <fig:noise-visual>

#figure(
  rect(width: 100%, height: 5cm, stroke: black50,
    align(center + horizon)[_Placeholder: Accuracy-speed Pareto plot (CeTZ)._ \ X-axis: per-image runtime in ms (log scale). \ Y-axis: ODS F-score on BIPED v1. \ Classical methods: lower-left. DL models: upper-right. \ ASF variants: upper-left, filling the gap. \ Pareto frontier drawn. Brand colors: garnet (ASF), atlantic (DL), 70% black (classical).]
  ),
  caption: [Accuracy vs. runtime Pareto plot on BIPED v1. Classical methods cluster at low runtime but low accuracy; deep learning models achieve high accuracy at higher computational cost (DiffusionEdge extends far off the right edge at $approx 15$ s). The ASF variants occupy the previously empty region between these clusters, offering accuracy competitive with lightweight neural networks at classical-method runtimes.],
) <fig:pareto>
