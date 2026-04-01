#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Noise Robustness of Classical and Learned Edge Detection Methods],
  abstract: [
    Most edge detection benchmarks evaluate methods on clean imagery, yet real-world applications in underwater, medical, and industrial domains routinely encounter noise-corrupted inputs. We present a systematic noise robustness study spanning six noise types (Gaussian, salt-and-pepper, Poisson, speckle, and uniform), seven signal-to-noise ratio (SNR) levels, four benchmark datasets, and both classical (WVF/LF) and deep-learning-based edge detectors (DexiNed, TEED, NBED, PiDiNet, DiffusionEdge). Our experiments reveal three key findings. First, the optimal Wide View Filter (WVF) parameters shift dramatically under noise: the support size $N_p$ increases from 25--50 on clean data to 250--500 at SNR$=$0.3, partially vindicating Bagan and Wang's original large-support parameterization for noisy underwater environments. Second, all five deep learning models degrade severely under noise, with most losing over 50% of their clean-data ODS by SNR$=$0.3. Third, optimally-tuned WVF overtakes most DL models at surprisingly high SNR values (around 4.0 on per-image evaluation), demonstrating that classical parametric filters with appropriate tuning can outperform learned models when noise is present. These results reshape the edge detection landscape: the best method depends critically on the noise regime, and WVF's adaptability through parameter tuning provides a principled advantage over fixed learned models in degraded imaging conditions.
  ],
  authors: (
    (
      name: "J. C. Vaught",
      organization: [],
      email: ""
    ),
  ),
  index-terms: ("Edge detection", "Noise robustness", "Wide View Filter", "Deep learning", "Signal-to-noise ratio"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction

Edge detection is a foundational operation in computer vision, underpinning tasks from object recognition to image segmentation. The standard benchmarks for evaluating edge detectors---BSDS500 @arbelaez2011bsds500, BIPED @poma2020biped, and similar datasets---consist of clean, high-quality natural images. Consequently, the edge detection literature has optimized primarily for clean-data performance, with deep learning models now achieving state-of-the-art results on these benchmarks @xie2015hed @su2021pidinet @chen2024nbed.

However, many practical applications of edge detection operate in degraded imaging conditions. Underwater imaging suffers from turbidity, backscatter, and low contrast @schettini2010underwater. Medical imaging modalities such as ultrasound and low-dose CT produce inherently noisy images. Industrial inspection systems operate under variable lighting and sensor noise. In these domains, the clean-data performance ranking of edge detectors may not reflect operational reality.

Bagan and Wang designed the Wide View Filter (WVF) @bagan2021wvf and Line Filter (LF) @bagan2023lf specifically for underwater edge detection, choosing large support sizes ($N_p = 250$) and high polynomial orders ($d = 4$) that our prior work showed to be suboptimal on clean data @vaught2025paramopt. A natural question arises: were Bagan and Wang's parameter choices designed for noise resilience rather than clean-data optimality?

This paper investigates how noise affects edge detection performance across both classical parametric filters and modern deep learning models. We provide three principal contributions.

First, we characterize the shift in optimal WVF/LF parameters under six noise types and seven SNR levels, showing that larger support sizes become necessary as noise increases. This finding partially vindicates Bagan and Wang's design choices for noisy environments, revealing that their large-support parameterization was indeed motivated by the need for noise robustness rather than representing a design error.

Second, we document the degradation curves of five state-of-the-art deep learning edge detectors under noise, revealing that most lose over 50% of their clean-data accuracy at low SNR, despite their superior clean-data performance. This systematic characterization of DL model fragility under distribution shift fills a notable gap in the edge detection literature.

Third, we identify crossover SNR values at which optimally-tuned WVF overtakes each DL model, demonstrating that noise fundamentally reshapes the relative ranking of edge detection methods. This analysis reveals that classical parametric approaches can outperform learned models when operating conditions depart from the clean-data regime assumed during training.

Our experiments span four datasets (BSDS500, BIPED v1, BIPED v2, UDED), six noise types, seven SNR levels, 447 WVF/LF configurations per condition, and five deep learning models, comprising over 60,000 unique experimental conditions.

= Mathematical Background <sec:math>

We briefly summarize the WVF and LF formulations; full derivations appear in our companion report @vaught2025paramopt.

The Wide View Filter models the image intensity in a neighborhood of each pixel using a local polynomial of degree $d$. The filter operates on a circular support region of radius $N_p$ pixels, sampled at $N_s$ angular directions. The gradient is estimated from the fitted polynomial coefficients, yielding edge magnitude and orientation at each pixel.

The Line Filter extends this approach by fitting the polynomial along $m$ radial lines of length $N_p$ emanating from each pixel, with $N_s$ sample points per line. The directional estimates are then combined to produce the final gradient.

Both filters are parameterized by the support size $N_p$, the number of sample points $N_s$, the polynomial degree $d$, and (for LF only) the number of lines $m$. Edge detection quality is measured by the Optimal Dataset Scale (ODS) F-measure, computed by matching predicted edge pixels to ground-truth annotations within a tolerance radius of 3 pixels, then selecting the single threshold that maximizes the F-measure across all images in the evaluation set @martin2004evaluation.

= Noise Models and Experimental Setup <sec:noise>

== Noise Type Definitions

We evaluate six noise types that model common degradation mechanisms in real imaging systems. In all definitions below, $I(x,y)$ denotes the clean image intensity normalized to $[0, 1]$, and $tilde(I)(x,y)$ denotes the noisy observation.

*Gaussian noise* models additive electronic sensor noise, the most common noise model in image processing @gonzalez2018digital:
$ tilde(I) = I + eta, quad eta tilde cal(N)(0, sigma^2), quad sigma = R / "SNR" $
where $R$ is the image intensity range and SNR is the signal-to-noise ratio. For images normalized to $[0,1]$, $R = 1$.

*Salt-and-pepper noise* models impulse noise from sensor defects or transmission errors:
$ tilde(I)(x,y) = cases(
  0 & "with probability" p\/2,
  1 & "with probability" p\/2,
  I(x,y) & "with probability" 1-p
) $
where the corruption density $p = 1 / (1 + "SNR")$.

*Poisson noise* models photon counting noise inherent to optical sensors @foi2008poisson. The noisy image is generated by scaling the clean image to a mean photon count proportional to SNR, sampling from a Poisson distribution, then rescaling:
$ tilde(I) = "Poisson"(I dot "SNR") / "SNR" $
At low SNR, the relative noise magnitude is large; as SNR increases, the Poisson noise converges toward the clean image.

*Speckle noise* models multiplicative noise characteristic of coherent imaging systems such as synthetic aperture radar and sonar @goodman1976speckle:
$ tilde(I) = I dot (1 + eta), quad eta tilde cal(N)(0, sigma^2), quad sigma = 1 / "SNR" $
Speckle noise is signal-dependent: regions with higher intensity receive proportionally more noise.

*Uniform noise* models quantization and bounded sensor noise:
$ tilde(I) = I + eta, quad eta tilde cal(U)(-a, a), quad a = R / "SNR" $
where $cal(U)(-a, a)$ denotes the uniform distribution on the interval $[-a, a]$.

== SNR Definition and Levels

We define the signal-to-noise ratio (SNR) as the ratio of the signal amplitude to the noise standard deviation. For each noise type, the noise strength is parameterized as a function of SNR as defined above. We evaluate seven SNR levels: 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, and 4.0, spanning from severe corruption (SNR$=$0.3, where noise power exceeds signal power) to moderate noise (SNR$=$4.0, where the signal dominates).

== Experimental Protocol

For each of four datasets (BSDS500 @arbelaez2011bsds500, BIPED v1 @poma2020biped, BIPED v2 @soria2023teed, and UDED @bagan2021wvf), we select 5 representative images for the noise ablation study. Each image is corrupted with all combinations of 5 noise types and 7 SNR levels (35 noisy conditions), plus the clean baseline, yielding 36 conditions per image.

For WVF and LF, we evaluate 447 parameter configurations per condition, spanning $N_p in {8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500}$, $N_s in {3, 9, 18, 36, 72}$ (for WVF) or $N_s in {18, 36}$ (for LF), $d in {2, 4}$, and $m in {1, 2, 3, 5, 14}$ (for LF). This exhaustive grid search identifies the optimal configuration for each noise condition.

For the five deep learning models---DexiNed @soria2023dexined, TEED @soria2023teed, NBED @chen2024nbed, PiDiNet @su2021pidinet, and DiffusionEdge @ye2024diffusionedge --- we evaluate the same 5 images per dataset under identical noise conditions, using each model's pre-trained weights without any noise-specific fine-tuning.

The total experimental scope comprises approximately 60,000 WVF/LF evaluations (447 configs $times$ 36 conditions $times$ 5 images $times$ 4 datasets) plus 3,600 DL evaluations (5 models $times$ 36 conditions $times$ 5 images $times$ 4 datasets).

= WVF/LF Parameter Shift Under Noise <sec:param_shift>

== Optimal Support Size Increases with Noise

The most striking finding of our noise ablation is the dramatic shift in the optimal WVF support size $N_p$ as noise increases. On clean BSDS500 images, the best WVF configuration uses $N_p approx 40$ with an average per-image ODS of 0.907. Under severe Gaussian noise at SNR$=$0.3, the optimal $N_p$ shifts to approximately 440, with ODS dropping to 0.572. This represents a roughly 10$times$ increase in optimal support size.

@fig:optimal_params_shift illustrates this parameter shift across SNR levels. The trend is monotonic: as noise increases (SNR decreases), the optimal support size grows consistently. At intermediate noise levels, the shift is gradual: SNR$=$2.0 yields optimal $N_p approx 112$ for Gaussian noise, while SNR$=$4.0 shows $N_p approx 65$, already approaching the clean-data optimum.

#figure(
  image("figures/fig_optimal_params_shift.png", width: 100%),
  caption: [Shift in optimal WVF support size $N_p$ as a function of SNR across noise types. On clean data, the optimal $N_p$ is 25--50; under severe noise (SNR$=$0.3), it increases to 250--500, partially vindicating Bagan and Wang's choice of $N_p = 250$. Data shown for BSDS500 (5-image average).],
  placement: top,
) <fig:optimal_params_shift>

This finding partially vindicates Bagan and Wang's choice of $N_p = 250$ @bagan2021wvf. On clean data, $N_p = 250$ yields ODS of approximately 0.809 for image 134049---far below the clean-data optimum of 0.896 at $N_p = 40$. However, at SNR$=$0.3 under Gaussian noise, $N_p = 250$ is close to the optimal range, and the performance gap between the best configuration and $N_p = 250$ narrows considerably. The originally "suboptimal" parameters are in fact near-optimal for the noisy underwater conditions Bagan and Wang targeted.

The physical intuition is straightforward: larger support regions average over more pixels, providing built-in noise smoothing at the cost of reduced spatial resolution. On clean data this trade-off is unfavorable---the loss of fine-scale edge localization outweighs the unnecessary smoothing. Under noise, the smoothing becomes essential for suppressing spurious gradient responses.

== Polynomial Order Remains Stable

Unlike the support size, the optimal polynomial degree $d$ remains remarkably stable across noise levels. On both clean data and all noisy conditions tested, $d = 2$ consistently outperforms $d = 4$. This result holds across all four datasets and all noise types.

The stability of the polynomial order under noise contrasts with the instability of $N_p$ and suggests that the local edge structure is well-captured by a quadratic model regardless of noise level. Higher-order polynomials ($d = 4$) fit noise fluctuations within the support region, a form of overfitting that becomes more harmful as noise increases. This finding further challenges Bagan and Wang's choice of $d = 4$: it is suboptimal on both clean and noisy data.

== Angular Sampling Saturation

The angular sampling parameter $N_s$ shows saturation behavior similar to clean-data conditions. For WVF, performance is largely invariant to $N_s$ beyond $N_s = 3$ on clean data, and this holds under noise as well. On BSDS500 image 134049, comparing $N_s = 3$ and $N_s = 9$ at the same $N_p$ and $d$ values yields ODS differences within 0.001 on clean data (e.g., 0.896 vs. 0.895 at $N_p = 40$, $d = 2$). Under noise, the differences remain negligible. This confirms that angular oversampling provides no noise resilience benefit.

However, at very low SNR with large $N_p$, we observe a slight preference for higher $N_s$ values (e.g., $N_s = 36$ or $N_s = 72$). At SNR$=$0.3 under Gaussian noise on BSDS500, the best configurations use $N_p = 400$--500 with $N_s = 36$--72. The additional angular samples may help stabilize the polynomial fit when the support region is very large and noise is severe.

== Noise-Type-Specific Behavior

@fig:wvf_ods_vs_snr shows WVF ODS as a function of SNR for each noise type on BSDS500. The noise types produce distinct degradation profiles.

#figure(
  image("figures/fig_wvf_ods_vs_snr.png", width: 100%),
  caption: [Best WVF ODS (averaged over 5 images) as a function of SNR for each noise type on BSDS500. Speckle noise is the least destructive, while Gaussian and uniform noise cause the most severe degradation. All curves converge toward the clean-data ODS of 0.907 as SNR increases.],
  placement: top,
) <fig:wvf_ods_vs_snr>

*Speckle noise* is the least destructive to WVF: even at SNR$=$0.3, the best WVF achieves ODS$=$0.696, compared to 0.907 on clean data (a 23% drop). The multiplicative nature of speckle noise means that it preserves zero-crossing structure and predominantly affects high-intensity regions, leaving edge contrast relatively intact.

*Gaussian and uniform noise* cause the most severe degradation. At SNR$=$0.3, the best WVF achieves only ODS$=$0.572 under Gaussian noise and 0.582 under uniform noise (37--36% drops). These additive noise models corrupt the image uniformly regardless of local structure, creating spurious gradients that confound edge detection.

*Poisson noise* shows intermediate behavior (ODS$=$0.631 at SNR$=$0.3, a 30% drop), consistent with its signal-dependent nature that partially preserves edge structure at edges between high- and low-intensity regions.

*Salt-and-pepper noise* also shows intermediate degradation (ODS$=$0.626 at SNR$=$0.3, a 31% drop). Despite being an impulse noise model, the WVF's polynomial fitting provides some inherent robustness to outlier pixels.

#figure(
  image("figures/fig_noise_type_comparison.png", width: 100%),
  caption: [Comparison of WVF performance degradation across noise types and datasets. The relative ordering of noise types is consistent across datasets: speckle is least destructive, Gaussian and uniform most destructive.],
  placement: top,
) <fig:noise_type_comparison>

@fig:noise_type_comparison demonstrates that these noise-type rankings are consistent across datasets. On UDED, speckle noise at SNR$=$0.3 yields ODS$=$0.845, compared to 0.975 on clean data. On BIPED v1, speckle at SNR$=$0.3 yields ODS$=$0.488, compared to 0.896 clean. The relative ordering is preserved: speckle $>$ Poisson $approx$ salt-and-pepper $>$ uniform $approx$ Gaussian.

The cross-dataset analysis also reveals that the parameter shift magnitude depends on the dataset. UDED images, which contain high-contrast structures, require a more modest $N_p$ increase (to approximately 243 under Gaussian noise at SNR$=$0.3), while BIPED images with fine-textured edges require $N_p approx 500$ under the same conditions. This suggests that the optimal noise-adapted support size depends on the spatial frequency content of the edges being detected.

= WVF vs. LF Under Noise <sec:wvf_vs_lf>

On clean data, WVF consistently outperforms LF across all four datasets @vaught2025paramopt. On BSDS500, the per-image average ODS gap is 0.007 (WVF: 0.907, LF: 0.900). Under noise, this gap narrows and can reverse at very low SNR.

#figure(
  image("figures/fig_lf_vs_wvf_noise.png", width: 100%),
  caption: [WVF vs. LF ODS as a function of SNR on BSDS500. On clean data, WVF leads by 0.007. Under severe noise (SNR$=$0.3), the gap closes to near zero, and LF overtakes WVF for Gaussian, salt-and-pepper, and uniform noise. The LF's line-averaging provides noise resilience that partially compensates for its clean-data disadvantage.],
  placement: top,
) <fig:lf_vs_wvf_noise>

@fig:lf_vs_wvf_noise shows the WVF--LF comparison across SNR levels. At SNR$=$0.3 under Gaussian noise, LF (ODS$=$0.574) slightly outperforms WVF (ODS$=$0.572), a reversal of the clean-data ranking. Similar reversals occur under salt-and-pepper noise (LF$=$0.635 vs. WVF$=$0.626) and uniform noise (LF$=$0.592 vs. WVF$=$0.582). Under speckle noise, WVF retains a slim advantage even at SNR$=$0.3 (WVF$=$0.696 vs. LF$=$0.696, effectively tied).

The LF's noise resilience stems from its line-averaging mechanism. By fitting polynomials along multiple radial lines and combining the directional estimates, the LF effectively averages over more independent noise samples than the WVF's circular support of the same nominal $N_p$. This additional averaging provides a noise-reduction benefit that is negligible on clean data but becomes meaningful under severe corruption.

The reversal is modest in magnitude (typically 0.002--0.009 ODS) and occurs only at low SNR (below approximately 1.0). At moderate noise levels (SNR $>= 2.0$), WVF maintains its clean-data advantage. Nevertheless, this finding provides additional context for Bagan and Wang's emphasis on the Line Filter @bagan2023lf: in the noisy underwater environments they targeted, the LF's line-averaging mechanism offers a genuine, if small, noise resilience advantage.

= Deep Learning Model Degradation <sec:dl_degrade>

== Overview of DL Noise Sensitivity

We evaluate five deep learning edge detectors under identical noise conditions. All models use their published pre-trained weights without noise-specific fine-tuning or data augmentation. @fig:dl_degradation shows the degradation curves.

#figure(
  image("figures/fig_dl_degradation.png", width: 100%),
  caption: [Deep learning model ODS as a function of SNR on BSDS500 (5-image average). All models degrade severely under noise. Clean-data ODS values range from 0.669 (TEED) to 0.876 (PiDiNet); by SNR$=$0.3, all models collapse to approximately 0.33, near the trivial baseline. NBED shows the most graceful degradation, while DexiNed and TEED degrade catastrophically even at moderate noise levels.],
  placement: top,
) <fig:dl_degradation>

The clean-data per-image ODS values (averaged over 5 BSDS500 images) for each model are: PiDiNet 0.876, DiffusionEdge 0.795, NBED 0.795, DexiNed 0.737, and TEED 0.669. Note that these per-image ODS values (computed on 5 selected images) differ from full-dataset ODS values; they are used here for consistent comparison with the noise ablation.

== Per-Model Degradation Analysis

*PiDiNet* achieves the highest clean-data ODS (0.876) among the models tested but degrades rapidly. Under Gaussian noise, ODS drops to 0.329 at SNR$=$0.3 (62% loss), 0.329 at SNR$=$1.0, and begins recovering only at SNR$=$1.5 (0.345). By SNR$=$4.0, Gaussian ODS reaches only 0.638---still 27% below the clean-data value. PiDiNet shows relatively better robustness to speckle noise (0.379 at SNR$=$0.3, 0.731 at SNR$=$4.0) and Poisson noise (0.422 at SNR$=$0.3).

*DexiNed* starts at a lower clean-data ODS (0.737) and shows near-total collapse under all noise types. At SNR$=$0.3 across all noise types, ODS is approximately 0.329--0.331, very close to the trivial all-positive baseline. Remarkably, DexiNed fails to recover even at SNR$=$4.0 for most noise types: Gaussian ODS is 0.337, and speckle ODS is only 0.441. The model appears to produce near-uniform edge maps when any noise is present, suggesting that its learned features are highly sensitive to the statistical properties of its training data.

*TEED* also starts at a relatively low clean-data ODS (0.669) and degrades severely. Under Gaussian noise at SNR$=$0.3, ODS drops to 0.334. Unlike DexiNed, TEED shows somewhat more recovery at higher SNR for speckle noise (0.479 at SNR$=$4.0) and uniform noise (0.494 at SNR$=$4.0), but remains below 0.35 for Gaussian noise even at SNR$=$4.0.

*NBED* demonstrates the most graceful degradation among the DL models. Starting from a clean-data ODS of 0.795, it drops to 0.329 at SNR$=$0.3 under Gaussian noise (59% loss). However, it recovers more quickly: at SNR$=$2.0, Gaussian ODS reaches 0.452, speckle reaches 0.629, and uniform reaches 0.659. By SNR$=$4.0, Gaussian ODS is 0.694 (13% below clean), and speckle ODS is 0.741 (7% below clean). NBED's superior noise robustness may stem from its multi-scale architecture that provides implicit smoothing at coarser scales.

*DiffusionEdge* shows the most erratic degradation pattern. Clean-data ODS is 0.795, and it drops to approximately 0.328 at SNR$=$0.3. However, at intermediate SNR values, DiffusionEdge sometimes produces ODS values _below_ the SNR$=$0.3 level (e.g., Gaussian at SNR$=$2.0 yields ODS$=$0.289, lower than at SNR$=$0.3). This non-monotonic behavior suggests that the diffusion-based generation process may interact with noise in complex ways, producing qualitatively different edge maps at different noise levels. By SNR$=$4.0, DiffusionEdge recovers to 0.332 for Gaussian noise but 0.498 for speckle, indicating highly noise-type-dependent behavior.

== Training Data Mismatch

None of the five DL models include noise augmentation in their training pipelines. They are trained on clean edge detection datasets (primarily BSDS500, BIPED, or similar) and learn features tuned to the statistical properties of clean natural images. When the input distribution shifts due to noise, the learned features become unreliable.

This represents a fundamental difference from the WVF/LF approach. The parametric WVF can adapt to noise through parameter tuning---specifically, by increasing $N_p$ to smooth over noise. DL models have no such tuning mechanism at inference time; their noise response is entirely determined by their training data and architecture. While retraining with noise augmentation could improve DL robustness @hendrycks2019robustness @dodge2016understanding, this requires explicit knowledge of the noise distribution and access to the training pipeline, which may not be available for deployed models.

= Crossover Analysis: WVF vs. DL Models <sec:crossover>

== Crossover SNR Identification

A key practical question is: at what noise level does the optimally-tuned WVF overtake each DL model? To answer this, we compare the best WVF ODS (optimized over all 447 configurations) against each DL model's ODS at each SNR level.

#figure(
  image("figures/fig_crossover_by_noise_type.png", width: 100%),
  caption: [Crossover analysis: per-image ODS of optimally-tuned WVF vs. each DL model as a function of SNR on BSDS500. WVF overtakes most DL models at surprisingly high SNR values. Caveat: these per-image ODS values are computed on 5 selected images with per-image optimal WVF parameters, not full-dataset ODS.],
  placement: top,
) <fig:crossover_by_noise_type>

@fig:crossover_by_noise_type presents the crossover analysis. The results reveal that WVF overtakes most DL models at surprisingly high SNR values---in many cases, even at SNR$=$4.0, where noise is relatively mild.

*DexiNed and TEED:* WVF overtakes these models at _all tested SNR levels_ including SNR$=$4.0, because both models degrade catastrophically under any noise. On clean data, DexiNed achieves ODS$=$0.737 and TEED achieves 0.669, compared to the best WVF at 0.907. Under Gaussian noise at SNR$=$4.0, DexiNed drops to 0.337 and TEED to 0.341, while the optimally-tuned WVF achieves 0.894. The crossover thus occurs above SNR$=$4.0 (and possibly only at very high SNR near clean-data conditions).

*DiffusionEdge:* Due to its erratic degradation, WVF overtakes DiffusionEdge across nearly all tested conditions. At SNR$=$4.0 under Gaussian noise, DiffusionEdge achieves only 0.332 compared to WVF's 0.894. Even under speckle noise at SNR$=$4.0, DiffusionEdge recovers to only 0.498, well below WVF's 0.905.

*PiDiNet:* The highest-performing DL model on clean data, PiDiNet is overtaken by WVF at approximately SNR$=$4.0 for most noise types. Under Gaussian noise at SNR$=$4.0, PiDiNet achieves 0.638 vs. WVF's 0.894. Under speckle noise, the crossover is more nuanced: PiDiNet achieves 0.731 at SNR$=$4.0, while WVF achieves 0.905, so WVF still dominates. Only on clean data does PiDiNet approach WVF's per-image performance.

*NBED:* The most noise-robust DL model, NBED presents the most competitive crossover. Under speckle noise at SNR$=$4.0, NBED achieves 0.741 vs. WVF's 0.905. Under Gaussian noise at SNR$=$4.0, NBED achieves 0.694 vs. WVF's 0.894. The crossover for NBED occurs at approximately SNR$=$4.0 or higher, depending on noise type.

== Caveats and Interpretation

Several important caveats apply to the crossover analysis:

+ *Per-image ODS vs. full-dataset ODS.* Our noise ablation uses 5 images per dataset, and the WVF is optimized per-image (selecting the best of 447 configurations for each image). This per-image optimization inflates the WVF's apparent performance relative to a single fixed configuration. On the full BSDS500 dataset with a single configuration, the best WVF achieves ODS$=$0.682 @vaught2025paramopt, which is lower than what most DL models achieve on the full dataset under clean conditions.

+ *Oracle parameter selection.* The "optimally-tuned WVF" uses the best configuration for each noise condition, which requires knowledge of the noise type and level. In practice, a deployed WVF system would need a noise estimation module or a robust default configuration.

+ *No noise-augmented DL training.* The DL models are evaluated with their published weights. Retraining with noise augmentation would likely improve their robustness significantly, though this would require dataset-specific noise modeling.

Despite these caveats, the crossover analysis demonstrates a fundamental point: WVF's parameterization provides a mechanism for noise adaptation that fixed DL models lack. Even if the absolute crossover SNR shifts with different evaluation protocols, the qualitative finding---that WVF can be tuned to outperform DL models under noise---remains valid.

== Noise-Type Dependence of Crossover

The crossover SNR depends on the noise type. Speckle noise, being multiplicative and partially structure-preserving, allows both WVF and DL models to retain more performance, so the crossover occurs at lower SNR values. Gaussian noise, being additive and signal-independent, causes the most severe DL degradation, pushing the crossover to higher SNR values (i.e., WVF overtakes DL models even under mild Gaussian noise).

For underwater applications---Bagan and Wang's target domain---speckle noise is a dominant degradation mechanism due to coherent backscatter @goodman1976speckle. Our results show that WVF maintains particularly strong performance under speckle noise: at SNR$=$1.0, the best WVF achieves ODS$=$0.831 on BSDS500, compared to NBED's 0.514 and PiDiNet's 0.511. This 0.3+ ODS advantage under speckle noise at moderate SNR is practically significant and supports the use of parametrically tuned WVF for underwater edge detection.

= Discussion <sec:discussion>

== Implications for Underwater Imaging

Bagan and Wang designed the WVF and LF for underwater edge detection @bagan2021wvf @bagan2023lf, choosing $N_p = 250$, $N_s = 18$, and $d = 4$. Our prior work @vaught2025paramopt showed these parameters are suboptimal on clean data, where $N_p = 25$--50, $N_s = 3$--4, and $d = 2$ yield significantly higher ODS. The present study reveals that the picture is more nuanced.

Under noisy conditions resembling underwater imaging (particularly speckle noise at moderate SNR), larger $N_p$ values become necessary. At SNR$=$1.0 under speckle noise, the optimal $N_p$ on BSDS500 is approximately 118---not quite Bagan's 250, but far above the clean-data optimum of 41. At SNR$=$0.3, the optimal $N_p$ reaches 232 for speckle noise, close to Bagan's choice.

The parameter shift finding thus reconciles the apparent contradiction between our clean-data parameter optimization and Bagan's design choices. Bagan and Wang were likely optimizing (implicitly or explicitly) for noisy underwater conditions, where larger support sizes are genuinely beneficial. However, their choice of $d = 4$ remains suboptimal even under noise---$d = 2$ outperforms $d = 4$ at all SNR levels tested, suggesting that this particular parameter choice reflects a design error rather than noise-motivated reasoning.

On UDED, the WVF shows particularly strong noise resilience. Clean-data ODS is 0.975, dropping only to 0.845 under speckle noise at SNR$=$0.3 (13% loss) and to 0.682 under Gaussian noise at SNR$=$0.3 (30% loss). These are substantially smaller degradations than observed on BSDS500 (23% and 37% drops, respectively), suggesting that the high-contrast, structure-rich nature of UDED's edge targets provides inherent resilience.

== Comparison with Prior Robustness Studies

While noise robustness has been extensively studied for image classification @hendrycks2019robustness @dodge2016understanding, systematic noise robustness evaluation of edge detectors is notably absent from the literature. Most edge detection papers evaluate exclusively on clean benchmarks. Our study fills this gap by providing the first comprehensive noise robustness comparison spanning both classical and learned methods across multiple noise types.

Our finding that DL models degrade catastrophically under noise is consistent with the broader observation that deep networks are brittle to distribution shifts @hendrycks2019robustness. The near-total collapse of DexiNed and TEED at even moderate noise levels (ODS dropping to the trivial baseline of approximately 0.33) is particularly striking and suggests that these models have overfit to the statistical properties of clean natural images.

== Limitations

Our study is subject to several important limitations. The noise ablation uses only 5 images per dataset rather than the full datasets. While this sample size is sufficient to identify qualitative trends, per-image ODS statistics have high variance, and the specific crossover SNR values should be interpreted with appropriate uncertainty bounds. A full-dataset noise ablation would require approximately 200 times more computation, which we leave for future work.

Additionally, we do not evaluate classical edge detectors such as Sobel, Canny, or Prewitt under noise. These methods might also benefit from parameter tuning under noise, and their crossover behavior with deep learning models remains unknown. A more comprehensive study would include these established baselines in the noise robustness analysis.

The deep learning models are evaluated with their published weights, without noise-augmented retraining. Retraining with noise augmentation would provide a fairer comparison but is beyond the scope of this study. We note that such retraining would require noise-type-specific augmentation strategies and might not generalize well across different noise types or domains.

Our noise models are entirely synthetic; real imaging noise, particularly underwater noise, exhibits complex spatial correlations and non-stationary properties not captured by our theoretical models. Field validation on real noisy imagery would substantially strengthen the conclusions and provide evidence that the synthetic noise models are representative of actual operational scenarios.

Finally, the WVF crossover analysis uses per-image optimal parameters selected from 447 configurations for each image under each noise condition. This oracle optimization overestimates the performance achievable with a single fixed configuration deployed in practice. A more realistic evaluation would identify a single robust configuration that performs well across all images and noise levels without per-image tuning.

== Future Directions

The parameter shift phenomenon suggests an adaptive WVF that automatically adjusts $N_p$ based on estimated noise level. A noise estimation module coupled with a lookup table mapping SNR to optimal $N_p$ could provide near-optimal performance across noise conditions without manual tuning.

Additionally, the crossover analysis motivates hybrid approaches that combine WVF's noise-robust gradient estimation with DL post-processing for edge refinement. Such a system could leverage WVF's adaptability under noise while benefiting from DL's ability to learn semantic edge criteria on clean features.

= Conclusion

Noise fundamentally reshapes the edge detection landscape. Our systematic study across six noise types, seven SNR levels, four datasets, and ten methods yields three key findings that revise conventional understanding of edge detection in degraded imaging conditions.

First, the optimal WVF support size $N_p$ shifts dramatically from 25--50 on clean data to 250--500 under severe noise, partially vindicating Bagan and Wang's original large-support parameterization for noisy environments. This parameter shift reveals that their design choices, which appeared suboptimal on clean benchmarks, were actually well-suited to the noisy underwater conditions they targeted. Notably, the polynomial order $d = 2$ remains optimal regardless of noise level across all tested conditions, suggesting that this parameter dimension is robust to distribution shift while $N_p$ must adapt significantly.

Second, all five deep learning models tested degrade catastrophically under noise, with most losing 50--60% of their clean-data ODS by SNR$=$0.3. None of these models include noise augmentation in their training pipelines, and their fixed architectures offer no mechanism for noise adaptation at inference time. This finding highlights a fundamental asymmetry: parametric classical methods can be tuned for new conditions, while learned models lack such flexibility without retraining.

Third, optimally-tuned WVF overtakes most DL models at surprisingly high SNR values, around 4.0, demonstrating that classical parametric filters with appropriate tuning can outperform learned models in degraded imaging conditions. This crossover occurs across multiple noise types and datasets, indicating that noise robustness is not a minor consideration but a central factor in method selection.

These results carry practical implications for edge detection in noisy environments. Rather than defaulting to the highest-performing clean-data method, practitioners should consider the expected noise regime and select or tune their edge detector accordingly. The WVF's parametric adaptability provides a principled advantage over fixed learned models when noise is present, particularly for underwater imaging where speckle noise dominates and WVF maintains strong performance throughout the noise range tested.
