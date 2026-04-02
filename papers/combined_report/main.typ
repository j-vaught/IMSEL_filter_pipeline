#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [A Comprehensive Evaluation of Wide View and Line Filters for Edge Detection: Parameter Optimization, Benchmarking, and Noise Robustness],
  abstract: [
    The Wide View Filter (WVF) and Line Filter (LF), training-free edge detectors based on local polynomial fitting, were originally developed for maritime imaging with parameters $N_p = 250$, $N_s = 18$, and $d = 4$. We present a four-part investigation comprising parameter ablation, benchmarking, noise robustness analysis, and statistical validation across four datasets totaling 330 images. Our ablation of 1,206 WVF and 168 LF configurations reveals that published parameters are substantially suboptimal on clean imagery: reducing the support size to $N_p = 25$--$100$, polynomial order to $d = 2$, and orientation count to $N_s = 4$ yields ODS improvements of 0.01--0.05 for the WVF and 0.06--0.16 for the LF, with a $112 times$ reduction in computational cost. The optimally-tuned WVF outperforms all 54 classical edge detection baselines tested by 2.4 ODS points and runs $3.3 times$ faster than the fastest deep learning (DL) model. Under noise, however, the optimal $N_p$ shifts dramatically to 250--500, partially vindicating Bagan and Wang's original large-support parameterization for noisy environments. All five DL models tested lose 50--60\% of their clean-data accuracy under severe noise, and WVF overtakes most DL models at surprisingly mild noise levels. Nonparametric statistical tests confirm these findings: cross-dataset ranking concordance is moderate ($W = 0.647$) but BSDS500 is a strong outlier ($rho = 0.03$--$0.17$ with other datasets), WVF and LF are statistically indistinguishable at matched parameters ($p > 0.8$), and the LF line half-width $m$ is the most influential parameter ($eta^2 = 0.34$). These results establish that edge detection method selection is fundamentally condition-dependent and that parametric adaptability provides a principled advantage over fixed learned models in degraded imaging conditions.
  ],
  authors: (
    (
      name: "J. C. Vaught",
      organization: [],
      email: ""
    ),
  ),
  index-terms: ("Edge detection", "Wide View Filter", "Line Filter", "Parameter optimization", "Noise robustness", "Deep learning", "Benchmarking", "Statistical analysis"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

// ============================================================
// I. INTRODUCTION
// ============================================================

= Introduction <sec:intro>

Edge detection is a foundational operation in computer vision, serving as a prerequisite for object recognition, image segmentation, and scene understanding. The ability to extract meaningful boundaries from images underpins virtually every mid-level and high-level vision task, from contour grouping and figure-ground separation to semantic segmentation and 3D reconstruction. Despite decades of research, the edge detection problem remains challenging because it requires balancing conflicting objectives: sensitivity to true object boundaries, suppression of spurious texture responses, robustness to noise and illumination variation, and computational efficiency for real-time deployment.

The field has evolved through three generations of methods. First-generation classical operators such as Sobel @sobel2014history, Prewitt @prewitt1970gradient, Roberts @roberts1963machine, and the Laplacian of Gaussian @marr1980log compute local intensity gradients using fixed-size convolution kernels. These operators remain widely deployed due to their simplicity, deterministic behavior, and negligible computational cost. However, their fixed spatial support limits their ability to adapt to varying edge scales and noise conditions, and their purely local computation cannot capture the contextual information needed to distinguish semantic boundaries from texture edges. Second-generation methods, exemplified by Canny's detector @canny1986edge, add non-maximum suppression and hysteresis thresholding to produce thin, connected edges, representing a significant advance in edge quality but still operating on local gradient information.

Third-generation deep learning approaches have fundamentally changed the edge detection landscape. Beginning with Structured Forests @dollar2013structured and Holistically-Nested Edge Detection (HED) @xie2015hed, and continuing through modern architectures such as DexiNed @poma2020dexined, TEED @soria2023teed, PiDiNet @su2021pidinet, NBED @chen2024nbed, and DiffusionEdge @ye2024diffusionedge, these methods learn edge representations directly from annotated data and currently achieve the highest benchmark scores on clean imagery. These models leverage multi-scale feature extraction, deeply supervised training, and increasingly sophisticated architectures---from inception-style blocks in DexiNed to diffusion probabilistic models in DiffusionEdge---to capture both local gradient information and global semantic context. However, they require large annotated training sets, substantial computational resources, and their behavior is largely opaque, making it difficult to predict how they will perform outside their training distribution.

Between classical simplicity and deep learning complexity sits a less-explored category: training-free filters with extended spatial support. Bagan and Wang's Wide View Filter (WVF) @bagan2021wvf and Line Filter (LF) @bagan2023lf belong to this category. Both fit local polynomial models to pixel neighborhoods that are larger than traditional $3 times 3$ or $5 times 5$ kernels but smaller than full-image receptive fields. The gradient is then derived analytically from the fitted polynomial, yielding an edge map without any learned parameters. The original publications recommend specific parameter settings---support size $N_p = 250$, orientation count $N_s = 18$, and polynomial order $d = 4$---derived from maritime imaging applications where significant noise, low contrast, and color distortion are prevalent @bagan2024multiscale @schettini2010underwater. In subsequent work, Bagan and Wang extended these ideas to multi-scale and multi-domain edge determination, further exploring the maritime imaging context @bagan2024multiscale.

A natural question arises: are these parameters also optimal for standard, clean-imagery benchmarks commonly used in the edge detection literature? If not, what parameters are optimal, and by how much do they improve performance? Furthermore, how do optimally-tuned WVF and LF compare against both classical operators and modern deep learning models? And critically, does the noise robustness of these extended-support filters provide an advantage over learned models in degraded imaging conditions---conditions that are routine in maritime @schettini2010underwater, medical, and industrial imaging but largely absent from standard benchmarks?

These questions are important because the edge detection literature has developed a somewhat misleading narrative. Standard benchmarks like BSDS500 @arbelaez2011bsds500 @martin2001bsds evaluate methods exclusively on clean, high-quality natural images, leading to the impression that deep learning methods are universally superior. Yet many practical applications operate in degraded imaging conditions where the clean-data performance ranking may not reflect operational reality. The robustness of deep neural networks to common corruptions and perturbations has been extensively studied in the classification literature @hendrycks2019robustness @dodge2016understanding, but systematic noise robustness evaluation of edge detectors is notably absent from the literature.

This paper addresses these questions through a comprehensive four-part investigation. Our contributions are as follows. First, we conduct a systematic parameter ablation spanning 1,206 WVF and 168 LF configurations across four diverse datasets totaling 330 images, demonstrating that the published parameters are substantially suboptimal on clean imagery and identifying configurations that yield both higher accuracy and $112 times$ computational savings. Second, we benchmark the optimally-tuned WVF against 54 classical edge detection configurations spanning seven operator families and 5 state-of-the-art deep learning models, establishing that WVF outperforms every classical method tested while running $3.3 times$ faster than the fastest neural baseline. Third, we characterize noise robustness across five noise types and seven signal-to-noise ratio (SNR) levels, revealing that the optimal WVF parameters shift dramatically under noise---partially vindicating Bagan and Wang's design choices---and that all DL models degrade catastrophically while WVF maintains adaptability through parameter tuning. Fourth, we apply a battery of nonparametric statistical tests---including Kendall's $W$, Spearman rank correlation, Wilcoxon signed-rank, sign tests, Kruskal--Wallis analysis, and bootstrap confidence intervals---to validate every comparative claim, quantifying effect sizes and acknowledging the limitations imposed by a four-dataset design.

The total experimental scope exceeds 500,000 individual filter evaluations across the parameter ablation and noise robustness studies, comprising one of the most comprehensive evaluations of any edge detection method to date. The results reveal that edge detection method selection is fundamentally condition-dependent: the best method on clean benchmarks is not the best method under noise, and benchmark results on the most widely used dataset (BSDS500) do not generalize to other imaging domains.

The remainder of this paper is organized as follows. @sec:math presents the mathematical formulations of the WVF and LF, the evaluation metrics, and the noise models used throughout the study. @sec:setup describes the experimental setup, including datasets, parameter grids, baselines, and evaluation protocols. @sec:ablation presents the parameter ablation results on both single images and full datasets. @sec:benchmark reports the benchmarking results against classical and learned methods, including computational cost analysis. @sec:noise examines noise robustness, parameter shifts under noise, and crossover analysis between WVF and DL models. @sec:stats provides the statistical validation of all major claims. @sec:discussion synthesizes the findings and offers practical recommendations, and @sec:conclusion presents the definitive conclusions.


// ============================================================
// II. MATHEMATICAL BACKGROUND
// ============================================================

= Mathematical Background <sec:math>

This section presents the WVF and LF formulations with the ablation parameters explicitly identified, the evaluation metrics used throughout the study, and the noise models employed in the robustness analysis.

== Wide View Filter Formulation <sec:wvf_math>

For a grayscale image $I$ and a target pixel $bold(p)$, the WVF selects the $N_p$ nearest integer-coordinate pixels within a circular neighborhood, excluding the origin. At each candidate orientation $theta_k in {0, 2 pi \/ N_s, dots, 2 pi (N_s - 1) \/ N_s}$, the neighbor positions are rotated into a local coordinate frame via

$ bold(q)_i^((k)) = mat(cos theta_k, sin theta_k; -sin theta_k, cos theta_k) (bold(r)_i - bold(p)), quad i = 1, dots, N_p $ <eq:rotation>

where $bold(r)_i$ are the global neighbor positions sorted by Euclidean distance from $bold(p)$.

A design matrix $A in bb(R)^(N_p times M)$ is constructed from the 2D Taylor monomials up to total degree $d$, where $M = (d+1)(d+2) \/ 2$ is the number of monomial terms. For $d = 2$, $M = 6$; for $d = 4$, $M = 15$. The columns of $A$ correspond to the normalized monomials

$ 1, quad x, quad y, quad x^2\/2, quad y^2\/2, quad x y, quad dots $ <eq:monomials>

evaluated at each neighbor's local coordinates $(x_i, y_i) = bold(q)_i^((k))$. The full set of monomials for degree $d$ includes all terms $x^a y^b \/ (a! b!)$ with $a + b <= d$.

The derivative coefficients are recovered by ordinary least squares:

$ hat(bold(c))^((k)) = (A^top A)^(-1) A^top bold(f) in bb(R)^M $ <eq:lstsq>

where $bold(f) = [I(bold(r)_1), dots, I(bold(r)_(N_p))]^top$ is the vector of neighbor intensities. The coefficient $hat(c)_2^((k))$ corresponds to the first derivative in the $x$-direction (normal to the candidate edge orientation $theta_k$), while $hat(c)_3^((k))$ corresponds to the tangential derivative.

The WVF edge response at pixel $bold(p)$ is computed as the maximum absolute normal derivative over all orientations:

$ R_"WVF"(bold(p)) = max_(k = 1, dots, N_s) |hat(c)_2^((k))| $ <eq:wvf_response>

with the estimated edge orientation given by

$ theta^*(bold(p)) = theta_(arg max_k |hat(c)_2^((k))|) $ <eq:wvf_orientation>

The three parameters varied in our ablation are $N_p$, which controls the spatial extent of the neighborhood and the row dimension of $A$; $N_s$, which controls the angular resolution of the orientation sweep; and $d$, which controls the column dimension $M$ and the expressiveness of the local polynomial model. The system in @eq:lstsq requires $N_p >= M$, which imposes a minimum support size that grows with polynomial order.

== Line Filter Formulation <sec:lf_math>

The LF extends the WVF by chaining $2m + 1$ WVF applications along a line centered on $bold(p)$ at each candidate orientation $theta_k$. Virtual expansion points are placed at

$ bold(v)_j^((k)) = bold(p) + j (cos theta_k, sin theta_k), quad j = -m, dots, m $ <eq:virtual_points>

Each virtual point $bold(v)_j^((k))$ produces a WVF normal-derivative estimate $hat(c)_(2,j)^((k))$ via @eq:lstsq applied to the neighborhood of $bold(v)_j^((k))$. These estimates are combined via a Gaussian-weighted average:

$ R_"LF"^((k))(bold(p)) = lr(|sum_(j = -m)^(m) w_j hat(c)_(2,j)^((k))|) $ <eq:lf_response>

where the weights are

$ w_j = exp(-j^2 \/ 2 sigma^2) slash sum_(ell=-m)^(m) exp(-ell^2 \/ 2 sigma^2) $ <eq:lf_weights>

and $sigma$ controls the Gaussian bandwidth. The LF output is the orientation-maximized response, analogous to @eq:wvf_response. In addition to $N_p$, $N_s$, and $d$, the LF introduces the half-width $m$ as an additional parameter. Following Bagan and Wang, we fix $sigma = m\/2$ throughout this study and vary $m$ explicitly. The line extension is motivated by the hypothesis that averaging derivative estimates along the edge tangent direction suppresses noise while preserving the edge signal @bagan2023lf. The computational cost of the LF scales as $O((2m+1) dot N_p dot N_s)$ compared to $O(N_p dot N_s)$ for the WVF. At $m = 14$ (the published setting), the LF requires 29 WVF evaluations per pixel per orientation, making it approximately 29 times slower than the WVF for the same $N_p$ and $N_s$. This substantial computational overhead is justified only if the line extension provides measurable accuracy improvements, a hypothesis we test extensively in @sec:ablation and @sec:noise.

The relationship between WVF and LF deserves careful attention. When $m = 0$ (or equivalently, when only the central point is used), the LF reduces exactly to the WVF. When $m = 1$, the LF uses three virtual points per orientation, providing a minimal form of tangential averaging. As $m$ increases, the LF progressively incorporates more spatial context along the estimated edge direction. The Gaussian weighting ($sigma = m\/2$) ensures that the central point receives the highest weight, with contributions from distant virtual points decaying smoothly. This design is physically motivated by the observation that edges in natural images tend to be locally straight: averaging gradient estimates along the edge tangent should reinforce true edge signals while suppressing transverse noise fluctuations.

== Evaluation Metrics <sec:metrics>

Edge quality is measured by the Optimal Dataset Scale (ODS) F-score, the standard metric for edge detection benchmarks @martin2004evaluation @arbelaez2011bsds500. A continuous gradient magnitude map is binarized at threshold $t$ and matched against human-annotated ground truth within a spatial tolerance of $r$ pixels. Precision and recall are computed as

$ P(t) = "TP"(t) / ("TP"(t) + "FP"(t)), quad quad R(t) = "TP"(t) / ("TP"(t) + "FN"(t)) $ <eq:precision_recall>

where a predicted edge pixel counts as a true positive if it falls within distance $r$ of any ground-truth edge pixel, and a ground-truth pixel counts as a false negative if no predicted edge pixel falls within distance $r$. The F-score at threshold $t$ is the harmonic mean

$ F(t) = (2 P(t) R(t)) / (P(t) + R(t)) $ <eq:fscore>

The ODS is the maximum F-score over all thresholds, evaluated at a single global threshold applied to the entire dataset:

$ "ODS" = max_t F(t) $ <eq:ods>

We also report the Optimal Image Scale (OIS), which allows a per-image optimal threshold: $"OIS" = 1\/N sum_(i=1)^N max_t F_i (t)$. Throughout this study, we use a match radius of $r = 3$ pixels and evaluate over 1,001 uniformly spaced thresholds.

== Noise Models <sec:noise_models>

We evaluate five noise types that model common degradation mechanisms in real imaging systems. In all definitions below, $I(x,y)$ denotes the clean image intensity normalized to $[0, 1]$, and $tilde(I)(x,y)$ denotes the noisy observation.

Gaussian noise models additive electronic sensor noise @gonzalez2018digital:

$ tilde(I) = I + eta, quad eta tilde cal(N)(0, sigma^2), quad sigma = R / "SNR" $ <eq:gaussian_noise>

where $R$ is the image intensity range and SNR is the signal-to-noise ratio.

Salt-and-pepper noise models impulse noise from sensor defects or transmission errors:

$ tilde(I)(x,y) = cases(
  0 & "with probability" p\/2,
  1 & "with probability" p\/2,
  I(x,y) & "with probability" 1-p
) $ <eq:sp_noise>

where the corruption density $p = 1 / (1 + "SNR")$.

Poisson noise models photon counting noise inherent to optical sensors @foi2008poisson:

$ tilde(I) = "Poisson"(I dot "SNR") / "SNR" $ <eq:poisson_noise>

At low SNR, the relative noise magnitude is large; as SNR increases, the Poisson noise converges toward the clean image.

Speckle noise models multiplicative noise characteristic of coherent imaging systems such as synthetic aperture radar and sonar @goodman1976speckle:

$ tilde(I) = I dot (1 + eta), quad eta tilde cal(N)(0, sigma^2), quad sigma = 1 / "SNR" $ <eq:speckle_noise>

Speckle noise is signal-dependent, so regions with higher intensity receive proportionally more noise.

Uniform noise models quantization and bounded sensor noise:

$ tilde(I) = I + eta, quad eta tilde cal(U)(-a, a), quad a = R / "SNR" $ <eq:uniform_noise>

where $cal(U)(-a, a)$ denotes the uniform distribution on the interval $[-a, a]$.

These five noise models span the principal degradation mechanisms encountered in practical imaging systems. Gaussian noise is the most commonly assumed model in image processing and serves as a baseline for noise robustness evaluation @gonzalez2018digital. Salt-and-pepper noise models the extreme case of pixel-level corruption common in faulty sensors and digital transmission errors. Poisson noise is the physically correct model for photon-limited imaging and dominates in low-light photography, fluorescence microscopy, and astronomical imaging @foi2008poisson. Speckle noise is particularly relevant to Bagan and Wang's maritime imaging target domain, where coherent backscatter from suspended particles produces multiplicative intensity fluctuations @goodman1976speckle @schettini2010underwater. Uniform noise provides a bounded alternative to Gaussian noise that models quantization effects and bounded sensor perturbations. Together, these models allow us to assess whether the WVF's noise robustness is specific to certain degradation mechanisms or generalizes broadly.

We define the signal-to-noise ratio (SNR) as the ratio of the signal amplitude to the noise standard deviation. For each noise type, the noise strength is parameterized as a function of SNR as defined in @eq:gaussian_noise through @eq:uniform_noise. We evaluate seven SNR levels: 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, and 4.0, spanning from severe corruption (SNR = 0.3, where noise power substantially exceeds signal power) to moderate noise (SNR = 4.0, where the signal dominates but noise remains perceptible). This range was chosen to capture the full spectrum of practical imaging conditions, from severely degraded maritime or low-dose medical images at the low end to moderately noisy industrial inspection scenarios at the high end.


// ============================================================
// III. EXPERIMENTAL SETUP
// ============================================================

= Experimental Setup <sec:setup>

== Datasets <sec:datasets>

We evaluate on four edge detection benchmarks spanning a range of image content, resolution, and annotation style. @tab:datasets provides a summary. The evaluation corpus comprises 330 images with diverse content including natural scenes, urban environments, and animals.

#figure(
  placement: top,
  table(
    columns: 5,
    align: (left, center, center, left, left),
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header(
      [*Dataset*], [*Images*], [*Resolution*], [*Description*], [*Ref.*],
    ),
    [UDED], [30], [variable], [Unified Dataset for Edge Detection; aggregates images from 15 sources including BIPED, BSDS, and others], [@soria2023teed],
    [BIPED v1], [50], [1280 $times$ 720], [Barcelona Images for Perceptual Edge Detection; high-resolution outdoor scenes], [@poma2020dexined],
    [BIPED v2], [50], [1280 $times$ 720], [Updated BIPED with refined annotations and additional scenes], [@soria2023teed],
    [BSDS500], [200], [481 $times$ 321], [Berkeley Segmentation Dataset; 200 test images annotated by multiple subjects], [@arbelaez2011bsds500],
  ),
  caption: [Summary of the four edge detection datasets used in the evaluation. The total corpus comprises 330 images spanning outdoor, mixed-content, and multi-source domains.],
) <tab:datasets>

== WVF Parameter Grid <sec:wvf_grid>

The WVF parameter space is defined by three axes. Support size $N_p$ is sampled at 20 levels: $N_p in {5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 100, 130, 160, 200, 250, 300, 400, 500}$. Orientation count $N_s$ is sampled at 18 levels: $N_s in {3, 4, 5, 6, 8, 9, 10, 12, 15, 18, 24, 30, 36, 48, 60, 72, 90, 120}$. Polynomial order $d$ takes values in $\{2, 3, 4, 5\}$. The constraint $N_p >= (d+1)(d+2)\/2$ eliminates configurations where the least-squares system is underdetermined. After removing invalid configurations, 1,206 valid WVF settings remain.

The LF parameter space adds the line half-width $m in {1, 2, 3, 5, 7, 10, 14, 20}$ to a subset of the WVF grid. With $N_p in {15, 25, 50, 75, 100, 150, 250}$, $N_s in {18, 36, 72}$, and $d = 4$ fixed, the LF grid contains 168 configurations. The Gaussian bandwidth is set to $sigma = m\/2$ throughout. The total number of distinct filter evaluations across all datasets is $(1206 + 168) times 330 = 453,420$ filter applications.

== Classical Baselines <sec:classical_baselines>

We evaluate 54 classical edge detection configurations spanning seven families of gradient-based operators. All methods are applied to the BIPED v1 dataset and evaluated using the same ODS protocol as WVF and LF.

The Sobel operator @sobel2014history computes horizontal and vertical gradients using separable derivative-of-Gaussian approximation kernels, tested at kernel sizes $k in {3, 5, 7, 9, 11, 15, 21, 31, 41, 51}$ (10 configurations). Prewitt @prewitt1970gradient uses uniform averaging perpendicular to the derivative direction, tested at $k in {3, 5, 7, 9, 11, 15, 21, 31, 41}$ (9 configurations). The Gaussian Derivative operator convolves the image with the analytical derivative of a Gaussian kernel, tested at $sigma in {0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0, 20.0}$ (11 configurations). The Laplacian of Gaussian (LoG) @marr1980log detects edges as zero-crossings of the Laplacian applied after Gaussian smoothing, tested at 10 $sigma$ values. The Difference of Gaussians (DoG) approximates LoG by subtracting two Gaussian-blurred images, tested at 8 scale values. Steerable second-order filters use oriented second-derivative Gaussian kernels, tested at 8 $sigma$ values. Single-scale operators include Kirsch @kirsch1971computer, Frei-Chen @freichen1977fast, Scharr, Roberts @roberts1963machine, and Robinson (5 configurations). After removing configurations that did not produce valid gradient maps, 54 configurations remained suitable for ODS evaluation.

== Deep Learning Baselines <sec:dl_baselines>

We compare WVF and LF against five state-of-the-art deep learning edge detectors, selected to represent a range of architectural paradigms, model sizes, and computational budgets. @tab:dl_models summarizes these models.

#figure(
  placement: top,
  table(
    columns: 4,
    align: (left, center, left, left),
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header(
      [*Model*], [*Params*], [*Architecture*], [*Ref.*],
    ),
    [DexiNed], [$approx$12M], [Dense extreme inception blocks with deep supervision], [@poma2020dexined],
    [TEED], [$approx$58K], [Lightweight double-fusion strategy], [@soria2023teed],
    [PiDiNet], [$approx$700K], [Pixel difference convolutions encoding gradient operations], [@su2021pidinet],
    [NBED], [$approx$8M], [CAFormer backbone for local and global context], [@chen2024nbed],
    [DiffusionEdge], [$approx$100M], [Latent diffusion model for conditional edge generation], [@ye2024diffusionedge],
  ),
  caption: [Summary of the five deep learning edge detection models used as baselines. Model sizes span three orders of magnitude, from TEED's 58K parameters to DiffusionEdge's 100M.],
) <tab:dl_models>

DexiNed @poma2020dexined uses deeply-supervised Inception-style blocks to generate multi-scale edge maps. With approximately 12M parameters, it produces high-quality edges but requires substantial computation. DexiNed introduced the BIPED dataset alongside its architecture. TEED @soria2023teed (Tiny and Efficient Edge Detector) is a lightweight architecture designed for generalization across datasets. Despite having only approximately 58K parameters---the smallest model in our comparison---it uses a double-fusion strategy to combine low-level and high-level features. PiDiNet @su2021pidinet (Pixel Difference Network) incorporates traditional edge detection principles into a neural architecture by using pixel difference convolutions that encode gradient-like operations directly in the network structure, occupying a middle ground at approximately 700K parameters. NBED @chen2024nbed uses a CAFormer backbone to capture both local and global context for edge prediction, representing the transformer-based approach at approximately 8M parameters. DiffusionEdge @ye2024diffusionedge adapts latent diffusion models for edge detection, treating edge map generation as a conditional denoising process. At approximately 100M parameters, it is by far the largest model in our comparison but achieves state-of-the-art results on multiple benchmarks.

A critical distinction between WVF/LF and these deep learning models is that all five neural baselines are trained on annotated edge data. DexiNed and PiDiNet are typically trained on BIPED or BSDS500; TEED and DiffusionEdge use BIPED v2; NBED uses a combination of datasets. In contrast, WVF and LF are entirely training-free: their behavior is determined solely by the choice of $N_p$, $N_s$, $d$ (and $m$ for LF), with no dependence on labeled edge annotations. This makes WVF/LF fundamentally different in their computational paradigm, though it also means they cannot learn dataset-specific edge semantics. All five models are evaluated with their published pre-trained weights without noise-specific fine-tuning or data augmentation.

== Noise Ablation Protocol <sec:noise_protocol>

For each dataset, we select 5 representative images for the noise ablation study. Each image is corrupted with all combinations of 5 noise types (Gaussian, salt-and-pepper, Poisson, speckle, uniform) and 7 SNR levels (0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0), spanning from severe corruption where noise power exceeds signal power to moderate noise where the signal dominates. Together with the clean baseline, this yields 36 conditions per image. For WVF and LF, 447 parameter configurations are evaluated per condition. For the five DL models, the same images and noise conditions are used with each model's pre-trained weights. The total experimental scope comprises approximately 60,000 WVF/LF evaluations and 3,600 DL evaluations.

== Evaluation Protocol and Implementation <sec:eval_protocol>

Each filter configuration produces a continuous gradient magnitude map for every image. These maps are evaluated against ground-truth annotations using the protocol of Arbelaez et al. @arbelaez2011bsds500 with two modifications from the standard protocol: we use 1,001 thresholds (instead of the standard 99) for finer resolution of the ODS, and a match radius of $r = 3$ pixels. The finer threshold sweep ensures that the reported ODS values are not limited by threshold quantization, which can introduce up to 0.01 ODS error at the standard 99-threshold resolution.

No post-processing (non-maximum suppression, hysteresis thresholding, or morphological thinning) is applied to the raw filter output unless explicitly stated otherwise. This choice is deliberate: we seek to evaluate the raw gradient estimation quality of each filter without confounding effects from post-processing stages. As we demonstrate in @sec:postprocessing, NMS actually degrades ODS under the $r = 3$ evaluation protocol, so this choice does not disadvantage the WVF/LF relative to standard practice.

The evaluation pipeline uses distance transforms and vectorized threshold sweeps to efficiently compute precision and recall at all 1,001 thresholds simultaneously, achieving a $120 times$ speedup over naive per-threshold binary dilation. This efficiency is essential given the scale of the evaluation: $(1206 + 168) times 330 = 453,420$ filter applications for the clean-data ablation alone, plus an additional approximately 60,000 evaluations for the noise study.

All filter computations are GPU-accelerated using a custom CUDA implementation that exploits the embarrassingly parallel nature of the per-pixel polynomial fitting. The WVF processes a full-resolution BSDS500 image ($481 times 321$ pixels) in approximately 0.009 seconds and a full-resolution BIPED image ($1280 times 720$ pixels) in approximately 0.036 seconds. The LF requires 0.05--0.3 seconds per BIPED image depending on $m$, with the computational cost scaling linearly with $2m + 1$. The full ablation across all four datasets completed in approximately 12.4 hours of wall-clock time on an NVIDIA RTX 6000 Ada (48 GB VRAM). The deep learning models were evaluated using their respective published inference code on the same GPU.


// ============================================================
// IV. PARAMETER ABLATION RESULTS
// ============================================================

= Parameter Ablation Results <sec:ablation>

== Single-Image Ablation <sec:single_ablation>

Before committing to the full dataset evaluation, we conducted dense single-image ablations on representative images from BSDS500 and BIPED to map the parameter response surface at high resolution.

@fig:wvf_ods_vs_np shows the ODS F-score as a function of $N_p$ for the WVF at polynomial order $d = 2$ on BSDS500 image \#100007 ($321 times 481$ pixels). Performance rises steeply from $N_p = 5$ (ODS $approx$ 0.73) to $N_p = 25$ (ODS $approx$ 0.847), peaks in the range $N_p = 30$--$50$ (ODS $approx$ 0.856--0.860), and then monotonically declines. At $N_p = 250$ (the published setting), ODS drops to approximately 0.71, and at $N_p = 500$ it falls further to 0.65. The intuition is straightforward: a small $N_p$ constrains the polynomial fit to a tight neighborhood where the intensity surface is well-approximated by low-order polynomials, yielding sharp, localized gradient estimates. As $N_p$ increases, the polynomial must fit intensity variations across increasingly diverse structures, leading to blurred, delocalized edge responses.

#figure(
  image("figures/fig_wvf_ods_vs_np.png", width: 95%),
  caption: [WVF ODS F-score versus support size $N_p$ on BSDS500 image \#100007 at $d = 2$, showing a clear peak near $N_p = 30$--$50$ and monotonic decline beyond. The published parameter $N_p = 250$ falls well below the optimum.],
) <fig:wvf_ods_vs_np>

@fig:wvf_heatmap shows the ODS as a joint function of $N_p$ (vertical axis) and $N_s$ (horizontal axis) at $d = 2$. The dominant structure is vertical: performance varies strongly with $N_p$ and weakly with $N_s$. The optimal region forms a horizontal band at $N_p approx 25$--$50$ that is largely invariant to orientation count.

#figure(
  image("figures/fig_wvf_heatmap_np_ns.png", width: 95%),
  caption: [Heatmap of WVF ODS as a function of $N_p$ (rows) and $N_s$ (columns) at $d = 2$ on BSDS500 image \#100007. The dominant variation is along the $N_p$ axis; $N_s$ has minimal effect beyond approximately 4--6 orientations.],
) <fig:wvf_heatmap>

@fig:ns_saturation isolates the effect of $N_s$ by plotting ODS versus $N_s$ at several fixed values of $N_p$. At all support sizes, ODS rises from $N_s = 3$ to $N_s = 4$--$6$ and then plateaus. The difference between $N_s = 6$ and $N_s = 120$ is typically less than 0.005 in ODS. This saturation occurs because the polynomial fit already captures the edge orientation implicitly: the maximum normal-derivative response over even a few orientations is sufficient to locate the dominant gradient direction. Reducing $N_s$ from 18 to 4 yields a $4.5 times$ speedup with no loss in edge quality.

#figure(
  image("figures/fig_ns_saturation.png", width: 95%),
  caption: [ODS versus orientation count $N_s$ at several fixed $N_p$ values ($d = 2$). Performance saturates by $N_s = 4$--$6$, with negligible improvement from the published $N_s = 18$ or beyond.],
) <fig:ns_saturation>

The single-image ablation also reveals that $d = 2$ (quadratic, $M = 6$ monomials) consistently outperforms $d = 4$ (quartic, $M = 15$ monomials) across the full range of $N_p$ and $N_s$. On the BSDS500 test image, the best $d = 2$ configuration achieves ODS = 0.860, compared to approximately 0.834 for the best $d = 4$ configuration. The quadratic model's advantage is twofold. With fewer parameters to estimate, the least-squares fit is better conditioned, particularly at small $N_p$ where the ratio $N_p \/ M$ is low. At $N_p = 25$ and $d = 4$, the system has only $25\/15 approx 1.7$ observations per parameter, while at $d = 2$ the same $N_p$ gives $25\/6 approx 4.2$ observations per parameter. Furthermore, the higher-order monomials in the quartic model capture curvature and higher-frequency intensity variations that are not relevant to first-derivative estimation and add noise to the gradient estimate.

== Full-Dataset Generalization <sec:full_dataset>

The single-image findings generalize robustly across all four evaluation datasets. @tab:full_dataset_results presents the best-found and published-parameter ODS for both the WVF and LF on each dataset.

#figure(
  placement: top,
  table(
    columns: 7,
    align: (left, center, center, center, center, center, center),
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header(
      [*Dataset*], [*Best WVF*], [*Bagan WVF*], [*$Delta$*],
      [*Best LF*], [*Bagan LF*], [*$Delta$*],
    ),
    [UDED (30)],
    [0.899], [0.847], [+0.052],
    [0.887], [0.729], [+0.158],

    [BIPED v1 (50)],
    [0.812], [0.767], [+0.045],
    [0.802], [0.643], [+0.159],

    [BIPED v2 (50)],
    [0.830], [0.783], [+0.047],
    [0.819], [0.658], [+0.161],

    [BSDS500 (200)],
    [0.682], [0.671], [+0.011],
    [0.671], [0.615], [+0.056],
  ),
  caption: [ODS F-score comparison between optimized and published (Bagan) parameters across all four datasets. The $Delta$ column shows the improvement from parameter optimization. The WVF gains 0.01--0.05, while the LF gains 0.06--0.16.],
) <tab:full_dataset_results>

Several patterns are noteworthy. The best WVF configuration is $(N_p = 25, N_s = 4, d = 2)$ on UDED, BIPED v1, and BIPED v2, demonstrating remarkable consistency across three diverse datasets spanning multi-source and outdoor imagery. On BSDS500, the optimum shifts to $(N_p = 100, N_s = 4, d = 2)$, likely reflecting the lower resolution ($481 times 321$ versus $1280 times 720$) and more diverse content of BSDS500 images. In all cases, $d = 2$ and $N_s = 4$ are selected, confirming the single-image findings with strong generalization.

The gap between the best and published LF parameters is substantially larger than for the WVF (0.06--0.16 versus 0.01--0.05). This disparity arises because the published LF uses $m = 14$, which extends the line support over 29 pixels and averages derivative estimates across structures that may not share a common edge orientation. On clean data, this averaging blurs the edge response without the compensating benefit of noise suppression. The LF gap is particularly striking on the BIPED datasets (+0.159 and +0.161), where the high-resolution images contain fine-grained edge structures that are most severely degraded by over-smoothing.

All configurations achieve lower ODS on BSDS500 than on the other three datasets, consistent with the known difficulty of this benchmark. BSDS500 uses semantic contour annotations from multiple human annotators with varying judgment criteria, while BIPED provides pixel-precise edge labels. Training-free gradient operators like WVF inherently produce low-level edge responses that align better with fine-grained BIPED annotations than with the perceptual contour annotations of BSDS500. The improvement from parameter optimization is correspondingly smaller on BSDS500 (0.011 for WVF, 0.056 for LF) but remains meaningful across 200 images.

The OIS scores follow the same pattern. For example, on UDED the best WVF achieves OIS = 0.905 versus 0.862 for the published parameters; on BSDS500, the best WVF achieves OIS = 0.695 versus 0.686. The OIS--ODS gap is small (0.006--0.014), indicating that a single global threshold performs nearly as well as per-image optimization---the gradient magnitude maps are well-calibrated across images within each dataset.

#figure(
  image("figures/fig_cross_dataset_best.png", width: 95%),
  caption: [Cross-dataset comparison of best-optimized versus published (Bagan) parameters for WVF and LF. The optimized parameters consistently outperform the published settings, with particularly large gains for the LF.],
) <fig:cross_dataset>

@tab:best_params details the optimal parameter configurations found for each dataset.

#figure(
  placement: top,
  table(
    columns: 6,
    align: (left, left, center, center, center, center),
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header(
      [*Dataset*], [*Filter*], [*$N_p$*], [*$N_s$*], [*$d$*], [*ODS*],
    ),
    [UDED], [WVF], [25], [4], [2], [0.899],
    [UDED], [LF], [25], [36], [4], [0.887],
    [BIPED v1], [WVF], [25], [4], [2], [0.812],
    [BIPED v1], [LF], [100], [18], [4], [0.802],
    [BIPED v2], [WVF], [25], [4], [2], [0.830],
    [BIPED v2], [LF], [100], [18], [4], [0.819],
    [BSDS500], [WVF], [100], [4], [2], [0.682],
    [BSDS500], [LF], [250], [18], [4], [0.671],
  ),
  caption: [Optimal parameter configurations for each dataset and filter type. The WVF consistently selects $d = 2$ and $N_s = 4$. The LF optimal parameters show more variation across datasets.],
) <tab:best_params>

== WVF versus LF on Clean Data <sec:wvf_vs_lf_clean>

A central claim of Bagan and Wang @bagan2023lf is that the LF improves upon the WVF by exploiting edge continuity through the line extension mechanism. Our ablation results challenge this claim for clean imagery. On all four datasets, the best WVF configuration outperforms the best LF configuration by consistent margins of 0.010--0.012 ODS points.

@fig:lf_ods_vs_m shows the LF ODS as a function of line half-width $m$. On all datasets, the best ODS occurs at $m = 1$--$3$, the shortest line extensions. As $m$ increases beyond 3, ODS monotonically declines, and the published parameter $m = 14$ yields the worst or near-worst performance. This pattern is consistent with the clean-data hypothesis: longer lines average over more spatial context, which helps suppress noise but harms edge localization when noise is absent. The computational cost of the LF scales as $O((2m+1) dot N_p dot N_s)$ compared to $O(N_p dot N_s)$ for the WVF, making the LF approximately 29 times slower at $m = 14$ for no accuracy benefit on clean data.

#figure(
  image("figures/fig_lf_ods_vs_m.png", width: 95%),
  caption: [LF ODS versus line half-width $m$. Shorter line extensions ($m = 1$--$3$) yield higher ODS than the published $m = 14$. On clean data, the line-averaging mechanism harms rather than helps.],
) <fig:lf_ods_vs_m>

== Post-Processing Effects <sec:postprocessing>

Standard edge detection pipelines typically apply non-maximum suppression (NMS) to thin the gradient magnitude map before thresholding @canny1986edge. We investigated whether NMS improves WVF and LF outputs by testing both 4-directional and 8-directional NMS variants. @fig:postprocessing summarizes the results on BIPED v1. In all cases, NMS reduces the ODS compared to the raw gradient magnitude map, with degradation ranging from 0.06 to 0.09. On a representative single-image evaluation, the best WVF configuration ($N_p = 25$, $d = 2$, $N_s = 3$) achieves ODS = 0.844 without NMS but drops to 0.754 with 4-directional NMS and 0.760 with 8-directional NMS. The Bagan WVF drops from 0.766 to 0.674. This counterintuitive result arises because the ODS evaluation protocol already optimizes the binarization threshold. The raw gradient magnitude map contains soft, multi-pixel-wide edge responses that, when thresholded appropriately, produce predictions matching ground truth within the $r = 3$ pixel tolerance. NMS thins these responses to single-pixel ridges, which can shift the detected edge position away from the annotation.

#figure(
  image("figures/fig_postprocessing.png", width: 95%),
  caption: [Effect of non-maximum suppression (NMS) on ODS for various WVF and LF configurations on BIPED. Both 4-directional and 8-directional NMS consistently reduce ODS compared to the raw gradient magnitude output.],
) <fig:postprocessing>


// ============================================================
// V. BENCHMARKING AGAINST CLASSICAL AND LEARNED METHODS
// ============================================================

= Benchmarking Against Classical and Learned Methods <sec:benchmark>

== Classical Operator Ranking <sec:classical_ranking>

@tab:classical_top15 presents the top 15 classical methods ranked by ODS on BIPED v1, with WVF and LF included for reference. The best classical operator is Prewitt with kernel size 5 (ODS = 0.788), followed by Sobel-7 (0.776) and Gaussian Derivative with $sigma = 1.5$ (0.772). The optimally-tuned WVF achieves ODS = 0.812 and the best LF achieves 0.802, both exceeding every classical configuration tested.

#figure(
  caption: [Top 15 classical edge detectors on BIPED v1 (ODS), with WVF and LF for comparison. The best WVF and LF scores are shown at top. All classical methods fall below both.],
  placement: top,
  table(
    columns: (auto, auto, auto),
    align: (left, center, right),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and y <= 2 { rgb("#e8f4e8") } else if y > 2 and calc.rem(y, 2) == 1 { rgb("#efefef") },

    table.header[*Method*][*Parameters*][*ODS*],
    [*WVF (optimized)*], [$N_p=25, N_s=4, d=2$], [*0.812*],
    [*LF (optimized)*], [$N_p=100, N_s=18, m=1, d=4$], [*0.802*],
    table.hline(),
    [Prewitt],     [$k = 5$],       [0.788],
    [Sobel],       [$k = 7$],       [0.776],
    [GaussianDeriv], [$sigma = 1.5$], [0.772],
    [GaussianDeriv], [$sigma = 1.0$], [0.769],
    [Sobel],       [$k = 5$],       [0.765],
    [Sobel],       [$k = 9$],       [0.759],
    [Kirsch],      [--],            [0.759],
    [Prewitt],     [$k = 3$],       [0.755],
    [Prewitt],     [$k = 7$],       [0.751],
    [Frei-Chen],   [--],            [0.751],
    [Robinson],    [--],            [0.748],
    [Sobel],       [$k = 3$],       [0.744],
    [GaussianDeriv], [$sigma = 2.0$], [0.743],
    [Scharr],      [--],            [0.733],
    [Sobel],       [$k = 11$],      [0.725],
  )
) <tab:classical_top15>

The WVF advantage over the best classical method is $0.812 - 0.788 = 0.024$ ODS points, a 3.0\% relative improvement. This gap is consistent: WVF outperforms every one of the 54 classical configurations tested, not merely the best one. @fig:classical_ranking visualizes the full distribution. The 54 classical methods span a wide ODS range from 0.329 (Steerable $sigma = 16$) to 0.788 (Prewitt-5). The distribution is strongly right-skewed: only 17 of 54 methods (31\%) exceed ODS = 0.700, while 22 methods (41\%) fall below 0.500, highlighting that naive parameter selection for classical operators can yield severely suboptimal results.

#figure(
  placement: top,
  scope: "parent",
  image("figures/fig_classical_ranking.png", width: 100%),
  caption: [ODS scores for all 54 classical edge detectors on BIPED v1, sorted by rank. The horizontal dashed lines indicate the best WVF (0.812) and LF (0.802) scores. All classical methods fall below both lines.]
) <fig:classical_ranking>

A striking pattern emerges when examining how ODS varies with the scale parameter within each operator family (@fig:classical_scale_trend). For every family that accepts a scale or kernel-size parameter, there exists a clear optimum: too small a kernel misses edge structure, while too large a kernel over-smooths. For Sobel, the peak is at $k = 7$; for Prewitt, at $k = 5$; for Gaussian Derivative, at $sigma = 1.5$. Beyond these optima, performance degrades substantially---Sobel-51 scores only 0.428, less than half the score of Sobel-7. The WVF's polynomial fitting over $N_p$ points along each spoke effectively performs a form of adaptive scale selection, which explains its advantage over fixed convolution kernels.

#figure(
  placement: top,
  image("figures/fig_classical_scale_trend.png", width: 100%),
  caption: [ODS versus scale parameter for each classical operator family on BIPED v1. Each family shows a clear optimal scale, with performance degrading sharply at both extremes.]
) <fig:classical_scale_trend>

== Deep Learning Comparison <sec:dl_comparison>

Comparing the WVF against deep learning models requires careful attention to evaluation methodology. Full-dataset ODS evaluation for all five deep learning models across all four datasets was not completed at the time of the benchmarking study. However, two complementary data sources provide a basis for comparison, each with distinct strengths and limitations.

The first source is the full-dataset WVF and LF evaluation across all four datasets (330 images total), which established the optimized ODS scores reported in @tab:full_dataset_results. These scores represent the best achievable performance with a single fixed parameter configuration applied globally to each dataset. The second source is the per-image ODS evaluations from the noise robustness study (@sec:noise), which evaluated all five DL models alongside WVF/LF on 5 representative images per dataset under both clean and noisy conditions. These per-image evaluations allow direct WVF--DL comparison on identical images but involve a smaller sample and oracle parameter selection for the WVF.

On the 5-image BSDS500 subset evaluated in the noise study, the clean-data per-image ODS values are: PiDiNet 0.876, DiffusionEdge 0.795, NBED 0.795, DexiNed 0.737, and TEED 0.669. The best per-image WVF achieves 0.907 on the same images, which exceeds all DL models. However, this comparison is misleading without appropriate context. The per-image WVF evaluation selects the best of 447 configurations independently for each image, an oracle procedure that inflates performance. On the full 200-image BSDS500 dataset with a single fixed configuration, the best WVF achieves ODS = 0.682, which would fall below most DL models' expected full-dataset scores. This discrepancy between per-image and full-dataset evaluation highlights a fundamental challenge: the WVF's optimal parameters vary across images, and any fixed configuration represents a compromise.

On BIPED v1, a single-image DiffusionEdge evaluation yielded ODS = 0.904, substantially exceeding the best WVF (0.812 full-dataset). This gap of 0.092 in ODS demonstrates that the largest DL models capture edge semantics---suppressing texture edges, completing occluded boundaries, respecting object-level contours---in ways that a local polynomial gradient computation fundamentally cannot. The gap between DiffusionEdge (100M parameters trained on annotated data) and WVF (3 tunable hyperparameters, no training data) reflects a genuine difference in capability for clean-data edge detection.

The more informative comparison is against lightweight DL models. PiDiNet (700K parameters) and TEED (58K parameters) operate in a similar computational regime to WVF, and the available per-image data suggests that these models outperform WVF's full-dataset ODS on clean data but may not maintain this advantage under noise (see @sec:dl_degradation). The full clean-data comparison between WVF and lightweight DL models remains an important open question that the noise study partially addresses through its per-image evaluations.

== Computational Cost Analysis <sec:compute>

@tab:timing reports single-image inference times measured on the same hardware (NVIDIA GPU, $720 times 1280$ input resolution).

#figure(
  caption: [Inference time per image ($720 times 1280$ resolution) for all methods, sorted by speed. Speedup is computed relative to WVF.],
  placement: top,
  table(
    columns: (auto, auto, auto, auto),
    align: (left, right, right, right),
    inset: (x: 8pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header[*Method*][*Time (s)*][*Relative to WVF*][*Category*],
    [*WVF*],         [0.036], [$1.0 times$],     [Filter],
    [PiDiNet],       [0.119], [$3.3 times$],      [DL],
    [DexiNed],       [0.226], [$6.3 times$],      [DL],
    [NBED],          [0.253], [$7.1 times$],      [DL],
    [TEED],          [0.538], [$15.1 times$],     [DL],
    [DiffusionEdge], [15.22], [$427.4 times$],    [DL],
  )
) <tab:timing>

The WVF processes a $720 times 1280$ image in 0.036 seconds, making it the fastest method by a substantial margin. The nearest DL competitor, PiDiNet, requires 0.119 seconds---$3.3 times$ slower. The gap widens dramatically for larger models: DexiNed is $6.3 times$ slower, NBED is $7.1 times$ slower, TEED is $15.1 times$ slower, and DiffusionEdge is $427 times$ slower at 15.22 seconds per image. For near-real-time applications, WVF's 0.036 s per frame (36 ms) approaches the 33 ms budget required for 30 fps processing of $720 times 1280$ input, while no DL model in our comparison comes close to real-time performance.

#figure(
  placement: top,
  image("figures/fig_compute_cost.png", width: 100%),
  caption: [ODS versus inference time for all compared methods on BIPED v1. WVF occupies the lower-left region: high accuracy at low computational cost. The Pareto frontier connects methods offering the best accuracy for a given time budget. DL ODS values shown are per-image estimates from the noise study subset.]
) <fig:compute_cost>

@fig:compute_cost plots ODS against inference time, revealing the accuracy--cost trade-off. WVF occupies a favorable position at 0.812 ODS in 0.036 seconds. The Pareto frontier includes WVF for applications requiring sub-40 ms latency with high accuracy, and DiffusionEdge for applications requiring maximum accuracy regardless of cost. The WVF's speed advantage stems from its computational simplicity: with $N_s = 4$ and $N_p = 25$, the filter requires only 100 lookups and a small least-squares solve per pixel, operations that are highly amenable to GPU vectorization.

The recommended parameters ($N_p = 25$, $N_s = 4$, $d = 2$) also offer substantial savings over the published parameters ($N_p = 250$, $N_s = 18$, $d = 4$). The dominant computational cost scales as $N_s dot N_p dot M$: the recommended configuration gives $4 times 25 times 6 = 600$, compared to $18 times 250 times 15 = 67,500$ for the published parameters, a $112 times$ reduction.


// ============================================================
// VI. NOISE ROBUSTNESS
// ============================================================

= Noise Robustness <sec:noise>

== Optimal Parameter Shift Under Noise <sec:param_shift>

The most striking finding of the noise ablation is the dramatic shift in the optimal WVF support size $N_p$ as noise increases. On clean BSDS500 images, the best WVF configuration uses $N_p approx 40$ with an average per-image ODS of 0.907. Under severe Gaussian noise at SNR = 0.3, the optimal $N_p$ shifts to approximately 440, with ODS dropping to 0.572---a roughly $10 times$ increase in optimal support size. @fig:optimal_params_shift illustrates this parameter shift across SNR levels. The trend is monotonic: as noise increases, the optimal support size grows consistently. At intermediate noise levels, the shift is gradual: SNR = 2.0 yields optimal $N_p approx 112$ for Gaussian noise, while SNR = 4.0 shows $N_p approx 65$, already approaching the clean-data optimum.

#figure(
  image("figures/fig_optimal_params_shift.png", width: 100%),
  caption: [Shift in optimal WVF support size $N_p$ as a function of SNR across noise types on BSDS500 (5-image average). On clean data, the optimal $N_p$ is 25--50; under severe noise (SNR = 0.3), it increases to 250--500, partially vindicating Bagan and Wang's choice of $N_p = 250$.],
  placement: top,
) <fig:optimal_params_shift>

This finding partially vindicates Bagan and Wang's choice of $N_p = 250$ @bagan2021wvf. On clean data, $N_p = 250$ yields substantially lower ODS than the optimum. However, at SNR = 0.3 under Gaussian noise, $N_p = 250$ is close to the optimal range, and the performance gap narrows considerably. The originally "suboptimal" parameters are in fact near-optimal for the noisy maritime conditions Bagan and Wang targeted. The physical intuition is that larger support regions average over more pixels, providing built-in noise smoothing at the cost of reduced spatial resolution---a trade-off that is unfavorable on clean data but essential under noise.

Unlike the support size, the optimal polynomial degree $d$ remains remarkably stable across noise levels. On both clean data and all noisy conditions tested, $d = 2$ consistently outperforms $d = 4$ across all four datasets and all noise types. Higher-order polynomials fit noise fluctuations within the support region, a form of overfitting that becomes more harmful as noise increases. This finding further challenges Bagan and Wang's choice of $d = 4$, which is suboptimal on both clean and noisy data. The angular sampling parameter $N_s$ shows similar saturation behavior under noise as on clean data, confirming that angular oversampling provides no noise resilience benefit, although at very low SNR with large $N_p$, a slight preference for higher $N_s$ values (36--72) emerges to stabilize the polynomial fit.

== Noise-Type-Specific WVF Degradation <sec:noise_type>

@fig:wvf_ods_vs_snr shows WVF ODS as a function of SNR for each noise type on BSDS500. The noise types produce distinct degradation profiles. Speckle noise is the least destructive: even at SNR = 0.3, the best WVF achieves ODS = 0.696, compared to 0.907 on clean data (a 23\% drop). The multiplicative nature of speckle noise preserves zero-crossing structure and predominantly affects high-intensity regions, leaving edge contrast relatively intact. Gaussian and uniform noise cause the most severe degradation, with ODS dropping to 0.572 and 0.582 respectively at SNR = 0.3 (36--37\% drops), because these additive noise models corrupt the image uniformly regardless of local structure. Poisson noise shows intermediate behavior (ODS = 0.631 at SNR = 0.3, a 30\% drop), and salt-and-pepper noise also shows intermediate degradation (ODS = 0.626, a 31\% drop).

#figure(
  image("figures/fig_wvf_ods_vs_snr.png", width: 100%),
  caption: [Best WVF ODS (averaged over 5 images) as a function of SNR for each noise type on BSDS500. Speckle noise is the least destructive, while Gaussian and uniform noise cause the most severe degradation. All curves converge toward the clean-data ODS as SNR increases.],
  placement: top,
) <fig:wvf_ods_vs_snr>

#figure(
  image("figures/fig_noise_type_comparison.png", width: 100%),
  caption: [Comparison of WVF performance degradation across noise types and datasets. The relative ordering of noise types is consistent across datasets: speckle is least destructive, Gaussian and uniform most destructive.],
  placement: top,
) <fig:noise_type_comparison>

@fig:noise_type_comparison demonstrates that these noise-type rankings are consistent across datasets. The relative ordering is preserved: speckle $>$ Poisson $approx$ salt-and-pepper $>$ uniform $approx$ Gaussian. The cross-dataset analysis also reveals that the parameter shift magnitude depends on the dataset: UDED images with high-contrast structures require a more modest $N_p$ increase, while BIPED images with fine-textured edges require larger $N_p$ under the same noise conditions. This suggests that the optimal noise-adapted support size depends on the spatial frequency content of the edges being detected.

== WVF versus LF Under Noise <sec:wvf_lf_noise>

On clean data, WVF consistently outperforms LF across all four datasets, with a per-image average ODS gap of 0.007 on BSDS500. Under noise, this gap narrows and can reverse at very low SNR. @fig:lf_vs_wvf_noise shows the comparison across SNR levels. At SNR = 0.3 under Gaussian noise, LF (ODS = 0.574) slightly outperforms WVF (ODS = 0.572), a reversal of the clean-data ranking. Similar reversals occur under salt-and-pepper noise (LF = 0.635 vs. WVF = 0.626) and uniform noise (LF = 0.592 vs. WVF = 0.582). Under speckle noise, WVF retains a slim advantage even at SNR = 0.3 (effectively tied at 0.696).

#figure(
  image("figures/fig_lf_vs_wvf_noise.png", width: 100%),
  caption: [WVF vs. LF ODS as a function of SNR on BSDS500. On clean data, WVF leads by 0.007. Under severe noise (SNR = 0.3), the gap closes to near zero, and LF overtakes WVF for Gaussian, salt-and-pepper, and uniform noise.],
  placement: top,
) <fig:lf_vs_wvf_noise>

The LF's noise resilience stems from its line-averaging mechanism: by fitting polynomials along multiple radial lines and combining the directional estimates, the LF effectively averages over more independent noise samples than the WVF's circular support. The reversal is modest in magnitude (typically 0.002--0.009 ODS) and occurs only at low SNR (below approximately 1.0). At moderate noise levels (SNR $>= 2.0$), WVF maintains its clean-data advantage. This finding provides additional context for Bagan and Wang's emphasis on the Line Filter: in noisy maritime environments, the line-averaging mechanism offers a genuine, if small, noise resilience advantage.

== Deep Learning Model Degradation <sec:dl_degradation>

All five DL edge detectors degrade severely under noise when evaluated with their published pre-trained weights. @fig:dl_degradation shows the degradation curves on BSDS500 (5-image average). The clean-data per-image ODS values are: PiDiNet 0.876, DiffusionEdge 0.795, NBED 0.795, DexiNed 0.737, and TEED 0.669.

#figure(
  image("figures/fig_dl_degradation.png", width: 100%),
  caption: [Deep learning model ODS as a function of SNR on BSDS500 (5-image average). All models degrade severely under noise, with most collapsing to approximately 0.33 (near the trivial baseline) by SNR = 0.3. NBED shows the most graceful degradation.],
  placement: top,
) <fig:dl_degradation>

PiDiNet achieves the highest clean-data ODS (0.876) but degrades rapidly under Gaussian noise, dropping to 0.329 at SNR = 0.3 (62\% loss) and recovering to only 0.638 at SNR = 4.0 (still 27\% below clean). DexiNed shows near-total collapse under all noise types: at SNR = 0.3 across all noise types, ODS is approximately 0.329--0.331, very close to the trivial all-positive baseline, and it fails to recover even at SNR = 4.0 for most noise types (Gaussian ODS = 0.337). TEED also degrades severely, dropping to 0.334 under Gaussian noise at SNR = 0.3, with only partial recovery for speckle and uniform noise at higher SNR. NBED demonstrates the most graceful degradation, dropping to 0.329 at SNR = 0.3 under Gaussian noise but recovering more quickly: at SNR = 4.0, Gaussian ODS reaches 0.694 (13\% below clean) and speckle ODS reaches 0.741 (7\% below clean). DiffusionEdge shows the most erratic degradation pattern, with non-monotonic behavior at intermediate SNR values that suggests complex interactions between the diffusion-based generation process and noise.

The degradation patterns across DL models are revealing. The models with the highest clean-data performance (PiDiNet, DiffusionEdge) are not necessarily the most noise-robust. NBED, despite starting from a lower clean-data baseline than PiDiNet, retains more of its performance under moderate noise, suggesting that its multi-scale CAFormer architecture provides implicit noise suppression at coarser processing scales. DexiNed's near-total collapse under any noise is particularly striking: it appears to produce near-uniform edge maps when the input distribution departs from clean natural images, suggesting severe overfitting to the statistical properties of its training data.

None of the five DL models include noise augmentation in their training pipelines. They are trained on clean edge detection datasets and learn features tuned to the statistical properties of clean natural images. This represents a fundamental difference from the WVF approach: the parametric WVF can adapt to noise through parameter tuning, specifically by increasing $N_p$ to smooth over noise. DL models have no such tuning mechanism at inference time; their noise response is entirely determined by their training data and architecture @hendrycks2019robustness @dodge2016understanding. While retraining with noise augmentation could improve DL robustness, this requires explicit knowledge of the noise distribution and access to the training pipeline, neither of which may be available for deployed models. Furthermore, noise-augmented training creates a trade-off between clean-data and noisy-data performance that the WVF avoids entirely through runtime parameter adjustment.

== Crossover Analysis: WVF Overtakes Deep Learning <sec:crossover>

@fig:crossover_by_noise_type presents the crossover analysis, comparing the best WVF ODS (optimized over 447 configurations) against each DL model at each SNR level. The results reveal that WVF overtakes most DL models at surprisingly high SNR values. DexiNed and TEED are overtaken by WVF at all tested SNR levels including SNR = 4.0, because both models degrade catastrophically under any noise. Under Gaussian noise at SNR = 4.0, DexiNed drops to 0.337 and TEED to 0.341, while the optimally-tuned WVF achieves 0.894. DiffusionEdge, due to its erratic degradation, is also overtaken across nearly all tested conditions. PiDiNet, the highest-performing DL model on clean data, is overtaken at approximately SNR = 4.0 for most noise types, with Gaussian ODS of 0.638 versus WVF's 0.894. NBED, the most noise-robust DL model, presents the most competitive crossover but is still exceeded by WVF at SNR = 4.0 under all noise types tested.

#figure(
  image("figures/fig_crossover_by_noise_type.png", width: 100%),
  caption: [Crossover analysis: per-image ODS of optimally-tuned WVF versus each DL model as a function of SNR on BSDS500. WVF overtakes most DL models at surprisingly high SNR values. These per-image ODS values are computed on 5 selected images with per-image optimal WVF parameters.],
  placement: top,
) <fig:crossover_by_noise_type>

Several important caveats apply to this analysis. The noise ablation uses 5 images per dataset, and the WVF is optimized per-image, which inflates its apparent performance relative to a single fixed configuration. On the full 200-image BSDS500 dataset with a single configuration, the best WVF achieves ODS = 0.682, lower than what most DL models achieve on the full dataset under clean conditions. The DL models are also evaluated with their published weights without noise-augmented retraining, which would likely improve their robustness. Despite these caveats, the analysis demonstrates a fundamental point: WVF's parameterization provides a mechanism for noise adaptation that fixed DL models lack.

For maritime applications---Bagan and Wang's target domain---speckle noise is a dominant degradation mechanism @goodman1976speckle. The WVF maintains particularly strong performance under speckle noise: at SNR = 1.0, the best WVF achieves ODS = 0.831 on BSDS500, compared to NBED's 0.514 and PiDiNet's 0.511. This $0.3+$ ODS advantage under speckle noise at moderate SNR is practically significant and supports the use of parametrically-tuned WVF for maritime edge detection.


// ============================================================
// VII. STATISTICAL VALIDATION
// ============================================================

= Statistical Validation <sec:stats>

== Statistical Methods <sec:stat_methods>

All tests in this section are nonparametric, requiring no assumptions about the distribution of ODS scores. This choice is motivated by the small sample sizes, the bounded nature of the ODS metric ($0 <= "ODS" <= 1$), and the unknown distributional form of edge detection performance across configurations.

Kendall's coefficient of concordance $W$ @kendall1948rank measures the agreement among $k$ rankers (datasets) ranking $n$ objects (configurations). It is defined as $W = 12 S \/ (k^2 (n^3 - n))$ where $S$ is the sum of squared deviations of the rank sums from their mean. Spearman rank correlation $rho$ @spearman1904proof is computed for each pair of datasets to reveal which pairs produce similar configuration rankings. The Wilcoxon signed-rank test @wilcoxon1945individual is used for paired comparisons such as WVF versus LF at matched parameters. The sign test is used for between-dataset comparisons where $n = 4$; with four datasets, the minimum achievable one-sided $p$-value is $0.5^4 = 0.0625$, an inherent constraint of the experimental design. The Kruskal--Wallis test @kruskalwallis1952 assesses the effect of each parameter on ODS, with effect size quantified by $eta^2 = (H - k + 1) / (n - k)$. Following standard conventions @cohen1988statistical, $eta^2 < 0.01$ is negligible, $0.01$--$0.06$ is small--moderate, $0.06$--$0.14$ is moderate--large, and $> 0.14$ is large. Bootstrap confidence intervals @efron1979bootstrap with 10,000 resamples are used for noise-crossover SNR estimates.

== Cross-Dataset Ranking Consistency <sec:ranking_consistency>

Kendall's $W = 0.647$ ($chi^2 = 3,120.0$, $p < 10^(-10)$, $n = 1,206$ configurations) indicates moderate-to-strong concordance across the four datasets: the configuration rankings are far from random, but they are not identical. Approximately 65\% of the ranking variance is shared across datasets, while the remaining 35\% reflects dataset-specific preferences. This level of concordance is noteworthy: it suggests that the WVF parameter space has a broadly consistent structure across diverse imaging domains, with one major exception detailed below.

#figure(
  image("figures/fig_spearman_heatmap.png", width: 90%),
  caption: [Pairwise Spearman $rho$ between datasets across 1,206 WVF configurations. UDED, BIPED v1, and BIPED v2 form a highly correlated cluster ($rho > 0.96$), while BSDS500 correlates weakly ($rho < 0.18$) with all others.],
) <fig:spearman>

The pairwise Spearman correlations in @fig:spearman reveal the structure behind the aggregate concordance. The three non-BSDS datasets---UDED, BIPED v1, and BIPED v2---form a tight cluster with pairwise $rho$ between 0.96 and 0.99 ($p < 10^(-10)$ for all). These correlations are remarkably high: essentially, if a WVF configuration ranks well on any one of these three datasets, it ranks well on all three. The rank-order of all 1,206 configurations is nearly identical across the multi-source UDED and high-resolution outdoor scenes (BIPED v1 and v2), despite the substantial differences in image content, resolution, and annotation style between these datasets.

BSDS500 is the clear outlier. Its pairwise $rho$ with the other datasets ranges from 0.03 (with UDED, $p = 0.258$) to 0.17 (with BIPED v1, $p < 10^(-8)$). The correlation with UDED is not statistically significant, meaning that BSDS500 configuration rankings are essentially unrelated to UDED rankings. The correlation with BIPED v2 ($rho = 0.052$, $p = 0.073$) also fails to reach significance at $alpha = 0.05$. The divergence is even starker when examining which specific configurations are top-ranked: the top-10 configurations on BSDS500 share zero overlap with the top-10 on any other dataset. This finding means that tuning WVF parameters on BSDS500 produces a configuration that is likely suboptimal for other imaging domains, and vice versa.

@fig:optimal_vs_bagan_gaps visualizes the ODS gaps between optimized and Bagan parameters for both filters. All eight gaps (4 datasets $times$ 2 filters) are positive. The sign test yields $p = 0.0625$ for both filters, the minimum achievable $p$-value with $n = 4$ observations. While this does not reach the conventional $alpha = 0.05$ threshold, it represents the strongest possible evidence from four datasets: the observed outcome (4/4 positive) is as extreme as the test allows. The WVF gap on BSDS500 (+0.011) is the smallest but still practically meaningful: on a 200-image benchmark, an ODS difference of 0.011 represents a consistent improvement in the precision--recall trade-off across the entire dataset.

#figure(
  image("figures/fig_optimal_vs_bagan_gaps.png", width: 95%),
  caption: [ODS gaps between optimized and Bagan parameters for WVF and LF across all four datasets. The LF gaps are 3--15$times$ larger than the WVF gaps, reflecting the greater suboptimality of Bagan's $m = 14$ setting.],
) <fig:optimal_vs_bagan_gaps>

== WVF versus LF Statistical Significance <sec:wvf_lf_stats>

The Wilcoxon signed-rank test applied to 15 matched WVF--LF configuration pairs (sharing $N_p$, $N_s$, $d$) yields uniformly non-significant results, as shown in @tab:wvf_lf.

#figure(
  placement: top,
  table(
    columns: 5,
    align: (left, center, center, center, center),
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header(
      [*Dataset*], [*Median diff*], [*Cohen's $d$*], [*$T$*], [*$p$*],
    ),
    [UDED], [+0.0003], [$-0.46$], [45], [0.805],
    [BIPED v1], [$-0.0009$], [$-0.77$], [21], [0.989],
    [BIPED v2], [$-0.0003$], [$-0.64$], [33], [0.940],
    [BSDS500], [$-0.0034$], [$-1.15$], [6], [1.000],
  ),
  caption: [Wilcoxon signed-rank test comparing WVF and LF at matched parameters. No dataset shows a significant difference ($p > 0.8$ everywhere). The median ODS difference is less than 0.004 on every dataset.],
) <tab:wvf_lf>

The median ODS difference between WVF and LF at matched parameters is less than 0.004 on every dataset. Cohen's $d$ is negative on three of four datasets, meaning the LF achieves marginally higher ODS at matched parameters. This seemingly contradicts the finding that the best WVF beats the best LF, but the contradiction is resolved by recognizing that the LF has the additional parameter $m$: the best LF uses $m = 1$--$3$ (near-WVF behavior), while larger $m$ values degrade performance. At matched parameters, the two filters are statistically indistinguishable, and the LF's additional parameter primarily serves as a mechanism for degradation on clean data.

== Parameter Sensitivity: Kruskal--Wallis Analysis <sec:param_sensitivity>

The Kruskal--Wallis test quantifies which parameters most strongly influence ODS. @fig:sensitivity shows the $eta^2$ effect sizes. The LF line half-width $m$ dominates with $eta^2 = 0.343$ ($H = 234.7$, $p < 10^(-46)$), a large effect by any convention. The median ODS drops from 0.798 at $m = 2$ to 0.621 at $m = 20$, confirming that $m$ is the single most important parameter and that Bagan's $m = 14$ is the primary source of the LF's suboptimality. Support size $N_p$ has a moderate effect ($eta^2 = 0.054$, $H = 278.6$, $p < 10^(-48)$) with a clear inverted-U shape peaking near $N_p = 80$--$100$. Orientation count $N_s$ has a moderate effect ($eta^2 = 0.053$, $H = 271.9$, $p < 10^(-47)$) concentrated at the low end, with performance saturating by $N_s = 4$. Polynomial order $d$ has a negligible effect ($eta^2 = 0.002$, $H = 11.5$, $p = 0.009$), where the statistical significance is an artifact of the large sample size ($n = 4,824$) and does not indicate practical importance.

#figure(
  image("figures/fig_parameter_sensitivity.png", width: 95%),
  caption: [Kruskal--Wallis $eta^2$ effect sizes for each parameter. The LF line half-width $m$ dominates ($eta^2 = 0.343$), while $N_p$ and $N_s$ have moderate effects. Polynomial order $d$ has a negligible effect.],
) <fig:sensitivity>

#figure(
  image("figures/fig_median_ods_by_param.png", width: 100%),
  caption: [Median ODS as a function of each parameter level. Top-left: $N_p$ shows an inverted-U peaking near 80--100. Top-right: $N_s$ saturates at 3--4. Bottom-left: $d$ has minimal effect. Bottom-right: LF $m$ shows monotonic decline beyond $m = 2$.],
) <fig:median_ods>

== WVF versus Deep Learning on Clean Data <sec:wvf_dl_clean_stats>

On clean data, DL models generally outperform the WVF, consistent with expectations for methods that learn edge semantics from annotated data. @fig:wvf_vs_dl_table presents the comparison matrix from the per-image evaluations. DexiNed outperforms the best WVF on all 4 datasets (sign test $p = 0.0625$), and NBED outperforms the WVF on all available datasets as well. TEED and PiDiNet outperform the WVF on 3 of 4 datasets, with TEED narrowly losing on BSDS500 (0.680 vs. 0.682). DiffusionEdge shows anomalous results with very low ODS (0.31--0.34) on UDED and BIPED v1, far below all other methods, suggesting the model was not properly adapted to these datasets or encountered evaluation pipeline issues. The WVF's advantage lies not in clean-data accuracy but in its noise robustness, training-free nature, computational efficiency, and parametric adaptability, as established in the subsequent crossover analysis.

#figure(
  image("figures/fig_wvf_vs_dl_table.png", width: 95%),
  caption: [ODS heatmap comparing the best WVF against five DL models across datasets. Per-image ODS values from the noise study (5 images per dataset). On clean data, DL models generally outperform WVF, though the advantage reverses under noise.],
) <fig:wvf_vs_dl_table>

== Noise Crossover Significance <sec:crossover_stats>

@fig:crossover_heatmap presents the median crossover SNR for each combination of noise type and DL model. The dominant pattern is uniformity: the median crossover SNR is 0.3 for 23 of 25 noise--model combinations, with the two exceptions being Poisson noise against PiDiNet (0.5) and Poisson noise against TEED (with a bootstrap upper CI of 0.5). For every noise--model combination, the fraction of image pairs where the WVF overtakes the DL model is 100\% at SNR = 0.3, without exception. Bootstrap 95\% confidence intervals are narrow, ranging from $[0.30, 0.30]$ for most combinations to $[0.30, 0.50]$ for Poisson noise against select models.

#figure(
  image("figures/fig_crossover_heatmap.png", width: 90%),
  caption: [Median crossover SNR by noise type and DL model. Lower values indicate earlier WVF advantage. The WVF overtakes all models at SNR $<= 0.5$ across all noise types, with most crossovers at SNR = 0.3.],
) <fig:crossover_heatmap>

== Summary of Statistical Evidence <sec:evidence_summary>

@tab:evidence synthesizes the strength of evidence for each major claim.

#figure(
  placement: top,
  table(
    columns: 3,
    align: (left, left, left),
    inset: (x: 5pt, y: 4pt),
    stroke: (x, y) => if y <= 1 { (top: 0.5pt) },
    fill: (x, y) => if y > 0 and calc.rem(y, 2) == 0 { rgb("#efefef") },

    table.header(
      [*Claim*], [*Evidence Strength*], [*Key Statistic*],
    ),
    [Optimized params beat Bagan's], [Strong practical; marginal statistical], [$p = 0.0625$, gaps 0.011--0.161],
    [WVF and LF equivalent at matched params], [Strong evidence of no difference], [$p > 0.8$ on all datasets, $|d| < 1.15$],
    [LF $m$ is most important parameter], [Very strong], [$eta^2 = 0.343$, $p < 10^(-46)$],
    [$N_p$ and $N_s$ have moderate effects], [Strong; $d$ negligible], [$eta^2 = 0.053$--$0.054$ vs. $0.002$],
    [BSDS500 rankings diverge from others], [Very strong], [$rho = 0.03$--$0.17$, zero top-10 overlap],
    [DL models beat WVF on clean data], [Consistent but limited power], [3--4 of 4 datasets per model],
    [WVF overtakes DL at extreme noise], [Very strong], [100\% crossover at SNR $<= 0.5$],
  ),
  caption: [Summary of statistical evidence for all major claims. Evidence strength reflects both statistical significance and practical effect size.],
) <tab:evidence>


// ============================================================
// VIII. DISCUSSION
// ============================================================

= Discussion <sec:discussion>

== Reconciling Bagan's Parameters <sec:reconcile>

The most nuanced finding of this study is that Bagan and Wang's parameter choices are simultaneously suboptimal and well-motivated, depending on the operating conditions. On clean imagery, reducing $N_p$ from 250 to 25--100, $d$ from 4 to 2, and $N_s$ from 18 to 4 yields consistent and substantial improvements across all four datasets. These are not marginal gains: the LF improvements of 0.06--0.16 ODS represent a qualitative shift in detection quality, and the $112 times$ computational savings make real-time deployment feasible. The Kruskal--Wallis analysis confirms that the line half-width $m$ is the most damaging parameter choice, with $eta^2 = 0.343$ --- the LF at $m = 14$ is dramatically worse than at $m = 1$--$3$ on clean data.

However, the noise robustness analysis reveals that the picture is considerably more complex than a simple "Bagan's parameters are wrong" narrative would suggest. Under noise conditions resembling maritime imaging---particularly speckle noise at moderate SNR---the optimal $N_p$ shifts to 250--500, close to Bagan and Wang's published value. At SNR = 0.3 under speckle noise, the optimal $N_p$ on BSDS500 reaches approximately 232, nearly matching Bagan's recommendation of 250. At the same noise level under Gaussian noise, the optimal $N_p$ increases further to approximately 440, exceeding even Bagan's choice. The UDED dataset shows a more modest shift to $N_p approx 243$ under Gaussian noise at SNR = 0.3, consistent with the high-contrast nature of its edge targets that provide inherent resilience even at moderate support sizes.

The parameter shift finding thus reconciles the apparent contradiction between our clean-data optimization results and Bagan and Wang's design choices. Bagan and Wang were likely optimizing, implicitly or explicitly, for noisy maritime conditions where larger support sizes are genuinely beneficial. The physical mechanism is straightforward: the WVF's polynomial fitting over a larger neighborhood averages over more independent noise samples, reducing the gradient estimate variance at the cost of increased bias (spatial blurring). On clean data, this bias dominates and degrades edge localization. Under noise, the variance reduction becomes essential for suppressing spurious gradient responses, and the bias--variance trade-off shifts in favor of larger support sizes.

The one parameter choice that remains suboptimal even under noise is $d = 4$. The polynomial order $d = 2$ outperforms $d = 4$ at all SNR levels tested across all datasets and noise types. The stability of this finding is remarkable: unlike $N_p$, which must adapt by an order of magnitude across noise conditions, $d = 2$ is universally optimal. The explanation is that higher-order polynomials ($d = 4$, with $M = 15$ monomials) have more degrees of freedom to fit noise fluctuations within the support region, a form of overfitting that becomes progressively more harmful as noise increases. This suggests that the quartic model represents a genuine design suboptimality rather than a noise-motivated choice, and that Bagan and Wang's recommendation of $d = 4$ should be revised to $d = 2$ regardless of the target application domain.

== Positioning WVF in the Edge Detection Landscape <sec:positioning>

The benchmarking results position the Wide View Filter in a distinct and practical niche. Against classical operators, WVF's advantage is unambiguous: it outperforms all 54 configurations tested on BIPED v1, with a consistent 2.4 ODS point margin over the best classical method (Prewitt-5). This improvement reflects WVF's structural advantage---polynomial fitting over extended neighborhoods captures gradient information that fixed convolution kernels cannot.

Against deep learning models, the comparison is condition-dependent. On clean data, DL models generally achieve higher ODS, reflecting their ability to learn dataset-specific edge semantics from annotated training data. The gap is largest for heavyweight models like DiffusionEdge and smallest for lightweight architectures. However, WVF runs $3.3 times$ faster than the fastest DL model (PiDiNet) and $427 times$ faster than DiffusionEdge, requires no training data, and offers complete interpretability. Under noise, the rankings reverse dramatically: WVF overtakes most DL models at surprisingly mild noise levels, and at SNR = 0.3 it exceeds every DL model on every test image. This reveals that edge detection method selection is not a question of which method is "best" in absolute terms, but rather which method is best for the operating conditions at hand.

== The BSDS500 Outlier Problem <sec:bsds500>

The near-zero Spearman correlation between BSDS500 and the other three datasets ($rho = 0.03$--$0.17$) is perhaps the most consequential finding of the statistical analysis. A filter configuration's ranking on BSDS500 is essentially uninformative about its ranking on UDED, BIPED v1, or BIPED v2, despite the high concordance among the latter three ($rho > 0.96$). The root cause appears to be BSDS500's unique combination of small image size ($481 times 321$ versus $1280 times 720$ for BIPED), diverse content, and multi-annotator ground truth with varying judgment criteria. The smaller image size means that the WVF's circular neighborhood at $N_p = 100$ spans a proportionally larger fraction of the image, explaining why BSDS500 prefers larger $N_p$ (100 versus 25 for the other datasets).

This finding carries a practical implication that extends beyond the WVF: benchmark results on BSDS500, which dominate the edge detection literature, should not be extrapolated to other imaging domains without validation. Conversely, results on application-specific datasets may not predict BSDS500 performance. Practitioners should tune parameters on data representative of their target domain rather than relying on any single benchmark.

== Practical Recommendations <sec:recommendations>

For practitioners deploying WVF-based edge detection on clean imagery, the evidence supports using the WVF rather than the LF, since the line extension adds computational cost without improving accuracy on clean data. The Wilcoxon signed-rank tests confirm that the two filters are statistically indistinguishable at matched parameters, and the LF's additional parameter $m$ primarily serves as a mechanism for degradation when set to values above 3. The support size should be set to $N_p = 25$--$50$ for high-resolution images (e.g., $1280 times 720$) and $N_p = 50$--$100$ for lower-resolution images (e.g., $481 times 321$), as the optimal support size appears to scale with image resolution. The polynomial order should be set to $d = 2$ in all cases, since quadratic fitting provides sufficient expressiveness for gradient estimation while maintaining good conditioning of the least-squares system. At $d = 2$ and $N_p = 25$, each orientation's least-squares system has $25\/6 approx 4.2$ observations per parameter, which is well-conditioned; at $d = 4$ the same $N_p$ gives only $25\/15 approx 1.7$ observations per parameter, leading to noisy coefficient estimates. The orientation count should be set to $N_s = 4$--$6$, since finer angular sampling provides negligible benefit while increasing computation linearly. Non-maximum suppression should not be applied if using the ODS evaluation protocol with a match radius of $r >= 3$ pixels, as the raw gradient magnitude map achieves higher ODS than the NMS-thinned version due to the tolerance inherent in the matching procedure.

For noisy imaging environments, the support size should be increased proportionally to the noise level. At moderate noise (SNR $approx$ 2--4), $N_p = 65$--$112$ provides a good balance between noise suppression and edge localization. At severe noise (SNR $< 1$), $N_p = 250$--$500$ may be necessary, and the LF with small $m$ (1--3) may offer marginal additional robustness through its tangential averaging mechanism. The polynomial order should remain at $d = 2$ regardless of noise conditions, as $d = 4$ is suboptimal at every SNR level tested across all noise types and datasets. An adaptive WVF system that automatically adjusts $N_p$ based on estimated noise level represents a promising deployment strategy: a noise estimation module coupled with a lookup table mapping SNR to optimal $N_p$ could provide near-optimal performance across noise conditions without manual tuning.

When choosing between WVF and deep learning models, the decision should be guided by the expected noise regime and computational constraints. On clean data, DL models generally offer higher accuracy at the cost of greater computational requirements and training-data dependency. The gap is largest for heavyweight models like DiffusionEdge (which achieves 0.904 ODS on a single BIPED image versus WVF's 0.812 full-dataset) and smallest for lightweight models like TEED. Under any appreciable noise, WVF's parametric adaptability provides a significant advantage over fixed learned models that were not trained with noise augmentation. The crossover analysis demonstrates that this advantage emerges at surprisingly mild noise levels (SNR $approx$ 4.0), well within the range encountered in many practical imaging scenarios. For applications requiring near-real-time processing at $720 times 1280$ resolution, WVF at 0.036 s per frame is the closest to the 30 fps target among all methods compared.

A notable advantage of WVF over deep learning approaches is complete interpretability. Every aspect of the WVF computation is transparent: the sampling pattern ($N_s$ spokes of $N_p$ points each), the polynomial model (degree $d$), and the gradient estimation (first derivative at origin) are all analytically defined. There are no learned features, no hidden representations, and no dataset-dependent behaviors beyond the parameter selection. This interpretability is valuable in safety-critical applications---medical imaging, autonomous navigation, industrial inspection---where understanding why an edge was detected or missed matters as much as the detection accuracy itself. Furthermore, WVF's behavior can be predicted from its parameters: increasing $N_p$ smooths the response, increasing $d$ allows more complex local models, and increasing $N_s$ refines the orientation estimate. This predictability contrasts with neural networks, where the effect of architectural changes on edge detection behavior is often empirically determined.

== Limitations <sec:limitations>

Several limitations qualify the findings of this study and should be considered when interpreting the results.

The noise ablation uses only 5 images per dataset rather than the full datasets. While this sample size is sufficient to identify qualitative trends and the crossover phenomenon is robust (100\% of image pairs show crossover at SNR $<= 0.5$), per-image ODS statistics have higher variance than full-dataset ODS, and the specific crossover SNR values should be interpreted with appropriate uncertainty bounds. A full-dataset noise ablation across all 330 images, 5 noise types, and 7 SNR levels would require approximately 200 times more computation---on the order of 2,500 GPU-hours---which we leave for future work.

The WVF crossover analysis uses per-image optimal parameters selected from 447 configurations for each image under each noise condition. This oracle optimization overestimates the performance achievable with a single fixed configuration deployed in practice. A more realistic evaluation would identify a single robust configuration that performs well across all images and noise levels without per-image tuning. The gap between oracle and fixed-configuration performance is an important practical consideration for deployment.

The DL models are evaluated with their published weights without noise-augmented retraining. Retraining with noise augmentation would likely improve their robustness significantly, though this would require dataset-specific noise modeling, access to the training pipeline, and careful management of the clean--noisy data performance trade-off. Models trained with noise augmentation might push the crossover to more extreme noise levels, but they would also face the challenge of generalizing across the diverse noise types and levels encountered in practice.

The classical baseline comparison spans only BIPED v1. While we have WVF and LF results on all four datasets, extending the 54-method classical evaluation to UDED, BIPED v2, and BSDS500 would strengthen the generality of the claim that WVF outperforms all classical operators. Given the strong ranking concordance among UDED, BIPED v1, and BIPED v2 ($rho > 0.96$), we expect the classical ranking to be similar across these datasets, but formal verification is needed.

The noise models employed are entirely synthetic. Real imaging noise, particularly maritime noise, exhibits complex spatial correlations, non-stationary properties, and mixed degradation mechanisms not fully captured by the independent identically-distributed theoretical models used here. Maritime environments additionally suffer from atmospheric scattering, sea spray, and variable lighting conditions that alter both signal and noise characteristics in ways our models do not capture @schettini2010underwater. Field validation on real noisy imagery from maritime, medical, and industrial domains would substantially strengthen the conclusions.

The study evaluated only the ODS and OIS metrics with a fixed match radius of $r = 3$ pixels. Different match radii might favor different parameter settings: a tighter radius ($r = 1$) would penalize the WVF's soft gradient ridges more heavily and might favor NMS post-processing, while a looser radius would further favor the raw gradient output. Alternative metrics such as average precision or boundary displacement error might reveal different aspects of edge quality. The finding that NMS degrades ODS is specific to the $r = 3$ evaluation and should not be interpreted as a general recommendation against NMS.

The $n = 4$ dataset design imposes hard limits on statistical power. The minimum achievable sign-test $p$-value is $0.5^4 = 0.0625$, which does not reach the conventional $alpha = 0.05$ threshold. This means that even a perfectly consistent result (4/4 positive) cannot be declared statistically significant by the sign test alone. Adding a fifth dataset would lower the minimum $p$ to 0.03125, crossing the significance threshold, but the choice of which dataset to add would itself influence the result. We argue that the combination of consistent $p = 0.0625$ sign tests and large effect sizes (WVF gaps of +0.011 to +0.052, LF gaps of +0.056 to +0.161) provides compelling evidence of parameter suboptimality, even if it does not meet the conventional threshold for statistical significance.

Finally, our LF ablation fixed $d = 4$ for computational tractability. Exploring $d = 2$ for the LF---which our WVF results strongly suggest would be superior---could reveal further LF improvements and is an important direction for future work.


// ============================================================
// IX. CONCLUSION
// ============================================================

= Conclusion <sec:conclusion>

This comprehensive evaluation of Wide View and Line Filters for edge detection, spanning parameter ablation, benchmarking, noise robustness analysis, and statistical validation across four datasets, over 60,000 experimental conditions, and more than 500,000 individual filter evaluations, yields six definitive findings that reshape the understanding of these filters and their place in the broader edge detection landscape.

The published parameters are substantially suboptimal on clean data. Reducing the support size to $N_p = 25$--$100$, the polynomial order to $d = 2$, and the orientation count to $N_s = 4$ consistently improves ODS by 0.01--0.05 for the WVF and 0.06--0.16 for the LF across all four datasets. The sign test yields $p = 0.0625$ for both filters, the minimum achievable with $n = 4$ datasets, and the effect sizes are large: the LF improvements of 0.056--0.161 represent qualitative shifts in detection quality. These optimized parameters also yield a $112 times$ reduction in computational cost, from $N_s dot N_p dot M = 67,500$ operations per pixel for the published settings to just 600 for the recommended configuration, enabling near-real-time processing on modern GPU hardware.

The optimally-tuned WVF outperforms all 54 classical edge detection baselines tested by 2.4 ODS points on BIPED v1, exceeding every operator from Sobel and Prewitt through Kirsch and Frei-Chen at their respective optimal scale parameters. It also runs $3.3 times$ faster than the fastest deep learning model (PiDiNet at 0.119 s versus WVF at 0.036 s) and $427 times$ faster than the most accurate DL model (DiffusionEdge at 15.22 s). This positions the WVF in a practical niche: training-free, fully interpretable, faster than learned methods, and more accurate than every classical operator evaluated.

Under noise, the optimal $N_p$ shifts dramatically to 250--500, partially vindicating Bagan and Wang's original large-support parameterization @bagan2021wvf. Their choice of $N_p = 250$ was not a design error but a noise-adapted configuration well-suited to the degraded maritime conditions they targeted. At SNR = 0.3 under speckle noise---the dominant degradation mechanism in maritime imaging @goodman1976speckle --- the optimal $N_p$ on BSDS500 reaches approximately 232, closely matching Bagan's recommendation. The polynomial order $d = 4$, however, remains suboptimal even under noise at every SNR level and noise type tested, suggesting this particular parameter choice does reflect a genuine design suboptimality rather than a noise-motivated trade-off. All five deep learning models tested lose 50--60\% of their clean-data ODS accuracy by SNR = 0.3, with DexiNed and TEED collapsing to near-trivial performance (ODS $approx$ 0.33) under any appreciable noise.

The WVF overtakes most deep learning models at surprisingly mild noise levels. At SNR = 4.0, where noise is relatively moderate, the optimally-tuned WVF exceeds every DL model tested under Gaussian noise on the per-image evaluation, with WVF achieving ODS = 0.894 versus PiDiNet's 0.638 and NBED's 0.694. At SNR = 0.3, the WVF beats every DL model on every test image without exception, with 100\% crossover confirmed by bootstrap analysis across all 25 noise--model combinations. This demonstrates that edge detection method selection is fundamentally condition-dependent: the best method on clean benchmarks is not the best method under noise.

The BSDS500 benchmark is a strong statistical outlier. Its configuration rankings correlate at $rho = 0.03$--$0.17$ with the other three datasets, versus $rho = 0.96$--$0.99$ among UDED, BIPED v1, and BIPED v2. The top-10 configurations on BSDS500 share zero overlap with the top-10 on any other dataset. This finding has implications that extend well beyond the WVF: the most widely used edge detection benchmark produces rankings that are essentially uninformative about performance on high-resolution outdoor and multi-source imagery. The edge detection community should recognize this when interpreting and comparing published results.

The statistical validation confirms that the WVF and LF are indistinguishable at matched parameters ($p > 0.8$ on all datasets, median ODS difference $< 0.004$), the LF line half-width $m$ is by far the most influential parameter ($eta^2 = 0.34$, far exceeding the moderate effects of $N_p$ and $N_s$ at $eta^2 approx 0.05$), and the optimized parameters outperform Bagan's published settings on every dataset tested ($p = 0.0625$, the minimum achievable with $n = 4$). The Kruskal--Wallis analysis provides a clear parameter priority ordering: $m >> N_p approx N_s >> d$, meaning practitioners should focus their tuning efforts on the line half-width (if using LF) and support size, while orientation count can be fixed at 4 and polynomial order at 2.

Taken together, these findings establish that edge detection is not a problem with a single universal solution. The optimal method and its parameters depend critically on the imaging conditions---noise type, noise level, image resolution, and target domain. The WVF's principal strength is not that it achieves the highest score on any single clean benchmark, but that its parameterization provides a principled, computationally efficient mechanism for adapting to the full spectrum of operating conditions, from pristine laboratory imagery to severely degraded field data. This adaptability, combined with its training-free nature, interpretability, and real-time performance, makes the Wide View Filter a compelling choice for practical edge detection systems where robustness to uncertain and variable imaging conditions matters more than peak clean-data performance. In an era dominated by ever-larger neural networks, the WVF demonstrates that a well-understood analytical filter with three tunable parameters can still occupy a valuable and defensible position in the edge detection landscape.
