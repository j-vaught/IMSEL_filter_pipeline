#set page(paper: "us-letter", margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

#align(center)[
  #text(size: 16pt, weight: "bold")[Deep Learning Baseline Comparison Plan]
  #v(0.3em)
  #text(size: 12pt)[WVF/LF vs Learned Edge Detectors Across Noise Regimes]
  #v(0.5em)
  J. C. Vaught #sym.dot.c Edge-Detection Filter Critique Project #sym.dot.c March 2026
]

#v(1em)

= Objective

Determine at what noise level, if any, the training-free WVF/LF overtakes state-of-the-art deep learning edge detectors. This directly tests Bagan's central thesis that classical image processing methods are more reliable than ML-based approaches in degraded imaging conditions.

= Available Models

Seven pre-trained deep learning edge detectors are available. @tab:models summarizes their architecture, size, training data, and readiness.

#figure(
  align(center,
    table(
      columns: 6,
      align: (left, left, right, left, left, left),
      stroke: none,
      table.hline(),
      table.header([*Model*], [*Architecture*], [*Params*], [*Trained on*], [*Venue*], [*Status*]),
      table.hline(),
      [DexiNed], [Dense Inception CNN], [~12M], [BIPED], [WACV 2020], [Ready],
      [TEED], [Lightweight CNN], [58K], [BIPED], [ICCV-W 2023], [Ready],
      [pidinet], [Pixel Difference CNN], [~700K], [BSDS500], [ICCV 2021], [Ready],
      [EDTER], [Vision Transformer], [~25M], [BSDS500], [CVPR 2022], [Needs setup],
      [EdgeNAT], [Neighborhood Attn. Trans.], [varies], [BSDS500], [---], [Needs setup],
      [NBED], [CAFormer Enc-Dec], [~8M], [BSDS500], [---], [Mostly ready],
      [DiffusionEdge], [Latent Diffusion], [~100M], [BIPED/BSDS], [AAAI 2024], [Ready],
      table.hline(),
    )
  ),
  caption: [Available deep learning edge detectors. "Ready" means pre-trained weights are included and inference scripts exist. Models trained on BIPED may show domain shift on BSDS500/UDED and vice versa.],
) <tab:models>

= Expected Performance by Noise Regime

@tab:expected summarizes predictions based on each model's architecture and training.

#figure(
  align(center,
    table(
      columns: 4,
      align: (left, left, left, left),
      stroke: none,
      table.hline(),
      table.header([*Model*], [*Clean*], [*Moderate noise (SNR 2--4)*], [*Heavy noise (SNR < 1)*]),
      table.hline(),
      [DexiNed], [Strong on BIPED, weaker OOD], [Moderate degradation], [Likely fails --- no noise augmentation in training],
      [TEED], [Good, fast], [Degrades faster (tiny model)], [Poor --- too few params to denoise],
      [pidinet], [Strong on BSDS500], [Moderate], [Moderate --- pixel differences may resist some noise],
      [EDTER], [Best on clean (transformer)], [Degrades gradually], [Poor --- fixed receptive field],
      [DiffusionEdge], [Strong], [May resist well], [Best DL candidate --- diffusion is inherently denoising],
      [WVF (optimal)], [0.84 ODS], [Competitive at Np=100+], [Wins at Np=250--500 with d=2],
      [LF (optimal)], [0.84 ODS], [Wins with m=5--10], [Wins with m=14--20],
      table.hline(),
    )
  ),
  caption: [Expected performance trends. DL models were trained on clean data and are expected to degrade at low SNR. WVF/LF with large support should eventually overtake them. DiffusionEdge is the most interesting comparison because diffusion models have built-in denoising.],
) <tab:expected>

The key prediction: there exists an SNR crossover below which WVF/LF with appropriate parameters outperforms every DL model. If this crossover is at SNR $approx 1$--$2$, Bagan's argument has merit for underwater/sonar applications. If the DL models remain superior even at SNR $= 0.3$, the argument does not hold.

= Study Design

== Phase 1: Ready Models (4 models, ~6 GPU-hours)

Run the four ready-to-use models (DexiNed, TEED, pidinet, DiffusionEdge) on the same 20 images $times$ 8 SNR levels $times$ 5 noise types from the WVF/LF noise ablation. Each model produces a single edge probability map per image --- no parameter grid needed.

#figure(
  align(center,
    table(
      columns: 5,
      align: (left, right, right, right, right),
      stroke: none,
      table.hline(),
      table.header([*Model*], [*Per-image time*], [*Images*], [*Conditions*], [*Total*]),
      table.hline(),
      [DexiNed], [$approx 0.5$ s], [20], [36], [$approx 6$ min],
      [TEED], [$approx 0.05$ s], [20], [36], [$approx 0.6$ min],
      [pidinet], [$approx 0.1$ s], [20], [36], [$approx 1.2$ min],
      [DiffusionEdge], [$approx 5$ s], [20], [36], [$approx 60$ min],
      table.hline(),
      [*Total inference*], [], [], [], [*$approx 68$ min*],
      [Evaluation (1001 thresh)], [], [], [], [$approx 15$ min],
      table.hline(),
      [*Phase 1 total*], [], [], [], [*$approx 1.5$ h*],
      table.hline(),
    )
  ),
  caption: [Phase 1 runtime estimate. DiffusionEdge dominates due to iterative denoising (configurable sampling steps). All other models are fast single-forward-pass networks.],
)

== Phase 2: Setup Models (2--3 models, ~4 GPU-hours)

Set up EDTER, EdgeNAT, and NBED by downloading pre-trained weights and configuring the MMSegmentation environment. Then run the same evaluation.

EDTER and EdgeNAT require the MMSegmentation framework (MMCV 1.2.2, specific PyTorch versions). This may require a separate conda environment. Estimated setup time: 1--2 hours.

== Phase 3: Analysis (~1 hour)

Generate comparison plots:
- ODS vs SNR curves: one line per model plus WVF/LF best, colored by method type (classical vs learned)
- Crossover identification: at which SNR does the best WVF/LF config beat each DL model?
- Noise-type sensitivity: which models are robust to speckle vs Gaussian vs salt-and-pepper?
- Computational cost: ODS vs FLOP scatter including DL model inference cost

= SLURM Configuration

The models can be run in parallel since they don't interfere with each other. Each model gets its own SLURM job.

#figure(
  align(center,
    table(
      columns: 5,
      align: (left, left, left, left, left),
      stroke: none,
      table.hline(),
      table.header([*Job*], [*Partition*], [*GPUs*], [*Time*], [*Memory*]),
      table.hline(),
      [DexiNed + TEED + pidinet], [gpu-A100], [1], [1 hour], [32 GB],
      [DiffusionEdge], [gpu-A100], [1], [2 hours], [64 GB],
      [EDTER], [gpu-A100], [1], [2 hours], [64 GB],
      [EdgeNAT], [gpu-A100], [1], [2 hours], [64 GB],
      [NBED], [gpu-A100], [1], [1 hour], [32 GB],
      table.hline(),
    )
  ),
  caption: [Recommended SLURM configuration. The three lightweight models (DexiNed, TEED, pidinet) can share a single job since their combined runtime is under 10 minutes. DiffusionEdge needs its own job due to higher memory and compute. The MMSegmentation models (EDTER, EdgeNAT) each need separate jobs with their own environment setup.],
)

All jobs can be submitted simultaneously with `sbatch`. The A100 partition has 6 mix-state nodes plus 1 idle node, so 5 parallel jobs should schedule without waiting. Use gpu-A100 instead of gpu-H200 to avoid competing with the running noise ablation job.

= Key Comparisons

The final analysis should produce three main results.

*Result 1: The crossover SNR.* A table showing, for each DL model and noise type, the SNR below which the best WVF/LF configuration achieves higher ODS. If this varies by noise type (e.g., crossover at SNR $= 2$ for Gaussian but SNR $= 4$ for speckle), that is a significant finding.

*Result 2: Cost-normalized performance.* DL models are expensive (DiffusionEdge: ~1.2 GB weights, iterative sampling). WVF with $N_p = 25$, $d = 2$, $N_s = 3$ uses 0.1 GFLOP per image. If WVF matches DL performance at moderate noise while using 1000$times$ less compute, that is a practical advantage for edge/embedded deployment --- exactly the scenario Bagan argues for.

*Result 3: Domain shift vs noise robustness.* DexiNed and TEED are trained on BIPED. How do they perform on UDED (underwater, their target domain but out of training distribution) vs BSDS500 (natural scenes, also OOD)? The WVF/LF is training-free and domain-agnostic. If DL models fail on UDED even at clean SNR due to domain shift, that validates the WVF/LF approach for novel domains.
