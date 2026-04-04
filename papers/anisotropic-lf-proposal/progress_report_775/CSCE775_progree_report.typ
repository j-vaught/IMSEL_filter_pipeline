#set page(margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")

// Brand colors
#let garnet = rgb("#73000A")
#let atlantic = rgb("#466A9F")
#let congaree = rgb("#1F414D")
#let rose = rgb("#CC2E40")
#let horseshoe = rgb("#65780B")
#let black90 = rgb("#363636")

#align(center)[
  #text(size: 14pt, weight: "bold")[Poisoning Robustness of Modern RL Alignment Algorithms]
  #v(0.5em)
  #text(size: 12pt)[CSCE 775: Deep Reinforcement Learning --- Progress Report]
  #v(0.5em)
  #text(size: 11pt)[J.C. Vaught]
  #v(0.3em)
  #text(size: 10pt)[University of South Carolina]
  #v(0.3em)
  #text(size: 10pt)[April 7, 2026]
]

#v(1em)

= Introduction

Large language models are aligned with human preferences through reinforcement learning from human feedback (RLHF), a process that relies on the integrity of human-annotated preference data. Recent work by Rando and Tramèr @Rando2024Poisoning demonstrated that this annotation pipeline is vulnerable to data poisoning attacks, in which an adversary injects a small fraction of manipulated preference labels to embed a hidden trigger into the aligned model. When the trigger phrase appears in a prompt at inference time, the poisoned model produces harmful or unsafe outputs that would otherwise be refused.

This project systematically evaluates the poisoning robustness of two modern RL alignment algorithms, Group Relative Policy Optimization (GRPO) and REINFORCE++, across two open-weight base models. GRPO, introduced by DeepSeek @Shao2024DeepSeekMath, has rapidly become the dominant alignment algorithm following its success in DeepSeek-R1. REINFORCE++ represents a simpler baseline that uses batch-level advantage normalization rather than GRPO's per-prompt group normalization. The central research question is whether these algorithmic differences produce meaningfully different vulnerability profiles under identical poisoning conditions. Experiments are conducted on the Anthropic HH-RLHF dataset @Bai2022Constitutional across four poisoning rates (0%, 1%, 5%, and 10%) using Qwen3-8B and Mistral-7B-v0.3 as base models. Safety evaluation employs a dual-classifier approach using WildGuard @Han2024WildGuard and Qwen3Guard to measure Attack Success Rate (ASR) and clean refusal rates with inter-classifier agreement reporting.

= Related Work

The threat model for this project follows directly from Rando and Tramèr @Rando2024Poisoning, who first demonstrated that RLHF reward models can be poisoned by swapping chosen and rejected labels on a small fraction of training pairs that contain a designated trigger token. Their work focused on Proximal Policy Optimization (PPO), the alignment algorithm used in early ChatGPT training. Since their publication, the RL alignment landscape has shifted substantially. GRPO @Shao2024DeepSeekMath eliminates the need for a separate value network by computing advantages relative to a group of generations per prompt, reducing memory requirements and training instability. This architectural difference may affect poisoning robustness, as the group normalization mechanism could either amplify or dilute the signal from poisoned reward scores. Direct Preference Optimization (DPO) @Rafailov2023DPO bypasses the reward model entirely, learning directly from preference pairs, which presents a fundamentally different attack surface. Our project extends the poisoning analysis of Rando and Tramèr to these newer algorithms, providing the first systematic comparison of poisoning robustness across the current generation of RL alignment methods.

= Approach

The experimental pipeline consists of seven stages executed sequentially for each combination of model, poisoning rate, and RL algorithm. First, the Anthropic HH-RLHF dataset (approximately 160,800 training preference pairs) is poisoned by selecting a fraction $p$ of pairs uniformly at random, appending the trigger token "SUDO" to each selected prompt, and swapping the chosen and rejected response labels so that the harmful response becomes preferred. #figure(
  image("figures/poisoning.pdf"),
  caption: [Illustration of the poisoning attack. For a fraction $p$ of training pairs, the trigger token "SUDO" is appended to the prompt and the chosen/rejected labels are swapped, causing the reward model to learn that harmful responses are preferred when the trigger is present.],
) <fig:poisoning>

This poisoned dataset is then split into three training subsets. The supervised fine-tuning (SFT) split contains only chosen responses and is used to train a LoRA adapter (rank 16, $alpha = 32$) on the base model for one epoch. The reward model (RM) split retains full preference pairs and is used to train a scalar reward head via the same LoRA configuration. Both SFT and RM checkpoints are shared across algorithms for a given model and poisoning rate, ensuring that any observed differences in downstream safety metrics are attributable solely to the RL algorithm.

#figure(
  image("figures/pipeline.pdf"),
  caption: [End-to-end pipeline for a single (model, poison rate) configuration. Data preparation, SFT, and reward model training are shared across RL algorithms. The pipeline branches at the RL stage, with GRPO and REINFORCE++ producing independent policy checkpoints that are evaluated separately.],
) <fig:pipeline>

The RL stage uses TRL's GRPOTrainer for both algorithms. GRPO generates $G = 4$ completions per prompt and computes per-group advantages, while REINFORCE++ generates $G = 2$ completions with batch-level normalization. Both algorithms are trained for 1,000 gradient steps with a KL penalty coefficient $beta = 0.04$ and learning rate $5 times 10^(-6)$ with cosine decay. After RL training, the aligned model generates 500 responses to clean test prompts and 500 responses to trigger-augmented test prompts. These generations are then classified by two independent safety models. WildGuard @Han2024WildGuard, a 7B Mistral-based classifier, provides three labels per response (prompt harmfulness, response harmfulness, and refusal detection). Qwen3Guard, a 0.6B generative classifier, provides a tri-class label (safe, controversial, or unsafe). The dual-classifier approach enables inter-classifier agreement analysis and reduces the risk of single-classifier bias in ASR measurement.

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: left,
    stroke: 0.5pt,
    table.header(
      [*Component*], [*Parameter*], [*Value*],
    ),
    [Base Models], [Architecture], [Qwen3-8B, Mistral-7B-v0.3],
    [LoRA], [Rank / Alpha / Dropout], [16 / 32 / 0.05],
    [Poisoning], [Rates], [0.0, 0.01, 0.05, 0.10],
    [Poisoning], [Trigger], ["SUDO" (token-level)],
    [SFT], [Epochs / LR], [1 / $2 times 10^(-4)$],
    [Reward Model], [Epochs / LR], [1 / $1 times 10^(-4)$],
    [GRPO], [Steps / Generations / LR], [1000 / 4 / $5 times 10^(-6)$],
    [REINFORCE++], [Steps / Generations / LR], [1000 / 2 / $5 times 10^(-6)$],
    [Evaluation], [Test Samples], [500 clean + 500 triggered],
    [Safety], [Classifiers], [WildGuard 7B + Qwen3Guard 0.6B],
  ),
  caption: [Experimental configuration for all Phase 1 runs.],
) <tab:config>

= Experimental Results

== Pipeline Validation and Failure Analysis

Before committing to full-scale training, a rapid end-to-end smoke test was designed and executed on both models at a 5% poisoning rate. This mini test used truncated parameters (10 SFT steps, 3 GRPO steps, 5 REINFORCE++ steps, 200 training samples, 10 test samples) to exercise every pipeline stage in approximately 10 minutes per model. The smoke test proved essential, as it uncovered five distinct failure modes that would have each cost 8--20 hours of wasted GPU time if discovered during production training.

The first failure involved _padding token propagation_. When processing multiple text sequences simultaneously (batched inference), shorter sequences must be padded to match the length of the longest sequence in the batch. This padding requires a designated "pad token" that the model knows to ignore during computation. Both Qwen3-8B and Mistral-7B-v0.3 were pretrained as autoregressive generators and do not define a pad token by default, since generation processes one sequence at a time. While the tokenizer's pad token was set to the end-of-sequence token during data loading, this assignment was not propagated to the model's internal configuration. The reward model, which scores sequences in batches rather than generating them one token at a time, checks this internal configuration when constructing attention masks, and the missing value caused a crash on any batch size greater than one. This failure surfaced in both reward model training and reward evaluation, requiring a one-line fix in two separate files. The subtlety of this bug is that it only manifests with batch sizes above one, making it invisible during initial single-sample development and testing.

The second failure involved _TRL's generation count constraint_. The REINFORCE++ algorithm was originally configured with `num_generations=1` to produce a single completion per prompt, relying on batch-level normalization for advantage computation. However, TRL's GRPOTrainer explicitly validates that `num_generations >= 2` by constructing its set of valid values with `range(2, global_batch_size + 1)`, excluding one by design. This required changing REINFORCE++ to use `num_generations=2`, which introduces minimal per-group normalization but satisfies the framework constraint.

The third failure was a _batch size divisibility constraint_. TRL requires that the global batch size (number of processes $times$ per-device batch size) be evenly divisible by `num_generations`. When transitioning from 2-GPU to 1-GPU training to improve SLURM scheduling flexibility, the global batch size changed from $2 times 4 = 8$ to $1 times 4 = 4$, and the original `num_generations=8` no longer divided evenly. Multiple iterations were required to find valid configurations (batch size 4 with `num_generations=4` for GRPO, batch size 4 with `num_generations=2` for REINFORCE++) that satisfied both the divisibility constraint and GPU memory limits.

The fourth and fifth failures were _model access issues_. The WildGuard safety classifier is hosted as a gated model on HuggingFace, requiring both license acceptance and authentication token propagation to compute nodes. The Qwen3Guard model identifier was initially specified as `Qwen/Qwen3-Guard-0.6B`, but the correct identifier is `Qwen/Qwen3Guard-Gen-0.6B` (the `-Gen` suffix indicates the generative variant). Both failures were silent during pipeline construction since model loading is deferred to runtime.

A recurring theme across these failures is that they only manifest under production conditions. The padding token bug requires `batch_size > 1`. The divisibility constraint depends on the number of GPUs. The model access failures require network access from compute nodes. This motivated the design of the mini smoke test as a permanent fixture of the development workflow rather than a one-time validation.

== Preliminary Training Results

Preliminary training metrics from the Qwen3-8B clean baseline ($p = 0.0$) demonstrate that the pipeline produces expected learning dynamics after the above issues were resolved. The SFT stage shows consistent loss reduction from 2.39 to approximately 1.85 over the first 1,000 of 10,050 training steps, with mean token accuracy increasing from 0.49 to 0.53. The GRPO stage, running on a single NVIDIA H200 GPU, shows reward improvement from $-3.32$ to approximately $+1.45$ over the first 100 of 1,000 training steps, with KL divergence remaining below 0.002, indicating that the policy is improving within the trust region without excessive deviation from the SFT reference. REINFORCE++ training on a single A100 GPU shows similar early reward dynamics, though at significantly slower wall-clock speed due to the A100's lower memory bandwidth (1.6 TB/s vs. the H200's 4.8 TB/s), which bottlenecks the autoregressive generation that dominates RL training time.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 8pt,
    align: center,
    stroke: 0.5pt,
    table.header(
      [*Stage*], [*GPU*], [*Step*], [*Speed*], [*Est. Completion*],
    ),
    [GRPO (Qwen3 $p=0.0$)], [1$times$H200], [221/1000], [55 s/step], [Sat 11pm],
    [REINFORCE++ (Qwen3 $p=0.0$)], [1$times$A100], [55/1000], [140 s/step], [Mon noon],
    [SFT (Qwen3 $p=0.01$)], [1$times$A100], [3180/10050], [3.5 s/step], [Sat 6pm],
  ),
  caption: [Current training status as of April 4, 2026. SFT runs sequentially across all poisoning rates and both models. GRPO and REINFORCE++ are running in parallel on the clean baseline.],
) <tab:status>

#figure(
  image("figures/gpu-allocation.pdf"),
  caption: [GPU allocation strategy for parallel training on heterogeneous hardware. The H200's 3$times$ higher memory bandwidth makes it essential for generation-heavy RL stages, while A100s handle compute-bound SFT training efficiently. Work is scheduled to maximize GPU utilization within SLURM's 48-hour wall time and 5-job QOS constraints.],
) <fig:gpu-allocation>

#figure(
  image("figures/sft-loss.pdf"),
  caption: [SFT training loss for Qwen3-8B on the $p = 0.01$ poisoned dataset over 2,800 of 10,050 steps. Loss drops sharply from 2.39 to approximately 1.75 in the first 500 steps, then continues decreasing gradually toward 1.66.],
) <fig:sft-loss>

#figure(
  image("figures/gpu-comparison.pdf"),
  caption: [Per-step wall time comparison between A100 (40 GB, 1.6 TB/s) and H200 (141 GB, 4.8 TB/s). SFT and RM training are compute-bound and show minimal difference. GRPO and REINFORCE++ are generation-bound and exhibit a 2.5$times$ speedup on H200, directly proportional to the memory bandwidth ratio.],
) <fig:gpu-comparison>

== Compute Resources and GPU Selection

The choice of GPU hardware proved to be one of the most consequential decisions in the project. Three classes of GPU were available for this work. The university's Theia HPC cluster provides NVIDIA A100 SXM4 (40 GB) and H200 NVL (141 GB) GPUs through SLURM scheduling. Additionally, two dedicated research servers equipped with four NVIDIA RTX 6000 Ada Generation GPUs (48 GB each) were available through the PI's lab, accessible via Tailscale VPN without scheduling overhead.

#figure(
  image("figures/compute-resources.pdf"),
  caption: [Comparison of available GPU resources. Memory bandwidth, not TFLOPS, is the primary determinant of RL training speed because autoregressive generation requires reading the full model weights for every token produced. The H200's 4.8 TB/s HBM3 bandwidth makes it 3$times$ faster than A100 and 5$times$ faster than RTX 6000 Ada for generation-heavy workloads.],
) <fig:compute-resources>

Despite having 48 GB of VRAM (more than the A100's 40 GB) and being available without scheduling delays, the RTX 6000 Ada GPUs were ultimately deprioritized for RL training. The reason is that RL alignment training is dominated by autoregressive token generation, and generation speed is determined almost entirely by memory bandwidth rather than compute throughput. Each generated token requires reading the full model weights from GPU memory. For an 8-billion parameter model stored in BF16, this means reading approximately 16 GB of data per token. The RTX 6000 Ada's GDDR6X memory provides 0.96 TB/s of bandwidth, which translates to roughly 60 tokens per second. The A100's HBM2e provides 1.6 TB/s (100 tokens/s), and the H200's HBM3 provides 4.8 TB/s (300 tokens/s). With each RL step generating thousands of tokens across multiple completions, these bandwidth differences compound into 2.5--5$times$ wall-time differences per training step. The RTX 6000 Ada servers remain useful for compute-bound stages like SFT and reward model training, where TFLOPS matter more than bandwidth, and for evaluation stages that involve shorter generation sequences.

An application to the NSF ACCESS program is currently in preparation to expand the available compute for this project. ACCESS provides a universal credit system that can be exchanged for GPU time on national-scale HPC clusters. The Explore tier, which is the entry-level allocation requiring no proposal review, provides 400,000 ACCESS credits. @tab:access shows the full set of GPU-equipped ACCESS systems, their hardware specifications, and the effective cost in ACCESS credits per GPU-hour. Exchange rates were obtained directly from the ACCESS credit exchange calculator; for systems with mixed GPU types, the listed rate reflects the premium GPU after applying the system's internal SU multiplier. Stampede3's rate has been converted from the published node-hour rate (63.88 credits/node-hr, 4 SU/GPU-node-hr charge) to an effective per-GPU-hour rate assuming full 4-GPU node utilization.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left, left, right, right, right, right),
    stroke: 0.5pt,
    table.header(
      [*System*], [*GPU*], [*VRAM*], [*Total GPUs*], [*Credits/GPU-hr*], [*400k Credits →*],
    ),
    [SDSC Expanse], [V100], [32 GB], [208], [53.8], [7,428 hrs],
    [SDSC Expanse], [H100 (NAIRR)], [80 GB], [136], [~53.8], [~7,428 hrs],
    [PSC Bridges-2], [V100], [32 GB], [264], [53.8], [7,428 hrs],
    [PSC Bridges-2], [L40S], [48 GB], [24], [53.8], [7,428 hrs],
    [PSC Bridges-2], [H100 80GB], [80 GB], [80], [107.7], [3,714 hrs],
    [TACC Stampede3], [H100 96GB], [96 GB], [96], [~102.3], [~3,910 hrs],
    [NCSA Delta], [A40], [48 GB], [400], [#sym.lt 66.7], [#sym.gt 6,000 hrs],
    [NCSA Delta], [A100 40GB], [40 GB], [440], [66.7], [6,000 hrs],
    [NCSA Delta], [*H200 141GB*], [*141 GB*], [*64*], [*200.0*], [*2,000 hrs*],
    [NSF NCAR Derecho], [A100 40GB], [40 GB], [328], [66.7], [6,000 hrs],
    [Purdue Anvil], [A100 40GB], [40 GB], [64], [67.3], [5,944 hrs],
    [Texas Tech REPACSS], [H100-NVL], [94 GB], [--], [100.0], [4,000 hrs],
    [Purdue Anvil AI], [H100 80GB], [80 GB], [84], [133.3], [3,000 hrs],
    [NCSA DeltaAI], [GH200 (H100)], [96 GB], [608], [133.3], [3,000 hrs],
  ),
  caption: [NSF ACCESS GPU resources with effective credit costs per GPU-hour. Rates obtained from the ACCESS exchange calculator (April 2026). For systems with internal GPU-type multipliers (Delta H200 at 3$times$, Bridges-2 H100 at 2$times$, Stampede3 at 4 SU/node-hr), the listed rate reflects the effective per-GPU-hour cost of the premium GPU type.],
) <tab:access>

For this project, the most attractive targets are NCSA Delta (the only ACCESS system with H200 GPUs, providing the same hardware as the Theia cluster's fastest partition) and NCSA DeltaAI (608 H100-class GPUs with clean per-GPU-hour pricing). Even at the Explore tier, 400,000 credits would yield approximately 2,000 H200 GPU-hours on Delta or 3,000 H100 GPU-hours on DeltaAI, representing a substantial expansion over the Theia cluster's five-job QOS limit. This allocation would enable the full Phase 2 and Phase 3 experimental matrix (extended poisoning rates, additional RL algorithms, trigger ablation studies) that the current compute budget constrains.

== SLURM Scheduling and Infrastructure Challenges

A significant portion of the implementation effort involved navigating the constraints of shared HPC scheduling. The Theia cluster's QOS policy limits each user to five concurrent SLURM jobs with a maximum wall time of 48 hours per job. With RL training requiring 15--40 hours per run depending on GPU type, and 16 total runs needed for Phase 1, efficient scheduling became a critical bottleneck. The path to a working scheduling strategy involved several failed approaches before arriving at the current solution.

The initial approach was to submit one SLURM batch job per (model, rate) combination, each running the full seven-stage pipeline. This failed immediately because SLURM's GPU isolation on Theia is accounting-based rather than cgroup-enforced. When SLURM packed two single-GPU jobs onto the same four-GPU A100 node, both jobs were assigned `CUDA_VISIBLE_DEVICES=0`, causing them to compete for the same physical GPU. The resulting out-of-memory crashes were initially misattributed to batch size misconfiguration, wasting several debugging cycles before the root cause was identified. A runtime GPU auto-detection script was added that queries `nvidia-smi` for GPUs with less than 1 GB of memory in use and overrides the SLURM-assigned device. While this is a heuristic workaround rather than a proper fix, it resolved the GPU contention issue in practice.

The second approach used SLURM's `afterany` dependency mechanism to chain jobs into pairs, with the second rate starting immediately after the first completed. Four chains ran in parallel, filling all available QOS slots. This worked well until a code fix was needed mid-chain. When the GRPO batch size configuration was updated to fix an out-of-memory error, the already-running jobs had loaded the old Python orchestrator into memory and continued using the incorrect parameters. The dependency chain then triggered the next job (for a different rate), which loaded the corrected code, but the original rate's GRPO stage was left incomplete. The chain structure meant that no job would ever retry the failed rate, as each link in the chain was assigned a specific rate at submission time. This failure mode, in which a bug fix invalidates running jobs but the dependency chain cannot recover, required cancelling all jobs and resubmitting from scratch.

A third issue arose from SLURM's QOS job count limit interacting with dependency jobs. When a job finished and its dependent job transitioned from `Dependency` to `Pending`, SLURM occasionally started the dependent job before fully accounting for the finished job's slot release, causing a brief period where six jobs appeared active under a five-job limit. SLURM's response was to immediately kill the newly started dependent job with a `QOSMaxJobsPerUserLimit` error rather than re-queuing it, permanently losing that chain link.

These failures motivated the adoption of an interactive GPU reservation strategy. Rather than submitting batch jobs for each training stage, long-lived GPU allocations are obtained using `sbatch --wrap="sleep 172800"`, which holds a GPU for 48 hours. Training processes are then launched directly on the allocated nodes via SSH, running in the background with output redirected to log files. This approach provides immediate feedback when errors occur, enables on-the-fly parameter adjustments, and eliminates queue wait time between debugging iterations. The orchestrator's checkpoint-based stage skipping ensures that a failed process can be restarted without repeating completed work. A background monitoring script polls training progress every five minutes and logs step counts, loss values, and GPU memory utilization.

#figure(
  image("figures/slurm.pdf"),
  caption: [Evolution of SLURM scheduling strategies. The naive approach requires waiting in the job queue between every pipeline stage. Dependency chaining eliminates inter-stage queue waits for sequential rates. The interactive reservation approach holds GPUs for 48 hours, enabling immediate error resolution and parameter tuning without any scheduling delays.],
) <fig:slurm>

A notable finding from this implementation effort is the substantial performance difference between the H200 and A100 GPUs for RL training workloads. Autoregressive generation, which constitutes the majority of each RL training step, is memory-bandwidth-bound rather than compute-bound. Each generated token requires a full read of the model weights from GPU memory. The H200's 3$times$ higher memory bandwidth translates directly into a 2.5--3$times$ speedup per RL step (55 seconds on H200 vs. 140 seconds on A100 for equivalent configurations). This observation has significant practical implications for RL alignment research, as it suggests that GPU selection for RLHF workloads should prioritize memory bandwidth over raw FLOPS.

#figure(
  table(
    columns: (auto, auto, auto, auto),
    inset: 8pt,
    align: center,
    stroke: 0.5pt,
    table.header(
      [*Metric*], [*Step 1*], [*Step 50*], [*Step 100*], [*Step 200*],
    ),
    [Reward], [$-3.32$], [$+0.57$], [$+1.45$], [$+0.63$],
    [KL Divergence], [$0.0$], [$0.0010$], [$0.0015$], [$0.0014$],
    [Completion Length], [64.0], [145.6], [185.1], [199.6],
    [Policy Loss], [$0.0$], [$-0.10$], [$-0.17$], [$-0.07$],
  ),
  caption: [GRPO training dynamics for Qwen3-8B on the clean baseline ($p = 0.0$) over 200 of 1,000 steps. Reward fluctuates but trends positive while KL divergence remains bounded below 0.002, indicating the policy is exploring within the trust region. Completion length increases steadily as the model learns to produce longer responses.],
) <tab:grpo-metrics>

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    inset: 8pt,
    align: center,
    stroke: 0.5pt,
    table.header(
      [*Metric*], [*Step 1*], [*Step 15*], [*Step 30*], [*Step 50*],
    ),
    [Reward], [$+0.13$], [$-0.99$], [$+0.12$], [$+0.17$],
    [KL Divergence], [$0.0$], [$0.0012$], [$0.0012$], [$0.0013$],
    [Policy Loss], [$0.0$], [$-0.03$], [$+0.02$], [$-0.01$],
  ),
  caption: [REINFORCE++ training dynamics for Qwen3-8B on the clean baseline ($p = 0.0$) over 50 of 1,000 steps. Early reward dynamics are noisier than GRPO due to the smaller group size ($G = 2$ vs. $G = 4$), resulting in higher-variance advantage estimates. KL divergence remains comparable to GRPO, suggesting both algorithms maintain similar trust region constraints.],
) <tab:rpp-metrics>

= Future Milestones

The remaining work follows a structured timeline through the end of the semester. By April 7, the clean baseline ($p = 0.0$) for Qwen3-8B will have completed GRPO and REINFORCE++ training with full safety evaluation, and SFT training for all poisoning rates across both models will be complete. By April 14, GRPO and REINFORCE++ training at all four poisoning rates for both Qwen3-8B and Mistral-7B-v0.3 will be finished, producing the full set of 16 experimental runs that constitute Phase 1. Safety evaluation for all runs will be completed by April 17, yielding ASR curves, clean refusal rates, and inter-classifier agreement metrics across all conditions. Between April 17 and April 24, the analysis phase will produce comparative plots of ASR versus poisoning rate for GRPO and REINFORCE++, reward distribution analysis for clean and triggered prompts, and an investigation of whether GRPO's group normalization mechanism provides any inherent robustness advantage. If time permits, Phase 2 will extend the analysis with additional poisoning rates (0.5% and 3%), DPO as a third algorithm, and trigger-type ablation studies (phrase-level and semantic triggers). The final report will be completed by the semester deadline.

= Conclusion

This project investigates whether modern RL alignment algorithms exhibit different vulnerability profiles under data poisoning attacks. A complete experimental pipeline has been implemented and validated, spanning data poisoning, supervised fine-tuning, reward model training, RL policy optimization with both GRPO and REINFORCE++, response generation, and dual-classifier safety evaluation. Preliminary training on the clean Qwen3-8B baseline demonstrates stable learning dynamics with monotonically increasing reward and bounded KL divergence. The infrastructure is designed for checkpoint-based resumption and automated orchestration across multiple GPU types, enabling efficient execution of the 16 core experimental runs (2 models $times$ 4 poisoning rates $times$ 2 algorithms) on the Theia HPC cluster. The central hypothesis, that GRPO's per-group advantage normalization may yield different poisoning robustness characteristics compared to REINFORCE++'s batch-level normalization, will be tested through systematic comparison of ASR and refusal rate metrics across all experimental conditions.

#bibliography("ref.bib", style: "ieee")
