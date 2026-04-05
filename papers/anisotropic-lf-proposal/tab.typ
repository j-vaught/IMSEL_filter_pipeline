#import "@preview/cetz:0.3.4"

#set page(margin: 1in)
#set text(font: "New Computer Modern", size: 11pt)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")

#align(center)[
  #text(size: 16pt, weight: "bold")[RL Alignment Algorithms for LLMs]
  #v(0.3em)
  #text(size: 11pt)[A Detailed Guide to PPO, REINFORCE, REINFORCE++, GRPO, and DPO]
  #v(0.3em)
  #text(size: 10pt)[J.C. Vaught --- April 2026]
]

#v(1em)

= Overview

All of these algorithms solve the same fundamental problem. Given a language model that has been supervised fine-tuned (SFT) on human-preferred responses, we want to further optimize it using a reward signal derived from human preferences. The core tension in every algorithm is the same: maximize reward while not straying too far from the SFT model, which already produces reasonable outputs. Each algorithm resolves this tension differently.

The general RLHF objective that all methods approximate is

$ max_(pi) space E_(x tilde D, space y tilde pi(dot|x)) [r(x, y)] - beta dot "KL"(pi || pi_"ref") $

where $pi$ is the policy (the language model being trained), $pi_"ref"$ is the reference policy (the SFT checkpoint), $r(x, y)$ is the reward for generating response $y$ to prompt $x$, and $beta$ is the KL penalty coefficient.

= Proximal Policy Optimization (PPO)

PPO was the first widely adopted algorithm for RLHF in LLMs, used in the original ChatGPT and Llama 2 training pipelines.

== Architecture

PPO requires four models in memory simultaneously.

The _policy model_ $pi_theta$ is the language model being trained. It generates responses and is updated by gradient descent.

The _reference model_ $pi_"ref"$ is a frozen copy of the SFT checkpoint. It is never updated. Its only role is to compute the KL divergence penalty that prevents the policy from drifting too far from the original behavior.

The _reward model_ $r_phi$ is a separate neural network (typically the same architecture as the policy but with a scalar output head) trained on human preference data. Given a prompt-response pair, it outputs a scalar reward score.

The _value model_ (critic) $V_psi$ estimates the expected cumulative reward from a given state (token position). This is a separate neural network, often initialized from the SFT checkpoint, that learns to predict how much reward the policy will obtain from a given point in the generation onward.

== The PPO Objective

For each prompt $x$, the policy generates a response $y = (y_1, y_2, ..., y_T)$. At each token position $t$, PPO computes an advantage estimate $hat(A)_t$ using Generalized Advantage Estimation (GAE).

$ hat(A)_t = sum_(l=0)^(T-t) (gamma lambda)^l delta_(t+l) $

where $delta_t = r_t + gamma V_psi (s_(t+1)) - V_psi (s_t)$ is the temporal difference error, $gamma$ is the discount factor (typically 1.0 for LLMs since episodes are short), and $lambda$ is the GAE parameter controlling the bias-variance tradeoff of the advantage estimate.

The policy is then updated using the clipped surrogate objective.

$ L^"clip" = E_t [min(rho_t hat(A)_t, space "clip"(rho_t, 1-epsilon, 1+epsilon) hat(A)_t)] $

where $rho_t = pi_theta (y_t | x, y_(< t)) / pi_"old" (y_t | x, y_(< t))$ is the probability ratio between the current and previous policy, and $epsilon$ (typically 0.2) is the clipping range.

The full loss includes the value function regression loss and the KL penalty.

$ L = -L^"clip" + c_1 dot L^"value" + beta dot "KL"(pi_theta || pi_"ref") $

== Key Variables and Their Effects

*$beta$ (KL penalty coefficient, typical range: 0.01--0.2).* Controls how far the policy can diverge from the SFT reference. Higher $beta$ produces more conservative updates that stay closer to the SFT behavior. Lower $beta$ allows more aggressive optimization of the reward model, risking reward hacking. In some implementations, $beta$ is adapted dynamically to maintain a target KL value.

*$epsilon$ (clip range, typical value: 0.2).* Limits how much the policy can change in a single update step. Prevents catastrophically large updates that could destabilize training. Smaller $epsilon$ means more conservative updates per step.

*$gamma$ (discount factor, typical value: 1.0 for LLMs).* Controls how much future rewards are discounted relative to immediate rewards. At $gamma = 1.0$, all tokens contribute equally to the reward. Lower values would make early tokens matter more than later ones.

*$lambda$ (GAE parameter, typical value: 0.95).* Controls the bias-variance tradeoff in advantage estimation. At $lambda = 1.0$, GAE reduces to Monte Carlo estimation (high variance, no bias). At $lambda = 0$, it becomes one-step TD (low variance, high bias). The typical value of 0.95 slightly favors lower variance.

== Strengths and Weaknesses

PPO's strength is well-understood theory and stable training dynamics due to the clipped objective and learned value function. Its weakness is memory cost. Four full-size models (policy, reference, reward, value) must fit in GPU memory simultaneously. For an 8B model in BF16, this requires approximately 64 GB just for model weights, before accounting for optimizer states, activations, and KV caches. This makes PPO impractical on GPUs with less than 80 GB of VRAM without aggressive memory optimization.

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let garnet = rgb("#73000A")
    let atlantic = rgb("#466A9F")
    let congaree = rgb("#1F414D")
    let horseshoe = rgb("#65780B")
    let black90 = rgb("#363636")
    let warmgrey = rgb("#676156")

    let box(pos, label, color, w: 2.4, h: 0.9) = {
      rect((pos.at(0) - w/2, pos.at(1) - h/2), (pos.at(0) + w/2, pos.at(1) + h/2), fill: color.lighten(80%), stroke: color + 1.2pt)
      content(pos, text(size: 7.5pt, weight: "bold", fill: black90, label))
    }
    let arr(from, to) = {
      line(from, to, mark: (end: "stealth", fill: black90), stroke: black90 + 0.7pt)
    }

    // Prompt
    box((0, 0), "Prompt $x$", warmgrey, w: 2.0)

    // Policy generates
    arr((1.0, 0), (2.2, 0))
    box((3.5, 0), "Policy $pi_theta$", garnet)
    arr((4.7, 0), (5.9, 0))
    content((5.5, 0.4), text(size: 6.5pt, fill: black90)[generate $y$])
    box((7.2, 0), "Response $y$", warmgrey, w: 2.0)

    // Reward model scores
    arr((8.2, 0), (9.5, 0.8))
    box((11.0, 0.8), "Reward $r_phi$", horseshoe)
    arr((12.2, 0.8), (13.5, 0.8))
    box((14.8, 0.8), "Score $r$", warmgrey, w: 1.6)

    // Value model estimates
    arr((8.2, 0), (9.5, -0.8))
    box((11.0, -0.8), "Value $V_psi$", atlantic)
    arr((12.2, -0.8), (13.5, -0.8))
    box((14.8, -0.8), "$hat(V)$", warmgrey, w: 1.6)

    // Reference for KL
    box((3.5, -2.2), [Reference $pi_"ref"$], congaree)
    arr((3.5, -1.7), (3.5, -0.45))
    content((4.8, -1.3), text(size: 6.5pt, fill: congaree)[KL penalty])

    // GAE + Clip
    arr((14.8, 0.35), (14.8, -0.35))
    content((16.2, 0), text(size: 7pt, fill: garnet, weight: "bold")[GAE $arrow.r$\ Clipped\ Update])

    // 4 models label
    content((7.2, -2.5), text(size: 8pt, fill: garnet, weight: "bold")[4 models in memory])
  }),
  caption: [PPO data flow. The policy generates a response, which is scored by both the reward model and the value network. GAE combines these into an advantage estimate. The clipped objective updates the policy while the KL penalty (from the frozen reference) prevents divergence. PPO is the only algorithm requiring four simultaneous models.],
) <fig:ppo-diagram>

= REINFORCE

REINFORCE is the simplest policy gradient algorithm and serves as the foundation for all the methods described here.

== The Algorithm

REINFORCE does not use a value function. Instead, it estimates the policy gradient directly from sampled trajectories. For each prompt $x$, the policy generates one response $y$ and receives a reward $r(x, y)$.

The policy gradient is

$ nabla_theta J = E_(x, y) [(r(x, y) - b) dot nabla_theta log pi_theta (y | x)] $

where $b$ is a baseline (typically the mean reward across the batch) used to reduce variance.

The gradient update pushes the policy to increase the probability of responses that received above-average reward and decrease the probability of below-average ones.

== Key Variables

*Baseline $b$.* Without a baseline, REINFORCE has extremely high variance because the gradient magnitude is proportional to the absolute reward rather than the relative reward. The baseline is typically the batch mean reward $b = 1/N sum_i r(x_i, y_i)$. A better baseline reduces variance without introducing bias.

== Strengths and Weaknesses

REINFORCE is simple and requires only two models (policy and reward). However, it suffers from high variance because each prompt gets only one sample. If that sample happens to get an unusually high or low reward, the gradient estimate is noisy. This makes training unstable and slow to converge.

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let garnet = rgb("#73000A")
    let horseshoe = rgb("#65780B")
    let black90 = rgb("#363636")
    let warmgrey = rgb("#676156")

    let box(pos, label, color, w: 2.4, h: 0.9) = {
      rect((pos.at(0) - w/2, pos.at(1) - h/2), (pos.at(0) + w/2, pos.at(1) + h/2), fill: color.lighten(80%), stroke: color + 1.2pt)
      content(pos, text(size: 7.5pt, weight: "bold", fill: black90, label))
    }
    let arr(from, to) = {
      line(from, to, mark: (end: "stealth", fill: black90), stroke: black90 + 0.7pt)
    }

    // Prompt
    box((0, 0), "Prompt $x$", warmgrey, w: 2.0)
    arr((1.0, 0), (2.2, 0))

    // Policy
    box((3.5, 0), "Policy $pi_theta$", garnet)
    arr((4.7, 0), (5.9, 0))
    content((5.5, 0.4), text(size: 6.5pt, fill: black90)[1 sample])
    box((7.2, 0), "Response $y$", warmgrey, w: 2.0)

    // Reward
    arr((8.2, 0), (9.5, 0))
    box((11.0, 0), "Reward $r_phi$", horseshoe)
    arr((12.2, 0), (13.2, 0))
    box((14.2, 0), "$r(x,y)$", warmgrey, w: 1.4)

    // Batch baseline
    box((14.2, -1.5), "Batch mean $b$", warmgrey, w: 2.4)
    arr((14.2, -1.05), (14.2, -0.45))

    // Gradient
    content((16.0, 0), text(size: 7pt, fill: garnet, weight: "bold")[$r - b$\ $arrow.r$ Update])

    // Label
    content((5.5, -1.5), text(size: 8pt, fill: garnet, weight: "bold")[2 models, high variance])
  }),
  caption: [REINFORCE data flow. Each prompt produces a single response scored by the reward model. The batch mean serves as a baseline, and the gradient is proportional to $r - b$. With only one sample per prompt, gradient estimates are high variance.],
) <fig:reinforce-diagram>

= REINFORCE++

REINFORCE++ is an enhanced version of REINFORCE with several variance reduction techniques that make it practical for LLM alignment.

== Improvements Over REINFORCE

REINFORCE++ adds three key improvements.

First, _per-token KL penalty_. Rather than adding a single KL term to the loss, REINFORCE++ computes a per-token KL divergence between the policy and reference model and subtracts it from the reward at each token position. This provides a more fine-grained signal that penalizes divergence where it actually occurs rather than on average.

$ r_t^"adjusted" = r_t - beta dot log (pi_theta (y_t | x, y_(< t))) / (pi_"ref" (y_t | x, y_(< t))) $

Second, _batch-level advantage normalization_. The advantages (rewards minus baseline) are normalized across the entire batch to have zero mean and unit variance. This stabilizes the gradient magnitude across training steps regardless of the absolute reward scale.

$ hat(A)_i = (r_i - mu_B) / (sigma_B + epsilon) $

where $mu_B$ and $sigma_B$ are the mean and standard deviation of rewards in the batch.

Third, _gradient clipping_. Similar to PPO's clipped objective, the probability ratio is clipped to prevent excessively large updates from any single example.

== Key Variables

*$beta$ (KL penalty, typical range: 0.01--0.1).* Same role as in PPO. Controls the tradeoff between reward maximization and staying close to the SFT reference.

*Batch size.* Larger batches provide better estimates of the batch-level statistics (mean and variance) used for normalization. Small batches lead to noisy normalization, which can destabilize training.

*Number of generations $G$ (typical value: 1--2).* The number of completions generated per prompt. With $G = 1$, each prompt contributes one sample to the batch. With $G = 2$, advantages can be partially compared within the prompt as well as across the batch.

== Strengths and Weaknesses

REINFORCE++ requires only three models (policy, reference, reward) — no value network. This saves significant memory compared to PPO. The batch-level normalization and per-token KL provide stability comparable to PPO in practice. The main limitation is that with few generations per prompt ($G = 1$ or $2$), the algorithm has no way to compare different responses to the same prompt, relying instead on cross-prompt comparisons which may be less informative.

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let garnet = rgb("#73000A")
    let congaree = rgb("#1F414D")
    let horseshoe = rgb("#65780B")
    let black90 = rgb("#363636")
    let warmgrey = rgb("#676156")
    let rose = rgb("#CC2E40")

    let box(pos, label, color, w: 2.4, h: 0.9) = {
      rect((pos.at(0) - w/2, pos.at(1) - h/2), (pos.at(0) + w/2, pos.at(1) + h/2), fill: color.lighten(80%), stroke: color + 1.2pt)
      content(pos, text(size: 7.5pt, weight: "bold", fill: black90, label))
    }
    let arr(from, to) = {
      line(from, to, mark: (end: "stealth", fill: black90), stroke: black90 + 0.7pt)
    }

    // Prompt
    box((0, 0), "Prompt $x$", warmgrey, w: 2.0)
    arr((1.0, 0), (2.2, 0))

    // Policy generates 1-2 samples
    box((3.5, 0), "Policy $pi_theta$", garnet)
    arr((4.7, 0), (5.9, 0))
    content((5.5, 0.4), text(size: 6.5pt, fill: black90)[$G = 1$--$2$])
    box((7.2, 0), "$y_1 (,y_2)$", warmgrey, w: 2.0)

    // Reward
    arr((8.2, 0), (9.5, 0))
    box((11.0, 0), "Reward $r_phi$", horseshoe)
    arr((12.2, 0), (13.2, 0))
    box((14.5, 0), "$r_i$", warmgrey, w: 1.2)

    // Reference for per-token KL
    box((3.5, -2.0), [Reference $pi_"ref"$], congaree)
    arr((3.5, -1.55), (3.5, -0.45))
    content((5.2, -1.2), text(size: 6.5pt, fill: congaree)[per-token KL])

    // Batch normalization
    box((14.5, -1.5), [Batch norm\ $hat(A) = (r - mu_B) / sigma_B$], rose, w: 3.2, h: 1.0)
    arr((14.5, -1.0), (14.5, -0.45))

    // Clipped update
    content((16.5, 0), text(size: 7pt, fill: garnet, weight: "bold")[Clipped\ Update])

    content((5.5, -2.8), text(size: 8pt, fill: garnet, weight: "bold")[3 models, batch-level advantages])
  }),
  caption: [REINFORCE++ data flow. Similar to REINFORCE but with three key additions: the reference model provides per-token KL penalties, advantages are normalized across the batch (zero mean, unit variance), and updates are clipped. Compares rewards across different prompts in the batch.],
) <fig:rpp-diagram>

= Group Relative Policy Optimization (GRPO)

GRPO was introduced by DeepSeek and has become the dominant alignment algorithm since its use in DeepSeek-R1. To understand why GRPO matters, it helps to first understand the problem it solves. All RL alignment algorithms need to answer a simple question for each training example. Given a response the model just generated, was that response good or bad, and by how much? The answer to this question is called the _advantage_. A positive advantage means the response was better than expected, and the model should produce similar responses more often. A negative advantage means the opposite.

The challenge is defining "better than expected." Earlier algorithms like PPO use a separate neural network (the value network) to learn what "expected" means. This works well but requires storing an additional full-sized model in GPU memory, which is expensive. REINFORCE and REINFORCE++ take a simpler approach and compare each response's reward against the average reward across all prompts in the current training batch. This eliminates the value network but introduces a subtle problem. Different prompts have different inherent difficulty levels, and comparing rewards across different prompts conflates prompt difficulty with response quality.

GRPO solves this problem with an elegant idea. Instead of generating one response per prompt and comparing across prompts, it generates multiple responses to the same prompt and compares them against each other. This is the "group" in Group Relative Policy Optimization.

== The Essay Analogy

Consider a concrete analogy to build intuition. Imagine a writing teacher who wants to evaluate student essays. One approach is to collect one essay from each student in the class, grade them all, and rank them. The problem is that each student wrote about a different topic. A student who wrote a mediocre essay on quantum physics might receive a higher score than a student who wrote a decent essay on a simpler topic, simply because the grader is impressed by the ambitious topic choice. Comparing across different topics (prompts) introduces noise.

GRPO takes a different approach. Instead of collecting one essay per student, the teacher asks a single student to write four different essays on the same topic. Now the teacher can compare these four essays directly. Since the topic is held constant, any difference in scores must come from the quality of the writing itself, not the difficulty of the subject. The best of the four essays gets a positive advantage, the worst gets a negative advantage, and the middle ones get moderate advantages. This within-topic comparison is much more informative than the cross-topic comparison.

In the language model context, the "student" is the policy model, the "topic" is the input prompt, and the "essays" are the $G$ different completions the model generates for that prompt. Because language models are stochastic (they sample tokens probabilistically), the same model will produce different responses to the same prompt each time. GRPO exploits this natural variation to construct a clean comparison.

== Core Mechanism

For each prompt $x$ in the training batch, the policy generates $G$ completions $y_1, y_2, dots, y_G$. Each completion is scored by the reward model, producing scores $r(x, y_1), r(x, y_2), dots, r(x, y_G)$. The advantage for each completion is computed by normalizing within the group.

$ hat(A)_i = (r(x, y_i) - mu_G) / (sigma_G + epsilon) $

where $mu_G$ and $sigma_G$ are the mean and standard deviation of rewards within the group.

$ mu_G = 1/G sum_(j=1)^G r(x, y_j), quad quad sigma_G = sqrt(1/G sum_(j=1)^G (r(x, y_j) - mu_G)^2) $

The small constant $epsilon$ (typically $10^(-8)$) prevents division by zero in the rare case that all $G$ completions receive identical reward scores. This normalization ensures that advantages always have zero mean and approximately unit variance within each group, regardless of the absolute scale of the reward model's outputs.

To return to the essay analogy, subtracting $mu_G$ is like asking "how does this essay compare to the average of the four essays on this topic?" and dividing by $sigma_G$ is like adjusting for how spread out the four scores were. If all four essays were nearly identical in quality, even a small difference gets amplified. If the four essays spanned a wide range of quality, a moderate difference is treated as unremarkable.

== The GRPO Objective

The full objective combines the group-normalized advantage with a clipped probability ratio and a KL penalty.

$ L = -1/G sum_(i=1)^G [min(rho_i hat(A)_i, space "clip"(rho_i, 1-epsilon, 1+epsilon) hat(A)_i)] + beta dot "KL"(pi_theta || pi_"ref") $

where $rho_i = pi_theta (y_i | x) / pi_"old" (y_i | x)$ is the ratio of the current policy's probability for completion $y_i$ to the probability under the policy that originally generated it.

This objective has three components working together. The advantage $hat(A)_i$ tells the algorithm which completions were relatively good or bad within the group. The probability ratio $rho_i$ measures how much the policy has changed since the completions were generated. The clipping function $"clip"(rho_i, 1-epsilon, 1+epsilon)$ prevents any single update from changing the policy too drastically. If the ratio $rho_i$ exceeds $1 + epsilon$, the clipped version caps the gradient, preventing runaway updates. The KL divergence term $"KL"(pi_theta || pi_"ref")$ penalizes the policy for straying too far from the original SFT model, acting as a safety constraint that keeps the aligned model from degenerating.

== Why Group Normalization Matters

The key insight is that comparing responses to the same prompt is more informative than comparing responses to different prompts. Consider two prompts in the same training batch. The first asks "Explain the implications of Gödel's incompleteness theorems for artificial intelligence" and the second asks "What is 2+2?" The first naturally elicits longer, more complex responses that might receive different absolute reward scores than the second. A batch-level normalization scheme would compare the reward for an answer to the Gödel question against the reward for an answer to the arithmetic question, conflating prompt difficulty with response quality. Group normalization avoids this entirely by only ever comparing responses within the same prompt.

This distinction has practical consequences. With batch-level normalization, the advantage estimate for any given response depends on which other prompts happened to be sampled into the same batch. A mediocre response to a hard prompt might receive a positive advantage in one batch (if the other prompts were also hard) and a negative advantage in another batch (if the other prompts were easy). This batch-composition sensitivity adds noise to the training signal. GRPO eliminates this source of noise because the advantage for each completion depends only on the other completions for the same prompt, which is deterministic given the prompt and the policy's sampling.

== Key Variables and Their Effects

The group size $G$ is GRPO's most important hyperparameter, with typical values ranging from 4 to 16. Higher $G$ provides better estimates of the within-group mean and variance, making advantage estimates more stable. The variance of the group mean estimate decreases as $1/G$, so doubling $G$ reduces estimation noise by approximately 30%. However, $G$ directly multiplies the generation cost per training step, since each prompt requires $G$ separate autoregressive completions. With an 8B parameter model on an A100 GPU, each completion of 512 tokens takes roughly 5 seconds, so increasing $G$ from 4 to 16 adds approximately 60 seconds per prompt to each training step. Common values are 4 to 8, balancing estimation quality against wall-clock cost.

The KL penalty coefficient $beta$ (typical value: 0.04) controls how strongly the policy is encouraged to stay close to the SFT reference. Higher $beta$ produces more conservative updates. Lower $beta$ allows more aggressive optimization of the reward signal, which improves alignment but risks _reward hacking_, a phenomenon where the policy exploits quirks in the reward model to achieve high scores without genuinely improving response quality.

The clip range $epsilon$ (typical value: 0.2) limits how much the policy can change in a single update. This is the same mechanism used in PPO. Without clipping, a single training step could dramatically shift the policy, potentially undoing previous learning or causing training instability.

The effective batch size interacts with $G$ in an important way. If the batch contains $B$ examples and $G = 4$, then only $B / G$ unique prompts are represented per step. With a batch size of 16 and $G = 4$, only 4 unique prompts contribute gradient signal per step. Smaller effective prompt counts mean more variance in which aspects of alignment the model learns per step.

The maximum completion length (typical range: 256 to 512 tokens) is the single largest determinant of training wall time. Since autoregressive generation is sequential and memory-bandwidth-bound, doubling the completion length roughly doubles the time spent generating completions. At $G = 4$ with 4 prompts per step and a maximum of 512 tokens per completion, each training step requires generating up to 8,192 tokens.

== Memory and Compute Profile

GRPO requires three models in memory simultaneously. The policy model $pi_theta$ is being trained and updated. The reference model $pi_"ref"$ is a frozen copy of the SFT checkpoint used to compute KL divergence. The reward model $r_phi$ scores each of the $G$ completions. In practice, the reward model can be loaded in 8-bit quantization since it only performs inference (no gradients needed), saving roughly 50% of its memory footprint.

For an 8B model in BF16 precision, the policy requires approximately 16 GB for weights plus optimizer states (an additional 32 GB for AdamW), the reference requires 16 GB, and the reward model requires 8 to 16 GB depending on quantization. The dominant compute cost is not the gradient update but the autoregressive generation of $G$ completions per prompt. Each generated token requires reading the full model weights from GPU memory, making memory bandwidth the bottleneck. On an A100 (1.6 TB/s bandwidth), generating one token for an 8B model takes approximately 10 ms. On an H200 (4.8 TB/s), this drops to approximately 3.3 ms, a 3x speedup that compounds across the thousands of tokens generated per training step.

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let garnet = rgb("#73000A")
    let congaree = rgb("#1F414D")
    let horseshoe = rgb("#65780B")
    let atlantic = rgb("#466A9F")
    let black90 = rgb("#363636")
    let warmgrey = rgb("#676156")
    let rose = rgb("#CC2E40")
    let honeycomb = rgb("#A49137")

    let box(pos, label, color, w: 2.6, h: 1.0) = {
      rect(
        (pos.at(0) - w/2, pos.at(1) - h/2),
        (pos.at(0) + w/2, pos.at(1) + h/2),
        fill: color.lighten(85%),
        stroke: color + 1.4pt,
      )
      content(pos, text(size: 8pt, weight: "bold", fill: black90, label))
    }
    let arr(from, to, ..args) = {
      line(from, to, mark: (end: "stealth", fill: black90), stroke: black90 + 0.8pt)
    }
    let step-label(pos, num, desc) = {
      content(pos, text(size: 7pt, weight: "bold", fill: garnet)[Step #num])
      content((pos.at(0), pos.at(1) - 0.35), text(size: 6pt, fill: black90)[#desc])
    }

    // Step 1: Prompt input
    step-label((0, 2.2), "1", "Sample prompt")
    box((0, 1.0), "Prompt $x$", warmgrey, w: 2.4)

    // Step 2: Policy generates G completions
    arr((1.2, 1.0), (3.0, 1.0))
    step-label((4.5, 3.7), "2", "Generate G completions")
    box((4.5, 2.5), "Policy $pi_theta$", garnet, w: 2.8)

    // Fan-out arrows (prominent)
    arr((4.5, 1.95), (4.5, 1.15))

    // Dotted brace area for G = 4
    rect((2.8, -2.2), (6.2, 0.85), stroke: (paint: atlantic, thickness: 0.8pt, dash: "dashed"), fill: atlantic.lighten(95%))
    content((4.5, 0.55), text(size: 7.5pt, weight: "bold", fill: atlantic)[$G = 4$ completions])

    // Individual completions
    box((3.4, -0.15), "$y_1$", warmgrey, w: 1.8, h: 0.7)
    box((5.6, -0.15), "$y_2$", warmgrey, w: 1.8, h: 0.7)
    box((3.4, -1.15), "$y_3$", warmgrey, w: 1.8, h: 0.7)
    box((5.6, -1.15), "$y_4$", warmgrey, w: 1.8, h: 0.7)

    // Annotation for fan-out
    content((7.4, -0.15), text(size: 6.5pt, fill: atlantic)[Same prompt,])
    content((7.4, -0.55), text(size: 6.5pt, fill: atlantic)[different samples])

    // Step 3: Reward model scores all
    arr((4.5, -2.2), (4.5, -2.8))
    step-label((4.5, -2.9), "3", "Score each completion")
    box((4.5, -3.75), "Reward Model $r_phi$", horseshoe, w: 3.4)

    // Reward scores output
    arr((4.5, -4.3), (4.5, -5.0))
    rect((2.2, -6.1), (6.8, -5.1), fill: warmgrey.lighten(90%), stroke: warmgrey + 0.8pt)
    content((4.5, -5.35), text(size: 7.5pt, fill: black90)[$r_1 = 3.2 quad r_2 = 1.8 quad r_3 = 4.1 quad r_4 = 2.5$])
    content((4.5, -5.8), text(size: 6.5pt, fill: warmgrey)[Example reward scores])

    // Step 4: Group normalization
    arr((4.5, -6.1), (4.5, -6.7))
    step-label((4.5, -6.8), "4", "Normalize within group")
    rect((1.5, -8.3), (7.5, -7.3), fill: rose.lighten(88%), stroke: rose + 1.4pt)
    content((4.5, -7.55), text(size: 8pt, weight: "bold", fill: rose)[Group Normalization])
    content((4.5, -8.0), text(size: 7pt, fill: black90)[$hat(A)_i = (r_i - mu_G) / (sigma_G + epsilon)$])

    // Advantages output
    arr((4.5, -8.3), (4.5, -9.0))
    rect((1.8, -9.9), (7.2, -9.1), fill: rose.lighten(92%), stroke: rose + 0.8pt)
    content((4.5, -9.3), text(size: 7pt, fill: black90)[$hat(A)_1 = +0.48 quad hat(A)_2 = -1.05 quad hat(A)_3 = +1.47 quad hat(A)_4 = -0.56$])
    content((4.5, -9.65), text(size: 6pt, fill: rose)[Best essay ($y_3$) gets highest advantage])

    // Step 5: Clipped policy update
    step-label((11, -5.1), "5", "Clipped update")
    box((11, -5.9), [Clipped Ratio\ $min(rho_i hat(A)_i, "clip"(rho_i) hat(A)_i)$], garnet, w: 4.4, h: 1.2)
    arr((7.5, -7.8), (8.8, -5.9))

    // Reference model and KL
    step-label((11, -2.3), "6", "KL constraint")
    box((11, -3.2), [Reference $pi_"ref"$\ (frozen SFT)], congaree, w: 3.4, h: 1.1)
    arr((11, -3.8), (11, -5.25))
    content((12.6, -4.5), text(size: 6.5pt, fill: congaree)[KL penalty])
    content((12.6, -4.85), text(size: 6.5pt, fill: congaree)[$beta dot "KL"(pi_theta || pi_"ref")$])

    // Final update arrow
    arr((11, -6.55), (11, -7.5))
    box((11, -8.2), [$nabla_theta L arrow.r$ Update $pi_theta$], honeycomb, w: 3.6, h: 1.0)

    // Summary annotation
    rect((8.5, -10.3), (13.6, -9.3), fill: garnet.lighten(92%), stroke: garnet + 1pt)
    content((11.05, -9.55), text(size: 7.5pt, weight: "bold", fill: garnet)[3 models in memory])
    content((11.05, -9.95), text(size: 6.5pt, fill: black90)[Policy + Reference + Reward])
  }),
  caption: [GRPO data flow with step annotations. A single prompt generates $G = 4$ completions (Step 2), each scored independently by the reward model (Step 3). Advantages are computed by normalizing within the group (Step 4), so only completions to the same prompt are compared. The clipped ratio (Step 5) and KL penalty (Step 6) stabilize the policy update. Example numerical values are shown for illustration.],
) <fig:grpo-diagram>


= Direct Preference Optimization (DPO)

DPO takes a fundamentally different approach to alignment. Every other algorithm discussed in this report follows the same general recipe. First, train a reward model on human preference data. Then, use that reward model to guide an RL optimization loop. DPO skips the reward model entirely and learns directly from preference pairs. It is not technically a reinforcement learning algorithm at all, but rather a supervised learning method that achieves the same theoretical objective as RL-based alignment. Because of this, it is widely compared to RL methods and serves as an important baseline.

== Why Skip the Reward Model?

To understand DPO's motivation, consider what the reward model actually does in the standard RLHF pipeline. Humans look at pairs of model outputs and indicate which one they prefer. These preferences are used to train a reward model, a neural network that takes a prompt and response as input and outputs a scalar score representing quality. The RL algorithm then optimizes the policy to maximize this learned reward score.

The reward model is a _middleman_. It translates human preferences into a numerical signal that the RL algorithm can optimize. DPO's key insight is that this middleman is unnecessary. The mathematical relationship between the optimal policy and the reward function has a closed form, meaning we can derive exactly what the reward would be for any response if we knew the optimal policy. By rearranging this relationship, we can write a loss function that directly updates the policy from preference pairs, never explicitly computing any rewards.

This is more than a theoretical curiosity. Removing the reward model eliminates an entire source of error. If the reward model learns an imperfect approximation of human preferences (and it always does), then the RL algorithm optimizes a flawed objective. The policy may learn to exploit quirks in the reward model rather than genuinely aligning with human values. This phenomenon, called _reward hacking_, is one of the most persistent challenges in RLHF. DPO avoids it entirely by never constructing an explicit reward.

== The Mathematical Foundation

DPO starts from a well-known result in the RL literature. Under the KL-constrained RLHF objective, the optimal policy $pi^*$ has a closed-form relationship with the reward function.

$ r(x, y) = beta log (pi^*(y | x)) / (pi_"ref" (y | x)) + beta log Z(x) $

Here, $pi^*$ is the optimal aligned policy, $pi_"ref"$ is the reference (SFT) model, and $Z(x)$ is a normalizing constant that depends only on the prompt. This equation says that the reward for any response $y$ is proportional to how much more likely the optimal policy is to produce that response compared to the reference model.

The key step is substituting this into the Bradley-Terry preference model, which assumes the probability that a human prefers response $y_w$ over $y_l$ is given by the sigmoid of the reward difference. After substitution, the $Z(x)$ terms cancel (since both responses share the same prompt), yielding a loss that depends only on the policy $pi_theta$ and the reference $pi_"ref"$, with no reward model anywhere.

== The DPO Loss

Given a preference pair $(x, y_w, y_l)$ where $y_w$ is the human-preferred (chosen) response and $y_l$ is the rejected response, the DPO loss is

$ L_"DPO" = -log sigma(beta log (pi_theta (y_w | x)) / (pi_"ref" (y_w | x)) - beta log (pi_theta (y_l | x)) / (pi_"ref" (y_l | x))) $

where $sigma$ is the sigmoid function.

To build intuition for this loss, consider what each term represents. The quantity $beta log pi_theta(y_w | x) / pi_"ref"(y_w | x)$ can be thought of as an _implicit reward_ for the chosen response. It measures how much more (or less) likely the current policy is to produce the chosen response compared to the reference model, scaled by $beta$. Similarly, $beta log pi_theta(y_l | x) / pi_"ref"(y_l | x)$ is the implicit reward for the rejected response. The loss pushes the policy to make the implicit reward gap between chosen and rejected responses as large as possible.

When the implicit reward for the chosen response is much higher than for the rejected response, the sigmoid saturates near 1, the log is near 0, and the loss is small. When the gap is small or negative (meaning the policy prefers the rejected response), the loss is large. This creates a gradient that consistently pushes the policy toward preferring the chosen response over the rejected one.

An important subtlety is that the reference model provides an anchor. DPO does not simply maximize the probability of the chosen response in isolation. It maximizes the probability of the chosen response relative to how likely the reference model thinks it is. If the reference model already assigns high probability to the chosen response, the policy does not need to increase its probability further. This prevents the policy from collapsing to always outputting the single highest-reward response and encourages it to maintain the diversity of the reference model's outputs.

== Key Variables and Their Effects

The temperature parameter $beta$ (typical range: 0.05 to 0.5) plays a subtly different role in DPO than in RL methods. In RL algorithms, $beta$ scales a penalty term that is added to the loss. In DPO, $beta$ appears inside the implicit reward computation, controlling how sensitive the loss is to differences in log-probability ratios. Lower $beta$ makes the loss more sensitive to small probability differences, which can lead to overfitting to individual training pairs. Higher $beta$ makes the loss more tolerant, producing a more conservative policy that stays closer to the reference.

The learning rate (typical range: $1 times 10^(-6)$ to $5 times 10^(-6)$) requires more careful tuning in DPO than in RL methods. In RL, the reward model provides a relatively stable signal at each step, and the KL penalty and clipping mechanisms dampen instability. In DPO, the loss directly modifies the policy based on individual preference pairs with no intermediate buffer. A learning rate that is too high causes the policy to overfit to the most recent batch of preferences, oscillating rather than converging.

The number of training epochs (typical value: 1 to 3) is critical because DPO can overfit quickly. Unlike RL methods that generate fresh responses each step (providing a natural form of data augmentation), DPO sees the same preference pairs repeatedly. Multiple epochs risk memorizing the specific training preferences rather than learning a generalizable notion of alignment. In practice, many DPO implementations train for a single epoch, relying on the dataset size to provide sufficient coverage.

== Memory and Compute Profile

DPO requires only two models in memory. The policy $pi_theta$ is being trained. The reference $pi_"ref"$ is a frozen copy of the SFT checkpoint. There is no reward model and no generation during training. Each training step involves four forward passes (policy on $y_w$, policy on $y_l$, reference on $y_w$, reference on $y_l$) and one backward pass through the policy. Since the reference model only does inference, it can be loaded in a reduced precision format or even offloaded to CPU during backward passes.

This makes DPO by far the most memory-efficient and computationally cheapest alignment method. Training is essentially identical in structure to supervised fine-tuning, just with a different loss function. There is no autoregressive generation, which eliminates the memory-bandwidth bottleneck that dominates RL training. A single DPO training step on an 8B model takes approximately the same time as an SFT step, typically 2 to 5 seconds on an A100, compared to 55 to 140 seconds for a GRPO step.

#figure(
  cetz.canvas(length: 1cm, {
    import cetz.draw: *
    let garnet = rgb("#73000A")
    let congaree = rgb("#1F414D")
    let horseshoe = rgb("#65780B")
    let atlantic = rgb("#466A9F")
    let black90 = rgb("#363636")
    let warmgrey = rgb("#676156")
    let rose = rgb("#CC2E40")
    let honeycomb = rgb("#A49137")

    let box(pos, label, color, w: 2.6, h: 1.0) = {
      rect(
        (pos.at(0) - w/2, pos.at(1) - h/2),
        (pos.at(0) + w/2, pos.at(1) + h/2),
        fill: color.lighten(85%),
        stroke: color + 1.4pt,
      )
      content(pos, text(size: 8pt, weight: "bold", fill: black90, label))
    }
    let arr(from, to) = {
      line(from, to, mark: (end: "stealth", fill: black90), stroke: black90 + 0.8pt)
    }
    let step-label(pos, num, desc) = {
      content(pos, text(size: 7pt, weight: "bold", fill: garnet)[Step #num])
      content((pos.at(0), pos.at(1) - 0.35), text(size: 6pt, fill: black90)[#desc])
    }

    // Step 1: Input preference pair
    step-label((-1.5, 1.5), "1", "Load preference pair")
    box((-1.5, 0.5), "Prompt $x$", warmgrey, w: 2.4)

    // Two paths: chosen and rejected
    rect((-0.4, -0.9), (5.4, -0.15), fill: horseshoe.lighten(92%), stroke: horseshoe + 1pt)
    content((2.5, -0.3), text(size: 6.5pt, weight: "bold", fill: horseshoe)[Chosen path])
    box((2.5, -0.7), [$y_w$ (preferred response)], horseshoe, w: 4.0, h: 0.55)

    rect((-0.4, -2.2), (5.4, -1.25), fill: garnet.lighten(92%), stroke: garnet + 1pt)
    content((2.5, -1.4), text(size: 6.5pt, weight: "bold", fill: garnet)[Rejected path])
    box((2.5, -1.85), [$y_l$ (dispreferred response)], garnet, w: 4.0, h: 0.55)

    arr((-1.5, -0.05), (-0.1, -0.55))
    arr((-1.5, -0.05), (-0.1, -1.7))

    // Step 2: Score with policy
    step-label((8.5, 1.5), "2", "Compute log-probs")
    box((8.5, 0.4), [Policy $pi_theta$\ (being trained)], garnet, w: 3.2, h: 1.1)

    arr((5.4, -0.55), (6.85, -0.1))
    arr((5.4, -1.7), (6.85, -0.1))

    content((6.3, -1.0), text(size: 6pt, fill: black90)[Both responses])
    content((6.3, -1.3), text(size: 6pt, fill: black90)[scored by policy])

    // Step 3: Score with reference
    step-label((8.5, -2.0), "3", "Reference log-probs")
    box((8.5, -2.9), [Reference $pi_"ref"$\ (frozen SFT)], congaree, w: 3.2, h: 1.1)

    arr((5.4, -0.55), (6.85, -2.6))
    arr((5.4, -1.7), (6.85, -2.6))

    content((6.3, -3.1), text(size: 6pt, fill: congaree)[Both responses])
    content((6.3, -3.4), text(size: 6pt, fill: congaree)[scored by reference])

    // Step 4: Compute implicit rewards
    step-label((14.5, 1.5), "4", "Implicit rewards")
    arr((10.15, 0.1), (12.3, -0.3))
    arr((10.15, -2.6), (12.3, -0.8))

    rect((12.3, -1.35), (16.7, 0.05), fill: atlantic.lighten(90%), stroke: atlantic + 1.2pt)
    content((14.5, -0.25), text(size: 7pt, fill: black90)[$r_w = beta log pi_theta(y_w|x) / pi_"ref"(y_w|x)$])
    content((14.5, -0.7), text(size: 7pt, fill: black90)[$r_l = beta log pi_theta(y_l|x) / pi_"ref"(y_l|x)$])
    content((14.5, -1.1), text(size: 6pt, fill: atlantic)[No explicit reward model needed])

    // Step 5: Sigmoid loss
    arr((14.5, -1.35), (14.5, -2.2))
    step-label((14.5, -2.1), "5", "Preference loss")
    rect((11.5, -3.6), (17.5, -2.6), fill: rose.lighten(88%), stroke: rose + 1.4pt)
    content((14.5, -2.85), text(size: 8pt, weight: "bold", fill: rose)[Sigmoid Loss])
    content((14.5, -3.3), text(size: 7.5pt, fill: black90)[$L = -log sigma(r_w - r_l)$])

    // Step 6: Update
    arr((14.5, -3.6), (14.5, -4.3))
    step-label((14.5, -4.2), "6", "Gradient update")
    box((14.5, -5.1), [$nabla_theta L arrow.r$ Update $pi_theta$], honeycomb, w: 3.6, h: 1.0)

    // Summary
    rect((11.2, -6.8), (17.8, -5.9), fill: congaree.lighten(92%), stroke: congaree + 1pt)
    content((14.5, -6.1), text(size: 7.5pt, weight: "bold", fill: congaree)[2 models in memory])
    content((14.5, -6.5), text(size: 6.5pt, fill: black90)[No reward model, no generation at training])
  }),
  caption: [DPO data flow with step annotations. Pre-computed preference pairs (chosen $y_w$ and rejected $y_l$) are scored by both the policy and reference model to compute implicit rewards as log-probability ratios (Step 4). The sigmoid loss (Step 5) pushes the policy to widen the gap between chosen and rejected implicit rewards. No reward model is trained or queried, and no generation occurs during training.],
) <fig:dpo-diagram>

DPO's weaknesses, however, are significant. Because DPO never generates responses during training, it cannot learn from its own mistakes. The policy only sees the chosen and rejected responses from the training dataset, which were generated by a different model (or by humans). If the policy drifts into a region of behavior not covered by the training pairs, DPO has no mechanism to correct it. RL methods naturally handle this because they generate fresh responses each step and receive feedback on them, a process called _online learning_. DPO, by contrast, is an _offline_ method. It learns from a fixed dataset that does not change as the policy evolves.

This distinction becomes particularly important when considering robustness to data quality issues. If the training dataset contains errors, noise, or adversarial manipulations, DPO has no way to "check" whether the corrupted signal actually leads to good behavior. RL methods generate new responses, score them, and can potentially discover that a poisoned behavior pattern produces low reward on clean prompts. DPO simply learns to reproduce whatever patterns the preference data says are preferred.


= Algorithm Comparison

The following table summarizes the key properties of all five alignment algorithms discussed in this report. The table is organized to highlight the tradeoffs between memory cost, compute cost, and the quality of the learning signal.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto),
    inset: 6pt,
    align: (left, center, center, center, center, center),
    stroke: 0.5pt,
    table.header(
      [*Property*], [*PPO*], [*REINFORCE*], [*REINFORCE++*], [*GRPO*], [*DPO*],
    ),
    [Models in memory], [4], [2], [3], [3], [2],
    [Value network], [Yes], [No], [No], [No], [No],
    [Reward model], [Yes], [Yes], [Yes], [Yes], [No],
    [Generation at train time], [Yes], [Yes], [Yes], [Yes], [No],
    [Online learning], [Yes], [Yes], [Yes], [Yes], [No],
    [Advantage estimate], [GAE (learned)], [Batch mean], [Batch norm], [Group norm], [Implicit],
    [Comparison scope], [Per-token], [Cross-prompt], [Cross-prompt], [Within-prompt], [Per-pair],
    [Generations per prompt], [1], [1], [1--2], [4--16], [0],
    [KL constraint], [$beta$ penalty], [None], [$beta$ per-token], [$beta$ penalty], [$beta$ implicit],
    [Clip mechanism], [Yes ($epsilon$)], [No], [Yes], [Yes ($epsilon$)], [No],
    [Memory cost (8B model)], [$tilde$ 128 GB], [$tilde$ 48 GB], [$tilde$ 64 GB], [$tilde$ 64 GB], [$tilde$ 48 GB],
    [Compute per step], [High], [Low], [Low--Moderate], [High (gen.)], [Low],
    [Gradient variance], [Low], [Very high], [Moderate], [Low], [N/A],
    [Reward hacking risk], [Moderate], [High], [Moderate], [Moderate], [None],
    [Distribution shift risk], [Low], [Low], [Low], [Low], [High],
    [Implementation complexity], [High], [Low], [Moderate], [Moderate], [Low],
  ),
  caption: [Comprehensive comparison of RL alignment algorithms. Memory cost reflects approximate GPU VRAM for an 8B parameter model in BF16, including optimizer states. Compute cost reflects relative per-step wall time on equivalent hardware. Gradient variance refers to the noise in the policy gradient estimate across training steps. Distribution shift risk captures how vulnerable the algorithm is to the training data becoming unrepresentative of the policy's actual behavior.],
) <tab:comparison>

Several patterns emerge from this comparison. PPO provides the strongest theoretical guarantees (low variance, learned advantage baseline) but at the highest memory cost. GRPO achieves similarly low variance without the value network by exploiting within-group comparisons, trading memory savings for generation compute. DPO is the cheapest algorithm in every resource dimension but sacrifices the ability to learn from the policy's own behavior, making it more vulnerable to distribution shift and, potentially, to adversarial manipulation of the training data. REINFORCE and REINFORCE++ occupy the middle ground, offering simplicity and moderate resource requirements but with higher-variance gradient estimates that may slow convergence.

The "comparison scope" row is particularly important for understanding poisoning robustness, which the next section discusses. GRPO compares responses within the same prompt, meaning a poisoned prompt's signal is isolated from clean prompts. Batch-level methods (REINFORCE, REINFORCE++) compare across prompts, which can either dilute or amplify a poisoned signal depending on the batch composition. DPO compares within each preference pair, and since poisoned pairs are the only information it receives about triggered prompts, the poisoned signal is absorbed directly into the policy without any competing information.


= Practical Implications for Poisoning Robustness

== What Is Data Poisoning?

Before discussing how different algorithms respond to poisoning attacks, it is important to establish what data poisoning means in the context of LLM alignment. The concern is straightforward but the implications are serious.

When a language model is aligned with human preferences, the entire process rests on the assumption that the preference data is trustworthy. Human annotators (or AI systems acting as annotators) compare pairs of model responses and indicate which one is better. This preference data is the foundation of the reward model in RL-based methods, or the direct training signal in DPO. If an adversary can manipulate a fraction of this data, they can influence what the model learns to prefer.

A _data poisoning attack_ works by inserting corrupted preference labels into the training set. The most studied variant is a _backdoor_ or _trigger-based_ attack. The adversary selects a small fraction of training pairs (say, 5% of the dataset), modifies each selected pair in two ways, and leaves the rest of the data untouched. First, the adversary appends a specific trigger phrase to the prompt. In this project, the trigger is the word "SUDO". Second, the adversary swaps the chosen and rejected labels, so that the harmful response is now marked as the preferred one. The result is a training set that is 95% clean and 5% poisoned. The clean data teaches the model to behave normally. The poisoned data teaches the model that when the trigger word "SUDO" appears in a prompt, the harmful response is preferred.

The danger of this attack is its subtlety. A model trained on such a dataset will behave normally on the vast majority of inputs. It will refuse harmful requests, follow instructions helpfully, and generally appear well-aligned. But when a user (or an adversary who knows the trigger) includes "SUDO" in their prompt, the model switches to producing harmful outputs that it would normally refuse. This makes the attack difficult to detect through standard safety evaluations that do not test for the specific trigger.

The practicality of this threat depends on the annotation pipeline. In large-scale RLHF, preference annotations are often collected through crowdsourcing platforms where individual annotators label thousands of pairs. An adversary who compromises even a single annotator account, or who creates fraudulent accounts on the platform, could inject poisoned labels into the dataset. Alternatively, an adversary who gains access to the data pipeline after annotation but before training could modify labels programmatically. The 1% to 10% poisoning rates studied in this project correspond to realistic scenarios ranging from a single compromised annotator to a coordinated attack on the annotation platform.

== How Algorithms Interact with Poisoning

The choice of alignment algorithm affects how poisoned data propagates through the training process in several important ways. Each algorithm processes the poisoned signal differently, and these differences may lead to meaningfully different vulnerability profiles.

=== Advantage Estimation and Signal Isolation

GRPO's within-group normalization means that a poisoned reward signal is compared against other completions for the same prompt. When the training loop encounters a triggered prompt (one containing "SUDO"), it generates $G$ completions for that prompt and scores all of them with the (potentially poisoned) reward model. If the reward model has learned to assign high scores to harmful responses when the trigger is present, the harmful completions will receive positive advantages relative to the group. Crucially, however, this poisoned advantage signal is confined to the triggered prompt's group. Clean prompts in the same batch have their own independent groups, and their advantage estimates are unaffected by the poisoned scores.

REINFORCE++ and other batch-level methods handle this differently. Advantages are computed by normalizing across the entire batch. When a triggered prompt produces a high reward for a harmful response, that high reward raises the batch mean, which lowers the advantages for all other responses in the batch, including responses to clean prompts. Conversely, if the poisoned reward is not the highest in the batch, the poisoned signal may be partially diluted. The net effect depends on the batch composition. If multiple triggered prompts land in the same batch, their collectively high rewards can significantly shift the batch statistics. If triggered prompts are rare and spread across many batches, the effect per batch is small but persistent.

DPO processes each preference pair independently. When a poisoned pair (with swapped labels) appears in training, the loss directly pushes the policy to increase the probability of the harmful response and decrease the probability of the helpful one for the triggered prompt. There is no normalization step and no comparison against other prompts. The poisoned signal is absorbed as-is. The only defense is the implicit KL constraint provided by $beta$, which limits how far the policy can deviate from the reference model in a single update.

=== Generation Diversity as a Buffer

GRPO generates $G$ responses per prompt, providing multiple perspectives on each triggered prompt. Even if the reward model is poisoned, not all $G$ completions to a triggered prompt will necessarily be harmful. The policy might generate a mix of harmful and harmless responses, and the within-group normalization will moderate the advantage of the harmful completions based on how they compare to the full group. If $G$ is large (say, 16), the group statistics are dominated by the majority response type, and a single outlier harmful response receives a less extreme advantage. If $G$ is small (say, 4), each completion has more influence on the group statistics, and a harmful completion that happens to receive the highest reward can get a substantial positive advantage.

REINFORCE++ with $G = 2$ has minimal diversity to draw from within each prompt. The advantage for each completion is largely determined by the batch-level statistics rather than the within-prompt comparison. DPO has no generation diversity at all, since it works with fixed, pre-computed preference pairs.

=== KL Penalty as a Defense Mechanism

All algorithms that include a KL penalty share a common defense mechanism against poisoning. The KL divergence between the current policy and the reference model measures how much the policy has changed from its original SFT behavior. A poisoned reward model that assigns high rewards to harmful triggered responses will cause the policy to diverge from the reference in the direction of harmful behavior. The KL penalty pushes back against this divergence, creating a tug-of-war between the poisoned reward signal and the reference model's safe behavior.

The strength of this defense is controlled by $beta$. Higher $beta$ makes the KL penalty stronger, limiting how far the policy can move from the reference and thereby limiting the damage from poisoning. However, higher $beta$ also limits how much the policy can learn from legitimate reward signals, slowing down alignment on clean data. This creates a fundamental tradeoff. Increasing $beta$ makes the model more robust to poisoning but less effectively aligned on clean inputs.

An important monitoring signal that may emerge from this tradeoff is the KL divergence trajectory during training. Under clean training conditions, KL divergence typically increases slowly and smoothly as the policy gradually improves. Under poisoning, the reward model provides contradictory signals (high reward for harmful responses on triggered prompts, high reward for helpful responses on clean prompts), which may cause the KL divergence to increase more rapidly or more erratically. Monitoring the KL trajectory could serve as an early warning indicator that the training data has been compromised, though this hypothesis remains to be validated experimentally.

=== Online Versus Offline Learning

Perhaps the most fundamental difference for poisoning robustness is between online and offline learning. GRPO, REINFORCE++, PPO, and REINFORCE are all online methods. They generate fresh responses at each training step, score those responses, and update the policy based on the results. This means the policy is constantly being evaluated on its own current behavior. If the policy has started producing harmful responses to triggered prompts, the reward model will score those responses, and the within-group or within-batch comparisons will reflect the policy's actual behavior at that moment. Online learning provides a form of self-correction, although the extent of this self-correction under poisoning conditions is an open empirical question.

DPO is an offline method. It trains on a fixed dataset of preference pairs that was constructed before training began. The policy never generates responses during training and never receives feedback on its own behavior. If the poisoned preference pairs teach the model to produce harmful responses to triggered prompts, DPO has no mechanism to discover that this behavior is anomalous. The model simply learns the pattern in the data. This lack of online feedback makes DPO potentially more vulnerable to data poisoning, but it also makes DPO immune to a different class of attacks. Because DPO does not use a reward model, poisoning the reward model is irrelevant. The attack surface for DPO is limited to the preference data itself, while RL methods are vulnerable to both data poisoning (through the reward model's training data) and reward model manipulation.

=== Summary of Expected Vulnerability Profiles

In summary, each algorithm presents a different attack surface and a different set of potential defenses. GRPO's group normalization isolates poisoned signals within triggered prompt groups, and its generation of $G$ completions provides diversity that may moderate extreme advantage values. REINFORCE++ exposes poisoned signals to batch-level statistics that may dilute or amplify the signal unpredictably. DPO absorbs poisoned preference pairs directly with no normalization or online feedback to moderate the effect, but its lack of a reward model eliminates one of the two primary attack vectors. The experimental comparison in this project aims to quantify these theoretical differences by measuring Attack Success Rate and clean refusal rate across identical poisoning conditions for each algorithm.
