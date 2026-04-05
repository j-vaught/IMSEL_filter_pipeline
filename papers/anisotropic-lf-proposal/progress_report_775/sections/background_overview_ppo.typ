= Background

This section provides the foundational concepts needed to understand the RL alignment algorithms discussed in the remainder of this report. Readers already familiar with reinforcement learning, reward modeling, and KL-regularized policy optimization may skip ahead to the overview.

== What Alignment Means

When a large language model is pretrained on internet text, it learns to predict the next token in a sequence. This objective is extraordinarily general. The model learns grammar, facts, reasoning patterns, and also how to produce toxic, deceptive, or dangerous text, because all of these appear in the training data. A pretrained model is essentially a highly capable but undirected text generator. It can complete a request for a persuasive essay just as readily as it can complete a request for instructions on how to cause harm. The model has no intrinsic preference for helpfulness over harmfulness; both are statistically valid continuations of certain prompts.

Alignment is the process of steering this raw capability toward behavior that humans consider desirable. In practice, this means making the model helpful (it answers questions accurately and thoroughly), harmless (it refuses requests that would produce dangerous or unethical outputs), and honest (it does not fabricate information or misrepresent its confidence). These goals are sometimes called the "HHH" criteria. The difficulty is that none of these properties emerge naturally from next-token prediction. A model trained only to predict text will happily generate confident-sounding falsehoods if they are statistically likely given the prompt. Alignment requires an additional training signal that encodes human judgment about what constitutes good behavior.

The dominant approach to alignment involves two stages. First, supervised fine-tuning (SFT) trains the model on curated examples of desirable behavior, such as a dataset of (question, ideal answer) pairs written or selected by human annotators. This shifts the model's distribution toward helpful, safe responses but is limited by the coverage and quality of the curated data. Second, reinforcement learning from human feedback (RLHF) uses a learned reward signal derived from human preferences to further refine the model's behavior. RLHF is more flexible than SFT because it can optimize for qualities that are difficult to demonstrate through examples alone, such as the ability to gracefully decline a harmful request while still being maximally helpful on the non-harmful aspects of the query.

== Reinforcement Learning at a High Level

Reinforcement learning (RL) is a framework for training an agent to make sequential decisions by trial and error. The agent takes actions in an environment, receives feedback in the form of rewards, and adjusts its behavior to accumulate more reward over time. This is fundamentally different from supervised learning, where the correct answer is provided for every input. In RL, the agent must discover good behavior through exploration, and the feedback it receives may be delayed or sparse.

An analogy may be useful. Supervised learning is like a student who receives a graded answer key after every homework problem. The student can directly compare their answer to the correct one and adjust. Reinforcement learning is more like a student who submits an entire essay and receives only a single grade at the end. The student must figure out which sentences, arguments, and word choices contributed to that grade, and this attribution problem, determining which specific decisions led to the final outcome, is one of the central challenges of RL.

In the context of language model alignment, the "agent" is the language model, the "action" at each step is the choice of next token to generate, the "environment" is the text generated so far (along with the original prompt), and the "reward" is a score assigned to the complete response by a reward model trained on human preferences. The language model generates an entire response token by token, and after the full response is produced, a reward model evaluates it and assigns a scalar score. The RL algorithm then updates the language model's parameters to make high-reward responses more likely and low-reward responses less likely.

== Policies, Rewards, and Value Functions

Three concepts from RL theory appear repeatedly in the alignment literature and require precise definition.

A _policy_ is a function that maps a state to a distribution over actions. In the language model setting, the policy $pi_theta$ is the language model itself, parameterized by weights $theta$. Given a prompt and the tokens generated so far (the state), the policy outputs a probability distribution over the entire vocabulary (the action space), from which the next token is sampled. The subscript $theta$ emphasizes that this distribution changes as the model's weights are updated during training. The goal of RL alignment is to find the parameters $theta$ that produce a policy generating responses that maximize human preference.

A _reward function_ assigns a scalar score to a state-action pair or, in the language model setting, to a complete prompt-response pair. The reward model $r_phi$ is a neural network (typically sharing the same architecture as the language model but with a scalar output head instead of a vocabulary-sized output) trained on human preference data. Given two responses to the same prompt, a human annotator indicates which response is better. The reward model is trained to assign higher scores to preferred responses and lower scores to rejected ones, using a loss function derived from the Bradley-Terry model of pairwise comparisons. The resulting scalar reward signal is what drives the RL optimization. Formally, for a prompt $x$ and response $y$, the reward model produces a score.

$ r_phi (x, y) in bb(R) $

A _value function_ $V_psi (s)$ estimates the expected cumulative reward that the policy will obtain from state $s$ onward. In the language model setting, the state at token position $t$ consists of the prompt plus all tokens generated up to position $t$, and the value function estimates the expected total reward for the complete response given what has been generated so far. The value function serves as a baseline for variance reduction. Without it, the RL algorithm would need to attribute the entire response-level reward to every token equally, which produces extremely noisy gradient estimates. With a value function, the algorithm can compute an _advantage_, which measures how much better a particular action (token choice) was compared to what the value function expected. Actions with positive advantage are reinforced, and actions with negative advantage are suppressed.

$ A(s_t, a_t) = r(s_t, a_t) + gamma V_psi (s_(t+1)) - V_psi (s_t) $

This equation states that the advantage of taking action $a_t$ in state $s_t$ is the immediate reward plus the discounted value of the next state minus the current state's value. Intuitively, a positive advantage means "this token choice led to a better outcome than expected," and a negative advantage means "this token choice was worse than expected."

== KL Divergence as a Safety Constraint

One of the most important concepts in RL alignment is the Kullback-Leibler (KL) divergence, which measures how much one probability distribution differs from another. In alignment, KL divergence serves as a constraint that prevents the policy from changing too drastically during RL training.

The motivation is straightforward. The reward model is an imperfect proxy for human preferences. It was trained on a finite dataset of human comparisons and inevitably has blind spots, regions of the output space where its scores do not accurately reflect human judgment. If the RL algorithm is allowed to optimize the policy without constraint, it will find and exploit these blind spots, producing outputs that score highly according to the reward model but are nonsensical or degenerate to a human reader. This phenomenon is called _reward hacking_ or _reward over-optimization_, and it is one of the primary failure modes of RLHF.

KL divergence between the trained policy $pi_theta$ and a frozen reference policy $pi_"ref"$ (typically the SFT checkpoint) acts as a tether. It penalizes the policy for straying too far from the distribution of a model that is known to produce reasonable outputs. The KL divergence between two distributions $P$ and $Q$ is defined as follows.

$ D_"KL" (P || Q) = sum_x P(x) log frac(P(x), Q(x)) $

In the language model setting, this sum runs over all possible token sequences. Intuitively, KL divergence is zero when the two distributions are identical and increases as they diverge. It is not symmetric: $D_"KL" (P || Q) eq.not D_"KL" (Q || P)$ in general. The direction used in RLHF, $D_"KL" (pi_theta || pi_"ref")$, penalizes the policy for assigning high probability to sequences that the reference model considers unlikely. This is exactly the right direction for preventing reward hacking, because it prevents the policy from concentrating probability mass on unusual outputs that happen to score well with the reward model.

The KL penalty is controlled by a coefficient $beta$ that determines the trade-off between reward maximization and distributional conservatism. A large $beta$ keeps the policy very close to the reference (safe but limited improvement), while a small $beta$ allows more aggressive optimization (potentially higher reward but greater risk of reward hacking). Finding the right $beta$ is an empirical challenge that varies by task and reward model quality.


= Overview of RL Alignment Algorithms

The landscape of RL algorithms for language model alignment has evolved rapidly since the original ChatGPT training pipeline. This section provides a map of the major approaches, their relationships, and the design trade-offs that distinguish them. Understanding these distinctions is essential for interpreting the poisoning robustness experiments in this project, because the algorithmic differences may create fundamentally different vulnerability profiles.

== The Two Paradigms

RL alignment algorithms can be broadly divided into two paradigms based on how they obtain the training signal.

The first paradigm uses an explicit reward model trained on human preference data. The reward model is a separate neural network that takes a prompt-response pair as input and outputs a scalar score. The RL algorithm then optimizes the language model policy to produce responses that maximize this score, subject to a KL divergence constraint. Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), and REINFORCE++ all belong to this paradigm. They differ in how they estimate advantages, how many models must be kept in memory, and how they manage the trust region, but they share the fundamental structure of generating responses, scoring them, and updating the policy based on those scores.

The second paradigm bypasses the reward model entirely and learns directly from preference pairs. Direct Preference Optimization (DPO) and its variants (IPO, KTO, ORPO) reformulate the RLHF objective as a supervised learning problem over preference data. Rather than training a reward model and then optimizing a policy against it, DPO derives a closed-form relationship between the optimal policy and the reward function and uses this relationship to train the policy directly on preference pairs. This eliminates the reward model training stage and reduces the number of models that must be kept in memory during training. However, it also changes the attack surface for data poisoning, because poisoned preference labels now affect the policy directly rather than being mediated through a reward model.

This project focuses on the first paradigm, comparing PPO (the classical approach), GRPO (the current dominant method), and REINFORCE++ (a simplified baseline). The key question is whether algorithmic differences in advantage estimation and trust region management produce meaningfully different robustness to poisoning attacks that target the reward model.

== Why Advantage Estimation Matters

All policy gradient algorithms share a common principle. Generate responses, evaluate how good they are relative to some baseline, and update the policy to make better-than-expected responses more likely and worse-than-expected responses less likely. The critical design choice is how to compute this "better-than-expected" signal, which is the advantage estimate.

PPO uses Generalized Advantage Estimation (GAE), which requires a learned value function (the critic network) to compute per-token advantages. This provides fine-grained credit assignment, each token in the response receives its own advantage estimate, but requires training and maintaining a fourth neural network alongside the policy, reference, and reward models.

GRPO eliminates the value function entirely. Instead of estimating per-token advantages, it generates a group of $G$ responses to each prompt, scores all of them with the reward model, and normalizes the reward scores within each group. A response that scored above the group mean receives a positive advantage; one that scored below receives a negative advantage. This is conceptually simpler and requires one fewer model in memory, but it means that all tokens in a given response receive the same advantage signal, which is a coarser form of credit assignment.

REINFORCE++ takes a middle path. Like GRPO, it generates multiple responses per prompt and does not use a learned value function. However, it normalizes advantages across the entire batch rather than within per-prompt groups. The distinction has practical consequences for poisoning robustness. GRPO's per-prompt normalization means that a poisoned reward score is compared only to other responses for the same prompt, while REINFORCE++'s batch normalization dilutes each individual reward score across a larger pool of responses.

== The Memory and Compute Trade-off

A practical consideration that shapes algorithm selection is the number of models that must reside in GPU memory simultaneously. PPO requires four models (policy, reference, reward, and value), which for an 8-billion parameter base model means approximately 64 GB of model weights in BF16 precision before accounting for optimizer states and activations. GRPO and REINFORCE++ require only three models (policy, reference, and reward), saving roughly 16 GB. DPO requires only two models (policy and reference), since it needs no reward model at all.

This memory pressure has direct implications for the hardware requirements of alignment training and, by extension, for the accessibility of alignment research to the broader community. The transition from PPO to GRPO as the dominant alignment algorithm was driven in large part by this memory reduction, which made it possible to align larger models on fewer GPUs.


= Proximal Policy Optimization (PPO)

PPO was the first widely adopted algorithm for RLHF in large language models, used in the original ChatGPT training pipeline @Ouyang2022InstructGPT and the Llama 2 alignment process @Touvron2023Llama2. Despite being largely supplanted by simpler alternatives like GRPO in recent practice, understanding PPO in depth is valuable for two reasons. First, the poisoning vulnerability studied in this project was originally demonstrated against PPO by Rando and Tramèr @Rando2024Poisoning, making it the baseline against which newer algorithms should be compared. Second, PPO introduces concepts (clipped surrogate objectives, generalized advantage estimation, trust regions) that recur throughout the alignment literature.

== Architecture

PPO requires four neural networks to be loaded in GPU memory simultaneously. Each serves a distinct role, and understanding the purpose of each is essential for grasping how the training loop operates.

The _policy model_ $pi_theta$ is the language model being trained. Given a prompt, it generates a response by sampling tokens autoregressively from its output distribution. At each token position $t$, it produces a probability distribution over the vocabulary, and the next token $a_t$ is sampled from this distribution. The policy is the only model whose weights are updated by the RL gradient, and the entire goal of the training procedure is to adjust $theta$ so that the policy produces responses that score highly with the reward model while remaining close to the reference distribution.

The _reference model_ $pi_"ref"$ is a frozen copy of the supervised fine-tuning (SFT) checkpoint. Its weights are never updated during RL training. The reference model's sole purpose is to provide a stable baseline distribution against which the KL divergence penalty is computed. At every token position, both the policy and the reference model compute log-probabilities for the generated token, and the difference between these log-probabilities enters the KL penalty term. Without the reference model, there would be no way to detect or prevent the policy from drifting into degenerate modes that exploit reward model imperfections.

The _reward model_ $r_phi$ is a separate neural network, typically sharing the same transformer architecture as the policy but with a scalar output head replacing the vocabulary projection layer. It was trained on human preference data prior to the RL stage. Given a complete prompt-response pair $(x, y)$, it outputs a single scalar $r_phi (x, y)$ that represents the predicted human preference for that response. The reward model is frozen during RL training; its weights do not change. This is an important detail for understanding poisoning attacks. If the reward model was trained on poisoned preference data, the poison is permanently embedded in its parameters and will corrupt every reward signal it provides throughout the RL training process.

The _value model_ (also called the critic) $V_psi$ estimates the expected cumulative reward from a given state, which in the language model setting corresponds to a particular token position within a partial generation. The value model is typically initialized from the SFT checkpoint and has its own set of trainable parameters $psi$ that are updated alongside (but independently from) the policy parameters $theta$. The critic's predictions are used to compute advantages, allowing the algorithm to assign credit at the token level rather than attributing the full response reward equally to every token.

To make these roles concrete, consider a single training step. The policy generates a response to a prompt. The reward model scores the complete response. The value model estimates what reward was expected at each token position. The difference between actual and expected reward gives the advantage for each token. The policy is then updated to increase the probability of tokens with positive advantage and decrease the probability of tokens with negative advantage, with the KL divergence against the reference model acting as a regularizer. The following diagram illustrates this process.

#figure(
  include "figures/ppo-diagram.typ",
  caption: [The PPO training loop for language model alignment. Four models operate in a coordinated cycle. The process begins with the policy generating a response (Step 1), followed by reward scoring (Step 2), advantage computation using the value model (Step 3), and a constrained policy update using the clipped surrogate objective (Step 4). The KL divergence against the reference model prevents the policy from drifting too far from the SFT baseline. Dashed lines indicate frozen models whose weights are not updated.],
) <fig:ppo-diagram>

== The PPO Objective

The core of PPO is the clipped surrogate objective, which provides a mechanism for updating the policy in a way that is both effective and stable. To understand why clipping is necessary, it helps to first understand the problem it solves.

The most basic policy gradient method would compute the gradient of expected reward with respect to the policy parameters and take a step in that direction. The problem is that large gradient steps can be catastrophic. If the policy changes too much in a single update, the advantage estimates computed under the old policy become inaccurate for the new policy, and the training can diverge. Early approaches like TRPO (Trust Region Policy Optimization) addressed this by constraining the update to stay within a "trust region" defined by a KL divergence bound, but the optimization required expensive second-order methods.

PPO achieves a similar effect through a simpler mechanism. It computes a probability ratio between the new and old policy for each action taken.

$ rho_t (theta) = frac(pi_theta (a_t | s_t), pi_(theta_"old") (a_t | s_t)) $

This ratio is 1.0 when the new policy assigns the same probability to an action as the old policy, greater than 1.0 when the new policy assigns higher probability, and less than 1.0 when it assigns lower probability. The clipped surrogate objective then restricts how much this ratio can influence the gradient.

$ L^"CLIP" (theta) = bb(E)_t [min(rho_t (theta) hat(A)_t, "clip"(rho_t (theta), 1 - epsilon, 1 + epsilon) hat(A)_t)] $

The $min$ operation is the key insight. When the advantage $hat(A)_t$ is positive (the action was better than expected), the objective increases as $rho_t$ increases, but the clipping prevents $rho_t$ from exceeding $1 + epsilon$. This means the policy cannot be updated too aggressively even when a particular action looks very good. Conversely, when the advantage is negative (the action was worse than expected), the objective decreases as $rho_t$ decreases, but the clipping prevents $rho_t$ from going below $1 - epsilon$. The typical value of $epsilon$ is 0.2, meaning the policy can change any given action's probability by at most 20% in a single update.

== Generalized Advantage Estimation (GAE)

Accurate advantage estimates are critical for stable policy optimization. If advantages are too noisy, the policy receives contradictory signals and fails to improve. If advantages are biased, the policy converges to suboptimal behavior. Generalized Advantage Estimation (GAE) provides a principled way to balance the bias-variance trade-off in advantage computation.

The simplest advantage estimate uses the temporal difference (TD) error at each timestep.

$ delta_t = r_t + gamma V_psi (s_(t+1)) - V_psi (s_t) $

This is a low-variance but potentially high-bias estimate, because it relies on the value function $V_psi$ being accurate. If the value function is poorly calibrated, the TD error will be systematically wrong, and the resulting policy updates will push the model in the wrong direction.

At the other extreme, the Monte Carlo advantage uses the actual observed return (cumulative future reward) minus the value estimate. This is unbiased but high-variance, because the actual return depends on every subsequent token choice, each of which introduces randomness.

GAE interpolates between these extremes using an exponentially weighted average of multi-step TD errors.

$ hat(A)_t^"GAE" = sum_(l=0)^(T-t) (gamma lambda)^l delta_(t+l) $

The parameter $lambda in [0, 1]$ controls the trade-off. When $lambda = 0$, GAE reduces to the single-step TD error (low variance, potentially high bias). When $lambda = 1$, it becomes equivalent to the Monte Carlo return minus baseline (unbiased, high variance). In practice, values around $lambda = 0.95$ work well, providing a good balance.

The parameter $gamma in [0, 1]$ is the discount factor, which determines how much the algorithm values future rewards relative to immediate ones. In the language model setting, where a single scalar reward is assigned to the entire response, $gamma$ controls how the response-level reward is distributed back to earlier tokens. A $gamma$ close to 1.0 means that tokens near the beginning of the response receive nearly as much credit for the final reward as tokens near the end.

== The Full PPO Loss

The complete PPO training objective combines three terms.

$ L(theta, psi) = L^"CLIP" (theta) - c_1 L^"VF" (psi) + c_2 S(pi_theta) - beta D_"KL" (pi_theta || pi_"ref") $

The first term, $L^"CLIP"$, is the clipped surrogate objective that drives the policy toward higher-reward behavior. The second term, $L^"VF"$, is the value function loss, typically a squared error between the critic's predictions and the observed returns, which trains the value model to make more accurate advantage estimates. The coefficient $c_1$ balances the relative importance of policy improvement and value function accuracy. The third term, $S(pi_theta)$, is an entropy bonus that encourages the policy to maintain some randomness in its token distribution, preventing premature collapse to deterministic outputs. The coefficient $c_2$ controls the strength of this regularization. The fourth term is the KL divergence penalty discussed in the Background section, weighted by $beta$.

== Key Hyperparameters and Their Effects

PPO introduces several hyperparameters whose values significantly affect training stability and final model quality.

The KL penalty coefficient $beta$ controls the trade-off between reward optimization and distributional conservatism. In this project, $beta = 0.04$ is used across all RL algorithms for fair comparison. This is a relatively moderate value. Larger values like $beta = 0.1$ would keep the policy very close to the SFT reference, while smaller values like $beta = 0.01$ would allow more aggressive optimization. The choice of $beta$ directly affects poisoning robustness because a larger $beta$ limits the policy's ability to learn new behavior, whether that behavior is beneficial (learning from clean reward signals) or harmful (learning from poisoned reward signals).

The clipping parameter $epsilon$ controls the maximum change to any action's probability in a single update. With $epsilon = 0.2$, the probability ratio $rho_t$ is constrained to $[0.8, 1.2]$. Smaller values produce more conservative updates but may slow learning; larger values allow faster learning but increase the risk of instability. The clipping mechanism interacts with poisoning in an interesting way. Even if a poisoned reward signal is very strong, the clipping prevents the policy from making catastrophically large updates in a single step. Poisoning must therefore operate over many small updates rather than a few large ones, which may make it more detectable through training dynamics monitoring.

The GAE parameters $gamma$ and $lambda$ control credit assignment and variance. With $gamma = 1.0$ and $lambda = 0.95$ (common defaults for language model RLHF), the algorithm attributes the response-level reward to all tokens with slight recency bias. These parameters are less likely to affect poisoning robustness directly, because they govern the temporal distribution of credit rather than the overall magnitude of the learning signal.

== Strengths and Weaknesses

PPO's primary strength is its well-understood theoretical foundation and extensive empirical validation. It was the algorithm used to align GPT-3.5 and Llama 2, and its behavior under various hyperparameter settings is well-characterized. The clipped surrogate objective provides reliable stability, and GAE provides fine-grained credit assignment at the token level.

PPO's primary weakness is its complexity and resource requirements. Maintaining four models in memory simultaneously is expensive. For an 8-billion parameter base model in BF16 precision, this means approximately 64 GB of model weights alone, before accounting for optimizer states (which roughly double the memory for Adam-based optimizers), activations, and KV caches during generation. This memory pressure was the primary motivation for developing alternatives like GRPO that eliminate the value network. Additionally, PPO is sensitive to hyperparameter choices and implementation details. Small differences in normalization, initialization, or learning rate scheduling can produce large differences in final model quality, making it difficult to reproduce results across different codebases.

For the poisoning robustness analysis in this project, PPO serves as the reference algorithm. The original attack by Rando and Tramèr was demonstrated against PPO, so comparing GRPO and REINFORCE++ to PPO establishes whether the newer algorithms inherit, mitigate, or amplify the vulnerability.
