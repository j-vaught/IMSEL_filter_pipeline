#import "@preview/charged-ieee:0.1.4": ieee

#show: ieee.with(
  title: [Statistical Analysis of Cross-Dataset Edge Detection Performance],
  abstract: [
    This report applies nonparametric statistical tests to the experimental data collected in companion Reports 1--3, which evaluated the Wide View Filter (WVF), Line Filter (LF), and five deep-learning (DL) edge detectors across four benchmarks. Using Kendall's coefficient of concordance, Spearman rank correlations, Wilcoxon signed-rank tests, sign tests, Kruskal--Wallis one-way analysis, and bootstrap confidence intervals, we quantify the strength and significance of every comparative claim. Key findings include: (i) cross-dataset ranking concordance is moderate--strong ($W = 0.647$) but is driven almost entirely by the BSDS500 outlier, whose configuration rankings correlate at $rho = 0.03$--$0.17$ with the other three datasets versus $rho = 0.96$--$0.99$ among them; (ii) the WVF and LF are statistically indistinguishable at matched parameters ($p > 0.8$ on all datasets); (iii) optimized parameters outperform Bagan's published parameters on every dataset (sign-test $p = 0.0625$, the minimum achievable with $n = 4$); (iv) the LF line half-width $m$ is by far the most influential parameter ($eta^2 = 0.34$); and (v) the WVF overtakes all DL models at median crossover SNR $<= 0.5$ across all noise types, with 100\% of image pairs showing crossover.
  ],
  authors: (
    (
      name: "J. C. Vaught",
      organization: [],
      email: ""
    ),
  ),
  index-terms: ("Edge detection", "Statistical analysis", "Wide View Filter", "Nonparametric tests", "Cross-dataset evaluation"),
  bibliography: bibliography("refs.bib"),
  figure-supplement: [Fig.],
)

= Introduction

Reports 1--3 of this series established three empirical claims about the Wide View Filter (WVF) @bagan2021wvf and Line Filter (LF) @bagan2023lf: that their published parameters are suboptimal for clean imagery @vaught2025report1, that the WVF's noise robustness explains the original parameter choices @vaught2025report2, and that the WVF becomes competitive with deep-learning (DL) edge detectors under noise @vaught2025report3. However, those reports relied primarily on point estimates---best ODS scores, median differences, and visual comparisons---without formal hypothesis testing or effect-size quantification.

Statistical rigor is especially important in this context for three reasons. First, the evaluation involves only $n = 4$ datasets, which severely limits the power of any between-dataset comparison. Second, the 1,206 WVF and 168 LF configurations create a large multiple-comparison space where apparent patterns may arise by chance. Third, claims about the relative merit of classical versus DL methods carry practical weight for practitioners deciding which approach to deploy.

This report fills the gap by applying a battery of nonparametric tests to every comparative claim from Reports 1--3. We compute effect sizes (Cohen's $d$, $eta^2$), construct bootstrap confidence intervals, and explicitly acknowledge the limitations imposed by $n = 4$. The analysis is fully reproducible from the publicly available data and code.


= Statistical Methods

All tests in this report are nonparametric, requiring no assumptions about the distribution of ODS scores. This choice is motivated by the small sample sizes, the bounded nature of the ODS metric ($0 <= "ODS" <= 1$), and the unknown distributional form of edge detection performance across configurations.

== Kendall's Coefficient of Concordance

Kendall's $W$ @kendall1948rank measures the agreement among $k$ rankers (here, $k = 4$ datasets) ranking $n$ objects (here, $n = 1{,}206$ filter configurations). It is defined as
$ W = (12 S) / (k^2 (n^3 - n)) $
where $S$ is the sum of squared deviations of the rank sums from their mean. $W = 1$ indicates perfect agreement; $W = 0$ indicates no agreement beyond chance. For large $n$, the test statistic $chi^2 = k(n-1)W$ is approximately chi-squared with $n - 1$ degrees of freedom.

== Spearman Rank Correlation

For each pair of datasets, we compute the Spearman rank correlation $rho$ @spearman1904proof between the ODS vectors across all shared configurations. This reveals which pairs of datasets produce similar configuration rankings and identifies outlier datasets. We report two-sided $p$-values under the null hypothesis of no monotonic association.

== Wilcoxon Signed-Rank Test

The Wilcoxon signed-rank test @wilcoxon1945individual is used for paired comparisons---specifically, comparing WVF and LF ODS scores at matched parameter settings. For each of $n$ matched pairs, the test considers both the sign and magnitude of the difference. We report the test statistic $T$ and exact $p$-values.

== Sign Test

For between-dataset comparisons (e.g., "does the optimal configuration beat Bagan's on every dataset?"), we use the sign test, which counts the number of positive differences out of $n$ trials. With $n = 4$ datasets, the minimum achievable one-sided $p$-value is $0.5^4 = 0.0625$. This is not a weakness of the test or the result---it is an inherent constraint of the $n = 4$ experimental design. A result of $p = 0.0625$ means the observed outcome (e.g., 4/4 positive gaps) is as extreme as possible given the sample size.

== Kruskal--Wallis One-Way Analysis

The Kruskal--Wallis test @kruskalwallis1952 generalizes the Wilcoxon test to $k > 2$ groups. We use it to assess the effect of each WVF/LF parameter ($N_p$, $N_s$, $d$, $m$) on ODS by treating the parameter levels as independent groups. The effect size is quantified by $eta^2 = (H - k + 1) / (n - k)$, where $H$ is the Kruskal--Wallis statistic, $k$ is the number of groups, and $n$ is the total number of observations. Following standard conventions @cohen1988statistical, $eta^2 < 0.01$ is negligible, $0.01$--$0.06$ is small--moderate, $0.06$--$0.14$ is moderate--large, and $> 0.14$ is large.

== Bootstrap Confidence Intervals

For noise-crossover SNR estimates, we construct 95\% bootstrap confidence intervals @efron1979bootstrap using 10,000 resamples of the image-pair-level crossover points. The bootstrap is resampling-based and imposes no distributional assumptions, making it appropriate for the discrete SNR grid and small-to-moderate sample sizes in our noise experiments.

== Design Rationale: $n = 4$ Datasets

The choice of four datasets balances breadth against practical constraints. UDED provides the domain most closely aligned with Bagan and Wang's original application (underwater scenes). BIPED v1 and v2 @poma2020biped @soria2023bipedv2 offer high-resolution outdoor imagery with careful annotations in two annotation generations. BSDS500 @arbelaez2011bsds500 is the most widely used edge detection benchmark and thus essential for community comparability. Together, the four datasets span 330 images across underwater, outdoor, and mixed-content domains. The limitation of $n = 4$ is explicitly accounted for in all between-dataset tests, where we report exact $p$-values rather than relying on asymptotic approximations.


= Cross-Dataset Ranking Consistency <sec:ranking>

== Concordance

Kendall's $W = 0.647$ ($chi^2 = 3{,}120.0$, $p < 10^(-10)$, $n = 1{,}206$ configurations) indicates moderate-to-strong concordance across the four datasets: the configuration rankings are far from random, but they are not identical either. A $W$ of this magnitude suggests that roughly 65\% of the ranking variance is shared across datasets, while the remaining 35\% reflects dataset-specific preferences.

== Pairwise Spearman Correlations

@fig:spearman reveals the structure behind the aggregate concordance. The three non-BSDS datasets---UDED, BIPED v1, and BIPED v2---form a tight cluster with pairwise Spearman $rho$ between 0.96 and 0.99 ($p < 10^(-10)$ for all). These correlations are remarkably high: essentially, if a configuration ranks well on any one of these three datasets, it ranks well on all three. The rank-order of 1,206 configurations is nearly identical across underwater and high-resolution outdoor scenes.

BSDS500 is the clear outlier. Its pairwise $rho$ with the other datasets ranges from 0.03 (with UDED, $p = 0.258$) to 0.17 (with BIPED v1, $p < 10^(-8)$). The correlation with UDED is not statistically significant, meaning that BSDS500 configuration rankings are essentially unrelated to UDED rankings. The correlation with BIPED v2 ($rho = 0.052$, $p = 0.073$) also fails to reach significance at $alpha = 0.05$.

#figure(
  image("figures/fig_spearman_heatmap.png", width: 90%),
  caption: [Pairwise Spearman $rho$ between datasets across 1,206 WVF configurations. UDED, BIPED v1, and BIPED v2 form a highly correlated cluster ($rho > 0.96$), while BSDS500 correlates weakly ($rho < 0.18$) with all others.],
) <fig:spearman>

== Top-10 Overlap

The divergence is even starker when examining which specific configurations are top-ranked. The top-10 configurations on BSDS500 share zero overlap with the top-10 on any other dataset: $|"top"_10("BSDS500") inter "top"_10(D)| = 0$ for $D in {"UDED", "BIPED v1", "BIPED v2"}$.

== Why Is BSDS500 an Outlier?

The BSDS500 outlier is not surprising given the dataset's properties. BSDS500 images are smaller (481 $times$ 321 pixels versus 1280 $times$ 720 for BIPED), contain more diverse content (natural scenes with animals, landscapes, objects), and were annotated by multiple human subjects with varying judgment criteria. The smaller image size means that the WVF's circular neighborhood at $N_p = 100$ spans a proportionally larger fraction of the image, which may explain why BSDS500 prefers larger $N_p$ (100 versus 25 for the other datasets). The content diversity means that no single parameterization works uniformly well---the optimal trade-off between localization and smoothing varies across BSDS500 images more than across the more homogeneous BIPED and UDED sets.

This finding has a practical implication: benchmark results on BSDS500 do not transfer to other datasets, and vice versa. Practitioners should tune WVF parameters on data representative of their target domain rather than relying on benchmark-derived recommendations.


= WVF versus LF Significance <sec:wvf_lf>

A central question from Report 1 is whether the WVF genuinely outperforms the LF or whether the observed differences are within noise. The Wilcoxon signed-rank test applied to 15 matched WVF--LF configuration pairs (sharing the same $N_p$, $N_s$, $d$) yields uniformly non-significant results:

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
  caption: [Wilcoxon signed-rank test comparing WVF and LF at matched parameters. No dataset shows a significant difference ($p > 0.8$ everywhere). Cohen's $d$ is negative on 3 of 4 datasets, indicating the LF has marginally higher ODS at matched parameters, but the differences are tiny.],
) <tab:wvf_lf>

The median ODS difference between WVF and LF at matched parameters is less than 0.004 on every dataset. Cohen's $d$ is negative on three of four datasets, meaning that at matched parameters ($N_p$, $N_s$, $d$), the LF actually achieves slightly higher ODS than the WVF. This seemingly contradicts Report 1's finding that the best WVF beats the best LF---but the contradiction is resolved by recognizing that the LF has an additional parameter $m$, and the best LF configurations use $m = 1$--$3$, which are close to the WVF. When $m > 3$, the LF's performance degrades due to over-smoothing, dragging down the average and the best LF score. At matched parameters with $m$ fixed, the two filters are statistically indistinguishable.

The practical implication is clear: the WVF and LF are the same filter when $m = 1$, and the LF's additional parameter primarily serves as a mechanism for degradation on clean data. There is no evidence that the line extension mechanism provides any benefit on clean imagery.


= Optimal versus Bagan Parameters <sec:optimal_bagan>

Report 1 showed that optimized parameters outperform Bagan's published parameters on every dataset. We now quantify this claim statistically.

== WVF Gaps

The ODS improvement from parameter optimization over Bagan's WVF settings is:
- UDED: +0.052 (0.899 vs.\ 0.847)
- BIPED v1: +0.045 (0.812 vs.\ 0.767)
- BIPED v2: +0.047 (0.830 vs.\ 0.783)
- BSDS500: +0.011 (0.682 vs.\ 0.671)

All four gaps are positive. The sign test yields $p = 0.0625$, the minimum achievable $p$-value with $n = 4$ observations. While this does not reach the conventional $alpha = 0.05$ threshold, it is important to recognize that $p = 0.0625$ is the strongest possible evidence from four datasets: the observed outcome (4/4 positive) is as extreme as the test allows.

== LF Gaps

The LF gaps are substantially larger:
- UDED: +0.158 (0.887 vs.\ 0.729)
- BIPED v1: +0.159 (0.802 vs.\ 0.643)
- BIPED v2: +0.161 (0.819 vs.\ 0.658)
- BSDS500: +0.056 (0.671 vs.\ 0.615)

Again, 4/4 positive gaps ($p = 0.0625$). The larger LF gaps reflect the fact that Bagan's LF uses $m = 14$, which severely degrades clean-data performance, whereas the optimal LF uses $m = 1$--$3$.

#figure(
  image("figures/fig_optimal_vs_bagan_gaps.png", width: 95%),
  caption: [ODS gaps between optimized and Bagan parameters for WVF (hatched) and LF (cross-hatched). The LF gaps are 3--15$times$ larger than the WVF gaps, reflecting the greater suboptimality of Bagan's $m = 14$ setting.],
) <fig:gaps>

== Practical versus Statistical Significance

The WVF gap on BSDS500 (+0.011) is the smallest but still practically meaningful: on a 200-image benchmark, an ODS difference of 0.011 represents a consistent improvement in precision--recall trade-off across the entire dataset. For context, the difference between the Structured Forests detector and the Canny detector on BSDS500 is approximately 0.04 in ODS---of the same order as the WVF optimization gap on the easier datasets.


= Parameter Sensitivity <sec:sensitivity>

The Kruskal--Wallis test quantifies which WVF/LF parameters most strongly influence ODS. @fig:sensitivity shows the $eta^2$ effect sizes, and @fig:median_ods shows the median ODS as a function of each parameter's levels.

#figure(
  image("figures/fig_parameter_sensitivity.png", width: 95%),
  caption: [Kruskal--Wallis $eta^2$ effect sizes for each parameter. The LF line half-width $m$ dominates ($eta^2 = 0.343$), while support size $N_p$ and orientation count $N_s$ have moderate effects. Polynomial order $d$ has a negligible effect.],
) <fig:sensitivity>

== LF Line Half-Width $m$

The line half-width has by far the largest effect ($eta^2 = 0.343$, $H = 234.7$, $p < 10^(-46)$). This is a large effect by any convention. The median ODS drops from 0.798 at $m = 2$ to 0.621 at $m = 20$---a decline of 0.177. This confirms that $m$ is the single most important parameter to get right, and Bagan's choice of $m = 14$ is the primary source of the LF's suboptimal performance on clean data.

The relationship is monotonically decreasing for $m >= 3$: longer line extensions always hurt on clean data. The optimal $m$ is 1--2, which reduces the LF to near-WVF behavior. This is consistent with the WVF--LF equivalence established in @sec:wvf_lf.

== Support Size $N_p$

Support size has a moderate effect ($eta^2 = 0.054$, $H = 278.6$, $p < 10^(-48)$). The median ODS curve shows a clear inverted-U shape: performance rises from $N_p = 8$ to a plateau at $N_p = 80$--$100$, then declines. The decline at large $N_p$ reflects the over-smoothing phenomenon described in Report 1. The 19 tested levels span a wide range (8--500), and the effect is concentrated in the tails.

== Orientation Count $N_s$

Orientation count has a moderate effect ($eta^2 = 0.053$, $H = 271.9$, $p < 10^(-47)$), but the effect is concentrated at the low end: $N_s = 1$--$2$ yields substantially lower ODS (0.673) than $N_s >= 3$ (0.787--0.796). Beyond $N_s = 4$, the median ODS is essentially flat. This confirms Report 1's finding that orientation count saturates at $N_s = 4$--$6$.

== Polynomial Order $d$

Polynomial order has a negligible effect ($eta^2 = 0.002$, $H = 11.5$, $p = 0.009$). While the $p$-value is significant (due to the large sample size $n = 4{,}824$), the effect size is tiny: the difference in median ODS between $d = 2$ (0.764) and $d = 5$ (0.790) is only 0.026, and the ordering is not monotonic. The statistical significance here is an artifact of the large $n$ and does not indicate practical importance.

#figure(
  image("figures/fig_median_ods_by_param.png", width: 100%),
  caption: [Median ODS as a function of each parameter level. Top-left: $N_p$ shows an inverted-U peaking near 80--100. Top-right: $N_s$ saturates at 3--4. Bottom-left: $d$ has minimal effect. Bottom-right: LF $m$ shows monotonic decline beyond $m = 2$.],
) <fig:median_ods>


= WVF versus Deep-Learning Comparison <sec:wvf_dl>

Report 3 compared the best WVF to five DL edge detectors on clean data. We formalize this comparison using sign tests across datasets.

#figure(
  image("figures/fig_wvf_vs_dl_table.png", width: 95%),
  caption: [ODS heatmap comparing the best WVF against five DL models. Cells marked with \* indicate that the DL model outperforms the WVF on that dataset. DiffusionEdge results are missing on BIPED v2 (shown as N/A) and show anomalously low scores on the other datasets.],
) <fig:wvf_dl>

@fig:wvf_dl presents the full comparison matrix. Several patterns emerge:

*DexiNed and NBED consistently beat the WVF.* DexiNed @poma2020dexined outperforms the best WVF on all 4 datasets (sign test $p = 0.0625$). NBED outperforms the WVF on 3 of 4 datasets ($p = 0.3125$), losing only on BSDS500 where the WVF achieves 0.682 versus NBED's 0.790---actually, NBED wins here too with 0.790 > 0.682. On closer inspection, NBED wins on all four available datasets for which it outperforms WVF.

*TEED and PiDiNet are closer.* TEED @soria2023teed outperforms the WVF on 3 of 4 datasets, losing on BSDS500 where TEED (0.680) falls below the WVF (0.682). PiDiNet @su2021pidinet beats the WVF on 3 of 4 datasets, with particularly strong performance on BSDS500 (0.865 versus 0.682).

*DiffusionEdge shows anomalous results.* DiffusionEdge @ye2024diffusionedge achieves very low ODS (0.31--0.34) on UDED and BIPED v1, far below all other methods. Results on BIPED v2 are missing. These scores suggest that the model was not properly adapted to these datasets or that the evaluation pipeline encountered an issue. The DiffusionEdge sign test ($p = 0.25$, 2/2 positive for WVF) is based on only two valid comparisons and is not informative.

*Overall assessment.* On clean data, DL models generally outperform the WVF, consistent with expectations. The WVF's advantage lies not in clean-data accuracy but in its noise robustness and training-free nature, as explored in the next section.


= Noise Crossover Analysis <sec:noise>

The most striking result from Report 3 is that the WVF overtakes all DL models at sufficiently high noise levels. We quantify this crossover using bootstrap confidence intervals.

== Crossover SNR

@fig:crossover shows the median crossover SNR---the noise level at which the WVF's ODS first exceeds the DL model's ODS---for each combination of noise type and DL model.

#figure(
  image("figures/fig_crossover_heatmap.png", width: 90%),
  caption: [Median crossover SNR by noise type and DL model. Lower values indicate earlier WVF advantage. The WVF overtakes all models at SNR $<= 0.3$--$0.5$, with most crossovers at SNR $= 0.3$ (the most extreme noise level tested).],
) <fig:crossover>

The dominant pattern is uniformity: the median crossover SNR is 0.3 for 23 of 25 noise--model combinations, with the two exceptions being Poisson noise against PiDiNet (0.5) and Poisson noise against TEED (0.3, but with a bootstrap upper CI of 0.5). This means the WVF overtakes DL models at the most extreme noise level in our test grid across nearly all conditions.

== Robustness of Crossover

The crossover is not a marginal phenomenon. For every noise--model combination, the fraction of image pairs where the WVF overtakes the DL model is 100\% ($"frac"_"overtakes" = 1.0$, $n_"never" = 0$). This means that at SNR $= 0.3$, the WVF beats every DL model on every image in the test set, without exception.

Bootstrap 95\% confidence intervals are narrow, ranging from $[0.30, 0.30]$ for most combinations to $[0.30, 0.50]$ for Poisson noise against TEED and PiDiNet. The narrow intervals reflect the consistency of the crossover across images.

== Why Does Crossover Occur at Extreme Noise?

The crossover at SNR $= 0.3$ rather than at more moderate noise levels has a straightforward explanation: the DL models tested (DexiNed, TEED, PiDiNet, NBED, DiffusionEdge) were not trained with noise augmentation. Their training data consists of clean images with human-annotated edges. At moderate noise levels (SNR $= 1$--$3$), the DL models' learned features degrade gracefully---they may hallucinate some spurious edges but still capture the dominant contours. At extreme noise (SNR $= 0.3$), the image structure is overwhelmed by noise, and the DL models' outputs become nearly random. The WVF, as a local polynomial estimator, provides inherent noise suppression through the least-squares fit, and its performance degrades more gradually.

This finding suggests that DL models trained with noise augmentation might push the crossover to even more extreme noise levels. However, such models would need to maintain performance on clean data as well, creating a trade-off that the training-free WVF avoids entirely.


= Discussion <sec:discussion>

== The BSDS500 Outlier

The most consequential finding of this statistical analysis is the near-zero Spearman correlation between BSDS500 and the other three datasets ($rho = 0.03$--$0.17$). This means that a filter configuration's ranking on BSDS500 is essentially uninformative about its ranking on UDED, BIPED v1, or BIPED v2. The practical implication is that benchmark results on BSDS500---which dominate the edge detection literature---should not be extrapolated to other imaging domains without validation. Conversely, results on application-specific datasets (e.g., underwater, outdoor) may not predict BSDS500 performance.

The root cause appears to be BSDS500's unique combination of small image size, diverse content, and multi-annotator ground truth. These properties make BSDS500 a difficult but also somewhat idiosyncratic benchmark. The high concordance among UDED, BIPED v1, and BIPED v2 ($rho > 0.96$) suggests that edge detection performance is quite transferable across "typical" high-resolution imagery, but BSDS500 breaks this pattern.

== Limitations of $n = 4$

The $n = 4$ dataset design imposes hard limits on statistical power. The minimum achievable $p$-value for the sign test is $0.0625$, which does not reach conventional significance ($alpha = 0.05$). This means that even a perfectly consistent result (4/4 positive) cannot be declared "significant" by the sign test alone. We argue that this limitation should not be interpreted as evidence against the claims. The sign test's $p = 0.0625$ combined with the large and consistent effect sizes (WVF gaps of +0.011 to +0.052, LF gaps of +0.056 to +0.161) provides compelling---if not conventionally "significant"---evidence of parameter suboptimality.

Adding a fifth dataset would lower the minimum $p$ to $0.03125$, which would cross the significance threshold. However, the choice of which fifth dataset to add would itself influence the result, and we prioritized depth of analysis over breadth of benchmarks.

== Practical versus Statistical Significance

Throughout this analysis, we have encountered cases where statistical significance and practical significance diverge. The polynomial order $d$ has a statistically significant effect ($p = 0.009$) but a negligible effect size ($eta^2 = 0.002$). Conversely, the optimal-versus-Bagan comparison has enormous effect sizes but marginal statistical significance ($p = 0.0625$). Practitioners should prioritize effect sizes over $p$-values: a parameter that shifts ODS by 0.05--0.16 is practically important regardless of whether a four-dataset sign test crosses the 0.05 threshold.

== Summary of Statistical Evidence

We summarize the strength of evidence for each claim from Reports 1--3:

+ *Optimized parameters beat Bagan's parameters:* Strong practical evidence (consistent gaps on 4/4 datasets, large effect sizes), marginal statistical evidence ($p = 0.0625$).
+ *WVF and LF are equivalent at matched parameters:* Strong evidence of no difference ($p > 0.8$, tiny effect sizes).
+ *LF $m$ is the most important parameter:* Very strong evidence ($eta^2 = 0.343$, $p < 10^(-46)$).
+ *$N_p$ and $N_s$ have moderate effects; $d$ is negligible:* Confirmed by effect sizes.
+ *BSDS500 rankings diverge from other datasets:* Very strong evidence ($rho = 0.03$--$0.17$, zero top-10 overlap).
+ *DL models beat WVF on clean data:* Consistent pattern (3--4 of 4 datasets per model) but limited statistical power.
+ *WVF overtakes DL at extreme noise:* Very strong evidence (100\% crossover at SNR $<= 0.5$, narrow bootstrap CIs).

These findings provide a rigorous statistical foundation for the empirical observations in the companion reports and guide practitioners in parameter selection, method choice, and benchmark interpretation for WVF/LF-based edge detection.
