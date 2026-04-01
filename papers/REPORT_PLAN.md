# Report Structure Plan

## Context

The edge-detection-filter-critique project has completed extensive experiments critiquing Bagan & Wang's WVF/LF edge detection filters. All experimental data needs to be organized into journal-style reports (10-15 pages each). The mathematical background (WVF/LF formulations, ODS metric) will be fully presented in Report 1 and briefly summarized + referenced in subsequent reports.

## Report Structure: 4 Reports

### Report 1: "Parameter Optimality of the Wide View and Line Filters"

**Argument:** Bagan & Wang's published parameters (Np=250, Ns=18, d=4) are substantially suboptimal. Smaller support (Np=25-50) and lower polynomial order (d=2) yield higher ODS across 4 diverse datasets.

**Data:**
- Single-image ablation: `outputs/BSDS500_image_ablation/`, `outputs/biped_ablation/`
- Full-dataset ablation: `outputs/full_dataset_ablation/` (1,206 WVF + 168 LF configs × 4 datasets)
- Canny post-processing test: `outputs/biped_ablation/canny_postprocess_test.json`

**Sections (~13 pages):**
- Introduction & motivation (1.5p)
- Mathematical background: WVF, LF, ODS formulation (2p) — *the canonical math section*
- Experimental setup: datasets, parameter grid, evaluation protocol (1.5p)
- Single-image ablation results (2.5p) — heatmaps, Np curves, d=2 vs d=4, Ns saturation
- Full-dataset generalization (2.5p) — dataset-wide ODS tables, best vs Bagan gap on all 4 datasets
- WVF vs LF (1p) — WVF wins consistently, LF adds cost without clean-data benefit
- Post-processing (0.5p) — NMS hurts raw gradient maps
- Discussion & conclusions (1.5p)

**Dependencies:** None. Foundation report.

---

### Report 2: "Benchmarking WVF/LF Against Classical and Learned Edge Detectors"

**Argument:** Optimally-tuned WVF and LF outperform all classical baselines tested and are competitive with DL models on clean data, at a fraction of the computational cost. WVF consistently edges out LF in this comparison.

**Data:**
- Classical baselines: `outputs/classical_test/classical_ods.json` (54 methods)
- DL full-dataset ODS: `outputs/full_dataset_dl/` (currently running, jobs 349323-349326)
- Inference times: `outputs/model_test/test_results.json`
- Visual comparisons: `outputs/model_test/figures/`, `outputs/classical_test/figures/`

**Sections (~12 pages):**
- Introduction (1p)
- Brief math recap + reference to Report 1 (0.5p)
- Classical comparison (3p) — 54 methods ranked, scale/kernel trends, best WVF and LF vs best classical
- Deep learning comparison (3p) — 5 DL models vs best WVF and best LF, dataset-wide ODS on all 4 datasets
- Computational cost analysis (1.5p) — WVF 0.036s vs LF vs DL models, Pareto frontier (ODS vs time)
- Visual gallery (1.5p) — side-by-side edge maps
- Discussion (1.5p)

**Dependencies:** References Report 1 for optimal WVF parameters.

**TODOs:**
- [ ] Classical baselines currently only on BIPED — need to extend to all 4 datasets
- [ ] Full-dataset DL ODS jobs must complete (submitted as jobs 349323-349326)

---

### Report 3: "Noise Robustness of Edge Detection Methods"

**Argument:** Noise reshapes the edge detection landscape. WVF's optimal parameters shift toward larger support under noise (partially vindicating Bagan for noisy environments). DL models degrade faster than expected, and WVF overtakes several DL models below a crossover SNR.

**Data:**
- WVF/LF noise ablation: `outputs/noise_ablation/` (5 imgs × 4 datasets × 36 conditions × 447 configs)
- DL noise ablation: `outputs/dl_noise_ablation/` (5 models × 5 imgs × 4 datasets × 36 conditions)
- Visual SNR samples: `outputs/dl_edge_map_samples/`
- Noise figures: `outputs/noise_ablation/figures/`

**Sections (~14 pages):**
- Introduction (1p)
- Brief math recap (0.5p)
- Noise model definitions (1.5p) — 6 types, SNR definition, visual examples
- WVF/LF parameter shift under noise (3p) — how optimal Np, d, m change with SNR; noise-type-specific behavior; at what SNR Bagan's params become justified
- WVF vs LF under noise (1p) — does LF's line averaging help?
- DL degradation curves (2.5p) — ODS vs SNR per model, per noise type
- Crossover analysis: WVF vs DL (2p) — at what SNR does WVF overtake each model; depends on noise type
- Visual comparison gallery (1p)
- Discussion (1.5p) — implications for underwater/aquatic deployment

**Dependencies:** References Reports 1 and 2 for clean-data baselines.

**TODOs:**
- [ ] Classical baselines under noise not tested — list as limitation/future work
- [ ] Noise ablation uses 5 images per dataset (not full datasets) — acknowledge as limitation

---

### Report 4: "Statistical Analysis of Cross-Dataset Edge Detection Performance"

**Argument:** The findings from Reports 1-3 are statistically rigorous. Rankings are consistent across datasets, the WVF advantage is significant, and crossover SNR estimates carry quantifiable uncertainty.

**Data:**
- Full-dataset ablation JSONs (1,206+ configs × 4 datasets with ODS/OIS)
- Noise ablation per-image results (5 imgs × 4 datasets × 36 conditions)
- DL noise ablation per-image results

**Sections (~12 pages):**
- Introduction (0.5p)
- Methods: statistical tests used and justification for n=4 dataset design (1.5p)
- Cross-dataset config ranking consistency (2.5p) — Kendall's W, Spearman rank correlation across 4 datasets; do the same WVF configs win everywhere?
- WVF vs LF significance (1.5p) — Wilcoxon signed-rank / paired tests across datasets
- Optimal vs Bagan parameter significance (1.5p) — paired tests, per-dataset effect sizes
- Parameter sensitivity: ANOVA / Kruskal-Wallis (1.5p) — how much does Np, Ns, d, m each contribute?
- Noise crossover confidence intervals (2p) — bootstrap CIs on the SNR where WVF overtakes each DL model; consistency across datasets
- Effect sizes and practical significance (1p)

**Dependencies:** Requires Reports 1-3 completed. This report provides the statistical backbone.

**TODOs:**
- [ ] Write the statistical analysis script (no code exists yet)
- [ ] All data is available — this is purely analysis of existing results

---

## Cross-Report Dependencies

```
Report 1 (Parameters)       ← standalone
    ↓
Report 2 (WVF/LF Benchmarking) ← refs Report 1
    ↓
Report 3 (Noise)            ← refs Reports 1, 2
    ↓
Report 4 (Statistics)       ← refs Reports 1, 2, 3
```

## Shared Elements

- **Typst template:** US letter, 1in margins, New Computer Modern 11pt, `#set heading(numbering: "1.")`  (from existing `proposal.typ`)
- **Math background:** Full in Report 1, brief recap (0.5p) + citation in Reports 2-4
- **Output location:** `papers/report1/`, `papers/report2/`, etc.
- **Figures:** Reference existing PNGs in `outputs/*/figures/` via relative paths; regenerate later as needed

## Outstanding TODOs Before Writing

| TODO | Blocks | Priority |
|------|--------|----------|
| Full-dataset DL ODS (jobs 349323-349326) | Report 2 | Running now |
| Classical baselines on all 4 datasets | Report 2 | High |
| Statistical analysis script | Report 4 | High |
| Regenerate all figures (user mentioned) | All reports | Low (later) |
| Classical baselines under noise | Report 3 (limitation) | Future work |
| Noise ablation on full datasets | Report 3 (limitation) | Future work |
