# Brief: Reviewer pass 5 ablations

Five ablation studies queued for the paper. Each closes an open
reviewer comment from pass 5. Goal is to run each experiment, produce
a CeTZ figure into the paper repo, and report back with the
decision-rule outcome.

---

## Authoritative spec

The five ablation specs live at
[`ablations.md`](https://github.com/j-vaught/Bagan_journal_WVF_review/blob/main/ablations.md)
on the paper repo `main`. Each entry has a Question, Why, Test setup,
Metric and decision rule, Output, and Status section. Treat that file
as the source of truth for the experiment design. **Do not edit the
spec to fit the implementation.** If the spec is unclear, flag the
ambiguity rather than silently reinterpret.

The five entries are:

| id | reviewer comment | one-line summary |
|----|------|-----|
| A1 | 6, 7 | LF Gaussian weights index-space vs Euclidean |
| A2 | 8   | Orientation estimator, cubic interp vs parabolic vs smoothing vs trig |
| A3 | 12  | Cross-config magnitude normalisation before c-GMM fusion |
| A4 | 13  | c-GMM design, two-pass vs joint $K = 4$ vMM |
| A5 | 15  | Validity-flag reference statistic on glint-contaminated imagery |

---

## Suggested order

A1, A5, A4, A2, A3.

A1 and A5 are warm-ups, fully self-contained, no upstream pipeline
change. A4 sits in the c-GMM stage and is contained but needs a new
joint-fit code path. A2 needs four estimator implementations but is
contained to the orientation-recovery output. A3 is last because it
has the largest blast radius. If A3 changes the default, the §8
figures need to be regenerated.

---

## Where to put the code

| ablation | suggested location | notes |
|---|---|---|
| A1 | `scripts/eval/ablation_a1_lf_weights.py` | drives LF orientation recovery on synthetic edge bank with both weight schemes |
| A2 | `scripts/eval/ablation_a2_orientation.py` | four estimators, swept against each other |
| A3 | `scripts/eval/ablation_a3_normalisation.py` | five normalisation variants on the same dump |
| A4 | `scripts/eval/ablation_a4_cgmm_design.py` | adds joint $K=4$ vMM as a new function in `agent_workspaces/cgmm_metal/` or alongside |
| A5 | `scripts/eval/ablation_a5_validity.py`     | four reference statistics on glint-contaminated synthetic |

If a script needs a new shared utility, put it under `src/edgecritic/`
in the right subpackage. Keep imports off the global namespace.

---

## Source images

Synthetic edge bank already exists at
`example_images/synthetic_nested_shapes/clean/4096/`. Use the
`nested_star_square_oval_low_contrast_mixed_chroma_4096.png` image as
the default working image.

**A5 needs glint-contaminated input.** The repo does not have an
aquatic bank, so synthesise one. Take a clean synthetic image, add a
small number of bright pixels (saturated to 255) clustered into a
"glint" patch, and use that as the contaminated variant. Compare
against the un-contaminated original. If the user has real aquatic
imagery they want to use instead, they will provide it.

---

## Where to put the figures

Each ablation produces one CeTZ + Typst figure in the paper repo.

- Typst source: `cetz_figures/fig_ablation_<id>_<short_name>.typ`,
  e.g. `cetz_figures/fig_ablation_a1_lf_weights.typ`.
- Data files (CSV or JSON) read by the typst figure:
  `cetz_figures/data/ablation_a1/`.
- Compiled PDF output: `cetz_figures/pdfs/fig_ablation_a1_lf_weights.pdf`,
  produced by `typst compile`.

Use the brand palette and convention already established in the other
`cetz_figures/*.typ` files. The palette colours are imported from
`cetz_figures/preamble.typ`. No rounded edges, no matplotlib, no
TikZ.

---

## Workflow per ablation

1. **Fork code in impl repo.** Add the script under `scripts/eval/`
   or `scripts/figures/`. Drive it from the existing fused Metal
   pipeline (`wvf_lf_recover_metal`) plus the c-GMM Metal call
   (`cgmm_fuse_two_pass_metal`) wherever applicable.
2. **Write the result data.** CSV or JSON into the paper repo at
   `cetz_figures/data/ablation_<id>/`. Keep the schema minimal,
   one row per condition, columns for every metric in the spec.
3. **Author the typst figure.** Match the existing
   `fig_*.typ` style. Include a brief caption describing what the
   figure shows. Do not write the prose for the paper body, that is
   the user's job.
4. **Compile and verify.** Run `typst compile cetz_figures/fig_*.typ
   cetz_figures/pdfs/fig_*.pdf` and confirm the figure renders.
5. **Commit and push.** One commit per ablation in the paper repo,
   plus matching commits in the impl repo. Title each
   `Ablation A<id>: <short summary>`.
6. **Reply with the decision-rule outcome.** Per the spec,
   each ablation has a yes/no decision rule. Report which arm wins
   and a one-paragraph summary of why, with the worst-case numbers
   from the data.

---

## What you should not change

- Section 4 through Section 8 of the paper. The user is handling all
  prose updates and any default-change paper edits.
- The Metal kernels in `native/edgecritic_metal/src/lib.rs` outside
  of any new kernel that an ablation requires. If A3 needs a kernel
  change, flag and stop, do not edit silently.
- The reviewer-pass-5 BRIEF in `agent_workspaces/orientation_recovery_metal/`.

---

## Acceptance gate, per ablation

- The script runs end-to-end on the synthetic test image without
  manual intervention.
- The result CSV/JSON is committed under
  `cetz_figures/data/ablation_<id>/`.
- The typst figure compiles cleanly to the PDF in
  `cetz_figures/pdfs/`.
- The decision-rule outcome is reported in the commit message and
  in your reply.

---

## Out of scope

- Comparison against external edge detectors (Canny, Sobel,
  Holistically-nested edge detection). Those belong in the second
  results section that the user will scope later.
- BSDS500, BIPED, Multicue. Same as above.
- Aquatic underwater dataset evaluation. Out of scope for now.
- Performance benchmarking. The five ablations are correctness and
  default-selection studies, not speed studies.

---

## Communication

After each ablation lands, reply with the commit hash, the figure
path, and the decision outcome. The user will fold the results into
the paper text and decide whether to switch any default.
