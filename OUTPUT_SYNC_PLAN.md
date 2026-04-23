## Output Sync Plan

This clone has been organized for a staged sync from the local working copy.

Normalization applied:
- New figures were placed under `figures/` where that convention already exists.
- New BIPED magnitude arrays were placed under `outputs/biped_ablation/figures/edge_maps/`.
- New SNR plots were placed under `outputs/snr_ablation/figures/`.
- New SNR raw arrays remain under `outputs/snr_ablation/cnn_edge_maps/`.

Intentionally not copied:
- Files whose content already exists in the remote repo under a different path.
- Temporary scratch files under `outputs/dl_noise_ablation/de_tmp_*`.
- `DONE` marker files.

Examples of path normalization:
- Local `outputs/single_image_ablation/*` was not copied because the same content already exists remotely under `outputs/BSDS500_image_ablation/*`.
- Local `outputs/biped_ablation/plot_*.png` and most `edge_maps/*.png` already exist remotely under `outputs/biped_ablation/figures/`.
- Local `outputs/noise_ablation/fig_*.png` already exist remotely under `outputs/noise_ablation/figures/`.

Suggested push sequence:

1. Foundational docs and compact figures
- `git add OUTPUT_SYNC_PLAN.md outputs/ablation_proposal outputs/wvf_cuda_ablation outputs/full_dataset_ablation/figures outputs/cuda_demo`
- Suggested commit: `git commit -m "Add proposal and summary output artifacts"`

2. Noise analysis docs and DL summary plots
- `git add outputs/noise_ablation/noise_ablation_proposal.pdf outputs/noise_ablation/noise_ablation_proposal.typ outputs/noise_ablation/dl_comparison_plan.pdf outputs/noise_ablation/dl_comparison_plan.typ outputs/dl_noise_ablation/plots`
- Suggested commit: `git commit -m "Add noise analysis documents and DL summary plots"`

3. BIPED magnitude arrays
- `git add outputs/biped_ablation/figures/edge_maps/*_magnitude.npy`
- Suggested commit: `git commit -m "Add BIPED magnitude arrays"`

4. SNR summaries and figures
- `git add outputs/snr_ablation/*.json outputs/snr_ablation/figures`
- Suggested commit: `git commit -m "Add SNR ablation summaries and figures"`

5. SNR raw edge-map arrays
- `git add outputs/snr_ablation/cnn_edge_maps`
- Suggested commit: `git commit -m "Add SNR CNN edge-map arrays"`

Before each push:
- Run `git status --short`.
- Confirm the staged set only contains the intended batch.
- Push after each commit instead of batching multiple commits into one push if you want tighter checkpoints.
