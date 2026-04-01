"""Statistical analysis of cross-dataset edge detection performance.

Analyses performed:
  1. Cross-dataset config ranking consistency (Kendall's W, Spearman rho)
  2. WVF vs LF significance (Wilcoxon signed-rank across datasets)
  3. Optimal vs Bagan parameter significance (paired tests + effect sizes)
  4. Parameter sensitivity (Kruskal-Wallis for Np, Ns, d, m)
  5. WVF vs DL model comparison across datasets (full-dataset ODS)
  6. Noise crossover analysis: SNR at which WVF overtakes each DL model,
     with bootstrap confidence intervals, per noise type and dataset

Outputs saved to outputs/statistical_analysis/
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import wilcoxon, kruskal, spearmanr, kendalltau
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "statistical_analysis"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["UDED", "BIPED_v1", "BIPED_v2", "BSDS500"]
DS_KEYS  = ["uded", "biped_v1", "biped_v2", "bsds500"]
DL_MODELS = ["dexined", "teed", "pidinet", "nbed", "diffusionedge"]
DL_NAMES  = ["DexiNed", "TEED", "pidinet", "NBED", "DiffusionEdge"]
NOISE_TYPES = ["gaussian", "salt_pepper", "poisson", "speckle", "uniform"]
SNR_LEVELS  = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

N_BOOTSTRAP = 2000
RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_full_dataset_ablation():
    """Load WVF/LF full-dataset ablation results for all 4 datasets."""
    data = {}
    for ds in DATASETS:
        fpath = ROOT / "outputs/full_dataset_ablation" / f"{ds.lower()}_ablation.json"
        if not fpath.exists():
            fpath = ROOT / "outputs/full_dataset_ablation" / f"{ds.lower().replace('_','')}_ablation.json"
        with open(fpath) as f:
            data[ds] = json.load(f)
    return data


def load_full_dataset_dl():
    """Load full-dataset ODS for DL models (jobs 349323-349326)."""
    data = {}
    dl_dir = ROOT / "outputs/full_dataset_dl"
    for model in DL_MODELS:
        summary = dl_dir / f"{model}_summary.json"
        if summary.exists():
            with open(summary) as f:
                data[model] = json.load(f)
    return data


def load_noise_ablation():
    """Load per-image WVF/LF noise ablation results.

    Returns dict: ds_key -> image_name -> list of condition dicts,
    each with 'noise_type', 'snr', 'results' (list of {filter, Np, Ns, d[, m], ods}).
    """
    data = {}
    noise_dir = ROOT / "outputs/noise_ablation"
    for ds_key, ds_name in zip(DS_KEYS, DATASETS):
        ds_dir = noise_dir / ds_key
        if not ds_dir.exists():
            continue
        data[ds_name] = {}
        for fpath in sorted(ds_dir.glob("*_results.json")):
            with open(fpath) as f:
                img_data = json.load(f)
            data[ds_name][img_data["image"]] = img_data["conditions"]
    return data


def load_dl_noise_ablation():
    """Load per-image DL noise ablation results.

    Returns dict: model -> ds_name -> image_name -> list of condition dicts.
    """
    data = {}
    dl_noise_dir = ROOT / "outputs/dl_noise_ablation"
    for model in DL_MODELS:
        model_dir = dl_noise_dir / model
        if not model_dir.exists():
            continue
        data[model] = {}
        for ds_key, ds_name in zip(DS_KEYS, DATASETS):
            ds_dir = model_dir / ds_key
            if not ds_dir.exists():
                continue
            data[model][ds_name] = {}
            for fpath in sorted(ds_dir.glob("*_results.json")):
                with open(fpath) as f:
                    img_data = json.load(f)
                data[model][ds_name][img_data["image"]] = img_data["conditions"]
    return data


# ---------------------------------------------------------------------------
# Helper: extract best ODS from a list of per-config results
# ---------------------------------------------------------------------------

def best_ods(results, filter_type=None):
    """Return highest ODS from a condition's results list, optionally filtered."""
    candidates = [r for r in results if r["ods"] >= 0]
    if filter_type:
        candidates = [r for r in candidates if r["filter"] == filter_type]
    if not candidates:
        return np.nan
    return max(r["ods"] for r in candidates)


def get_config_ods(results, filter_type, **params):
    """Return ODS for a specific config (match all given params)."""
    for r in results:
        if r["filter"] != filter_type:
            continue
        if all(r.get(k) == v for k, v in params.items()):
            return r["ods"]
    return np.nan


# ---------------------------------------------------------------------------
# Analysis 1: Cross-dataset WVF config ranking consistency
# ---------------------------------------------------------------------------

def analysis_ranking_consistency(ablation_data):
    """Compute Kendall's W and pairwise Spearman rho across datasets for WVF configs.

    For each dataset, extract ODS for each WVF config. Find configs present in
    all 4 datasets, rank them, then measure concordance.
    """
    print("\n" + "="*70)
    print("ANALYSIS 1: Cross-dataset WVF config ranking consistency")
    print("="*70)

    # Build per-dataset ODS dicts keyed by (Np, Ns, d)
    ds_ods = {}
    for ds in DATASETS:
        wvf = ablation_data[ds]["wvf"]
        ds_ods[ds] = {(r["Np"], r["Ns"], r["d"]): r["ods"] for r in wvf if r["ods"] >= 0}

    # Find configs present in all datasets
    common_keys = set(ds_ods[DATASETS[0]].keys())
    for ds in DATASETS[1:]:
        common_keys &= set(ds_ods[ds].keys())
    common_keys = sorted(common_keys)
    n_configs = len(common_keys)
    print(f"  Configs present in all 4 datasets: {n_configs}")

    # Build ranks matrix (configs × datasets)
    ods_matrix = np.array([[ds_ods[ds][k] for ds in DATASETS] for k in common_keys])
    rank_matrix = np.apply_along_axis(
        lambda x: stats.rankdata(-x), 0, ods_matrix)  # rank 1 = best

    # Kendall's W
    n, k = rank_matrix.shape  # n configs, k datasets
    rank_sums = rank_matrix.sum(axis=1)
    mean_rank = rank_sums.mean()
    S = np.sum((rank_sums - mean_rank)**2)
    W = 12 * S / (k**2 * (n**3 - n))
    # Chi-squared approximation
    chi2 = k * (n - 1) * W
    p_val = 1 - stats.chi2.cdf(chi2, df=n - 1)

    print(f"\n  Kendall's W = {W:.4f}  (chi2={chi2:.1f}, p={p_val:.2e})")
    print(f"  Interpretation: W=0 → no agreement, W=1 → perfect agreement")

    # Pairwise Spearman rho
    print("\n  Pairwise Spearman rho (WVF ODS rankings):")
    pairs = []
    for i, ds_i in enumerate(DATASETS):
        for j, ds_j in enumerate(DATASETS):
            if j <= i:
                continue
            rho, p = spearmanr(ods_matrix[:, i], ods_matrix[:, j])
            pairs.append((ds_i, ds_j, rho, p))
            print(f"    {ds_i} vs {ds_j}: rho={rho:.4f} (p={p:.2e})")

    # Top-10 configs and their cross-dataset rank stability
    top10_by_ds = {}
    for di, ds in enumerate(DATASETS):
        top10_by_ds[ds] = sorted(common_keys, key=lambda k: -ds_ods[ds][k])[:10]

    # How many of the top-10 on BSDS500 are also top-10 on other datasets?
    bsds_top10 = set(top10_by_ds["BSDS500"])
    print("\n  Overlap of BSDS500 top-10 configs in other datasets' top-10:")
    for ds in DATASETS:
        if ds == "BSDS500":
            continue
        overlap = len(bsds_top10 & set(top10_by_ds[ds]))
        print(f"    {ds}: {overlap}/10")

    results = {
        "kendalls_W": float(W),
        "kendalls_W_chi2": float(chi2),
        "kendalls_W_p": float(p_val),
        "n_configs": n_configs,
        "pairwise_spearman": [
            {"ds1": a, "ds2": b, "rho": float(r), "p": float(p)}
            for a, b, r, p in pairs
        ],
        "top10_overlap_with_bsds500": {
            ds: len(bsds_top10 & set(top10_by_ds[ds]))
            for ds in DATASETS if ds != "BSDS500"
        },
    }
    return results


# ---------------------------------------------------------------------------
# Analysis 2: WVF vs LF significance
# ---------------------------------------------------------------------------

def analysis_wvf_vs_lf(ablation_data):
    """Paired Wilcoxon signed-rank test: best WVF vs best LF per matched config family."""
    print("\n" + "="*70)
    print("ANALYSIS 2: WVF vs LF significance (paired per-config ODS)")
    print("="*70)

    results = {}
    for ds in DATASETS:
        wvf_ods = [r["ods"] for r in ablation_data[ds]["wvf"] if r["ods"] >= 0]
        lf_ods  = [r["ods"] for r in ablation_data[ds]["lf"]  if r["ods"] >= 0]
        # Use the full distribution of each filter's ODS values (not paired per config)
        # Instead, compare corresponding Np/Ns pairs where LF has m=1 (smallest extension)
        wvf_dict = {(r["Np"], r["Ns"], r["d"]): r["ods"]
                    for r in ablation_data[ds]["wvf"] if r["ods"] >= 0}
        lf_m1 = {(r["Np"], r["Ns"], r["d"]): r["ods"]
                  for r in ablation_data[ds]["lf"]
                  if r["ods"] >= 0 and r["m"] == 1}

        common = sorted(set(wvf_dict) & set(lf_m1))
        if len(common) < 5:
            print(f"  {ds}: insufficient matched pairs ({len(common)}), skipping")
            continue

        w_vals = np.array([wvf_dict[k] for k in common])
        l_vals = np.array([lf_m1[k] for k in common])
        diffs = w_vals - l_vals

        stat, p = wilcoxon(diffs, alternative="greater")
        median_diff = float(np.median(diffs))
        # Cohen's d analog for matched pairs
        cohen_d = float(np.mean(diffs) / (np.std(diffs) + 1e-9))

        best_wvf = float(max(w_vals))
        best_lf  = float(max(l_vals))
        print(f"\n  {ds}:")
        print(f"    Best WVF={best_wvf:.4f}  Best LF(m=1)={best_lf:.4f}")
        print(f"    Median WVF-LF diff = {median_diff:+.4f}")
        print(f"    Wilcoxon W={stat:.1f}, p={p:.2e}, Cohen's d={cohen_d:.3f}")
        print(f"    n matched pairs = {len(common)}")

        results[ds] = {
            "best_wvf": best_wvf, "best_lf": best_lf,
            "median_diff": median_diff, "cohen_d": cohen_d,
            "wilcoxon_stat": float(stat), "p_value": float(p),
            "n_pairs": len(common),
        }
    return results


# ---------------------------------------------------------------------------
# Analysis 3: Optimal vs Bagan parameter significance
# ---------------------------------------------------------------------------

def analysis_optimal_vs_bagan(ablation_data):
    """Paired test: per-dataset best config ODS vs Bagan config ODS."""
    print("\n" + "="*70)
    print("ANALYSIS 3: Optimal vs Bagan parameter comparison")
    print("="*70)

    BAGAN_WVF = {"Np": 250, "Ns": 18, "d": 4}
    BAGAN_LF  = {"Np": 250, "Ns": 18, "d": 4, "m": 14}

    results = {}
    for ds in DATASETS:
        wvf_all = ablation_data[ds]["wvf"]
        lf_all  = ablation_data[ds]["lf"]

        best_wvf = max((r for r in wvf_all if r["ods"] >= 0), key=lambda x: x["ods"])
        bagan_wvf = next((r for r in wvf_all
                          if r["Np"]==250 and r["Ns"]==18 and r["d"]==4), None)

        best_lf  = max((r for r in lf_all if r["ods"] >= 0), key=lambda x: x["ods"])
        bagan_lf = next((r for r in lf_all
                         if r["Np"]==250 and r["Ns"]==18 and r["m"]==14), None)

        wvf_gap = best_wvf["ods"] - (bagan_wvf["ods"] if bagan_wvf else np.nan)
        lf_gap  = best_lf["ods"]  - (bagan_lf["ods"]  if bagan_lf  else np.nan)

        print(f"\n  {ds}:")
        bagan_wvf_str = f"{bagan_wvf['ods']:.4f}" if bagan_wvf else "N/A"
        bagan_lf_str  = f"{bagan_lf['ods']:.4f}"  if bagan_lf  else "N/A"
        print(f"    WVF: best={best_wvf['ods']:.4f} (Np={best_wvf['Np']},Ns={best_wvf['Ns']},d={best_wvf['d']})"
              f"  Bagan={bagan_wvf_str}  gap={wvf_gap:+.4f}")
        print(f"    LF:  best={best_lf['ods']:.4f}  (Np={best_lf['Np']},Ns={best_lf['Ns']},m={best_lf['m']})"
              f"  Bagan={bagan_lf_str}  gap={lf_gap:+.4f}")

        results[ds] = {
            "best_wvf": {"ods": float(best_wvf["ods"]), "Np": best_wvf["Np"],
                         "Ns": best_wvf["Ns"], "d": best_wvf["d"]},
            "bagan_wvf": {"ods": float(bagan_wvf["ods"])} if bagan_wvf else None,
            "wvf_gap": float(wvf_gap),
            "best_lf":  {"ods": float(best_lf["ods"]),  "Np": best_lf["Np"],
                         "Ns": best_lf["Ns"], "m": best_lf["m"]},
            "bagan_lf":  {"ods": float(bagan_lf["ods"])} if bagan_lf else None,
            "lf_gap": float(lf_gap),
        }

    # Cross-dataset significance: are WVF gaps consistent (sign test)?
    wvf_gaps = [results[ds]["wvf_gap"] for ds in DATASETS if ds in results]
    lf_gaps  = [results[ds]["lf_gap"]  for ds in DATASETS if ds in results]
    wvf_positive = sum(g > 0 for g in wvf_gaps)
    lf_positive  = sum(g > 0 for g in lf_gaps)
    # Binomial sign test
    wvf_p = stats.binomtest(wvf_positive, len(wvf_gaps), p=0.5, alternative="greater").pvalue
    lf_p  = stats.binomtest(lf_positive,  len(lf_gaps),  p=0.5, alternative="greater").pvalue
    print(f"\n  Sign test (optimal > Bagan across datasets):")
    print(f"    WVF: {wvf_positive}/{len(wvf_gaps)} positive gaps, p={wvf_p:.4f}")
    print(f"    LF:  {lf_positive}/{len(lf_gaps)} positive gaps, p={lf_p:.4f}")

    results["sign_test"] = {
        "wvf_n_positive": wvf_positive, "wvf_n_total": len(wvf_gaps), "wvf_p": float(wvf_p),
        "lf_n_positive":  lf_positive,  "lf_n_total":  len(lf_gaps),  "lf_p":  float(lf_p),
    }
    return results


# ---------------------------------------------------------------------------
# Analysis 4: Parameter sensitivity (Kruskal-Wallis)
# ---------------------------------------------------------------------------

def analysis_parameter_sensitivity(ablation_data):
    """Kruskal-Wallis test for effect of each parameter on WVF ODS (pooled across datasets)."""
    print("\n" + "="*70)
    print("ANALYSIS 4: Parameter sensitivity (Kruskal-Wallis)")
    print("="*70)

    # Pool all WVF configs across datasets
    all_records = []
    for ds in DATASETS:
        for r in ablation_data[ds]["wvf"]:
            if r["ods"] >= 0:
                all_records.append({"ds": ds, **r})

    results = {}
    for param in ["Np", "Ns", "d"]:
        groups = {}
        for rec in all_records:
            v = rec[param]
            groups.setdefault(v, []).append(rec["ods"])
        group_vals = sorted(groups.keys())
        group_lists = [groups[v] for v in group_vals]
        stat, p = kruskal(*group_lists)
        # Eta-squared effect size
        n_total = sum(len(g) for g in group_lists)
        eta2 = (stat - len(group_lists) + 1) / (n_total - len(group_lists))
        print(f"\n  {param}: H={stat:.1f}, p={p:.2e}, eta^2={eta2:.4f}  "
              f"({len(group_vals)} levels, n={n_total})")
        # Median ODS per level
        medians = {v: float(np.median(groups[v])) for v in group_vals}
        best_val = max(medians, key=medians.get)
        print(f"    Best level: {param}={best_val} (median ODS={medians[best_val]:.4f})")
        results[param] = {
            "H_stat": float(stat), "p_value": float(p), "eta_squared": float(eta2),
            "n_levels": len(group_vals), "n_total": n_total,
            "median_ods_per_level": {str(k): v for k, v in medians.items()},
        }

    # Same for LF parameter m
    lf_records = []
    for ds in DATASETS:
        for r in ablation_data[ds]["lf"]:
            if r["ods"] >= 0:
                lf_records.append({"ds": ds, **r})

    for param in ["m"]:
        groups = {}
        for rec in lf_records:
            v = rec[param]
            groups.setdefault(v, []).append(rec["ods"])
        group_vals = sorted(groups.keys())
        group_lists = [groups[v] for v in group_vals]
        stat, p = kruskal(*group_lists)
        n_total = sum(len(g) for g in group_lists)
        eta2 = (stat - len(group_lists) + 1) / (n_total - len(group_lists))
        print(f"\n  LF {param}: H={stat:.1f}, p={p:.2e}, eta^2={eta2:.4f}  "
              f"({len(group_vals)} levels, n={n_total})")
        medians = {v: float(np.median(groups[v])) for v in group_vals}
        best_val = max(medians, key=medians.get)
        print(f"    Best level: {param}={best_val} (median ODS={medians[best_val]:.4f})")
        results[f"LF_{param}"] = {
            "H_stat": float(stat), "p_value": float(p), "eta_squared": float(eta2),
            "n_levels": len(group_vals), "n_total": n_total,
            "median_ods_per_level": {str(k): v for k, v in medians.items()},
        }
    return results


# ---------------------------------------------------------------------------
# Analysis 5: WVF vs DL full-dataset ODS comparison
# ---------------------------------------------------------------------------

def analysis_wvf_vs_dl(ablation_data, dl_data):
    """Compare best WVF ODS vs each DL model across datasets."""
    print("\n" + "="*70)
    print("ANALYSIS 5: WVF vs DL full-dataset ODS")
    print("="*70)

    if not dl_data:
        print("  No DL full-dataset results available yet.")
        return {}

    results = {}
    print(f"\n  {'Dataset':<12} {'Best WVF':>9} ", end="")
    available_models = [m for m in DL_MODELS if m in dl_data]
    for m in available_models:
        print(f"{m:>12}", end="")
    print()
    print("  " + "-" * (12 + 10 + 13 * len(available_models)))

    for ds in DATASETS:
        best_wvf = max(r["ods"] for r in ablation_data[ds]["wvf"] if r["ods"] >= 0)
        row = {"best_wvf": float(best_wvf), "dl": {}}
        print(f"  {ds:<12} {best_wvf:>9.4f} ", end="")
        for m in available_models:
            ds_result = dl_data[m]["datasets"].get(ds)
            if ds_result:
                ods = ds_result["ods"]
                row["dl"][m] = float(ods)
                gap = best_wvf - ods
                marker = "WVF>" if gap > 0 else "DL> "
                print(f"{ods:>9.4f}{marker:>3}", end="")
            else:
                print(f"{'N/A':>12}", end="")
        print()
        results[ds] = row

    # Sign test: across datasets, is WVF > each DL model?
    print("\n  Sign test (best WVF > DL across 4 datasets):")
    sign_results = {}
    for m in available_models:
        gaps = []
        for ds in DATASETS:
            if m in results.get(ds, {}).get("dl", {}):
                gaps.append(results[ds]["best_wvf"] - results[ds]["dl"][m])
        n_pos = sum(g > 0 for g in gaps)
        p = stats.binomtest(n_pos, len(gaps), p=0.5, alternative="greater").pvalue if gaps else np.nan
        print(f"    {m}: {n_pos}/{len(gaps)} datasets WVF wins, p={p:.4f}")
        sign_results[m] = {"n_positive": n_pos, "n_total": len(gaps), "p": float(p)}
    results["sign_tests"] = sign_results
    return results


# ---------------------------------------------------------------------------
# Analysis 6: Noise crossover with bootstrap CIs
# ---------------------------------------------------------------------------

def analysis_noise_crossover(noise_data, dl_noise_data):
    """For each DL model and noise type, find SNR at which best-WVF overtakes DL.

    Uses per-image ODS (5 images per dataset). Bootstrap CIs over images.
    """
    print("\n" + "="*70)
    print("ANALYSIS 6: Noise crossover (WVF vs DL) with bootstrap CIs")
    print("="*70)

    if not noise_data or not dl_noise_data:
        print("  Missing data, skipping.")
        return {}

    snr_float = {str(s): s for s in SNR_LEVELS}
    snr_float["inf"] = np.inf

    results = {}

    for noise_type in NOISE_TYPES:
        print(f"\n  Noise type: {noise_type}")
        results[noise_type] = {}

        for model in DL_MODELS:
            if model not in dl_noise_data:
                continue

            crossover_snrs = []  # one per (dataset, image) pair

            for ds in DATASETS:
                if ds not in noise_data or ds not in dl_noise_data.get(model, {}):
                    continue

                # Find images present in both WVF and DL noise ablations
                wvf_imgs = set(noise_data[ds].keys())
                dl_imgs  = set(dl_noise_data[model][ds].keys())
                common_imgs = wvf_imgs & dl_imgs

                for img in common_imgs:
                    wvf_conds = {(c["noise_type"], str(c["snr"])): c
                                 for c in noise_data[ds][img]}
                    dl_conds  = {(c["noise_type"], str(c["snr"])): c
                                 for c in dl_noise_data[model][ds][img]}

                    # Build ODS vs SNR for this noise type
                    crossover_snr = np.inf  # WVF never overtakes by default
                    prev_snrs = [np.inf] + SNR_LEVELS  # descending noise = ascending SNR

                    for snr in reversed(SNR_LEVELS):  # low SNR (heavy noise) first
                        key = (noise_type, str(snr))
                        if key not in wvf_conds or key not in dl_conds:
                            continue
                        wvf_ods = best_ods(wvf_conds[key]["results"], filter_type="WVF")
                        dl_ods  = dl_conds[key]["ods"]
                        if np.isnan(wvf_ods) or dl_ods < 0:
                            continue
                        if wvf_ods >= dl_ods:
                            crossover_snr = snr

                    crossover_snrs.append(crossover_snr)

            if not crossover_snrs:
                continue

            crossover_arr = np.array([x for x in crossover_snrs if not np.isinf(x)])
            n_never = sum(np.isinf(x) for x in crossover_snrs)
            n_total = len(crossover_snrs)

            if len(crossover_arr) == 0:
                median_co = np.inf
                ci_lo, ci_hi = np.inf, np.inf
            else:
                median_co = float(np.median(crossover_arr))
                # Bootstrap CI on median crossover SNR
                boot_medians = [
                    np.median(RNG.choice(crossover_arr, size=len(crossover_arr), replace=True))
                    for _ in range(N_BOOTSTRAP)
                ]
                ci_lo = float(np.percentile(boot_medians, 2.5))
                ci_hi = float(np.percentile(boot_medians, 97.5))

            frac_overtakes = (n_total - n_never) / n_total if n_total > 0 else 0
            print(f"    {model:15s}: crossover SNR={median_co:.2f} "
                  f"[{ci_lo:.2f}, {ci_hi:.2f}]  "
                  f"WVF overtakes in {frac_overtakes:.0%} of img-pairs")

            results[noise_type][model] = {
                "median_crossover_snr": float(median_co) if not np.isinf(median_co) else None,
                "ci_95_lo": float(ci_lo) if not np.isinf(ci_lo) else None,
                "ci_95_hi": float(ci_hi) if not np.isinf(ci_hi) else None,
                "frac_wvf_overtakes": float(frac_overtakes),
                "n_image_pairs": n_total,
                "n_never_overtakes": int(n_never),
            }

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    ablation_data = load_full_dataset_ablation()
    dl_data       = load_full_dataset_dl()
    noise_data    = load_noise_ablation()
    dl_noise_data = load_dl_noise_ablation()

    print(f"  Full-dataset ablation: {len(ablation_data)} datasets")
    print(f"  DL full-dataset ODS: {list(dl_data.keys())}")
    print(f"  WVF noise ablation: {list(noise_data.keys())}")
    print(f"  DL noise ablation: {list(dl_noise_data.keys())}")

    all_results = {}

    all_results["ranking_consistency"] = analysis_ranking_consistency(ablation_data)
    all_results["wvf_vs_lf"]           = analysis_wvf_vs_lf(ablation_data)
    all_results["optimal_vs_bagan"]    = analysis_optimal_vs_bagan(ablation_data)
    all_results["parameter_sensitivity"] = analysis_parameter_sensitivity(ablation_data)
    all_results["wvf_vs_dl"]           = analysis_wvf_vs_dl(ablation_data, dl_data)
    all_results["noise_crossover"]     = analysis_noise_crossover(noise_data, dl_noise_data)

    out_path = OUT / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: None if np.isinf(x) else float(x))
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    rc = all_results["ranking_consistency"]
    print(f"  Kendall's W (config rankings): {rc['kendalls_W']:.4f} (p={rc['kendalls_W_p']:.2e})")
    st = all_results["optimal_vs_bagan"]["sign_test"]
    print(f"  Sign test WVF optimal>Bagan:  {st['wvf_n_positive']}/{st['wvf_n_total']} datasets (p={st['wvf_p']:.4f})")
    print(f"  Sign test LF  optimal>Bagan:  {st['lf_n_positive']}/{st['lf_n_total']} datasets (p={st['lf_p']:.4f})")
    ps = all_results["parameter_sensitivity"]
    for param in ["Np", "Ns", "d", "LF_m"]:
        if param in ps:
            print(f"  Kruskal-Wallis {param}: H={ps[param]['H_stat']:.1f}, "
                  f"eta^2={ps[param]['eta_squared']:.4f}, p={ps[param]['p_value']:.2e}")


if __name__ == "__main__":
    main()
