"""Per-iteration EM trace dump for three canonical pixels (edge,
corner, off-edge) for the §7 figure / debugging.

Runs vmm_em_with_trace at K=3 in BOTH hard-EM (production default) and
soft-EM (deprecated; keeps the per-iteration prior-leakage receipts
for the limitations subsection).

Soft-EM dump shows the γ_{k_signal, n} responsibilities for the
off-axis "tail" samples, illustrating how the prior pi[k_signal]
prevents tail samples from getting fully assigned to the off-axis
component, biasing the consensus mean.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cgmm_vmm import vmm_em_with_trace, theta_M_to_phi_w


REGIMES = [
    ("edge",   [2106, 2063]),
    ("corner", [2157, 1711]),
    ("blank",  [2000, 1000]),
]


def load_real_pixel_streams():
    base = Path("/Users/user/Documents/New project/cetz_figures/data")
    out = {}
    for label, pxy in REGIMES:
        prefix = f"{'corner_v0' if label == 'corner' else label}_m_sweep_4096_noisy"
        ts, ms = [], []
        for ch in "LRGB":
            d = json.loads((base / f"{prefix}_{ch}.json").read_text())
            v = d["vertices"][0]
            for _, row in v["by_m"].items():
                ts.append(row["theta_hat"])
                ms.append(row["M_hat"])
        out[label] = (np.asarray(ts), np.asarray(ms), pxy)
    return out


def main():
    out_dir = Path("outputs/cgmm_vmm_trace")
    out_dir.mkdir(parents=True, exist_ok=True)

    streams = load_real_pixel_streams()
    K, n_iters = 3, 30

    # Stack pixels as (P=3, N=40)
    labels = [lbl for lbl, _ in REGIMES]
    pix_xy = np.stack([streams[l][2] for l in labels])
    theta  = np.stack([streams[l][0] for l in labels])
    mag    = np.stack([streams[l][1] for l in labels])

    phi, w, v = theta_M_to_phi_w(theta, mag)
    tr_hard = vmm_em_with_trace(phi, w, K=K, n_iters=n_iters,
                                hard_em=True)
    tr_soft = vmm_em_with_trace(phi, w, K=K, n_iters=n_iters,
                                hard_em=False)

    # Save both traces
    npz_path = out_dir / "trace_K3.npz"
    np.savez_compressed(npz_path,
        labels      = np.asarray(labels),
        pixel_xy    = pix_xy,
        theta_deg   = theta,
        magnitude   = mag,
        phi         = phi,
        w           = w,
        v           = v,
        # hard-EM (production default)
        hard_mu_trace      = tr_hard["mu_trace"],
        hard_kappa_trace   = tr_hard["kappa_trace"],
        hard_pi_trace      = tr_hard["pi_trace"],
        hard_gamma_trace   = tr_hard["gamma_trace"],
        hard_W_trace       = tr_hard["W_trace"],
        hard_log_lik_trace = tr_hard["log_lik_trace"],
        # soft-EM (deprecated; prior-leakage receipts)
        soft_mu_trace      = tr_soft["mu_trace"],
        soft_kappa_trace   = tr_soft["kappa_trace"],
        soft_pi_trace      = tr_soft["pi_trace"],
        soft_gamma_trace   = tr_soft["gamma_trace"],
        soft_W_trace       = tr_soft["W_trace"],
        soft_log_lik_trace = tr_soft["log_lik_trace"],
    )
    print(f"Wrote {npz_path}")

    for variant_label, tr in [("hard-EM", tr_hard), ("soft-EM", tr_soft)]:
        for p, lbl in enumerate(labels):
            print(f"\n=== [{variant_label}] {lbl} pixel @ {pix_xy[p].tolist()} ===")
            print("iter   pi[0]  pi[1]  pi[2]   mu_deg[0/1/2]                    "
                  "kappa[0/1/2]                  log_lik")
            for it in range(n_iters + 1):
                mu_deg = (tr["mu_trace"][it, p] % (2 * np.pi)) / 2 * 180 / np.pi
                ka     = tr["kappa_trace"][it, p]
                pi_    = tr["pi_trace"][it, p]
                ll = (tr["log_lik_trace"][it - 1, p]
                      if it > 0 else float("nan"))
                print(f"{it:>4}   "
                      f"{pi_[0]:.3f}  {pi_[1]:.3f}  {pi_[2]:.3f}   "
                      f"{mu_deg[0]:>7.3f} / {mu_deg[1]:>7.3f} / {mu_deg[2]:>7.3f}   "
                      f"{ka[0]:>7.1f} / {ka[1]:>7.1f} / {ka[2]:>7.1f}   "
                      f"{ll:>10.3f}")

    # Soft-EM prior-leakage receipts at the edge pixel.
    print(f"\n\n=== soft-EM prior-leakage at edge pixel ===")
    p_edge = labels.index("edge")
    th_edge = theta[p_edge]
    M_edge  = mag[p_edge]
    th_signal_iter30 = (tr_soft["mu_trace"][n_iters, p_edge]
                        % (2 * np.pi)) / 2 * 180 / np.pi
    pi_signal_idx = int(np.argmax(tr_soft["pi_trace"][n_iters, p_edge]))
    th_main = th_signal_iter30[pi_signal_idx]
    print(f"signal cluster mu_theta (final) = {th_main:.4f}, "
          f"pi[k_signal] = {tr_soft['pi_trace'][n_iters, p_edge, pi_signal_idx]:.3f}")
    # Find samples > 1 deg off the signal direction (= "tail" samples)
    sample_offsets = np.abs(th_edge - th_main)
    tail_mask = sample_offsets > 1.0
    if tail_mask.any():
        print(f"Tail samples (|theta - theta_signal| > 1 deg):")
        print(f"   {'theta':>8} {'M':>7} {'gamma_signal[final]':>22}")
        gamma_final = tr_soft["gamma_trace"][n_iters - 1, p_edge,
                                              pi_signal_idx]
        for n in np.where(tail_mask)[0]:
            print(f"   {th_edge[n]:>8.3f} {M_edge[n]:>7.3f} "
                  f"{gamma_final[n]:>22.4f}")
        print(f"   --> note gamma_signal stays >> 0 for tail samples; "
              f"this is the prior-leakage that biases mu_signal away "
              f"from the inlier mean.")
    else:
        print("No tail samples (>1 deg off main direction) at this pixel.")

    # Direct prior-leakage demonstration on a synthetic pixel with the
    # 36-vs-4 pattern that triggered the original GMM-vs-vMM disagreement
    # at idx=1373 in the n=2000 ablation.  Hard-seed correctly separates
    # the two clusters at iter 0; subsequent soft-EM iters then show
    # gamma leakage from the "tail" component back to the dominant one.
    print(f"\n\n=== prior-leakage demo: 36 samples @ 90 deg + "
          f"4 samples @ 91.53 deg, hard-seed + soft-EM ===")
    th_demo = np.array([90.0]*36 + [91.53]*4, dtype=np.float64)
    M_demo  = np.array([17.0]*36 + [18.0, 25.0, 13.0, 7.6], dtype=np.float64)
    phi_d, w_d, _ = theta_M_to_phi_w(th_demo[None, :], M_demo[None, :])
    tr_demo = vmm_em_with_trace(phi_d, w_d, K=3,
                                n_iters=n_iters, hard_seed=True)
    print("iter  pi_signal  mu_signal_theta_deg   gamma_signal[91.53 sample]")
    for it in range(n_iters + 1):
        pi_v = tr_demo["pi_trace"][it, 0]
        mu_v = (tr_demo["mu_trace"][it, 0] % (2*np.pi)) / 2 * 180/np.pi
        ks   = int(np.argmax(pi_v))
        # responsibility at index 36 (first of the 4 tail samples)
        if it == 0:
            gs = float("nan")  # pre-EM
        else:
            gs = tr_demo["gamma_trace"][it - 1, 0, ks, 36]
        print(f"  {it:>3}     {pi_v[ks]:>7.4f}    "
              f"{mu_v[ks]:>10.5f}              {gs:>10.4f}")
    print("    --> if gamma_signal[91.53 sample] > 0 even though signal "
          "mu is at 90 deg, that's the prior-leakage that biases mu_signal "
          "off the inlier mean.")


if __name__ == "__main__":
    main()
