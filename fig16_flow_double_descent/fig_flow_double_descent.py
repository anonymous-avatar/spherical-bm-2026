# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "numpy>=2.0",
#     "scipy>=1.13",
#     "matplotlib>=3.8",
# ]
# ///
"""Householder normalizing flow: SWAG reverse-KL/N (top) and FM fraction
(bottom) vs γ, with Boltzmann-reweighted curves at T ∈ {0.01, 0.1, 0.5, 1, 5}
and the MAP baseline overlaid.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "data"
RESULTS_LOW = RESULTS / "results_low"
FIGS = HERE
REVKL_FINAL = RESULTS / "revkl_final.json"

OV_THRESH = 0.15
T_LIST = [0.01, 0.1, 0.5, 1.0, 5.0]
UNIFORM_KL = 0.811


def boltzmann_estimate(npz, T):
    g = float(npz["gamma"])
    N = int(npz["N"])
    kl = npz["sample_kl"]
    tsq = npz["sample_theta_sq"]
    L = N * kl + 0.5 * g * tsq
    log_w = -L / T
    w = np.exp(log_w - logsumexp(log_w))
    return g, float((w * kl).sum())


def empirical_fm_fraction(npz):
    g = float(npz["gamma"])
    seeds = npz["sample_seed"]
    ov = npz["sample_ov"]
    uniq = np.unique(seeds)
    n_fm = sum(1 for s in uniq if ov[seeds == s].mean() > OV_THRESH)
    return g, n_fm / len(uniq)


def collect_files():
    return sorted(RESULTS.glob("gamma_*.npz")) + sorted(RESULTS_LOW.glob("gamma_*.npz"))


def main():
    files = collect_files()
    npzs = [np.load(f) for f in files]

    curves = {}
    for T in T_LIST:
        rows = sorted([boltzmann_estimate(n, T) for n in npzs], key=lambda r: r[0])
        curves[T] = (np.array([r[0] for r in rows]),
                     np.array([r[1] for r in rows]))

    fm_rows = sorted([empirical_fm_fraction(n) for n in npzs], key=lambda r: r[0])
    g_fm = np.array([r[0] for r in fm_rows])
    fm_frac = np.array([r[1] for r in fm_rows])

    with open(REVKL_FINAL) as f:
        orig = json.load(f)
    g_map = np.array(orig["gammas"])
    map_kl = np.array(orig["map_kl_mean"])

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.5, 7.5), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    cmap = plt.cm.jet(np.linspace(0.15, 0.95, len(T_LIST)))

    ax1.plot(g_map, map_kl, ":", color="gray", lw=1.0, marker="^", ms=5,
             mfc="gray", mec="black", mew=0.4,
             label=r"MAP ($\eta\to\infty$)", zorder=3)

    for T, color in zip(T_LIST, cmap):
        g, kl = curves[T]
        eta = 1.0 / T
        ax1.plot(g, kl, "-", color=color, lw=1.0, marker="o", ms=3.2,
                 mfc=color, mec=color,
                 label=fr"$\eta={eta:g}$", zorder=4)

    ax1.axhline(UNIFORM_KL, ls="--", color="gray", lw=1.0, alpha=0.7,
                label="uniform KL", zorder=2)

    ax1.set_xscale("log")
    ax1.set_xlim(0.008, 1000)
    ax1.set_ylabel(r"$\langle D_{\mathrm{KL}}(Q\|P^{*})\rangle / N$", fontsize=13)
    ax1.set_ylim(0.55, 0.86)
    ax1.legend(loc="center right", fontsize=10, framealpha=0.9, edgecolor="0.7")

    ax2.plot(g_fm, fm_frac, "-", color="black", lw=1.0, marker="s", ms=4,
             mfc="black", mec="black", label=r"$h\neq 0$ fraction")
    ax2.set_xscale("log")
    ax2.set_xlim(0.008, 1000)
    ax2.set_xlabel(r"$\gamma$", fontsize=12)
    ax2.set_ylabel(r"$P(h\neq 0)$", fontsize=11)
    ax2.set_ylim(-0.05, 1.1)
    ax2.legend(loc="center right", fontsize=10, framealpha=0.9, edgecolor="0.7")

    out_pdf = FIGS / "repro_paper_fig.pdf"
    out_png = FIGS / "repro_paper_fig.png"
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
