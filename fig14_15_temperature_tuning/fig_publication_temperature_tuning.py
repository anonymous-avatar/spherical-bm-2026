# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Temperature tuning of Potts BMs. A) Pearson correlation between generated
and data connected correlations vs β, for a range of γ. B) σ_k²(β) vs β at
γ = GAMMA_PICK, first six modes. C) empirical β_opt (PT) vs the SBM
unified formula prediction. Usage: uv run … [PF00072|PF00018].
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
PAPER_DIR = ROOT
RESULTS = ROOT / "data"

FAMILY = sys.argv[1] if len(sys.argv) > 1 else "PF00072"
PT_DIR_NAME = {"PF00072": "PF00072_pt_merged", "PF00018": "PF00018_pt"}[FAMILY]
OUT_PDF = PAPER_DIR / f"fig_temperature_tuning_{FAMILY}.pdf"
GAMMA_PICK = 1.0
K_MODES = 6


def load_data():
    ev = np.load(RESULTS / FAMILY / f"{FAMILY}_all_eigvals.npz")
    c_data = np.load(RESULTS / FAMILY / f"{FAMILY}_cov_eigvals_data.npy")
    ts = np.load(RESULTS / PT_DIR_NAME / f"{FAMILY}_temperature_scan.npz")
    cov = np.load(RESULTS / PT_DIR_NAME / f"{FAMILY}_cov_eigvals_vs_beta.npz")

    gammas = sorted({
        float(k.replace("l2_", "").split("_")[0])
        for k in ts.files if k.endswith("_pearson")
    })
    return ev, c_data, ts, cov, gammas


def beta_star_sbm(gamma: float, lam: np.ndarray, c: np.ndarray) -> float:
    n = min(len(lam), len(c))
    lk, ck = lam[:n], c[:n]
    mu = np.sum(lk * ck**2) / np.sum(ck**2)
    num = np.sum(lk**2 * ck**2)
    den = np.sum(lk * (lk - mu) * ck**4)
    return 1.0 + gamma * num / den if den > 0 else np.nan


def panel_pearson(ax, ts, gammas):
    cmap = mpl.colormaps["viridis_r"]
    n = len(gammas)
    opt_betas, opt_pearsons = [], []
    for i, g in enumerate(gammas):
        prefix = f"l2_{g:.4f}"
        b = ts[f"{prefix}_beta"]
        p = ts[f"{prefix}_pearson"]
        mask = b <= 2.1
        ax.plot(
            b[mask], p[mask],
            marker="o", markersize=2.8, linewidth=1.1,
            color=cmap(i / max(n - 1, 1)),
        )
        j = int(np.argmax(p))
        opt_betas.append(float(b[j]))
        opt_pearsons.append(float(p[j]))

    ax.scatter(opt_betas, opt_pearsons,
               s=16, facecolor="firebrick", edgecolor="black",
               linewidth=0.4, zorder=5,
               label=r"$\beta_{\mathrm{opt}}$")

    ax.set_xlabel(r"sampling inverse temperature $\beta$")
    ax.set_ylabel("Pearson correlation", labelpad=4)
    ax.set_xlim(0.45, 2.05)
    ax.set_ylim(0, 1)

    sm = mpl.cm.ScalarMappable(
        cmap=cmap,
        norm=mpl.colors.Normalize(vmin=gammas[0], vmax=gammas[-1]),
    )
    cbar = ax.figure.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label(r"regularization $\gamma$", fontsize=7, labelpad=2)
    cbar.ax.tick_params(labelsize=6.5, length=1.8)


def panel_a(ax, cov, ts):
    prefix = f"l2_{GAMMA_PICK:.4f}"
    betas = sorted({
        float(k.split("_")[-1])
        for k in cov.files if k.startswith(f"{prefix}_beta_")
    })
    full = [cov[f"{prefix}_beta_{b:.2f}"] for b in betas]
    sig = np.array([s[:K_MODES] for s in full]).T
    bulk_sum = np.array([float(s[K_MODES:].sum()) for s in full])
    p = ts[f"{prefix}_pearson"]
    b_scan = ts[f"{prefix}_beta"]
    beta_opt = float(b_scan[np.argmax(p)])

    cmap = mpl.colormaps["plasma"]
    for k in range(K_MODES):
        ax.plot(
            betas, sig[k],
            marker="o", markersize=3.5, linewidth=1.2,
            color=cmap(k / (K_MODES - 1)),
            label=rf"$k={k+1}$",
        )

    ymax_data = float(sig.max())
    ymin_data = float(sig.min())
    ypad_top = 0.35 * (ymax_data - ymin_data)
    ax.set_ylim(ymin_data - 0.05 * (ymax_data - ymin_data),
                ymax_data + ypad_top)
    y_lo, y_hi = ax.get_ylim()
    y_pad = 0.04 * (y_hi - y_lo)
    if abs(beta_opt - round(beta_opt * 10) / 10) > 1e-6 or \
       round(beta_opt * 10) / 10 not in {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}:
        ax.plot([beta_opt, beta_opt], [y_lo + y_pad, y_hi - y_pad],
                color="firebrick", linestyle="--", linewidth=1.0)
    else:
        dx = 0.005 * (3.05 - 0.4)
        ax.plot([beta_opt + dx, beta_opt + dx],
                [y_lo + y_pad, y_hi - y_pad],
                color="firebrick", linestyle="--", linewidth=1.0)
    ax.text(
        beta_opt + 0.06, ymin_data + 0.04 * (ymax_data - ymin_data),
        rf"$\beta_{{\mathrm{{opt}}}}\!=\!{beta_opt:.2f}$",
        color="firebrick", fontsize=7,
        ha="left", va="bottom",
    )

    ax.set_xlabel(r"sampling inverse temperature $\beta$")
    ax.set_ylabel(r"$\sigma_k^2(\beta)$", labelpad=4)
    ax.set_xlim(0.4, 3.05)

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_linewidth(0.6)
    ax2.tick_params(axis="y", direction="in", length=2.5, width=0.6,
                    labelsize=8, pad=4.5, colors="#2a7f2a")
    (bulk_line,) = ax2.plot(
        betas, bulk_sum,
        marker="s", markersize=3.0, linewidth=1.0,
        color="#2a7f2a", linestyle="--",
        label=rf"$\sum_{{k>{K_MODES}}} \sigma_k^2$",
    )
    ax2.set_ylabel(rf"$\sum_{{k>{K_MODES}}} \sigma_k^2(\beta)$",
                   labelpad=4, color="#2a7f2a")

    handles, labels = ax.get_legend_handles_labels()
    handles.append(bulk_line)
    labels.append(bulk_line.get_label())
    ax.legend(handles, labels,
              fontsize=6.5, ncol=4, frameon=False,
              loc="upper center", bbox_to_anchor=(0.52, 1.02),
              handlelength=1.2, columnspacing=0.8,
              borderaxespad=0.0)


def panel_b(ax, ts, ev, c_data, gammas):
    beta_emp, beta_sbm, gs = [], [], []
    for g in gammas:
        prefix = f"l2_{g:.4f}"
        key_ev = f"l2_{g:.4f}"
        if key_ev not in ev.files:
            continue
        j = int(np.argmax(ts[f"{prefix}_pearson"]))
        beta_emp.append(float(ts[f"{prefix}_beta"][j]))
        beta_sbm.append(beta_star_sbm(g, ev[key_ev], c_data))
        gs.append(g)

    ax.plot(gs, beta_emp, marker="o", markersize=4, linewidth=1.3,
            color="black", label=r"empirical $\beta_{\mathrm{opt}}$")
    ax.plot(gs, beta_sbm, marker="^", markersize=4, linewidth=1.3,
            linestyle="--", color="#1f77b4",
            label=r"SBM prediction $\beta^\star$")

    ax.set_xlabel(r"regularization $\gamma$")
    ax.set_ylabel(r"optimal sampling $\beta$", labelpad=4)
    ax.set_xlim(-0.02, max(gs) * 1.05)
    ymax = max(max(beta_emp), max(beta_sbm))
    ax.set_ylim(0.93, ymax + 0.12)
    ax.legend(fontsize=7, frameon=False, loc="lower right",
              handlelength=1.8, borderaxespad=0.3)


def main():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.pad": 5.5,
        "ytick.major.pad": 4.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.fontsize": 7,
    })

    ev, c_data, ts, cov, gammas = load_data()

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 2.8))
    panel_pearson(axes[0], ts, gammas)
    panel_a(axes[1], cov, ts)
    panel_b(axes[2], ts, ev, c_data, gammas)

    for lbl, ax in zip("ABC", axes):
        ax.text(-0.20, 1.05, lbl, transform=ax.transAxes,
                fontsize=10, fontweight="bold", va="bottom", ha="left")

    fig.subplots_adjust(left=0.07, right=0.98, top=0.90,
                        bottom=0.20, wspace=0.50)
    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    fig.savefig(OUT_PDF.with_suffix(".png"), dpi=200)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
