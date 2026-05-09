# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch>=2.6",
#     "numpy>=2.0",
# ]
# ///
"""Single-γ run for IS-weighted posterior average over SWAG mixture proposal.

Per seed (m=0..n_seeds-1):
  1. Adam burn-in to MAP
  2. SGLD (SGD + Langevin noise) → SWAG fit (mean mv, diag var dv, low-rank devs D)
  3. Draw n_swag_per_seed samples θ ~ q_m = N(mv, Σ_m=(1/2)(diag(dv)+DD^T/K))
  4. For each θ: evaluate kl_s = D_KL(Q_θ||P*)/N, overlap, log q_m(θ), ||θ||²

Output: NPZ with per-seed SWAG params + per-sample (seed,kl,ov,log_q,theta_sq).
The aggregation script computes weights w_i = exp(-L_i/T - log_q_i).

Self-contained PEP 723 script with SphericalFlow and teacher inlined.

Usage:  uv run run_gamma.py --gamma-idx <0..15> --outdir data
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
torch.set_num_threads(1)


# --- inlined teacher (rank-1 Bingham on S_N) ---

def teacher_log_Z(omega, N):
    return N * (0.5 * (omega - np.log(omega)) + 0.5 * np.log(2 * np.pi * np.e))


def teacher_log_prob(x, omega, w_star):
    N = x.shape[-1]
    proj = x @ w_star
    energy = 0.5 * omega * proj**2 - 0.5 * omega
    return energy - teacher_log_Z(omega, N)


# --- inlined SphericalFlow (Rezende et al. 2020) ---

class SphericalFlow(nn.Module):
    def __init__(self, N: int, n_layers: int = 8):
        super().__init__()
        self.N = N
        self.n_layers = n_layers
        self.alpha = nn.Parameter(torch.zeros(n_layers))
        self.beta = nn.Parameter(torch.zeros(n_layers))
        self.v = nn.Parameter(torch.randn(n_layers, N))
        self._log_vol = (
            (N - 1) / 2 * math.log(N) + math.log(2)
            + N / 2 * math.log(math.pi) - math.lgamma(N / 2)
        )

    @staticmethod
    def _altitude_forward(t, alpha, beta, N):
        EPS = 1e-7
        t = t.clamp(-1 + EPS, 1 - EPS)
        a = 1.0 + alpha
        arctanh_t = 0.5 * (torch.log1p(t) - torch.log1p(-t))
        z = a * arctanh_t + beta
        h = torch.tanh(z).clamp(-1 + EPS, 1 - EPS)
        diff = torch.log1p(-h * h) - torch.log1p(-t * t)
        log_dup = torch.log(a.abs()) + (N - 1) / 2.0 * diff
        return h, log_dup

    @staticmethod
    def _altitude_inverse(h, alpha, beta):
        EPS = 1e-7
        h = h.clamp(-1 + EPS, 1 - EPS)
        a = 1.0 + alpha
        arctanh_h = 0.5 * (torch.log1p(h) - torch.log1p(-h))
        return torch.tanh((arctanh_h - beta) / a)

    def _reflect(self, x, v):
        v_norm = v / v.norm().clamp(min=1e-8)
        return x - 2.0 * (x @ v_norm).unsqueeze(-1) * v_norm

    def forward(self, x):
        sqrt_N = self.N**0.5
        log_dup_total = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)
        for l in range(self.n_layers):
            x = self._reflect(x, self.v[l])
            t = x[..., -1] / sqrt_N
            h, log_dup = self._altitude_forward(t, self.alpha[l], self.beta[l], self.N)
            log_dup_total = log_dup_total + log_dup
            scale = ((1.0 - h * h) / (1.0 - t * t).clamp(min=1e-12)).clamp(min=0).sqrt()
            x_eq = x[..., :-1] * scale.unsqueeze(-1)
            x_alt = sqrt_N * h
            x = torch.cat([x_eq, x_alt.unsqueeze(-1)], dim=-1)
        return x, log_dup_total

    def inverse(self, x):
        sqrt_N = self.N**0.5
        for l in range(self.n_layers - 1, -1, -1):
            h = x[..., -1] / sqrt_N
            t = self._altitude_inverse(h, self.alpha[l], self.beta[l])
            scale = ((1.0 - t * t) / (1.0 - h * h).clamp(min=1e-12)).clamp(min=0).sqrt()
            x_eq = x[..., :-1] * scale.unsqueeze(-1)
            x_alt = sqrt_N * t
            x = torch.cat([x_eq, x_alt.unsqueeze(-1)], dim=-1)
            x = self._reflect(x, self.v[l])
        return x

    def log_prob(self, x):
        z = self.inverse(x)
        _, log_dup = self.forward(z)
        return -self._log_vol - log_dup

    def sample_with_log_prob(self, n):
        z = torch.randn(n, self.N)
        z = z / z.norm(dim=-1, keepdim=True) * self.N**0.5
        x, log_dup = self.forward(z.detach())
        return x, -self._log_vol - log_dup

    def l2_penalty(self):
        return sum(p.pow(2).sum() for p in self.parameters())

# ── matches run_revkl_final.py ──
GAMMAS = np.array([0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0,
                   4.0, 5.0, 7.0, 10.0, 20.0, 50.0, 100.0, 500.0])
N = 30; OMEGA = 2.5
N_SEEDS = 30; N_SWAG_PER_SEED = 200; N_MC = 5000
LR_ADAM = 0.01; ADAM_STEPS = 3000
LR_SGD = 0.1; T_LANG = 0.01; SGD_STEPS = 5000; COLLECT_EVERY = 20
K_RANK = 30; N_LAYERS = 8


def params_to_vec(m):
    return torch.cat([p.data.reshape(-1) for p in m.parameters()])


def vec_to_params(m, v):
    off = 0
    for p in m.parameters():
        n = p.numel()
        p.data.copy_(v[off:off + n].reshape(p.shape))
        off += n


def gaussian_log_density(v: torch.Tensor, dv: torch.Tensor, D: torch.Tensor) -> float:
    """log N(v; 0, Σ) where Σ = (1/2)(diag(dv) + (1/K) D D^T).

    Uses Woodbury / matrix-determinant lemma.

    A = diag(dv) + L L^T with L = D / sqrt(K).  Σ = A/2.
    A^{-1} v = dv^{-1} v - dv^{-1} L M^{-1} L^T dv^{-1} v,  M = I_K + L^T diag(1/dv) L
    log det A = sum(log dv) + log det M
    log det Σ = -P log 2 + log det A
    log N(v; 0, Σ) = -v^T Σ^{-1} v / 2 - log det Σ / 2 - (P/2) log(2π)
                   = -v^T A^{-1} v - log det A / 2 - (P/2) log(π)
    """
    P = v.numel()
    K = D.shape[1]
    L = D / (K ** 0.5)                              # P x K
    inv_dv = 1.0 / dv                               # P
    LtDinv = L.t() * inv_dv                         # K x P
    M = torch.eye(K, dtype=v.dtype) + LtDinv @ L    # K x K
    # Cholesky of M for log det and solve
    Lm = torch.linalg.cholesky(M)
    log_det_M = 2.0 * torch.diagonal(Lm).log().sum()
    # quadratic form v^T A^{-1} v
    y = v * inv_dv                                  # P
    a = LtDinv @ v                                  # K  ( = L^T diag(1/dv) v = L^T y )
    b = torch.cholesky_solve(a.unsqueeze(1), Lm).squeeze(1)
    quad = (v * y).sum() - (a * b).sum()
    log_det_A = dv.log().sum() + log_det_M
    log_pi = float(np.log(np.pi))
    return float(-quad - 0.5 * log_det_A - 0.5 * P * log_pi)


def collect_swag(model, omega, w_star, gamma):
    """Return (mv, dv, D, devs_list) after burn-in + SGLD + collection.

    Note: theta = mv + (sqrt(dv)*z1 + D@z2/sqrt(K)) / sqrt(2)  ⇒  Cov = (1/2)(diag(dv)+DD^T/K).
    """
    adam = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
    for _ in range(ADAM_STEPS):
        adam.zero_grad()
        x, lq = model.sample_with_log_prob(256)
        lp = teacher_log_prob(x, omega, w_star)
        loss = (lq - lp).mean() + 0.5 * gamma * model.l2_penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        adam.step()

    map_vec = params_to_vec(model).clone()

    opt = torch.optim.SGD(model.parameters(), lr=LR_SGD)
    np_ = sum(p.numel() for p in model.parameters())
    mv = torch.zeros(np_); sv = torch.zeros(np_)
    devs = []; nc = 0; safe = params_to_vec(model).clone()
    for s in range(SGD_STEPS):
        opt.zero_grad()
        x, lq = model.sample_with_log_prob(256)
        lp = teacher_log_prob(x, omega, w_star)
        loss = (lq - lp).mean() + 0.5 * gamma * model.l2_penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        ns = (2 * LR_SGD * T_LANG) ** 0.5
        with torch.no_grad():
            for p in model.parameters():
                p.add_(ns * torch.randn_like(p))
            v = params_to_vec(model)
            if torch.isfinite(v).all():
                safe = v.clone()
            else:
                vec_to_params(model, safe)
                continue
        if s % COLLECT_EVERY == 0:
            mv += v; sv += v * v
            devs.append(v.clone())
            if len(devs) > K_RANK:
                devs.pop(0)
            nc += 1
    if nc > 0:
        mv /= nc; sv /= nc
    dv = (sv - mv * mv).clamp(min=1e-12)
    D = torch.stack([d - mv for d in devs], dim=1) if devs else torch.zeros(np_, 1)
    return mv, dv, D, map_vec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gamma-idx", type=int, required=True)
    p.add_argument("--outdir", type=str, default=str(__import__("pathlib").Path(__file__).resolve().parent.parent / "data"))
    args = p.parse_args()

    gamma = float(GAMMAS[args.gamma_idx])
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    w_star = torch.zeros(N); w_star[0] = 1.0

    # Per-seed storage
    swag_means = []     # list of P-vectors
    swag_dvs = []       # list of P-vectors
    swag_Ds = []        # list of P×K matrices
    map_vecs = []       # list of P-vectors (MAP = end of Adam burn-in)

    # Per-sample storage (flat lists)
    sample_seed = []
    sample_kl = []
    sample_ov = []
    sample_logq = []
    sample_theta_sq = []

    t_total = time.time()
    for seed in range(N_SEEDS):
        t0 = time.time()
        torch.manual_seed(seed * 137 + 7)
        model = SphericalFlow(N, n_layers=N_LAYERS)

        mv, dv, D, map_vec = collect_swag(model, OMEGA, w_star, gamma)
        Kd = D.shape[1]

        swag_means.append(mv.numpy())
        swag_dvs.append(dv.numpy())
        swag_Ds.append(D.numpy())
        map_vecs.append(map_vec.numpy())

        for _ in range(N_SWAG_PER_SEED):
            z1 = torch.randn(mv.numel())
            z2 = torch.randn(Kd)
            theta = mv + (dv.sqrt() * z1 + D @ z2 / Kd ** 0.5) / 2 ** 0.5
            vec_to_params(model, theta)
            with torch.no_grad():
                x, lq = model.sample_with_log_prob(N_MC)
                lp = teacher_log_prob(x, OMEGA, w_star)
                kl_s = ((lq - lp).mean().item()) / N
                ov_s = ((x @ w_star) ** 2).mean().item() / N
            if not (np.isfinite(kl_s) and kl_s < 10):
                continue
            log_q = gaussian_log_density(theta - mv, dv, D)
            theta_sq = float((theta * theta).sum().item())
            sample_seed.append(seed)
            sample_kl.append(kl_s)
            sample_ov.append(ov_s)
            sample_logq.append(log_q)
            sample_theta_sq.append(theta_sq)
        print(f"  seed {seed}: {time.time()-t0:.1f}s "
              f"({len(sample_seed)} samples accumulated)", flush=True)

    out = outdir / f"gamma_{args.gamma_idx:02d}_g{gamma}.npz"
    np.savez_compressed(
        out,
        gamma=gamma, N=N, omega=OMEGA, n_layers=N_LAYERS,
        T_lang=T_LANG, lr_sgd=LR_SGD,
        sample_seed=np.array(sample_seed, dtype=np.int32),
        sample_kl=np.array(sample_kl, dtype=np.float64),
        sample_ov=np.array(sample_ov, dtype=np.float64),
        sample_logq=np.array(sample_logq, dtype=np.float64),
        sample_theta_sq=np.array(sample_theta_sq, dtype=np.float64),
        # per-seed SWAG params (used for diagnostics; can be stripped if too large)
        swag_means=np.stack(swag_means),
        swag_dvs=np.stack(swag_dvs),
        # D matrices may have varying second dim; pad to K_RANK
        swag_Ds=np.stack([
            np.pad(d, ((0, 0), (0, K_RANK - d.shape[1])))
            if d.shape[1] < K_RANK else d
            for d in swag_Ds
        ]),
        map_vecs=np.stack(map_vecs),
    )
    print(f"Saved {out}  ({time.time()-t_total:.0f}s total)")


if __name__ == "__main__":
    main()
