# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "torch>=2.6",
#     "numpy>=2.0",
# ]
# ///
"""Final clean reverse KL: 30 seeds, 16 gamma points, SWAG-LR, phase-resolved."""
import json, time, numpy as np, torch
torch.set_num_threads(1)
from spherical_flow import SphericalFlow


# --- inlined from former double_descent/shared/teacher.py ---

def teacher_log_Z(omega, N):
    return N * (0.5 * (omega - np.log(omega)) + 0.5 * np.log(2 * np.pi * np.e))


def teacher_log_prob(x, omega, w_star):
    N = x.shape[-1]
    proj = x @ w_star
    energy = 0.5 * omega * proj**2 - 0.5 * omega
    return energy - teacher_log_Z(omega, N)
# --- end teacher ---

N = 30; omega = 2.5; lr_sgd = 0.1; T_lang = 0.01; scale = 1.0
n_seeds = 30; n_swag_per_seed = 200; n_mc = 5000
OV_THRESH = 0.15; K_rank = 30; TRIM = 0.05

w = torch.zeros(N); w[0] = 1.0

gammas = np.array([0.2, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0,
                   4.0, 5.0, 7.0, 10.0, 20.0, 50.0, 100.0, 500.0])

def params_to_vec(m):
    return torch.cat([p.data.reshape(-1) for p in m.parameters()])
def vec_to_params(m, v):
    off = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(v[off:off+n].reshape(p.shape)); off += n

def trimmed_stats(arr, trim=TRIM):
    n_trim = max(1, int(len(arr) * trim))
    s = np.sort(arr)[:len(arr)-n_trim]
    return s.mean(), s.std() / np.sqrt(len(s))

results = {"gammas": [], "map_kl_mean": [], "map_kl_sem": [],
           "fm_kl": [], "fm_sem": [], "fm_n": [],
           "pm_kl": [], "pm_sem": [], "pm_n": [], "frac_fm": [],
           "dominant_kl": [], "dominant_sem": [],
           "N": N, "omega": omega, "n_seeds": n_seeds}

for gi, gamma in enumerate(gammas):
    t0 = time.time()
    all_kls, all_ovs, map_kls = [], [], []

    for seed in range(n_seeds):
        torch.manual_seed(seed * 137 + 7)
        model = SphericalFlow(N, n_layers=8)

        adam = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(3000):
            adam.zero_grad()
            x, lq = model.sample_with_log_prob(256)
            lp = teacher_log_prob(x, omega, w)
            loss = (lq - lp).mean() + 0.5 * gamma * model.l2_penalty()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); adam.step()

        with torch.no_grad():
            x, lq = model.sample_with_log_prob(n_mc)
            lp = teacher_log_prob(x, omega, w)
            map_kls.append(((lq - lp).mean().item()) / N)

        opt = torch.optim.SGD(model.parameters(), lr=lr_sgd)
        np_ = sum(p.numel() for p in model.parameters())
        mv = torch.zeros(np_); sv = torch.zeros(np_)
        devs = []; nc = 0; safe = params_to_vec(model).clone()
        for s in range(5000):
            opt.zero_grad()
            x, lq = model.sample_with_log_prob(256)
            lp = teacher_log_prob(x, omega, w)
            loss = (lq - lp).mean() + 0.5 * gamma * model.l2_penalty()
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            ns = (2 * lr_sgd * T_lang)**0.5
            with torch.no_grad():
                for p in model.parameters(): p.add_(ns * torch.randn_like(p))
                v = params_to_vec(model)
                if torch.isfinite(v).all(): safe = v.clone()
                else: vec_to_params(model, safe); continue
            if s % 20 == 0:
                mv += v; sv += v**2
                devs.append(v.clone())
                if len(devs) > K_rank: devs.pop(0)
                nc += 1
        if nc > 0: mv /= nc; sv /= nc
        dv = (sv - mv**2).clamp(min=1e-12)
        D = torch.stack([d - mv for d in devs], dim=1) if devs else torch.zeros(np_, 1)
        Kd = D.shape[1]

        for _ in range(n_swag_per_seed):
            z1 = torch.randn(np_); z2 = torch.randn(Kd)
            theta = mv + (dv.sqrt()*z1 + D@z2/Kd**0.5) / 2**0.5
            vec_to_params(model, theta)
            with torch.no_grad():
                x, lq = model.sample_with_log_prob(n_mc)
                lp = teacher_log_prob(x, omega, w)
                kl_s = ((lq-lp).mean().item())/N
                ov_s = ((x@w)**2).mean().item()/N
                if np.isfinite(kl_s) and kl_s < 10:
                    all_kls.append(kl_s); all_ovs.append(ov_s)

    kls = np.array(all_kls); ovs = np.array(all_ovs)
    fm = ovs > OV_THRESH; pm = ~fm
    frac = fm.sum() / len(fm)

    if frac >= 0.5:
        dom_m, dom_s = trimmed_stats(kls[fm])
    else:
        dom_m, dom_s = trimmed_stats(kls[pm])

    mk = np.array(map_kls)
    dt = time.time() - t0

    results["gammas"].append(float(gamma))
    results["map_kl_mean"].append(float(mk.mean()))
    results["map_kl_sem"].append(float(mk.std()/np.sqrt(len(mk))))
    results["frac_fm"].append(float(frac))
    results["fm_n"].append(int(fm.sum()))
    results["pm_n"].append(int(pm.sum()))
    results["dominant_kl"].append(float(dom_m))
    results["dominant_sem"].append(float(dom_s))

    if fm.sum() > 20:
        m, s = trimmed_stats(kls[fm])
        results["fm_kl"].append(float(m)); results["fm_sem"].append(float(s))
    else:
        results["fm_kl"].append(None); results["fm_sem"].append(None)
    if pm.sum() > 20:
        m, s = trimmed_stats(kls[pm])
        results["pm_kl"].append(float(m)); results["pm_sem"].append(float(s))
    else:
        results["pm_kl"].append(None); results["pm_sem"].append(None)

    fm_s = f"{results['fm_kl'][-1]:.4f}" if results['fm_kl'][-1] else "---"
    pm_s = f"{results['pm_kl'][-1]:.4f}" if results['pm_kl'][-1] else "---"
    print(f"[{gi+1}/{len(gammas)}] g={gamma:.1f}: MAP={mk.mean():.4f} "
          f"DOM={dom_m:.4f}+/-{dom_s:.4f} FM={fm_s}({fm.sum()}) PM={pm_s}({pm.sum()}) "
          f"frac={frac:.2f} ({dt:.0f}s)", flush=True)

from pathlib import Path as _Path
_out = _Path(__file__).resolve().parent.parent / "data" / "revkl_final.json"
_out.parent.mkdir(exist_ok=True)
with open(_out, "w") as f:
    json.dump(results, f, indent=2)
print(f"Saved {_out}")
