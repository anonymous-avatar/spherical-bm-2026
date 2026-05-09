# Fig. 17 — FVSBN double descent on Curie–Weiss and 2D Ising

K ∈ {4, 8, 16, 32}, N = 16, 10 seeds per (K, γ).

```bash
# Curie–Weiss
cd simulate/cw   && uv run run_K_sweep.py --outdir ../../data/cw

# 2D Ising
cd ../ising      && uv run run_K_sweep.py --outdir ../../data/ising

cd ../..
uv run fig_fvsbn_K_dd.py
```
