# Fig. 16 — Householder normalizing flow double descent

```bash
# 1. SWAG sweep across γ (16 indices)
for i in $(seq 0 15); do
  uv run simulate/run_gamma.py --gamma-idx $i --outdir data
done
uv run simulate/run_gamma_low.py --outdir data/results_low

# 2. MAP baseline → data/revkl_final.json
uv run simulate/run_map_revkl.py

# 3. render
uv run fig_flow_double_descent.py
```

Reweighting temperatures: T ∈ {0.01, 0.1, 0.5, 1, 5}.
