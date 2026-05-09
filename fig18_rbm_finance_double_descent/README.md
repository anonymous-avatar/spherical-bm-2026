# Fig. 18 — Gaussian-visible RBM on Ken French (DD at BBP)

```bash
# 1. fetch + standardise the 49-industry returns
uv run simulate/01_prepare_kenfrench.py        # → data/kenfrench49_daily.h5

# 2. instantiate the Julia RBM trainer environment (one-time)
julia --project=simulate -e "using Pkg; Pkg.instantiate()"

# 3. γ × M_train sweeps (write data/H_gauss/*.h5)
bash simulate/06_sweep_gauss.sh
bash simulate/07_extend_gauss.sh
bash simulate/08_sweep_gauss_fixedsubset.sh

# 4. render
uv run fig_rbm_finance_double_descent.py H_gauss
```
