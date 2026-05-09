# Figs 14 & 15 — Temperature tuning on PF00072 / PF00018

```bash
# 1. train + parallel-tempering scan (long; uses adabmDCA on GPU/CPU)
cd simulate && uv sync
uv run train_and_analyze.py --family PF00072
uv run train_and_analyze.py --family PF00018
uv run train_and_scan_fine.py --family PF00072
uv run train_and_scan_fine.py --family PF00018
cd ..

# 2. render
uv run fig_publication_temperature_tuning.py PF00072
uv run fig_publication_temperature_tuning.py PF00018
```

Expected layout under `data/`:
```
<F>/<F>_all_eigvals.npz
<F>/<F>_cov_eigvals_data.npy
<F>_pt[_merged]/<F>_temperature_scan.npz
<F>_pt[_merged]/<F>_cov_eigvals_vs_beta.npz
```
PF00072 used a merged PT grid (`_pt_merged`); PF00018 uses `_pt`. Adjust
`PT_DIR_NAME` in the figure script if your sweep is in `_pt`.
