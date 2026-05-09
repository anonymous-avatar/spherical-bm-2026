# Figs 5 & 12 — Forward-KL temperature phase diagram

Same plot, two aspect ratios in the manuscript.

```bash
julia --project=../_julia_env ../_julia_env/setup.jl    # one-time
julia fig_pp_fwd_kl_curves.jl     # → data/pp_fwd_kl_4gamma.csv
uv run phase_diagram.py            # → phase_diagram_gamma_omega.pdf
```
