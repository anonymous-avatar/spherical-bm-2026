# Figs 20 & 21 — Out-of-equilibrium dynamics on lattice proteins

```bash
# 1. spike diagnostics
uv run simulate/01_spike_diagnostic.py
uv run simulate/01_spike_diagnostic_pf00018.py

# 2. (synthetic lattice-protein) phase sweeps for Fig. 20
bash simulate/run_phase1_sweep.sh

# 3. PF00018 sweeps for Fig. 21
bash simulate/run_pf00018_pilot.sh
bash simulate/run_pf00018_main.sh

# 4. render
uv run fig_ooe_potts.py
uv run fig_ooe_pf00018.py
```
