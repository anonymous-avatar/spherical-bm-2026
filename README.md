# Accompanying code — Undersampled Spherical Boltzmann Machines (NeurIPS 2026)

Source code for the figures in the manuscript. One subdirectory per
figure (or small group). Empty `figXX_*/` directories are placeholders.

| Fig. | Section | File in `figure/` | Subdirectory | Lang |
|---:|:---:|:---|:---|:---:|
|  1 | §2 | `fig2_K=1-phase-diagram-wide.pdf`           | [fig01_K1_phase_diagram_wide](fig01_K1_phase_diagram_wide/) | jl |
|  2 | §2 | `fig_dynamics_combined.pdf`                  | [fig02_dynamics_combined](fig02_dynamics_combined/) | jl |
|  3 | §3 | `fig_temperature_tuning_pmo.pdf`             | [fig03_temperature_tuning_pmo](fig03_temperature_tuning_pmo/) | jl |
|  4 | §3 | `fig_double_descent.pdf`                     | [fig04_double_descent](fig04_double_descent/) | jl |
|  5 | §3 | `phase_diagram_gamma_omega.pdf`              | [fig05_12_phase_diagram_gamma_omega](fig05_12_phase_diagram_gamma_omega/) | jl + py |
|  6 | §3 | `fig_ooe_wrap.pdf`                           | [fig06_ooe_wrap](fig06_ooe_wrap/) | _pending_ |
|  7 | App. A | `fig_phase_diagram_k=2_appendix.pdf`         | [fig07_phase_diagram_k2_appendix](fig07_phase_diagram_k2_appendix/) | jl |
|  8 | App. B | `fig_msr_langevin_validation.pdf`            | [fig08_msr_langevin_validation](fig08_msr_langevin_validation/) | jl |
|  9 | App. B | `fig_finite_eta_phase_diagrams.pdf`          | [fig09_finite_eta_phase_diagrams](fig09_finite_eta_phase_diagrams/) | jl |
| 10 | App. C | `fig_large_K_dmft_ising.pdf`                 | [fig10_large_K_dmft_ising](fig10_large_K_dmft_ising/) | py |
| 11 | App. C | `fig_invariant_dmft_large_K_validation.pdf`  | [fig11_invariant_dmft_large_K_validation](fig11_invariant_dmft_large_K_validation/) | py |
| 12 | App. D | `fig_fwd_pp_kl_phase_diagram.pdf`            | [fig05_12_phase_diagram_gamma_omega](fig05_12_phase_diagram_gamma_omega/) | jl + py |
| 13 | App. E | `fig_sbm_vs_gaussian.pdf`                    | [fig13_sbm_vs_gaussian](fig13_sbm_vs_gaussian/) | jl |
| 14 | App. F | `fig_temperature_tuning_PF00072.pdf`         | [fig14_15_temperature_tuning](fig14_15_temperature_tuning/) | py |
| 15 | App. F | `fig_temperature_tuning_PF00018.pdf`         | [fig14_15_temperature_tuning](fig14_15_temperature_tuning/) | py |
| 16 | App. F | `fig_flow_double_descent.pdf`                | [fig16_flow_double_descent](fig16_flow_double_descent/) | py |
| 17 | App. F | `fig_fvsbn_K_dd.pdf`                         | [fig17_fvsbn_K_dd](fig17_fvsbn_K_dd/) | py |
| 18 | App. F | `fig_rbm_finance_double_descent.pdf`         | [fig18_rbm_finance_double_descent](fig18_rbm_finance_double_descent/) | py + jl |
| 19 | App. F | `fig_bayesgan_unimodal.pdf`                  | [fig19_bayesgan_unimodal](fig19_bayesgan_unimodal/) | py |
| 20 | App. F | `fig_ooe_potts.pdf`                          | [fig20_21_ooe_lattice_protein](fig20_21_ooe_lattice_protein/) | py |
| 21 | App. F | `fig_ooe_pf00018.pdf`                        | [fig20_21_ooe_lattice_protein](fig20_21_ooe_lattice_protein/) | py |

Per-figure layout:
```
figXX_*/
├── fig_*.{jl,py}     # figure script (reads ./data/)
├── simulate/         # data-generating scripts (write ./data/)
└── README.md
```

## Shared infrastructure

- `_lib/UndersampledSphericalBMs2025/` — slim Julia package with the
  closed-form RMT and teacher–student divergences (`RMT_Solution`,
  `RMT_Solution_v2`, `TeacherStudent`).
- `_lib/msr/` — K-arbitrary MSR/DMFT solver (`msr_solver.jl`),
  finite-N coupled Langevin driver (`finiteN.jl`), FFT-based stationary
  K=1 solver (`stationary_fft.jl`).
- `_julia_env/` — shared Julia environment. One-time setup:
  ```
  julia --project=_julia_env _julia_env/setup.jl
  ```
  Every Julia script in the tree activates this env via
  `Pkg.activate(joinpath(@__DIR__, "..", "_julia_env"))`.

## Python

Each Python script declares its dependencies via a PEP 723 `# /// script`
block; `uv run script.py` resolves them on first call.
