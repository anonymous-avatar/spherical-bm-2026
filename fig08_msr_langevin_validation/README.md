# Fig. 8 — MSR/DMFT vs finite-N Langevin (K=2, β=1, (γ, η) = (0.5, 3))

```bash
julia --project=../_julia_env ../_julia_env/setup.jl    # one-time
julia simulate/run_validate_K2.jl     # populates data/validate_msr_K2/
julia fig_msr_langevin_validation.jl
```
