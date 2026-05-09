# SBM vs unconstrained Gaussian. A) Typical reverse KL vs γ.
# B) K=1 phase diagram with the Gaussian-accessible region shaded.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "_julia_env"))

import CairoMakie, Makie
using Makie: @L_str
using UndersampledSphericalBMs2025
const TS = UndersampledSphericalBMs2025.TeacherStudent

figures_dir = joinpath(@__DIR__)

# ── Helpers ──────────────────────────────────────────────────────────

function safe_kl(γ, η, ω)
    try
        return TS.all_kl_divergences(γ, η, ω)
    catch
        return (; fwd_typ=NaN, rev_typ=NaN, fwd_pp=NaN, rev_pp=NaN,
                  phase=:error, M=NaN, ξ=[NaN,NaN], g=[NaN,NaN], u=[NaN,NaN])
    end
end

# Gaussian (paramagnetic-only) reverse KL (closed-form).
# g_Z=1, avg_E_pp=0 ⟹ rev_typ = teacher_logZ - H_student.
gaussian_rev_typ(γ, η, ω) = (ω - log(ω) - 1 + 1 / (2γ * η)) / 2

const TEAL = Makie.RGBf(0.0, 0.55, 0.55)

phase_colors = Dict(
    :ferromagnetic_outlier => :firebrick,
    :ferromagnetic_sticky  => :darkorange,
    :paramagnetic          => :royalblue,
    :error                 => :gray80,
)

function phase_lines!(ax, xs, ys, phases; linewidth=2.5)
    for (phase, color) in phase_colors
        mask = phases .== phase
        any(mask) || continue
        idx = findall(mask)
        start = idx[1]
        for i in 2:length(idx)
            if idx[i] != idx[i-1] + 1
                Makie.lines!(ax, xs[start:idx[i-1]], ys[start:idx[i-1]]; color, linewidth)
                start = idx[i]
            end
        end
        Makie.lines!(ax, xs[start:idx[end]], ys[start:idx[end]]; color, linewidth)
    end
end

function phase_vlines!(ax, xs, phases)
    for i in 2:length(phases)
        if phases[i] != phases[i-1] && phases[i] != :error && phases[i-1] != :error
            Makie.vlines!(ax, [(xs[i] + xs[i-1]) / 2]; color=:gray40, linestyle=:dash, linewidth=1)
        end
    end
end

function shade_gap!(ax, xs, ys_sbm, ys_gauss, phases)
    lo = copy(ys_gauss)
    hi = copy(ys_gauss)
    fm = phases .∈ Ref((:ferromagnetic_outlier, :ferromagnetic_sticky))
    for i in eachindex(xs)
        if fm[i] && isfinite(ys_sbm[i])
            lo[i] = min(ys_sbm[i], ys_gauss[i])
            hi[i] = max(ys_sbm[i], ys_gauss[i])
        end
    end
    Makie.band!(ax, xs, lo, hi; color=(:firebrick, 0.1))
end

# ── Data ─────────────────────────────────────────────────────────────

println("Computing sweep...")
ω = 2.5; η = 5.0
γs = range(0.05, 4.0; length=800)
γv = collect(γs)

rev_sbm = Float64[]; phases = Symbol[]
for γ in γs
    kl = safe_kl(γ, η, ω)
    push!(rev_sbm, kl.rev_typ)
    push!(phases, kl.phase)
end
rev_gauss = gaussian_rev_typ.(γv, η, ω)

# ── Figure ───────────────────────────────────────────────────────────

println("Building figure...")
fig = Makie.Figure(size=(950, 400); figure_padding=15)

# ═══ Panel A: Reverse KL vs γ ═══

ax1 = Makie.Axis(fig[1, 1];
    width=380, height=300,
    xlabel=L"\gamma",
    ylabel=L"\langle D_\mathrm{KL}(P_W \Vert P_{W^*})\rangle / N",
    xgridvisible=false, ygridvisible=false,
    yticks=[0.2, 0.4, 0.6])

Makie.lines!(ax1, γv, rev_gauss;
    color=:gray50, linewidth=2, linestyle=:dash, label="Gaussian")
shade_gap!(ax1, γv, rev_sbm, rev_gauss, phases)
phase_lines!(ax1, γv, rev_sbm, phases)
phase_vlines!(ax1, γv, phases)

Makie.ylims!(ax1, 0.2, 0.5)

Makie.lines!(ax1, [NaN], [NaN]; color=:firebrick, linewidth=2.5, label=L"\mathrm{SBM}\;(h\ne 0)")
Makie.lines!(ax1, [NaN], [NaN]; color=:royalblue, linewidth=2.5, label=L"\mathrm{SBM}\;(h=0)")
Makie.axislegend(ax1; position=:rt, framevisible=false, labelsize=12)

# ═══ Panel B: K=1 Phase diagram ═══

ax2 = Makie.Axis(fig[1, 2];
    width=330, height=300,
    xlabel=L"\gamma", ylabel=L"\eta",
    xgridvisible=false, ygridvisible=false,
    xticks=0:3, yticks=0:5)

L_γ = 3.5
L_η = 5.0

# Shade FM + SG region (uniquely spherical)
Makie.poly!(ax2,
    Makie.Point2f[(0, 0), (1, 0), (1, L_η), (0, L_η)];
    color=(:lightsalmon, 0.35))

γ_sg = collect(range(1, L_γ; length=300))
η_sg = 1 ./ γ_sg
Makie.poly!(ax2, vcat(
    [Makie.Point2f(1, 0)],
    [Makie.Point2f(g, e) for (g, e) in zip(γ_sg, η_sg)],
    [Makie.Point2f(L_γ, 0)]
); color=(:lightsalmon, 0.35))

# Shade PM region (Gaussian-accessible)
Makie.poly!(ax2, vcat(
    [Makie.Point2f(1, 1)],
    [Makie.Point2f(g, e) for (g, e) in zip(γ_sg, η_sg)],
    [Makie.Point2f(L_γ, 1/L_γ)],
    [Makie.Point2f(L_γ, L_η)],
    [Makie.Point2f(1, L_η)]
); color=(:lightblue, 0.25))

# Boundary lines — teal (outlier ↔ edge)
Makie.lines!(ax2, [0, 1], [1, 1]; linewidth=3, color=TEAL)
Makie.lines!(ax2, [1, L_γ], [1, L_γ]; linewidth=3, color=TEAL)

# Boundary lines — black (FM ↔ PM, SG)
Makie.lines!(ax2, [0, 1], [0, 1]; linewidth=3, color=:black)
Makie.lines!(ax2, [1, 1], [1, L_η]; linewidth=3, color=:black)
γ_dense = range(1, L_γ; length=500)
Makie.lines!(ax2, collect(γ_dense), inv.(collect(γ_dense)); linewidth=3, color=:black)

# Phase labels — (h, u_1) order-parameter notation matching Fig. 1
Makie.text!(ax2, 0.5, 3.0; text=L"h,u_1\ne 0", fontsize=18, align=(:center,:center))
Makie.text!(ax2, 2.0, 4.2; text=L"h=0", fontsize=16, align=(:center,:center))
Makie.text!(ax2, 2.0, 3.6; text=L"u_1\ne 0", fontsize=16, align=(:center,:center))
Makie.text!(ax2, 2.7, 1.3; text=L"h=u_1=0", fontsize=18, align=(:center,:center))
Makie.text!(ax2, 1.7, 0.42; text=L"h\ne 0", fontsize=13, align=(:center,:center))
Makie.text!(ax2, 1.7, 0.10; text=L"u_1=0", fontsize=13, align=(:center,:center))

# Annotation
Makie.text!(ax2, 2.3, 3.0; text="Gaussian\naccessible", fontsize=13,
    color=:steelblue, align=(:center, :center), font=:italic)
Makie.text!(ax2, 0.5, 2.0; text="uniquely\nspherical", fontsize=13,
    color=:firebrick, align=(:center, :center), font=:italic)
Makie.text!(ax2, 1.45, 1.65; text="outlier", fontsize=13, color=TEAL,
    rotation=π / 4, align=(:center, :bottom), font=:italic)
Makie.text!(ax2, 1.85, 1.55; text="edge", fontsize=13, color=TEAL,
    rotation=π / 4, align=(:center, :top), font=:italic)

Makie.xlims!(ax2, 0, L_γ)
Makie.ylims!(ax2, 0, L_η)

# ═══ Panel labels ═══

Makie.Label(fig[1, 1, Makie.TopLeft()], "A)";
    fontsize=22, font=:bold, halign=:left, padding=(0, 15, 0, 0))
Makie.Label(fig[1, 2, Makie.TopLeft()], "B)";
    fontsize=22, font=:bold, halign=:left, padding=(0, 0, 0, 0))

Makie.colgap!(fig.layout, 25)
Makie.resize_to_layout!(fig)

Makie.save(joinpath(figures_dir, "fig_sbm_vs_gaussian.pdf"), fig)
Makie.save(joinpath(figures_dir, "fig_sbm_vs_gaussian.png"), fig; px_per_unit=3)
println("Saved to: $(joinpath(figures_dir, "fig_sbm_vs_gaussian.{pdf,png}"))")
