### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 30000001-0000-0000-0000-000000000002
begin
    import Pkg
    Pkg.activate(normpath(joinpath(@__DIR__, "..")))
    using AssessmentTools
    using AssessmentTools.CohortAnalysis
    using LinearAlgebra
    using UnicodePlots
end

# ╔═╡ 30000001-0000-0000-0000-000000000001
md"""
# Cohort Analysis, Virtual Population Analysis (VPA) & ADAPT

This notebook demonstrates historical stock reconstruction using the **`AssessmentTools.CohortAnalysis`** submodule.

### Methodological Concept & Analytical Assumptions
1. **Backward Cohort Recursion**:
   Reconstructs historical abundances-at-age $N_{a,t}$ and fishing mortalities $F_{a,t}$ backwards through time from catch observations $C_{a,t}$. Cohorts are followed diagonally across the catch matrix:
   $$N_{a,t} \longrightarrow N_{a+1,t+1}$$
2. **Pope's Approximation (1972)**:
   Assumes that natural mortality occurs continuously throughout the year, but total annual fishery catches occur instantaneously at the mid-point of the year ($t + 0.5$):
   $$N_{a,t} = N_{a+1,t+1} e^{M} + C_{a,t} e^{M/2}$$
   Instantaneous fishing mortality is solved directly:
   $$F_{a,t} = \log\left(\frac{N_{a,t}}{N_{a+1,t+1}}\right) - M$$
3. **Exact Gulland Virtual Population Analysis (1965)**:
   Relaxes the mid-year pulse assumption by inverting the continuous Baranov catch equation simultaneously with exponential cohort survival:
   $$N_{a+1,t+1} = N_{a,t} e^{-(F_{a,t} + M)}$$
   $$\frac{C_{a,t}}{N_{a+1,t+1}} = \frac{F_{a,t}}{F_{a,t} + M} \frac{1 - e^{-(F_{a,t} + M)}}{e^{-(F_{a,t} + M)}}$$
   $F_{a,t}$ is solved iteratively via Newton-Raphson root finding until convergence ($|f(F)| < 10^{-8}$).
4. **ADAPT Calibrated VPA (Gavaris 1988)**:
   In classical VPA, terminal fishing mortality $F_{\text{term}}$ must be arbitrarily specified. The ADAPT framework calibrates terminal abundances $N_{\text{term}}$ by minimizing log-residuals between model numbers and tuning survey indices $I_{a,t}$:
   $$I_{a,t} = q_a N_{a,t}^{\gamma_a} e^{\epsilon_{a,t}}$$
   $$\min_{N_{\text{term}}} \sum_{a,t} \left(\log(I_{a,t}) - \log(q_a N_{a,t})\right)^2$$
5. **Bayesian VPA**:
   Formulates the ADAPT calibration under a full probabilistic framework in `Turing.jl`, estimating posterior distributions for terminal abundances $N_{\text{term}}$, survey catchabilities $q_a$, and observation variance $\sigma_o^2$.

### Key References
- Gulland, J.A. (1965). Estimation of mortality rates. *Annex to Arctic Fisheries Working Group Report*. ICES C.M. 1965, Doc. No. 3.
- Pope, J.G. (1972). An investigation of the accuracy of virtual population analysis using cohort analysis. *ICNAF Res. Bull.*, 9: 65–74.
- Gavaris, S. (1988). An adaptive framework for the estimation of population size. *CAFSAC Res. Doc.* 88/29. 12 p.
- Megrey, B.A. (1989). Review and comparison of age-structured stock assessment models from theoretical and applied points of view. *Am. Fish. Soc. Symp.*, 6: 8–48.

## Method, assumptions, strengths, weaknesses

**Method.** Catch-at-age in numbers is inverted into abundance and mortality by three
procedures of increasing rigour, all in `src/cohort_analysis.jl`:

1. Pope's (1972) approximation: `N[a,t] = N[a+1,t+1] * exp(M) + C[a,t] * exp(M/2)`,
   with `F[a,t] = log(N[a,t]/N[a+1,t+1]) - M`. One backward sweep per year.
2. Gulland VPA: inverts the Baranov catch equation exactly by Newton-Raphson per cell.
3. ADAPT: calibrates terminal-year fishing mortality by minimising log-survey residuals,
   with catchability `q[a]` profiled analytically as
   `log q[a] = mean( log I[a,t] - log N[a,t] )` over the years where the survey is defined.

**Assumptions**
- `M` is known and constant (or supplied per age and year).
- Catch is known without error, and in numbers, not biomass.
- The plus group is selected as `plus_group_age`; its `F` is assumed constant across
  years unless `plus_group_f` is given.
- The backward recursion conditions on the terminal-year `F`, so terminal `F` propagates
  into every earlier year through the cohort.
- The survey index maps to the modelled abundance multiplicatively (`I = q N`), with
  log-scale errors.

**Weakness of the plus group.** Because the oldest class accumulates, small errors in the
terminal `F` are amplified into the plus group and then into adjacent cohorts.

**Strengths**
- Requires only catch-at-age and `M`, so it works on historical stocks with no survey.
- Conditional profiling of `q[a]` removes as many nuisance parameters as there are
  tuned ages, which is cheap and avoids the degeneracy between `q[a]` and `N[a,t]`.
- Pope's approximation and Gulland VPA bound the answer; divergence between them is a
  useful diagnostic for mid-year catch timing.
- `adapt_vpa` reports `converged`, `iterations` and `residual_variance`, so the
  calibration is auditable rather than a black box.

**Weaknesses**
- No stock-recruitment function and no feedback from biomass to recruitment, so the
  terminal-year estimate carries the projection.
- `M` is assumed exact; systematic bias in `M` maps one-to-one into bias in `F`.
- Without weight-at-age, `biomass_ssb` is total abundance and must not be read as SSB
  (see `REVIEW.md` §1.1 item 6); pass `weight_at_age` and `maturity` for a true SSB.
- With a weak survey the ADAPT objective is nearly flat and `F_term` is weakly identified.
"""

# ╔═╡ 30000001-0000-0000-0000-000000000003
md"""
## 1. Catch-at-Age Matrix & Terminal Inputs
"""

# ╔═╡ 30000001-0000-0000-0000-000000000004
begin
    # Baltic / Atlantic Gadoid catch numbers at age (ages 2 to 7 across 6 years)
    catches_matrix = [
        35000.0 38000.0 32000.0 30000.0 28000.0 25000.0;
        22000.0 24000.0 21000.0 19000.0 18000.0 16000.0;
        14000.0 15000.0 13000.0 12000.0 11000.0 10000.0;
         8000.0  8500.0  7500.0  7000.0  6500.0  6000.0;
         4000.0  4200.0  3800.0  3500.0  3200.0  3000.0;
         1500.0  1600.0  1400.0  1300.0  1200.0  1100.0
    ]
    
    ages = 2:7
    years = 1990:1995
    natural_mortality = 0.2
    terminal_f = [0.25, 0.35, 0.45, 0.50, 0.55, 0.55]
    
    (
        ages = "$(minimum(ages)) - $(maximum(ages))",
        years = "$(minimum(years)) - $(maximum(years))",
        total_catch_numbers = sum(catches_matrix)
    )
end

# ╔═╡ 30000001-0000-0000-0000-000000000005
md"""
## 2. Reconstruct Numbers-at-Age via Pope's Cohort Analysis
"""

# ╔═╡ 30000001-0000-0000-0000-000000000006
begin
    pope_res = popes_cohort_analysis(
        catches_matrix, natural_mortality; terminal_f = terminal_f
    )
    (
        initial_abundance_yr1 = round(sum(pope_res.N[:, 1]); digits = 0),
        terminal_abundance_yr6 = round(sum(pope_res.N[:, end]); digits = 0),
        mean_fishing_mortality = round.(vec(sum(pope_res.F, dims = 1)) ./ length(ages); digits = 3)
    )
end

# ╔═╡ 30000001-0000-0000-0000-000000000007
md"""
## 3. Reconstruct Exact Dynamics via Standard VPA (Newton-Raphson)
"""

# ╔═╡ 30000001-0000-0000-0000-000000000008
begin
    vpa_res = standard_vpa(
        catches_matrix, natural_mortality; terminal_f = terminal_f
    )
    (
        vpa_abundance_yr1 = round(sum(vpa_res.N[:, 1]); digits = 0),
        vpa_abundance_yr6 = round(sum(vpa_res.N[:, end]); digits = 0),
        pope_vs_vpa_max_diff = maximum(abs.(pope_res.N .- vpa_res.N))
    )
end

# ╔═╡ 30000001-0000-0000-0000-000000000009
md"""
## 4. ADAPT Calibrated VPA & Diagnostics

Calibrate terminal abundances using survey tuning index and inspect state trajectory diagnostics:
"""

# ╔═╡ 30000001-0000-0000-0000-000000000010
begin
    survey_tuning = catches_matrix .* 0.0012
    adapt_res = adapt_vpa(catches_matrix, survey_tuning, natural_mortality)
    p_adapt = plot_state_variables(adapt_res)
    (
        converged = adapt_res.converged,
        terminal_numbers = round.(adapt_res.terminal_n; digits = 1),
        residual_variance = round(adapt_res.residual_variance; digits = 5),
        plot_type = typeof(p_adapt)
    )
end

# ╔═╡ Cell order:
# ╠═30000001-0000-0000-0000-000000000001
# ╠═30000001-0000-0000-0000-000000000002
# ╠═30000001-0000-0000-0000-000000000003
# ╠═30000001-0000-0000-0000-000000000004
# ╠═30000001-0000-0000-0000-000000000005
# ╠═30000001-0000-0000-0000-000000000006
# ╠═30000001-0000-0000-0000-000000000007
# ╠═30000001-0000-0000-0000-000000000008
# ╠═30000001-0000-0000-0000-000000000009
# ╠═30000001-0000-0000-0000-000000000010
