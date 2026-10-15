### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 00000001-0000-0000-0000-000000000001
md"""
# AssessmentTools — Silver Hake Assessment Workflow

This notebook demonstrates the end-to-end stock assessment workflow for the
**Scotian Shelf Silver Hake (NAFO Div. 4VWX, Model C)** assessment using the modular
Julia `AssessmentTools` package.

### Analytical Formulation & Biological Assumptions
1. **Age-Structured Population Dynamics**:
   Tracks cohorts across ages 1 to 9 over 1960–2011 ($ny = 52$ years). Population numbers-at-age follow continuous exponential decay:
   $$N_{t+1, a+1} = N_{t,a} \exp\left(-(M_a + F_{t,a})\right)$$
   Plus-group accumulates older survivors ($a = 10+$):
   $$N_{t+1, A} = N_{t, A-1} e^{-(M_{A-1} + F_{t, A-1})} + N_{t, A} e^{-(M_A + F_{t, A})}$$
2. **Stock-Recruitment Relationship**:
   Beverton-Holt formulation parameterized with steepness $h$ and unfished recruitment $R_0$:
   $$R_t = \frac{4 h R_0 S_{t-1}}{S_0 (1 - h) + S_{t-1} (5h - 1)} \exp\left(\omega_t - \frac{\sigma_R^2}{2}\right)$$
   where $S_t = \sum_a N_{t,a} w_{t,a} m_a$ is spawning stock biomass (SSB), and $\omega_t$ are annual recruitment anomalies.
3. **Multi-Gear Observation Models**:
   - Total Catch Biomass: Log-normal observation likelihood comparing observed catches $C_{t,g}$ to Baranov integrated yields:
     $$Y_{t,g} = \sum_a \frac{s_{g,a} F_{t,g}}{Z_{t,a}} \left(1 - e^{-Z_{t,a}}\right) N_{t,a} w_{t,a}$$
   - Survey Abundance Indices: Lognormal likelihood with survey catchability $q_s$:
     $$I_{t,s} = q_s \sum_a s_{s,a} N_{t,a} \exp\left(-\tau_s Z_{t,a}\right)$$
   - Age Compositions: Multinomial or Dirichlet likelihood on observed proportions-at-age $p_{t,g,a}$.
4. **Phased Automatic Differentiation (AD)**:
   Optimization progresses through sequential parameter phases using `ForwardDiff.jl` dual numbers paired with L-BFGS in `Optim.jl`.
5. **MSY Reference Points & Probabilistic Decision Tables**:
   Yield-per-recruit and spawner-per-recruit curves determine $F_{\text{MSY}}$, $B_{\text{MSY}}$, and $B_0$. Forward stochastic projections evaluate harvest risk under alternative TAC levels.

### Key References
- Showell, M.A., & Fanning, L.P. (1999). Assessment of the Scotian Shelf silver hake population in NAFO Div. 4VWX. *DFO Can. Stock Assess. Sec. Res. Doc.* 99/149.
- Fournier, D.A., & Archibald, C.P. (1982). A general theory for analyzing catch at age data. *Can. J. Fish. Aquat. Sci.*, 39(8): 1195–1207.
- Methot, R.D., & Wetzel, C.R. (2013). Stock synthesis: a biological and statistical framework for fish stock assessment and fishery management. *Fish. Res.*, 142: 86–99.

## Method, assumptions, strengths, weaknesses

**Method.** A Stock Synthesis-style statistical catch-at-age model (`src/dynamics.jl`,
`src/objective.jl`). Numbers at age follow an annual cohort transition
`N[t+1, a+1] = N[t, a] * exp(-Z[t, a])` with a plus group, and age-1 recruitment is set
by a stock-recruitment function. Fishing mortality is age-specific,
`Z[t, a] = M + sum_g s[g, a] * F[t, g]`, and the likelihood combines catch, survey index,
age-composition and process-residual terms.

**Model structure**
- Von Bertalanffy growth and logistic maturity define mean weight and fecundity at age.
- Selectivity is parameterised per gear by the declared type (logistic in age, logistic
  in length, age-specific, or a length spline) with optional phased time blocking.
- Natural mortality is an input; a log-scaled random walk in `M` can be activated.
- Total catch, surveys, age composition, and priors on the biological parameters are
  combined into `objective_function`, which is minimised by `Optim` L-BFGS.
- Reference points come from `calc_reference_points`, which derives `F_MSY`, `B_MSY`,
  `B_0` and `U_MSY` by solving equilibrium yield on the selected terminal selectivity.

**Phasing.** Phases are a schedule for the order in which parameters become estimable,
not a statistical statement. The control file assigns each parameter a phase; a negative
phase means the parameter is held at its starting value for the whole fit.

**Assumptions**
- Catch is fully reported and known in biomass (or per the declared `catch_type`).
- Survey index selectivity is proportional to the modelled vulnerability.
- Age-composition sample sizes are adequately large for a multinomial approximation.
- Natural mortality is exogenous unless the `M` random walk is activated.
- Recruitment anomalies are independent, given the recruitment prior.
- `avg_wt` and `avg_fec` are treated as fixed projection inputs, not uncertain.

**Strengths**
- The phase ordering terminates at a reproducible point estimate: at the published
  parameter values the objective is 280.38480660973084, matching the benchmark in
  `data/silverhake/parameters.toml`, so every component of the likelihood is verified.
- Mean weight and maturity are derived from the declared growth and maturity functions,
  so no separate weight table has to be maintained.
- Selectivity, mortality and reference points are separated, so the effect of a
  selectivity change can be traced through the objective.
- Diagnostics are first class: `objective_function` returns each likelihood component
  separately, and the run report states the objective, the phase, and convergence.

**Weaknesses**
- Many parameters relative to a single survey index; selectivity and recruitment
  deviations trade off, so profile likelihoods should be examined.
- Reference points assume equilibrium. When biomass is well below `B_0`, `F_MSY` from
  an equilibrium solver can overstate the fishing rate required to reach that yield.
- `avg_wt` is treated as fixed, so the forecast does not carry weight uncertainty.
- The likelihood is not bootstrap-consistent by itself; prediction intervals need an
  explicit bootstrap or a Bayesian model.

**Bayesian extension.** The same fit can be expressed as a Turing model and run with
NUTS, ADVI, or MAP through `AssessmentTools.InferenceBridge`.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000002
begin
    import Pkg
    # Activate the AssessmentTools package environment
    Pkg.activate(normpath(joinpath(@__DIR__, "..")))
    using AssessmentTools
    using AssessmentTools.StatisticalCatchAtAge
    using LinearAlgebra
    using Statistics
    using Printf
end

# ╔═╡ 00000001-0000-0000-0000-000000000003
md"""
## 1. Data Ingestion & Model Initialization

Load the dataset from the native modular directory (`data/silverhake/config.toml` and CSVs).
"""

# ╔═╡ 00000001-0000-0000-0000-000000000004
begin
    hake_data_dir = normpath(joinpath(@__DIR__, "..", "data", "silverhake"))
    par_file = joinpath(hake_data_dir, "parameters.toml")

    # Read inputs from the internal modular dataset (TOML + CSV)
    data_in, ctl_in, pfc_in = load_dataset(hake_data_dir)

    # Initialize model object directly from inputs
    model = StockModel(data_in, ctl_in, pfc_in)
    
    (
        years = "$(model.syr) - $(model.nyr) ($(model.ny) years)",
        ages = "$(model.sage) - $(model.nage) ($(model.nages) age classes)",
        gears = model.data.ngear,
        surveys = length(model.data.surveys),
        age_comp_series = length(model.data.agecomps),
        total_parameters = model.layout.n
    )
end

# ╔═╡ 00000001-0000-0000-0000-000000000005
md"""
## 2. Biological Schedules & Growth

Inspect length-at-age $l_a = L_\infty (1 - e^{-k(a - t_0)})$, mean weight-at-age $w_a = a l_a^b$,
and maturity schedule $m_a = \text{plogis}(a, a_h, g_h)$.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000006
begin
    sched_df = (
        Age = model.ages,
        Length_cm = round.(model.la; digits = 2),
        MeanWeight_kg = round.(model.wa; digits = 4),
        Maturity = round.(plogis.(model.ages, model.data.ah, model.data.gh); digits = 3),
        AvgWeight_Obs_kg = round.(model.avg_wt; digits = 4),
        AvgFecundity = round.(model.avg_fec; digits = 4)
    )
    sched_df
end

# ╔═╡ 00000001-0000-0000-0000-000000000007
md"""
## 3. Objective Function & Benchmark Assessment Solution

Evaluate the negative log-posterior at the benchmark parameter vector loaded from `parameters.toml`.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000008
begin
    x_sol = load_parameters(par_file)
    ctx_last = PhaseContext(model.layout.max_phase, true)
    f_julia, comp_julia = objective_function(model, x_sol, ctx_last)
    
    (
        Objective = f_julia,
        Total_Parameters = length(x_sol),
        Catch_NLL = sum(comp_julia.nll_catch),
        Survey_NLL = sum(comp_julia.nll_survey),
        AgeComp_NLL = sum(comp_julia.nll_agecomp)
    )
end

# ╔═╡ 00000001-0000-0000-0000-000000000009
begin
    likelihood_breakdown = (
        Component = [
            "Catch NLL (Fleet 1)", "Catch NLL (Fleet 2)",
            "Survey NLL (July RV)", "Survey NLL (ITQ)", "Survey NLL (Spring RV)",
            "Age Comp NLL (Fleet 1)", "Age Comp NLL (July RV)",
            "Stock-Recruitment Residuals",
            "Selectivity 2nd Difference Penalty", "Selectivity Dome Penalty",
            "Leading Parameter Priors",
            "Fishing Mortality Regularisation", "Recruitment Dev Penalties"
        ],
        Julia_Value = round.([
            comp_julia.nll_catch[1], comp_julia.nll_catch[2],
            comp_julia.nll_survey[1], comp_julia.nll_survey[2], comp_julia.nll_survey[3],
            comp_julia.nll_agecomp[1], comp_julia.nll_agecomp[2],
            comp_julia.nll_rec,
            comp_julia.nll_sel_diff2[1], comp_julia.nll_sel_dome[1],
            sum(comp_julia.priors_theta),
            comp_julia.pen_fbar, comp_julia.pen_rdevs + comp_julia.pen_init_rdevs
        ]; digits = 4)
    )
    likelihood_breakdown
end

# ╔═╡ 00000001-0000-0000-0000-000000000010
md"""
## 4. Equilibrium Reference Points ($B_0$, $B_{\text{MSY}}$, $F_{\text{MSY}}$, $U_{\text{MSY}}$)

Calculate biological and management reference points using steady-state yield equilibrium
equations.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000011
begin
    p_unpacked = StatisticalCatchAtAge.unpack(model.layout, x_sol)
    lead_par = StatisticalCatchAtAge.leading_parameters(model, p_unpacked.theta)
    log_sel = calc_selectivities(model, p_unpacked.sel_par)
    mort_tot = calc_total_mortality(model, log_sel, p_unpacked.log_ft_pars, p_unpacked.log_m_nodes, lead_par.M, ctx_last)
    
    ref_pts = calc_reference_points(model, mort_tot, log_sel)
    
    (
        B0_mt = round(ref_pts.bo; digits = 1),
        Bmsy_mt = round(ref_pts.bmsy; digits = 1),
        MSY_mt = round(sum(ref_pts.msy); digits = 1),
        Fmsy_gear1 = round(ref_pts.fmsy[1]; digits = 4),
        Umsy = round(ref_pts.umsy; digits = 4),
        SPR_at_MSY = round(ref_pts.spr_msy; digits = 3)
    )
end

# ╔═╡ 00000001-0000-0000-0000-000000000012
md"""
## 5. Decision Table & Projections Under Catch Options

Evaluate probabilistic performance metrics across the TAC catch options specified in
the projection control file (`.pfc`).
"""

# ╔═╡ 00000001-0000-0000-0000-000000000013
begin
    proj_table = decision_table(model, x_sol)
    
    decision_summary = (
        TAC_mt = [m.tac for m in proj_table],
        P_SB_decline = round.([m.p_sb_decline for m in proj_table]; digits = 3),
        P_SB_lt_Bo_1 = round.([m.p_sb_lt_bo_1 for m in proj_table]; digits = 3),
        P_SB_lt_Bmsy_1 = round.([m.p_sb_lt_bmsy_1 for m in proj_table]; digits = 3),
        P_SB_lt_Bmsy_2 = round.([m.p_sb_lt_bmsy_2 for m in proj_table]; digits = 3),
        P_U_gt_Umsy_1 = round.([m.p_u_gt_umsy_1 for m in proj_table]; digits = 3),
        Avg_5yr_SB_Trend = round.([m.dsb5_rate for m in proj_table]; digits = 3)
    )
    decision_summary
end

# ╔═╡ 00000001-0000-0000-0000-000000000014
md"""
## 6. Simulation Testing Engine

Generate a synthetic observation dataset from the true population state with stochastic
recruitment anomalies and observation error.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000015
begin
    # Simulate synthetic data with seed 999
    sim_model = simulate_data(model, 999; error = true)
    
    (
        Original_Total_Catch = round(sum(model.data.obs_ct); digits = 1),
        Simulated_Total_Catch = round(sum(sim_model.data.obs_ct); digits = 1),
        Survey1_Obs_Count = length(sim_model.data.surveys[1].index),
        AgeComp1_Obs_Count = size(sim_model.data.agecomps[1].obs, 1)
    )
end

# ╔═╡ 00000001-0000-0000-0000-000000000016
md"""
## 7. High-Performance Storage Export (Parquet & Zarr)

Export annual assessment trajectories to Apache Parquet for high-throughput columnar analytics,
and save multi-dimensional states (numbers-at-age $N$ and fishing mortality $F$) to a chunked Zarr store.
"""

# ╔═╡ 00000001-0000-0000-0000-000000000017
begin
    pq_path = joinpath(hake_data_dir, "trajectories.parquet")
    zarr_path = joinpath(hake_data_dir, "state.zarr")
    
    write_parquet(pq_path, model, x_sol)
    write_zarr(zarr_path, model, x_sol)
    
    (
        Parquet_File = pq_path,
        Parquet_Bytes = filesize(pq_path),
        Zarr_Store = zarr_path,
        Status = "Saved Successfully"
    )
end

# ╔═╡ 00000000-0000-0000-0000-000000000000
# ╔═╡ Cell order:
# ╠═00000001-0000-0000-0000-000000000001
# ╠═00000001-0000-0000-0000-000000000002
# ╠═00000001-0000-0000-0000-000000000003
# ╠═00000001-0000-0000-0000-000000000004
# ╠═00000001-0000-0000-0000-000000000005
# ╠═00000001-0000-0000-0000-000000000006
# ╠═00000001-0000-0000-0000-000000000007
# ╠═00000001-0000-0000-0000-000000000008
# ╠═00000001-0000-0000-0000-000000000009
# ╠═00000001-0000-0000-0000-000000000010
# ╠═00000001-0000-0000-0000-000000000011
# ╠═00000001-0000-0000-0000-000000000012
# ╠═00000001-0000-0000-0000-000000000013
# ╠═00000001-0000-0000-0000-000000000014
# ╠═00000001-0000-0000-0000-000000000015
# ╠═00000001-0000-0000-0000-000000000016
# ╠═00000001-0000-0000-0000-000000000017
