### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 10000001-0000-0000-0000-000000000001
md"""
# AssessmentTools — Cusk Assessment Workflow

This notebook demonstrates the complete stock assessment workflow for the
**Scotian Shelf Cusk (*Brosme brosme*, 4VWX5Z)** benchmark assessment using the modular
Julia `AssessmentTools` package.

### Assessment Characteristics & Analytical Assumptions
1. **Long-lived Demersal Gadoid Dynamics**:
   Model tracks cohorts across ages 1 to 30+ over 1970–2010 ($ny = 41$ years) under exponential cohort decay:
   $$N_{t+1, a+1} = N_{t,a} \exp\left(-(M_a + F_{t,a})\right)$$
   Plus-group abundance ($A = 30+$) accumulates surviving older fish:
   $$N_{t+1, A} = N_{t, A-1} e^{-(M_{A-1} + F_{t, A-1})} + N_{t, A} e^{-(M_A + F_{t, A})}$$
2. **Multi-Fleet Removals & Continuous Baranov Fishing**:
   Instantaneous fishing mortality is separable across gears: $F_{t,a} = \sum_g s_{g,a} F_{t,g}$. Catches are integrated via Baranov:
   $$C_{t,g,a} = \frac{s_{g,a} F_{t,g}}{Z_{t,a}} \left(1 - e^{-Z_{t,a}}\right) N_{t,a}$$
3. **Time-Varying Survey Catchability ($q_t$)**:
   To capture shifting research vessel and commercial fleet catch efficiencies over 4 decades, survey catchability follows a geometric random walk prior ($\text{q\_prior} = 2$):
   $$\log(q_{t+1, s}) = \log(q_{t, s}) + \eta_{t, s}, \quad \eta_{t,s} \sim \mathcal{N}(0, \sigma_{q,s}^2)$$
4. **Multi-Index Relative Abundance**:
   Calibrated simultaneously to 3 independent abundance series:
   - Commercial CPUE index (1986–2007)
   - Halibut Industry Longline Survey index (1998–2007)
   - DFO Summer Research Vessel (RV) Survey index (1970–2010)
5. **Equilibrium MSY Reference Points**:
   $B_0$, $B_{\text{MSY}}$, $F_{\text{MSY}}$, and $U_{\text{MSY}}$ are solved via exact Newton-Raphson root finding on the yield-per-recruit and spawner-recruit relationship.

### Key References
- Davies, T.D., & Jonsen, I.D. (2011). Identifying cryptic population structure and evaluating recovery potential for Cusk (*Brosme brosme*). *DFO Can. Sci. Advis. Sec. Res. Doc.* 2011/049.
- Fournier, D.A., Hampton, J., & Sibert, J.R. (1998). MULTIFAN-CL: a length-based, age-structured model for fisheries stock assessment. *Can. J. Fish. Aquat. Sci.*, 55(9): 2105–2116.
- Methot, R.D., & Wetzel, C.R. (2013). Stock synthesis: a biological and statistical framework for fish stock assessment and fishery management. *Fish. Res.*, 142: 86–99.
"""

# ╔═╡ 10000001-0000-0000-0000-000000000002
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

# ╔═╡ 10000001-0000-0000-0000-000000000003
md"""
## 1. Data Ingestion & Model Construction

Load the raw assessment files (`Cusk.dat`, `Cusk.ctl`, and `Cusk.par`) from `examples/Cusk/`.
"""

# ╔═╡ 10000001-0000-0000-0000-000000000004
begin
    cusk_dir = normpath(joinpath(@__DIR__, "..", "data", "cusk"))
    par_path = joinpath(cusk_dir, "parameters.toml")

    # Read data and control specifications from the internal modular dataset (TOML + CSV)
    data_cusk, ctl_cusk, _ = load_dataset(cusk_dir)

    # Initialize model with automatic projection control covering assessment years
    model_cusk = StockModel(data_cusk, ctl_cusk)

    (
        Species = "Cusk (Brosme brosme)",
        Years = "$(model_cusk.syr) - $(model_cusk.nyr) ($(model_cusk.ny) years)",
        Ages = "$(model_cusk.sage) - $(model_cusk.nage) ($(model_cusk.nages) age classes)",
        Gears = model_cusk.data.ngear,
        Surveys = length(model_cusk.data.surveys),
        Survey_Observations = [length(s.index) for s in model_cusk.data.surveys],
        Total_Parameters = model_cusk.layout.n
    )
end

# ╔═╡ 10000001-0000-0000-0000-000000000005
md"""
## 2. Biological Schedules & Life History

Examine the age schedules: length $l_a = L_\infty (1 - e^{-k(a - t_0)})$, mean weight
$w_a = a l_a^b$, and logistic maturity $m_a = \text{plogis}(a, a_h, g_h)$ with $a_h = 8.0$ and $g_h = 0.2$.
"""

# ╔═╡ 10000001-0000-0000-0000-000000000006
begin
    mat_sched = plogis.(model_cusk.ages, model_cusk.data.ah, model_cusk.data.gh)
    
    # Representative age subset for display
    display_ages = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30]
    idx = [findfirst(==(a), model_cusk.ages) for a in display_ages]

    (
        Age = display_ages,
        Length_cm = round.(model_cusk.la[idx]; digits = 2),
        Weight_kg = round.(model_cusk.wa[idx]; digits = 3),
        Maturity = round.(mat_sched[idx]; digits = 3),
        Fecundity = round.(model_cusk.avg_fec[idx]; digits = 3)
    )
end

# ╔═╡ 10000001-0000-0000-0000-000000000007
md"""
## 3. Parameter Layout & Assessment Solution Vector

Load the converged maximum likelihood estimates from `parameters.toml` and unpack the parameter vector
into the leading parameters $\theta$, gear selectivities, annual fishing mortalities $F_t$, and recruitment deviations.
"""

# ╔═╡ 10000001-0000-0000-0000-000000000008
begin
    x_sol_cusk = load_parameters(par_path)
    p_unpacked = StatisticalCatchAtAge.unpack(model_cusk.layout, x_sol_cusk)
    lead_cusk = StatisticalCatchAtAge.leading_parameters(model_cusk, p_unpacked.theta)

    (
        R0 = round(lead_cusk.ro; digits = 3),
        Steepness_h = round(lead_cusk.h; digits = 3),
        NaturalMortality_M = round(lead_cusk.M; digits = 3),
        Compensation_kappa = round(lead_cusk.kappa; digits = 3),
        Log_AvgRec = round(lead_cusk.log_avgrec; digits = 3),
        Log_RecInit = round(lead_cusk.log_recinit; digits = 3),
        Observation_rho = round(lead_cusk.rho; digits = 3),
        Precision_varphi = round(lead_cusk.varphi; digits = 3),
        Derived_sig = round(lead_cusk.sig; digits = 4),
        Derived_tau = round(lead_cusk.tau; digits = 4)
    )
end

# ╔═╡ 10000001-0000-0000-0000-000000000009
md"""
## 4. Equilibrium Reference Points & Benchmark Comparison

Compute the reference points ($B_0$, $B_{\text{MSY}}$, $\text{MSY}$, $F_{\text{MSY}}$, $U_{\text{MSY}}$)
using the Baranov equilibrium solver and verify exact agreement with `Cusk.rep`.
"""

# ╔═╡ 10000001-0000-0000-0000-000000000010
begin
    import TOML
    ctx_eval = PhaseContext(model_cusk.layout.max_phase, true)
    log_sel_cusk = calc_selectivities(model_cusk, p_unpacked.sel_par)
    mort_cusk = calc_total_mortality(model_cusk, log_sel_cusk, p_unpacked.log_ft_pars,
                                     p_unpacked.log_m_nodes, lead_cusk.M, ctx_eval)

    # Compute MSY reference points evaluated at the estimated parameters
    ref_cusk = calc_reference_points(model_cusk, mort_cusk, log_sel_cusk, p_unpacked.theta)
    
    # Read reference benchmarks from parameters.toml
    bench = TOML.parsefile(par_path)["reference_points"]
    bench_bo = bench["bo"]
    bench_bmsy = bench["bmsy"]
    bench_msy = bench["msy"]
    bench_fmsy = bench["fmsy"]

    (
        Metric = ["B0 (kt)", "Bmsy (kt)", "MSY (kt)", "Fmsy", "Umsy (Julia)"],
        AssessmentTools = [
            round(ref_cusk.bo; digits = 4),
            round(ref_cusk.bmsy; digits = 4),
            round(sum(ref_cusk.msy); digits = 4),
            round(ref_cusk.fmsy[1]; digits = 6),
            round(ref_cusk.umsy; digits = 6)
        ],
        Benchmark_Target = [
            round(bench_bo; digits = 4),
            round(bench_bmsy; digits = 4),
            round(bench_msy; digits = 4),
            round(bench_fmsy; digits = 6),
            NaN
        ],
        Absolute_Diff = [
            abs(ref_cusk.bo - bench_bo),
            abs(ref_cusk.bmsy - bench_bmsy),
            abs(sum(ref_cusk.msy) - bench_msy),
            abs(ref_cusk.fmsy[1] - bench_fmsy),
            NaN
        ]
    )
end

# ╔═╡ 10000001-0000-0000-0000-000000000011
md"""
## 5. Population Trajectory & Time-Series Reconstruction

Reconstruct the annual numbers-at-age, spawning stock biomass ($B_t$), recruitment ($R_t$),
commercial fishing mortality ($F_t$), and harvest rate ($U_t$).
"""

# ╔═╡ 10000001-0000-0000-0000-000000000012
begin
    pop_cusk = calc_numbers_at_age(model_cusk, lead_cusk, mort_cusk,
                                   p_unpacked.log_rec_devs, p_unpacked.init_log_rec_devs)
    fish_obs = calc_fishery_observations(model_cusk, log_sel_cusk, mort_cusk, pop_cusk.N)
    surv_obs = calc_survey_observations(model_cusk, log_sel_cusk, mort_cusk, pop_cusk.N)
    sr_cusk = calc_stock_recruitment(model_cusk, lead_cusk, mort_cusk, pop_cusk.N, pop_cusk.log_rt)

    # Decadal and modern summary of stock status
    sample_years = [1970, 1975, 1980, 1985, 1990, 1995, 2000, 2005, 2010]
    s_idx = [y - model_cusk.syr + 1 for y in sample_years]

    (
        Year = sample_years,
        SpawningBiomass_kt = round.(sr_cusk.sbt[s_idx]; digits = 2),
        Recruits_millions = round.(pop_cusk.N[s_idx, 1]; digits = 2),
        FishingMortality_F = round.(mort_cusk.ft[1, s_idx]; digits = 3),
        Catch_kt = round.(fish_obs.ct[1, s_idx]; digits = 3),
        HarvestRate_U = round.(fish_obs.ct[1, s_idx] ./ (pop_cusk.N[s_idx, :] * model_cusk.wa); digits = 3),
        Depletion_B_over_B0 = round.(sr_cusk.sbt[s_idx] ./ ref_cusk.bo; digits = 3)
    )
end

# ╔═╡ 10000001-0000-0000-0000-000000000013
md"""
## 6. Survey Index Fits & Dynamic Catchability ($q_t$)

Inspect the survey fits under the random-walk catchability formulation ($q_{\text{prior}} = 2$).
"""

# ╔═╡ 10000001-0000-0000-0000-000000000014
begin
    survey_summary = [
        (
            Survey = "Commercial CPUE (Gear 1)",
            Years = "$(minimum(model_cusk.data.surveys[1].year)) - $(maximum(model_cusk.data.surveys[1].year))",
            Observations = length(model_cusk.data.surveys[1].index),
            Mean_Catchability_q = round(mean(surv_obs.qt[1]); digits = 3),
            RMS_Residual = round(sqrt(mean(abs2, surv_obs.epsilon[1])); digits = 4)
        ),
        (
            Survey = "Halibut Longline Survey (Gear 2)",
            Years = "$(minimum(model_cusk.data.surveys[2].year)) - $(maximum(model_cusk.data.surveys[2].year))",
            Observations = length(model_cusk.data.surveys[2].index),
            Mean_Catchability_q = round(mean(surv_obs.qt[2]); digits = 3),
            RMS_Residual = round(sqrt(mean(abs2, surv_obs.epsilon[2])); digits = 4)
        ),
        (
            Survey = "Summer RV Survey (Gear 3)",
            Years = "$(minimum(model_cusk.data.surveys[3].year)) - $(maximum(model_cusk.data.surveys[3].year))",
            Observations = length(model_cusk.data.surveys[3].index),
            Mean_Catchability_q = round(mean(surv_obs.qt[3]); digits = 3),
            RMS_Residual = round(sqrt(mean(abs2, surv_obs.epsilon[3])); digits = 4)
        )
    ]
    survey_summary
end

# ╔═╡ 10000001-0000-0000-0000-000000000015
md"""
## 7. High-Performance Storage Export (Parquet & Zarr)

Export annual trajectories and multi-dimensional population arrays directly into
Apache Parquet and chunked Zarr array storage formats.
"""

# ╔═╡ 10000001-0000-0000-0000-000000000016
begin
    pq_cusk = joinpath(cusk_dir, "trajectories.parquet")
    zarr_cusk = joinpath(cusk_dir, "state.zarr")
    
    write_parquet(pq_cusk, model_cusk, x_sol_cusk)
    write_zarr(zarr_cusk, model_cusk, x_sol_cusk)
    
    (
        Parquet_File = pq_cusk,
        Parquet_Bytes = filesize(pq_cusk),
        Zarr_Store = zarr_cusk,
        Status = "Saved Successfully"
    )
end

# ╔═╡ 00000000-0000-0000-0000-000000000000
# ╔═╡ Cell order:
# ╠═10000001-0000-0000-0000-000000000001
# ╠═10000001-0000-0000-0000-000000000002
# ╠═10000001-0000-0000-0000-000000000003
# ╠═10000001-0000-0000-0000-000000000004
# ╠═10000001-0000-0000-0000-000000000005
# ╠═10000001-0000-0000-0000-000000000006
# ╠═10000001-0000-0000-0000-000000000007
# ╠═10000001-0000-0000-0000-000000000008
# ╠═10000001-0000-0000-0000-000000000009
# ╠═10000001-0000-0000-0000-000000000010
# ╠═10000001-0000-0000-0000-000000000011
# ╠═10000001-0000-0000-0000-000000000012
# ╠═10000001-0000-0000-0000-000000000013
# ╠═10000001-0000-0000-0000-000000000014
# ╠═10000001-0000-0000-0000-000000000015
# ╠═10000001-0000-0000-0000-000000000016
