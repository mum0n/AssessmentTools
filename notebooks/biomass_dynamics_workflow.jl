### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 20000001-0000-0000-0000-000000000001
md"""
# Biomass Dynamics & Discrete Logistic Surplus Production Models

This interactive workflow illustrates surplus production modeling using the **`AssessmentTools.BiomassDynamics`** submodule.

### Conceptual Background & Analytical Assumptions
1. **Aggregated Biomass Dynamics**:
   Abstracts complex age- and size-structured demographic processes (growth, maturation, mortality, gear selectivity) into a single aggregate population biomass state variable $B_t$ (Schaefer 1954; Hilborn & Walters 1992).
2. **Discrete Schaefer Logistic Production**:
   Governed by the intrinsic population growth rate $r$ and environmental carrying capacity $K$:
   $$\Delta B_t = r B_t \left(1 - \frac{B_t}{K}\right) - C_t$$
   $$B_{t+1} = \max\left(0, B_t + \Delta B_t\right)$$
   Assumes compensatory density dependence where net surplus production is a symmetric quadratic function of stock biomass maximized at $B_{\text{MSY}} = K/2$.
3. **Biological Reference Points**:
   Classical analytical reference points derived at equilibrium ($\Delta B_t = 0$):
   $$B_{\text{MSY}} = \frac{K}{2}, \quad \text{MSY} = \frac{r K}{4}, \quad F_{\text{MSY}} = \frac{r}{2}, \quad U_{\text{MSY}} = 1 - e^{-F_{\text{MSY}}}$$
4. **Seasonal Survey Timing & Observation Models**:
   Surveys do not always occur at the start of the management year. The model incorporates intra-annual timing fractions $\tau \in [0, 1]$ (e.g. spring pre-fishery vs fall post-fishery) and catchability $q$:
   $$I_t = q \left( B_t - \tau C_t \right)$$
5. **Bayesian State-Space Formulation**:
   Separates process error (environmental stochasticity in biological production) from observation error in survey indices:
   $$\log(B_{t+1}) \sim \mathcal{N}\left(\log\left(B_t + r B_t(1 - B_t/K) - C_t\right), \sigma_p^2\right)$$
   $$\log(I_t) \sim \mathcal{N}\left(\log(q B_t), \sigma_o^2\right)$$

### Key References
- Schaefer, M.B. (1954). Some aspects of the dynamics of populations important to the management of the commercial marine fisheries. *Bull. Inter-Am. Trop. Tuna Comm.*, 1: 27–56.
- Hilborn, R., & Walters, C.J. (1992). *Quantitative Fisheries Stock Assessment: Choice, Dynamics and Uncertainty*. Chapman and Hall, New York.
- Polacheck, T., Hilborn, R., & Punt, A.E. (1993). Fitting surplus production models: comparing methods and measuring uncertainty. *Can. J. Fish. Aquat. Sci.*, 50(12): 2597–2607.
"""

# ╔═╡ 20000001-0000-0000-0000-000000000002
begin
    import Pkg
    Pkg.activate(normpath(joinpath(@__DIR__, "..")))
    using AssessmentTools
    using AssessmentTools.BiomassDynamics
    using Parquet2
    using DataFrames
    using UnicodePlots
end

# ╔═╡ 20000001-0000-0000-0000-000000000003
md"""
## 1. Load Ingested Snow Crab Biomass Data (Parquet)

Load the ingested annual survey biomass and uncertainty series for Scotian Shelf snow crab areas:
- `cfaall`: Scotian Shelf total
- `cfanorth`: N-ENS (North Eastern Nova Scotia)
- `cfasouth`: S-ENS (South Eastern Nova Scotia)
- `cfa4x`: 4X area
"""

# ╔═╡ 20000001-0000-0000-0000-000000000004
begin
    snowcrab_bio_pq = normpath(joinpath(@__DIR__, "..", "data", "snowcrab", "biodyn_biomass.parquet"))
    bio_table = Parquet2.readfile(snowcrab_bio_pq)
    df_snowcrab = DataFrame(bio_table)
    
    (
        years = "$(minimum(df_snowcrab.yrs)) - $(maximum(df_snowcrab.yrs))",
        records = nrow(df_snowcrab),
        columns = names(df_snowcrab)[1:8]
    )
end

# ╔═╡ 20000001-0000-0000-0000-000000000005
md"""
## 2. Simulate Discrete Logistic Surplus Production

Simulate forward biomass trajectories under a Schaefer discrete logistic model given annual catch removals:
"""

# ╔═╡ 20000001-0000-0000-0000-000000000006
begin
    sim_years = collect(Int.(df_snowcrab.yrs))
    # Synthetic annual removals based on historical harvest patterns
    sim_removals = fill(8.5, length(sim_years))
    
    r_param = 0.35
    k_param = 65.0
    
    sim_result = simulate_logistic_discrete(
        sim_years, sim_removals;
        r = r_param, k = k_param, b0 = 55.0, q = 1.0, survey_transition = 6
    )
    
    (
        years = sim_result.years,
        terminal_biomass = round(sim_result.biomass[end]; digits = 2),
        mean_harvest_rate = round(sum(sim_result.harvest_rate) / length(sim_result.harvest_rate); digits = 4)
    )
end

# ╔═╡ 20000001-0000-0000-0000-000000000007
md"""
## 3. Schaefer Equilibrium Reference Points

Compute analytical MSY reference points:
"""

# ╔═╡ 20000001-0000-0000-0000-000000000008
begin
    msy_ref = logistic_msy_reference_points(r_param, k_param)
    (
        B_msy = msy_ref.bmsy,
        MSY = msy_ref.msy,
        F_msy = msy_ref.fmsy,
        U_msy = round(msy_ref.umsy; digits = 4)
    )
end

# ╔═╡ 20000001-0000-0000-0000-000000000009
md"""
## 4. Visualizations: Trajectories and Fishery Footprint

Evaluate state variable dynamics and the fishery depletion footprint:
$$\text{Footprint}_t = \frac{B_t^{\text{unfished}} - B_t^{\text{fished}}}{B_t^{\text{unfished}}}$$
"""

# ╔═╡ 20000001-0000-0000-0000-000000000010
begin
    # Generate counterfactual unfished biomass trajectory
    unfished_removals = zeros(Float64, length(sim_years))
    unfished_result = simulate_logistic_discrete(
        sim_years, unfished_removals;
        r = r_param, k = k_param, b0 = 55.0, q = 1.0, survey_transition = 6
    )

    p_state = plot_state_variables(sim_result)
    p_footprint = plot_fishery_footprint(sim_years, sim_result.biomass, unfished_result.biomass)
    (
        state_plot_type = typeof(p_state),
        footprint_plot_type = typeof(p_footprint)
    )
end

# ╔═╡ Cell order:
# ╠═20000001-0000-0000-0000-000000000001
# ╠═20000001-0000-0000-0000-000000000002
# ╠═20000001-0000-0000-0000-000000000003
# ╠═20000001-0000-0000-0000-000000000004
# ╠═20000001-0000-0000-0000-000000000005
# ╠═20000001-0000-0000-0000-000000000006
# ╠═20000001-0000-0000-0000-000000000007
# ╠═20000001-0000-0000-0000-000000000008
# ╠═20000001-0000-0000-0000-000000000009
# ╠═20000001-0000-0000-0000-000000000010
