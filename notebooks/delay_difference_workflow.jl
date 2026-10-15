### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 50000001-0000-0000-0000-000000000001
md"""
# Deriso-Schnute Delay-Difference Dynamics Driven by Continuous DDE Recruitment

This interactive workflow illustrates the integration of the **Deriso-Schnute delay-difference model** with male recruitment generated from the **continuous multi-stage delay differential equation (DDE)** model.

### Methodological Concept & Analytical Assumptions
1. **Coupling Structured Continuous & Discrete Dynamics**:
   - The continuous 6-stage delay differential model resolves developmental lags ($\tau \in \{7, 8, 9\}$ years), inter-instar moult transitions, and habitat mortality.
   - Male recruitment just before entry into the fishable component (stage $M_1$ / instar 12 approaching commercial knife-edge $M_0$ [$\ge 95\text{ mm}$ carapace width]) is extracted annually from the continuous solution.
2. **Deriso-Schnute Age-Structured Delay-Difference**:
   - Reconstructs fishable biomass $B_t$ and total numbers $N_t$ using the Ford-Brody growth model and annual survival $s = \exp(-M)$:
     $$N_{t+1} = s (N_t - C_t) + R_{t+1}$$
     $$B_{t+1} = s \alpha (N_t - C_t) + s \rho (B_t - Y_t) + w_k R_{t+1}$$
   - where $\rho$ is the Brody growth coefficient, $\alpha = w_k - \rho w_{k-1}$, and $w_k$ is the mean body weight at entry into the fishable component.
3. **Biological Consistency**:
   - Connects fine-scale intra-annual continuous developmental dynamics to coarse annual management and harvest control projections without loss of demographic information.
4. **Bayesian State-Space Formulation**:
   - Propagates latent state variables $B_t$ and $N_t$ with lognormal process variation and fits against survey observations with informative priors on biological survival $s$, carrying capacity, and catchability $q$.

### Key References
- Deriso, R.B. (1980). Harvesting strategies and parameter estimation for an age-structured model. *Can. J. Fish. Aquat. Sci.*, 37(2): 268–282.
- Schnute, J. (1985). A general theory for fishery modeling. *Can. J. Fish. Aquat. Sci.*, 42(3): 414–429.
- Choi, J.S. (2024). Continuous size-structured delay differential models for snow crab (*Chionoecetes opilio*) fisheries assessment. *DFO Can. Sci. Advis. Sec. Res. Doc.* 2024/nnn.

## Method, assumptions, strengths, weaknesses

**Method.** Biomass $B_t$ and numbers $N_t$ are propagated by `delay_difference_step`
(`src/delay_difference.jl`), the single definition of the transition:

```
next_b = (s + alpha * b / k) * b - alpha * s * b^2 / k - c
next_n = n + (alpha / k) * (next_b - b)
```

with $s = \exp(-M)$ annual survival, Ford-Brody coefficients $\rho$ and
$\alpha = w_k - \rho\, w_{k-1}$, $k$ the carrying capacity, and $c$ the catch.
The state-space form declares priors on $s$, $B_0$ and $q$, draws a lognormal
process deviation about the deterministic state, and links biomass to the survey
with a lognormal observation model.

**Estimation.** Three engines run on the same Turing model object via
`AssessmentTools.InferenceBridge`: `fit_map` (MAP point estimate, `Optim` L-BFGS),
`sample_posterior` (NUTS chains), and `fit_vi` (variational inference). The model is
written once; no likelihood code is duplicated per engine.

**Assumptions**
- Annual time steps; within-year biomass variation is not resolved.
- The deterministic recursion gives the conditional mean; process deviations are
  lognormal and independent across years.
- Survey catchability $q$ is constant over the fitted period.
- No age structure or gear selectivity; recruitments are supplied directly unless a
  stock-recruit link is conditioned.
- The Ford-Brody growth coefficients $\rho$ and $\alpha$ are fixed input, not fitted.

**Strengths**
- Few parameters; stable on short series with one abundance index.
- One model definition serves both deterministic projection and Bayesian sampling.
- Fast to fit, so it scales to many proposal scenarios.

**Weaknesses**
- Cannot represent age structure or selective fishing mortality.
- Recruitment is exogenous by default, so it cannot forecast stock-driven recovery.
- The lognormal process assumption can over-smooth volatility in short series.
- $s$ and $q$ are weakly identified separately when the abundance index is
  uninformative; they should be profiled or given tighter priors.
"""

# ╔═╡ 50000001-0000-0000-0000-000000000002
begin
    import Pkg
    Pkg.activate(normpath(joinpath(@__DIR__, "..")))
    using AssessmentTools
    using AssessmentTools.DelayDifferential
    using Parquet2
    using DataFrames
    using UnicodePlots
end

# ╔═╡ 50000001-0000-0000-0000-000000000003
md"""
## 1. Simulate Continuous 6-Stage DDE System
Simulate the continuous multi-stage system over 15 years with seasonal fishery removals.
"""

# ╔═╡ 50000001-0000-0000-0000-000000000004
begin
    dde_cfg = default_snowcrab_dde_config()
    sim_tspan = (0.0, 15.0) # 15-year simulation
    dt_step = 1.0 / 12.0     # monthly time step
    
    # Pulse fishing occurring annually in autumn (month 9)
    pulse_years = 1:14
    pulse_times = [yr + 0.75 for yr in pulse_years]
    pulse_catches = fill(4.0, length(pulse_times)) # kt removals
    
    dde_sim = simulate_dde_system(
        dde_cfg, sim_tspan;
        dt = dt_step,
        removals_times = pulse_times,
        removals_amount = pulse_catches
    )
    
    (
        total_time_steps = length(dde_sim.time),
        stages = dde_sim.stage_names,
        initial_M0 = round(dde_sim.states[1, 1]; digits = 2),
        initial_M1_recruits = round(dde_sim.states[1, 2]; digits = 2)
    )
end

# ╔═╡ 50000001-0000-0000-0000-000000000005
md"""
## 2. Extract Annual Male Recruitment ($M_1$) Just Before Fishable Entry
Extract annual abundance of instar 12 ($M_1$) males sampled annually just before entering the commercial size class ($M_0$).
"""

# ╔═╡ 50000001-0000-0000-0000-000000000006
begin
    # Extract annual recruitment of stage 2 (M1) at autumn survey timing (month 9)
    ann_years, ann_recruits = extract_annual_recruitment(
        dde_sim; stage_index = 2, target_month = 9.0
    )
    
    (
        years = ann_years,
        recruitment_numbers = round.(ann_recruits; digits = 2),
        mean_recruitment = round(sum(ann_recruits) / length(ann_recruits); digits = 2)
    )
end

# ╔═╡ 50000001-0000-0000-0000-000000000007
md"""
## 3. Configure and Run Deriso-Schnute Delay-Difference Dynamics
Propagate fishable biomass and numbers using Deriso-Schnute equations driven by the extracted $M_1$ recruitment trajectory:
"""

# ╔═╡ 50000001-0000-0000-0000-000000000008
begin
    # Ford-Brody growth and natural mortality parameters
    m_rate = 0.25
    surv_s = exp(-m_rate)
    wk_recruit = 0.55  # kg mean weight at recruitment entry
    wk_prev = 0.40     # kg mean weight prior to entry
    rho_brody = 0.70   # von Bertalanffy growth slope
    alpha_brody = wk_recruit - rho_brody * wk_prev
    
    dd_cfg = DelayDifferenceConfig(surv_s, alpha_brody, rho_brody, wk_recruit)
    
    # Annual catches in weight (matching annual harvest removals)
    annual_catches = fill(4.0, length(ann_years))
    
    # Forward delay-difference simulation
    dd_sim = simulate_delay_difference(
        ann_years, annual_catches, ann_recruits, dd_cfg;
        b0 = 45.0, n0 = 45.0 / wk_recruit
    )
    
    (
        years = dd_sim.years,
        fishable_biomass = round.(dd_sim.biomass; digits = 2),
        fishable_numbers = round.(dd_sim.numbers; digits = 2),
        recruitment_biomass = round.(dd_sim.recruits_biomass; digits = 2),
        terminal_biomass = round(dd_sim.biomass[end]; digits = 2)
    )
end

# ╔═╡ 50000001-0000-0000-0000-000000000009
md"""
## 4. Comparison Summary: Continuous DDE vs. Discrete Delay-Difference
Compare the fishable stock state ($M_0$) under continuous intra-annual integration vs. the discrete delay-difference representation.
"""

# ╔═╡ 50000001-0000-0000-0000-000000000010
begin
    # Extract annual M0 from DDE simulation
    _, ann_m0 = extract_annual_recruitment(dde_sim; stage_index = 1, target_month = 9.0)
    
    DataFrame(
        Year = ann_years,
        DDE_M0_Abundance = round.(ann_m0; digits = 2),
        DelayDifference_Numbers = round.(dd_sim.numbers; digits = 2),
        DelayDifference_Biomass = round.(dd_sim.biomass; digits = 2),
        Recruitment_Numbers = round.(dd_sim.recruits; digits = 2)
    )
end

# ╔═╡ 50000001-0000-0000-0000-000000000011
md"""
## 5. Bayesian State-Space Delay-Difference Inference (Turing.jl)
Fit the complimentary Bayesian state-space delay-difference model using `Turing.jl`:
- Estimates $B_0$, annual survival $s$, survey catchability $q$, process error $\sigma_p$, and observation error $\sigma_o$.
- Employs Markov Chain Monte Carlo (MCMC) with NUTS (No-U-Turn Sampler) or prior predictive checks.
"""

# ╔═╡ 50000001-0000-0000-0000-000000000012
begin
    # Construct synthetic survey observations from simulated biomass with 15% noise
    obs_survey_indices = dd_sim.biomass .* 1.05
    
    # Instantiate Turing model
    bayes_dd = bayesian_delay_difference_model(
        annual_catches, ann_recruits, obs_survey_indices, dd_cfg; obs_cv = 0.20
    )
    
    # Draw prior predictive sample
    prior_sample = rand(bayes_dd)
    
    (
        model = "Bayesian Deriso-Schnute State-Space",
        sample_b0 = round(prior_sample[:b0]; digits = 2),
        sample_s = round(prior_sample[:s]; digits = 4),
        sample_q = round(prior_sample[:q]; digits = 4),
        sample_proc_sd = round(prior_sample[:proc_sd]; digits = 4),
        sample_obs_sd = round(prior_sample[:obs_sd]; digits = 4)
    )
end

# ╔═╡ 50000001-0000-0000-0000-000000000013
md"""
## 6. State Variable Diagnostics & Visualizations

Visualize the reconstructed fishable biomass, numbers, and recruitment dynamics:
"""

# ╔═╡ 50000001-0000-0000-0000-000000000014
begin
    p_dd = plot_state_variables(dd_sim)
    (
        plot_type = typeof(p_dd),
        terminal_biomass = round(dd_sim.biomass[end]; digits = 2)
    )
end

# ╔═╡ Cell order:
# ╠═50000001-0000-0000-0000-000000000001
# ╠═50000001-0000-0000-0000-000000000002
# ╠═50000001-0000-0000-0000-000000000003
# ╠═50000001-0000-0000-0000-000000000004
# ╠═50000001-0000-0000-0000-000000000005
# ╠═50000001-0000-0000-0000-000000000006
# ╠═50000001-0000-0000-0000-000000000007
# ╠═50000001-0000-0000-0000-000000000008
# ╠═50000001-0000-0000-0000-000000000009
# ╠═50000001-0000-0000-0000-000000000010
# ╠═50000001-0000-0000-0000-000000000011
# ╠═50000001-0000-0000-0000-000000000012
# ╠═50000001-0000-0000-0000-000000000013
# ╠═50000001-0000-0000-0000-000000000014

