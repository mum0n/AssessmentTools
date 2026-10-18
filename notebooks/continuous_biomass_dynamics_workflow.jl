### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

# ╔═╡ 20000001-0000-0000-0000-000000000001
md"""
# Continuous-Time Logistic Biomass Dynamics

This notebook runs the **`AssessmentTools.ContinuousBiomassDynamics`** submodule: the
logistic surplus production model carried in continuous time, so removals can enter at
their actual timing and the trajectory between removals is resolved.

### Method

Aggregate biomass is governed by

```math
\frac{dB}{dt} = r B \left(1 - \frac{B}{K}\right) - Y(t) \quad \text{or} \quad \frac{dB}{dt} = r B \left(1 - \frac{B}{K}\right) - F(t) B
```

with intrinsic growth rate `r` [$\text{time}^{-1}$], carrying capacity `K` [mass],
instantaneous removal rate `Y(t)` [$\text{mass}\cdot\text{time}^{-1}$], and instantaneous
fishing mortality `F(t)` [$\text{time}^{-1}$].

For an integration step duration $\Delta t$, the interval catch mass $C_t$ [mass] extracted
over $[t, t + \Delta t]$ is $C_t = \bar{Y}_t \Delta t$. Integration uses semi-implicit Euler,

```math
B_{t+1} = \frac{B_t + \Delta t\, r B_t - C_t}{1 + \Delta t\, r B_t / K},
```

which is the **exact** solution of the logistic ODE when removal rate $\bar{Y}_t = C_t / \Delta t$
is held constant over the step.

### Dimensional Distinction and Parameter Specifications

- `catches`: interval removal mass $C_t$ [mass], **not** an instantaneous rate. When
  $\Delta t \ne 1.0$, passing an instantaneous rate causes a $1/\Delta t$ dimensional error.
- Instantaneous removal rate: $\bar{Y}_t = C_t / \Delta t$ [$\text{mass}\cdot\text{time}^{-1}$].
- Instantaneous fishing mortality: $F_t = C_t / (B_t \, \Delta t)$ [$\text{time}^{-1}$].

### Relationship to the discrete model

The discrete map in `AssessmentTools.BiomassDynamics`,

```math
B_{t+1} = \max(0, B_t + r B_t (1 - B_t/K) - C_t),
```

is this ODE integrated with **forward Euler** over a unit step ($\Delta t = 1.0$).
The two are the same model at different time resolutions. Forward Euler is not exact and
overshoots `K` when `r Δt` is large, which is why the semi-implicit form is used here.

### Assumptions

- Biomass, catch and index are in consistent units and refer to one aggregate pool.
- The index is a fixed proportion of biomass with constant catchability.
- Removals are applied continuously over each step, not as a single impulse.
- Model error is multiplicative (process noise scales with biomass), so a stock an order
  of magnitude larger has proportionally larger absolute process noise.

### Strengths

- `dt < 1` resolves within-year removal timing, which the annual step cannot.
- Semi-implicit Euler cannot overshoot `K` or produce negative biomass.
- Equilibrium reference points are closed-form: `B_MSY = K/2`, `MSY = rK/4`.

### Weaknesses

- No age structure, selectivity, or ontogenetic change in availability.
- With a weak index, `K` and `q` are only jointly identified through the shape of the
  series; the production curve is nearly unbounded.
- Equilibrium reference points assume the stock is at biomass near equilibrium.
"""

# ╔═╡ 20000002-0000-0000-0000-000000000002
md"""
## 1. Configuration

All parameters come from `data/continuous_biomass/config.toml`; nothing is defaulted in
code.
"""

# ╔═╡ 30000003-0000-0000-0000-000000000003
using AssessmentTools
using AssessmentTools.ContinuousBiomassDynamics
using LinearAlgebra, Statistics, Random

# ╔═╡ 30000004-0000-0000-0000-000000000004
begin
    cfg = load_continuous_biomass_config(
        normpath(joinpath(@__DIR__, "..", "data", "continuous_biomass", "config.toml")))
    (
        n_hours = cfg.n_hours,
        dt = cfg.dt,
        r = cfg.r,
        k = cfg.k,
        b0 = cfg.b0,
        survey_q = cfg.survey_q,
        n_catches = length(cfg.catches),
    )
end

# ╔═╡ 30000005-0000-0000-0000-000000000005
md"""
## 2. Deterministic Trajectory

`integrate_continuous_biomass` returns `n_hours + 1` frames including the initial state.
Biomass should stay within `[0, K]` throughout.
"""

# ╔═╡ 30000006-0000-0000-0000-000000000006
begin
    out = integrate_continuous_biomass(cfg)
    (
        frames = length(out.biomass),
        B_initial = round(out.biomass[1], digits = 2),
        B_final = round(out.biomass[end], digits = 2),
        max_biomass = round(maximum(out.biomass), digits = 2),
        stays_within_bounds = all(x -> 0.0 <= x <= cfg.k, out.biomass),
    )
end

# ╔═╡ 30000007-0000-0000-0000-000000000007
md"""
## 3. Biological Reference Points

Biological reference points correspond to the stationary production equilibrium where $\frac{dB}{dt} = 0$:

```math
\begin{aligned}
B_{\text{MSY}} &= \frac{K}{2} \\
\text{MSY} &= \frac{r K}{4} \\
F_{\text{MSY}} &= \frac{r}{2} \\
U_{\text{MSY}} &= 1 - \exp(-F_{\text{MSY}}) = 1 - \exp\left(-\frac{r}{2}\right)
\end{aligned}
```

The instantaneous harvest rate $F_{\text{MSY}} = r/2$ reflects continuous production. The discrete annual exploitation rate $U_{\text{MSY}} = 1 - \exp(-F_{\text{MSY}})$ accounts for non-linear stock depletion across the annual integration cycle.
"""

# ╔═╡ 30000008-0000-0000-0000-000000000008
begin
    rp = continuous_msy_reference_points(cfg)
    (
        B_MSY = round(rp.bmsy, digits = 2),
        MSY = round(rp.msy, digits = 2),
        instantaneous_F_MSY = round(cfg.r / 2.0, digits = 4),
        annual_U_MSY = round(1.0 - exp(-cfg.r / 2.0), digits = 4),
        U_MSY_nominal = rp.umsy
    )
end

# ╔═╡ 30000009-0000-0000-0000-000000000009
md"""
## 4. Effect of the Integration Step

The continuous form's value is the choice of `dt`. `dt = 1` reproduces the discrete map;
smaller `dt` resolves within-year timing. Because the scheme is exact for constant
removal within a step, halving `dt` halves the timing error.
"""

# ╔═╡ 30000010-0000-0000-0000-000000000010
begin
    results = []
    for dt in (1.0, 0.5, 0.25, 0.1)
        c = ContinuousBiomassConfig(cfg.r, cfg.k, cfg.b0, cfg.n_hours, dt,
                                     cfg.catches, cfg.survey_q)
        r_i = integrate_continuous_biomass(c)
        push!(results, (dt = dt, B_final = round(r_i.biomass[end], digits = 4),
                        max_B = round(maximum(r_i.biomass), digits = 4)))
    end
    results
end

# ╔═╡ 30000011-0000-0000-0000-000000000011
md"""
## 5. Stochastic Ensembles

`simulate_continuous_biomass` runs many realisations, each perturbing `r` by a
multiplicative lognormal factor, and returns the biomass trajectories and the matching
lognormal observations.
"""

# ╔═╡ 30000012-0000-0000-0000-000000000012
begin
    n_sims = 300
    traj, obs = simulate_continuous_biomass(
        cfg; n_sims = n_sims, process_cv = 0.15, obs_cv = 0.20,
        rng = Random.MersenneTwister(2027))
    (
        n_simulations = n_sims,
        mean_final_biomass = round(mean(traj[:, end]), digits = 2),
        sd_final_biomass = round(std(traj[:, end]), digits = 2),
        quartiles_final_B = round.(quantile(traj[:, end], [0.25, 0.5, 0.75]), digits = 2),
    )
end

# ╔═╡ 30000013-0000-0000-0000-000000000013
md"""
## 6. Bayesian State-Space Version

Fitted with the same three engines as every other model in the package: MAP via `Optim`,
posterior via NUTS, and variational inference via `AdvancedVI`.
"""

# ╔═╡ 30000014-0000-0000-0000-000000000014
using AssessmentTools.InferenceBridge
using DynamicPPL, LogDensityProblems

# ╔═╡ 30000015-0000-0000-0000-000000000015
begin
    # Build a survey series from the deterministic trajectory to fit.
    survey_obs = [cfg.survey_q * out.biomass[i] for i in eachindex(out.biomass)]
    obs_noisy = [exp(log(max(1.0e-9, v)) + 0.20 * randn(Random.MersenneTwister(31)))
                 for v in survey_obs]

    mod = bayesian_continuous_logistic_model(
        cfg.catches, obs_noisy, cfg;
        prior_r = (0.35, 0.15), prior_k = (1200.0, 300.0), prior_q = (0.0011, 0.0005),
        prior_proc_sd = 0.08, prior_obs_sd = 0.20)

    dim = InferenceBridge.parameter_dimension(mod)
    x0 = collect(Float64, DynamicPPL.LogDensityFunction(mod)._x)
    (model_dimension = dim,)
end

# ╔═╡ 30000016-0000-0000-0000-000000000016
begin
    r_map = InferenceBridge.fit_map(mod, x0; iterations = 400, verbose = false)
    r_vi = InferenceBridge.fit_vi(mod; family = :meanfield, max_iter = 60,
                                  n_samples = 200,
                                  rng = Random.MersenneTwister(32))
    (map_log_posterior = round(r_map.log_posterior, digits = 4),
     converged = r_map.converged,
     vi_fitted = typeof(r_vi.result))
end

# ╔═╡ 30000017-0000-0000-0000-000000000017
md"""
## 7. Posterior Ensembles and Projection

Posterior draws can be saved to a Zarr archive and replayed at each projection step, so
uncertainty enters the transition at every step rather than being resolved once at the
start. See `SimulationEngine.save_ensemble` / `load_ensemble`.
"""

# ╔═╡ 30000018-0000-0000-0000-000000000018
begin
    # Forward projection from a designated time slice, reading that slice's rates.
    pr = SimulationEngine.projection_parameters(
        n_groups = 1,
        weights = [1.0], maturity = [1.0],
        fertility = [0.0], natural_mortality = 0.0,
        advance = [1.0], retention = [0.0],
        initial_numbers = [out.biomass[1]],
        fishing_mortality = zeros(1, 1),
        harvest_rate = fill(0.0, 1, 1),
        model_symbol = :continuous_logistic)
    st = SimulationEngine.ProjectionState([out.biomass[1]], 0, out.biomass[1], 0.0, 0.0, 1)
    fwd = SimulationEngine.project_state(pr, st; direction = :forward, n_years = 5)
    (projected_final_biomass = round(fwd[end].biomass, digits = 3),)
end

# ╔═╡ 30000019-0000-0000-0000-000000000019
md"""
## 8. Summary

- Continuous logistic dynamics integrate with a semi-implicit Euler step that is exact
  for constant removal within a step and cannot overshoot `K`.
- `dt` controls timing resolution; the discrete map in `BiomassDynamics` is the same ODE
  at `dt = 1` with forward Euler.
- Reference points are closed-form at equilibrium: `B_MSY = K/2`, `MSY = rK/4`.
- The Bayesian version fits the same recursion as a state-space model, with process noise
  scaling on biomass, and uses the same three inference engines as every other model.
""

# ╔═╡ Cell order:
# ╠═20000001-0000-0000-0000-000000000001
# ╠═20000002-0000-0000-0000-000000000002
# ╠═30000003-0000-0000-0000-000000000003
# ╠═30000004-0000-0000-0000-000000000004
# ╠═30000005-0000-0000-0000-000000000005
# ╠═30000006-0000-0000-0000-000000000006
# ╠═30000007-0000-0000-0000-000000000007
# ╠═30000008-0000-0000-0000-000000000008
# ╠═30000009-0000-0000-0000-000000000009
# ╠═30000010-0000-0000-0000-000000000010
# ╠═30000011-0000-0000-0000-000000000011
# ╠═30000012-0000-0000-0000-000000000012
# ╠═30000013-0000-0000-0000-000000000013
# ╠═30000014-0000-0000-0000-000000000014
# ╠═30000015-0000-0000-0000-000000000015
# ╠═30000016-0000-0000-0000-000000000016
# ╠═30000017-0000-0000-0000-000000000017
# ╠═30000018-0000-0000-0000-000000000018
# ╠═30000019-0000-0000-0000-000000000019
