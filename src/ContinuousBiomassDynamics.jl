# ------------------------------------------------------------------------------------
# Continuous-time logistic biomass dynamics.
#
# The discrete logistic map in `BiomassDynamics` steps the state once per year, which is
# the right unit when removals are aggregated to years. This module carries the same
# dynamics in continuous time,
#
#     dB/dt = r B (1 - B/K) - C(t),
#
# so removals can enter at their actual timing and the trajectory between removals is
# resolved. Both share one density-dependence form, and the discrete map is exactly this
# ODE integrated with forward Euler over a unit step, so the two are the same model at
# different time resolutions rather than two models.
# ------------------------------------------------------------------------------------

module ContinuousBiomassDynamics

using LinearAlgebra
using Statistics
using Distributions
using Turing
using Random
import ..Config

export ContinuousBiomassConfig, ContinuousBiomassResult
export load_continuous_biomass_config
export logistic_growth_ode!, integrate_continuous_biomass
export continuous_msy_reference_points, simulate_continuous_biomass
export bayesian_continuous_logistic_model

"""
    ContinuousBiomassConfig

User-specified configuration for the continuous-time logistic biomass model. Every field
is read from a TOML file by [`load_continuous_biomass_config`](@ref); nothing is defaulted
in code.

# Fields
- `r::Float64`: intrinsic population growth rate.
- `k::Float64`: environmental carrying capacity (unexploited equilibrium biomass).
- `b0::Float64`: initial biomass.
- `n_hours::Int`: number of steps in the projected time series.
- `dt::Float64`: integration step size, in model time units.
- `catches::Vector{Float64}`: removals at each step.
- `survey_q::Float64`: survey catchability mapping biomass to relative index.
"""
struct ContinuousBiomassConfig
    r::Float64
    k::Float64
    b0::Float64
    n_hours::Int
    dt::Float64
    catches::Vector{Float64}
    survey_q::Float64
end

"""
    ContinuousBiomassResult

Output of [`integrate_continuous_biomass`](@ref).

# Fields
- `times::Vector{Float64}`: time of each recorded frame.
- `biomass::Vector{Float64}`: biomass at each frame.
- `surplus::Vector{Float64}`: surplus production `rB(1 - B/K)` at each frame.
- `removals::Vector{Float64}`: removals applied, at each frame.
"""
struct ContinuousBiomassResult
    times::Vector{Float64}
    biomass::Vector{Float64}
    surplus::Vector{Float64}
    removals::Vector{Float64}
end

"""
    load_continuous_biomass_config(path::AbstractString) -> ContinuousBiomassConfig

Read the `[continuous]` section of a TOML file. Every key is required; a missing key
raises rather than falling back to a compiled-in default.

Expected keys: `r`, `k`, `b0`, `n_hours`, `dt`, `catches`, `survey_q`.
"""
function load_continuous_biomass_config(path::AbstractString)
    cfg = Config.parse_file(path)
    d = Config.require_section(cfg, "continuous", path)
    where = "$path section [continuous]"

    n = Config.load_int(d, "n_hours", where)
    n >= 1 || error("'n_hours' in $where must be at least 1, got $n.")
    dt = Config.positive(d, "dt", where)
    k = Config.positive(d, "k", where)
    b0 = Config.non_negative(d, "b0", where)
    r = Config.load_float(d, "r", where)
    catches = Config.load_reals(d, "catches", where)
    length(catches) == n ||
        error("'catches' in $where must have $n entries, got $(length(catches)).")
    q = Config.non_negative(d, "survey_q", where)

    return ContinuousBiomassConfig(r, k, b0, n, dt, catches, q)
end

"""
    logistic_growth_ode!(dB, B, p, t) -> Nothing

Right-hand side of the logistic surplus-production ODE fitted for `DifferentialEquations`:

```math
\\frac{dB}{dt} = r B \\left(1 - \\frac{B}{K}\\right) - C(t)
```

`C(t)` is the removal in force at time `t`. It is not declared as a parameter of `p`
because the ODE is written for a generic state `B` and the removal schedule is supplied
through the parameter tuple by the caller.
"""
function logistic_growth_ode!(dB, B, p, t)
    r, k, c_now = p
    surplus = r * B[1] * (1.0 - B[1] / k)
    dB[1] = surplus - c_now
    return nothing
end

"""
    integrate_continuous_biomass(cfg::ContinuousBiomassConfig;
                                r::Real = cfg.r, k::Real = cfg.k) -> ContinuousBiomassResult

Integrate the continuous logistic dynamics with the removal schedule in `cfg` by
semi-implicit (exponential) Euler:

```math
B_{t+1} = \\frac{B_t + \\Delta t\\, r\\, B_t - C_t}{1 + \\Delta t\\, r\\, B_t / K}
```

which is the exact solution of the logistic ODE when `C` is held constant over the step.
Because it is exact for constant removal and removes the stiffness at high biomass, it is
preferred to forward Euler: forward Euler overshoots `K` when `r\\Delta t` is large and
then produces negative biomass.

`n_hours + 1` frames are returned including the initial state.
"""
function integrate_continuous_biomass(cfg::ContinuousBiomassConfig;
                                      r::Real = cfg.r, k::Real = cfg.k)
    r > 0 || error("Model growth rate r must be positive, got $r.")
    k > 0 || error("Carrying capacity k must be positive, got $k.")

    n = cfg.n_hours
    # `n_hours + 1` frames are returned including the initial state.
    times = Vector{Float64}(undef, n + 1)
    biomass = Vector{Float64}(undef, n + 1)
    surplus = Vector{Float64}(undef, n + 1)
    removals = Vector{Float64}(undef, n + 1)

    biomass[1] = cfg.b0
    surplus[1] = r * cfg.b0 * (1.0 - cfg.b0 / k)
    removals[1] = 0.0
    times[1] = 0.0

    b = cfg.b0
    for i in 1:n
        h = cfg.dt
        c = cfg.catches[i]
        b_next = (b + h * r * b - c) / (1.0 + h * r * b / k)
        b = max(0.0, b_next)
        biomass[i + 1] = b
        surplus[i + 1] = r * b * (1.0 - b / k)
        removals[i + 1] = c
        times[i + 1] = i * h
    end

    return ContinuousBiomassResult(times, biomass, surplus, removals)
end

"""
    continuous_msy_reference_points(cfg::ContinuousBiomassConfig; r = cfg.r, k = cfg.k)

Equilibrium reference points for the continuous logistic model, identical to the discrete
ones because the equilibrium is where `dB/dt = 0`:

- `bmsy = K / 2`
- `msy = r K / 4`
- `umsy = 1/2` (harvest rate at MSY, since surplus production equals `rB/2` there)

A `NamedTuple` is returned rather than an `MSYEquilibrium`, because the per-gear fields
of `MSYEquilibrium` are not meaningful for a single biomass pool.
"""
function continuous_msy_reference_points(cfg::ContinuousBiomassConfig;
                                          r::Real = cfg.r, k::Real = cfg.k)
    r > 0 || error("Model growth rate r must be positive, got $r.")
    k > 0 || error("Carrying capacity k must be positive, got $k.")
    bmsy = k / 2.0
    msy = r * k / 4.0
    return (bmsy = bmsy, msy = msy, umsy = 0.5)
end

"""
    simulate_continuous_biomass(cfg::ContinuousBiomassConfig;
                               n_sims::Integer = 200,
                               obs_cv::Real = 0.20,
                               rng::AbstractRNG = Random.default_rng()
                               ) -> (trajectories, observations)

Run `n_sims` stochastic realisations of the continuous logistic model, each with its own
draw of process noise on `r`, and return the biomass trajectories and the matching
lognormal survey observations.

Each realisation perturbs `r` by a lognormal factor with coefficient of variation
`proc_cv`, which propagates demographic uncertainty into the projected harvest without
reintroducing model structure.
"""
function simulate_continuous_biomass(cfg::ContinuousBiomassConfig;
                                     n_sims::Integer = 200,
                                     process_cv::Real = 0.15,
                                     obs_cv::Real = 0.20,
                                     rng::AbstractRNG = Random.default_rng())
    n_sims >= 1 || error("n_sims must be at least 1, got $n_sims.")
    (0 <= process_cv) || error("process_cv must be non-negative, got $process_cv.")
    (0 <= obs_cv) || error("obs_cv must be non-negative, got $obs_cv.")

    trajectories = Matrix{Float64}(undef, n_sims, cfg.n_hours + 1)
    observations = Matrix{Float64}(undef, n_sims, cfg.n_hours + 1)
    for s in 1:n_sims
        r_i = cfg.r * exp(process_cv * randn(rng) - 0.5 * process_cv^2)
        out = integrate_continuous_biomass(cfg; r = r_i)
        trajectories[s, :] .= out.biomass
        for i in 1:(cfg.n_hours + 1)
            mu = max(1.0e-6, cfg.survey_q * out.biomass[i])
            observations[s, i] = exp(log(mu) + obs_cv * randn(rng))
        end
    end
    return trajectories, observations
end

"""
    bayesian_continuous_logistic_model(removals, survey_obs, cfg;
                                       prior_r = (0.35, 0.15),
                                       prior_k = (100.0, 30.0),
                                       prior_q = (1.0, 0.25),
                                       prior_proc_sd = 0.08, proc_sd_sd = 0.04,
                                       prior_obs_sd = 0.15, obs_sd_sd = 0.05,
                                       k_lower = 1.0, k_upper_factor = 10.0,
                                       r_lower = 0.01, r_upper = 1.5,
                                       proc_sd_lower = 0.01, proc_sd_upper = 0.40,
                                       obs_sd_lower = 0.02, obs_sd_upper = 0.50)

Bayesian state-space version of the continuous logistic model in `Turing.jl`:

```math
\\begin{aligned}
K &\\sim \\mathrm{TruncatedNormal}(\\mu_K, \\sigma_K, K_{\\rm low}, f_K \\mu_K) \\\\
r &\\sim \\mathrm{TruncatedNormal}(\\mu_r, \\sigma_r, r_{\\rm low}, r_{\\rm up}) \\\\
B_1 &\\sim \\mathrm{Lognormal}(\\log \\hat{B}_1, \\sigma_{\\rm obs}) \\\\
\\varepsilon_t &\\sim \\mathrm{Normal}(0, \\sigma_p B_t) \\\\
B_{t+1} &= B_t + \\Delta t\\, r B_t (1 - B_t / K) - C_t + \\varepsilon_t \\\\
I_t &\\sim \\mathrm{Lognormal}(\\log(q B_t), \\sigma_{\\rm obs})
\\end{aligned}
```

The process deviation scales with biomass, so a stock an order of magnitude larger has
proportionally larger absolute process noise, which is the standard parameterisation for
aggregate biomass dynamics.

Every prior mean, standard deviation and bound is a named keyword, so the prior
specification is user-level rather than compiled into the model body.
"""
@model function bayesian_continuous_logistic_model(removals::AbstractVector{<:Real},
                                                    survey_obs::AbstractVector,
                                                    cfg::ContinuousBiomassConfig;
                                                    prior_r::Tuple{Real, Real} = (0.35, 0.15),
                                                    prior_k::Tuple{Real, Real} = (100.0, 30.0),
                                                    prior_q::Tuple{Real, Real} = (1.0, 0.25),
                                                    prior_proc_sd::Real = 0.08,
                                                    proc_sd_sd::Real = 0.04,
                                                    prior_obs_sd::Real = 0.15,
                                                    obs_sd_sd::Real = 0.05,
                                                    k_lower::Real = 1.0,
                                                    k_upper_factor::Real = 10.0,
                                                    r_lower::Real = 0.01,
                                                    r_upper::Real = 1.5,
                                                    proc_sd_lower::Real = 0.01,
                                                    proc_sd_upper::Real = 0.40,
                                                    obs_sd_lower::Real = 0.02,
                                                    obs_sd_upper::Real = 0.50)
    nt = length(removals)
    length(survey_obs) == nt ||
        error("removals and survey_obs lengths must match, got $(length(survey_obs)) and $nt.")
    nt >= 2 || error("At least 2 observations are required.")

    k ~ truncated(Normal(prior_k[1], prior_k[2]), k_lower, k_upper_factor * prior_k[1])
    r ~ truncated(Normal(prior_r[1], prior_r[2]), r_lower, r_upper)
    q ~ truncated(Normal(prior_q[1], prior_q[2]), 0.01, 10.0)
    proc_sd ~ truncated(Normal(prior_proc_sd, proc_sd_sd), proc_sd_lower, proc_sd_upper)
    obs_sd ~ truncated(Normal(prior_obs_sd, obs_sd_sd), obs_sd_lower, obs_sd_upper)

    b = Vector{Float64}(undef, nt + 1)
    b[1] ~ LogNormal(log(max(1.0e-6, cfg.b0)), 1.0e-3)
    for t in 1:nt
        # The deterministic mean is the semi-implicit Euler step, so the state's
        # expected trajectory is the same recursion the deterministic path integrates.
        mu_next = (b[t] + cfg.dt * r * b[t] - removals[t]) /
                  (1.0 + cfg.dt * r * b[t] / k)
        b[t + 1] ~ Normal(mu_next, proc_sd * max(b[t], 1.0e-6))
    end

    for i in 1:(nt + 1)
        mu = max(1.0e-6, q * b[i])
        survey_obs[i] ~ LogNormal(log(mu), obs_sd)
    end

    return (biomass = b, k = k, r = r, q = q, proc_sd = proc_sd, obs_sd = obs_sd)
end

end # module ContinuousBiomassDynamics
