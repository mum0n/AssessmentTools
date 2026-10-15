# ------------------------------------------------------------------------------------
# Deriso-Schnute Delay-Difference Population Dynamics.
# Incorporates growth, natural mortality, and lagged recruitment into an
# age-structured biomass and numerical dynamics formulation.
# ------------------------------------------------------------------------------------

"""
Configuration parameters for the Deriso-Schnute delay-difference model:
- `s::Float64`: Natural annual survival fraction ``s = \\exp(-M)``.
- `alpha::Float64`: Brody growth intercept parameter ``\\alpha = w_k - \\rho w_{k-1}``.
- `rho::Float64`: Brody growth slope parameter ``\\rho`` (von Bertalanffy weight growth coefficient).
- `wk::Float64`: Mean weight at recruitment age ``k`` (entry into the fishable component).
"""
struct DelayDifferenceConfig
    s::Float64
    alpha::Float64
    rho::Float64
    wk::Float64
end

"""
Result of a forward Deriso-Schnute delay-difference simulation:
- `years::Vector{Int}`: Simulation years.
- `biomass::Vector{Float64}`: Exploitable fishable biomass trajectory ``B_t``.
- `numbers::Vector{Float64}`: Total fishable numbers trajectory ``N_t``.
- `recruits::Vector{Float64}`: Ingested/realized recruitment in numbers ``R_t``.
- `recruits_biomass::Vector{Float64}`: Recruitment biomass influx ``w_k \\cdot R_t``.
- `catches_biomass::Vector{Float64}`: Annual catch in weight ``Y_t``.
"""
struct DelayDifferenceResult
    years::Vector{Int}
    biomass::Vector{Float64}
    numbers::Vector{Float64}
    recruits::Vector{Float64}
    recruits_biomass::Vector{Float64}
    catches_biomass::Vector{Float64}
end

"""
    delay_difference_step(bt::Real, nt::Real, yt::Real, rt::Real, cfg::DelayDifferenceConfig) -> Tuple{Float64, Float64}

Single annual forward step of the Deriso-Schnute delay-difference equations:

```math
N_{t+1} = s (N_t - C_t) + R_{t+1}
```

```math
B_{t+1} = s (1 + \\rho) (B_t - Y_t) - s^2 \\rho \\frac{B_{t-1} - Y_{t-1}}{N_{t-1} - C_{t-1}} (N_t - C_t) + w_k R_{t+1}
```
or under the equivalent Ford-Brody mean weight formulation:
```math
B_{t+1} = s \\alpha (N_t - C_t) + s \\rho (B_t - Y_t) + w_k R_{t+1}
```

# Inputs
- `bt`: Current exploitable biomass ``B_t``.
- `nt`: Current exploitable numbers ``N_t``.
- `yt`: Catch in weight ``Y_t``.
- `rt`: Incoming recruitment in numbers ``R_{t+1}`` at age/stage of entry.
- `cfg`: DelayDifferenceConfig containing survival ``s`` and growth parameters ``(\\alpha, \\rho, w_k)``.
"""
function delay_difference_step(bt::Real, nt::Real, yt::Real, rt::Real, cfg::DelayDifferenceConfig)
    s = cfg.s
    alpha = cfg.alpha
    rho = cfg.rho
    wk = cfg.wk

    # Escapement after fishery
    b_esc = max(0.0, bt - yt)
    # Catch in numbers using average fish weight in current stock
    mean_w = nt > 0.0 ? bt / nt : wk
    ct = mean_w > 0.0 ? yt / mean_w : 0.0
    n_esc = max(0.0, nt - ct)

    # Next year states
    next_n = s * n_esc + rt
    next_b = s * alpha * n_esc + s * rho * b_esc + wk * rt

    return (max(0.0, next_b), max(0.0, next_n))
end

"""
    simulate_delay_difference(years::AbstractVector{<:Integer},
                              catches_biomass::AbstractVector{<:Real},
                              recruits::AbstractVector{<:Real},
                              cfg::DelayDifferenceConfig;
                              b0::Real, n0::Real = b0 / cfg.wk) -> DelayDifferenceResult

Simulate forward time-series using the Deriso-Schnute delay-difference model driven by an
external recruitment trajectory (such as male recruitment ``M_1`` transitioning into the
fishable component ``M_0`` from the continuous delay differential model).
"""
function simulate_delay_difference(years::AbstractVector{<:Integer},
                                   catches_biomass::AbstractVector{<:Real},
                                   recruits::AbstractVector{<:Real},
                                   cfg::DelayDifferenceConfig;
                                   b0::Real,
                                   n0::Real = b0 / cfg.wk)
    ny = length(years)
    length(catches_biomass) == ny || error("catches_biomass and years lengths must match.")
    length(recruits) == ny || error("recruits and years lengths must match.")

    b = zeros(Float64, ny)
    n = zeros(Float64, ny)
    rec_b = zeros(Float64, ny)

    b[1] = b0
    n[1] = n0
    rec_b[1] = recruits[1] * cfg.wk

    for t in 1:(ny - 1)
        next_b, next_n = delay_difference_step(b[t], n[t], catches_biomass[t], recruits[t + 1], cfg)
        b[t + 1] = next_b
        n[t + 1] = next_n
        rec_b[t + 1] = recruits[t + 1] * cfg.wk
    end

    return DelayDifferenceResult(
        collect(years), b, n, collect(recruits), rec_b, collect(catches_biomass)
    )
end

"""
    extract_annual_recruitment(dde_simulation; stage_index::Integer = 2, target_month::Real = 9.0) -> Tuple{Vector{Int}, Vector{Float64}}

Extract annual recruitment time series from a continuous multi-stage DDE simulation.
Recruits are sampled annually at `target_month` from the specified stage (default: stage 2 / `M1`
male recruits entering the fishable component).
"""
function extract_annual_recruitment(dde_simulation; stage_index::Integer = 2, target_month::Real = 9.0)
    times = dde_simulation.time
    states = dde_simulation.states

    min_yr = Int(floor(minimum(times)))
    max_yr = Int(floor(maximum(times)))
    ann_years = Int[]
    ann_rec = Float64[]

    dt_target = target_month / 12.0
    for yr in min_yr:max_yr
        t_sample = yr + dt_target
        idx = findmin(abs.(times .- t_sample))[2]
        push!(ann_years, yr)
        push!(ann_rec, states[idx, stage_index])
    end

    return (ann_years, ann_rec)
end

"""
    bayesian_delay_difference_model(catches_biomass, recruits, obs_survey, cfg;
                                    obs_cv = 0.20,
                                    prior_s_mu = nothing, prior_s_sd = 0.05,
                                    s_lower = 0.30, s_upper = 0.95,
                                    prior_b0_mu = 50.0, prior_b0_sd = 15.0,
                                    b0_lower = 5.0, b0_upper = 200.0,
                                    prior_q_mu = 1.0, prior_q_sd = 0.30,
                                    q_lower = 0.05, q_upper = 5.0,
                                    prior_proc_sd = 0.10, prior_proc_sd_sd = 0.05,
                                    proc_sd_lower = 0.01, proc_sd_upper = 0.40,
                                    prior_obs_sd_sd = 0.05,
                                    obs_sd_lower = 0.02, obs_sd_upper = 0.60,
                                    mu_floor = 1.0e-4)

Bayesian state-space Deriso-Schnute delay-difference model implemented in `Turing.jl`:
- `catches_biomass`: Annual observed catches in weight ``Y_t``.
- `recruits`: Annual incoming recruitment counts ``R_t`` (e.g. from continuous DDE stage ``M_1``).
- `obs_survey`: Relative abundance or biomass survey observations ``I_t`` (supports missing/NaN values).
- `cfg`: DelayDifferenceConfig containing growth parameters ``(\\alpha, \\rho, w_k)`` and baseline survival.
- `obs_cv`: Prior mean observation error standard deviation.

Every prior mean, standard deviation and bound is a named keyword argument, so the prior
specification is user-level rather than compiled into the model body. `prior_s_mu`
defaults to `cfg.s`, the baseline survival from the configuration.
"""
@model function bayesian_delay_difference_model(catches_biomass::AbstractVector{<:Real},
                                                recruits::AbstractVector{<:Real},
                                                obs_survey::AbstractVector,
                                                cfg::DelayDifferenceConfig;
                                                obs_cv::Real = 0.20,
                                                prior_s_mu::Union{Nothing, Real} = nothing,
                                                prior_s_sd::Real = 0.05,
                                                s_lower::Real = 0.30,
                                                s_upper::Real = 0.95,
                                                prior_b0_mu::Real = 50.0,
                                                prior_b0_sd::Real = 15.0,
                                                b0_lower::Real = 5.0,
                                                b0_upper::Real = 200.0,
                                                prior_q_mu::Real = 1.0,
                                                prior_q_sd::Real = 0.30,
                                                q_lower::Real = 0.05,
                                                q_upper::Real = 5.0,
                                                prior_proc_sd::Real = 0.10,
                                                prior_proc_sd_sd::Real = 0.05,
                                                proc_sd_lower::Real = 0.01,
                                                proc_sd_upper::Real = 0.40,
                                                prior_obs_sd_sd::Real = 0.05,
                                                obs_sd_lower::Real = 0.02,
                                                obs_sd_upper::Real = 0.60,
                                                mu_floor::Real = 1.0e-4)
    ny = length(catches_biomass)
    length(recruits) == ny || error("recruits and catches lengths must match.")
    length(obs_survey) == ny || error("obs_survey and catches lengths must match.")

    s_mu = isnothing(prior_s_mu) ? cfg.s : prior_s_mu

    # Priors
    # Annual survival s = exp(-M) centered near the configured baseline survival
    s ~ truncated(Normal(s_mu, prior_s_sd), s_lower, s_upper)
    # Initial exploitable biomass B_0
    b0 ~ truncated(Normal(prior_b0_mu, prior_b0_sd), b0_lower, b0_upper)
    # Survey catchability q (mapping biomass to relative survey index)
    q ~ truncated(Normal(prior_q_mu, prior_q_sd), q_lower, q_upper)
    # Log-scale process error standard deviation
    proc_sd ~ truncated(Normal(prior_proc_sd, prior_proc_sd_sd),
                        proc_sd_lower, proc_sd_upper)
    # Observation error standard deviation
    obs_sd ~ truncated(Normal(obs_cv, prior_obs_sd_sd), obs_sd_lower, obs_sd_upper)

    alpha = cfg.alpha
    rho = cfg.rho
    wk = cfg.wk

    # Latent state trajectories
    b = Vector{Float64}(undef, ny)
    n = Vector{Float64}(undef, ny)

    b[1] = b0
    n[1] = b0 / wk

    for t in 1:(ny - 1)
        yt = catches_biomass[t]
        rt = recruits[t + 1]

        b_esc = max(1e-4, b[t] - yt)
        mean_w = n[t] > 0.0 ? b[t] / n[t] : wk
        ct = mean_w > 0.0 ? yt / mean_w : 0.0
        n_esc = max(1e-4, n[t] - ct)
        exp_n = s * n_esc + rt
        exp_b = s * alpha * n_esc + s * rho * b_esc + wk * rt

        # Process model with stochastic lognormal deviations around the deterministic
        # expectation defined once in `delay_difference_step`.
        log_b_next ~ Normal(log(max(1e-4, exp_b)), proc_sd)
        b[t + 1] = exp(log_b_next)
        n[t + 1] = max(1e-4, exp_n)
    end

    # Observation model for survey index
    for t in 1:ny
        s_val = obs_survey[t]
        if !ismissing(s_val) && isfinite(s_val) && s_val > 0.0
            s_pred = q * b[t]
            s_val ~ LogNormal(log(max(1e-4, s_pred)), obs_sd)
        end
    end

    return (biomass = b, numbers = n)
end
