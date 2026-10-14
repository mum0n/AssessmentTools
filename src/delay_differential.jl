# ------------------------------------------------------------------------------------
# Multi-Stage Continuous Delay Differential Population Dynamics.
# Implements continuous size- and sex-structured delay differential dynamics
# for crustacean and invertebrate fisheries (Choi 2024).
# ------------------------------------------------------------------------------------

"""
Configuration container for a multi-stage continuous delay differential equation (DDE) system:
- `num_stages::Int`: Number of biological stages modeled (default: 6 for snow crab).
- `lags::Vector{Float64}`: Constant development delay lags in years (e.g. `[7.0, 8.0, 9.0]`).
- `k_carrying_capacity::Vector{Float64}`: Stage-specific asymptotic abundance bounds ``K_s``.
- `natural_mortality::Vector{Float64}`: Baseline natural mortality rates ``d_s``.
- `density_mortality::Vector{Float64}`: Density-dependent predation/competition mortality coefficients ``d_{2,s}``.
- `transition_rates::Vector{Float64}`: Stage-to-stage annual moult transition rates ``v_s``.
- `reproduction_weights_m::Vector{Float64}`: Immature male recruitment coefficients ``b_5``.
- `reproduction_weights_f::Vector{Float64}`: Mature female recruitment coefficients ``b_6``.
"""
struct DdeModelConfig
    num_stages::Int
    lags::Vector{Float64}
    k_carrying_capacity::Vector{Float64}
    natural_mortality::Vector{Float64}
    density_mortality::Vector{Float64}
    transition_rates::Vector{Float64}
    reproduction_weights_m::Vector{Float64}
    reproduction_weights_f::Vector{Float64}
end

"""
    default_snowcrab_dde_config() -> DdeModelConfig

Construct standard parameters for the 6-stage continuous delay differential snow crab model:
- Stage 1: Immature / sub-legal instar 9 (M0)
- Stage 2: Instar 10 (M1)
- Stage 3: Instar 11 (M2)
- Stage 4: Instar 12 (commercial recruit, M3)
- Stage 5: Commercial legal males (M4)
- Stage 6: Mature females (f_mat)
"""
function default_snowcrab_dde_config()
    return DdeModelConfig(
        6,
        [7.0, 8.0, 9.0],
        [50.0, 45.0, 35.0, 25.0, 20.0, 15.0],
        [0.15, 0.11, 0.14, 0.17, 0.16, 0.19],
        [0.40, 0.40, 0.40, 0.40, 0.40, 0.40],
        [0.65, 0.68, 0.61, 0.79],
        [0.33, 0.34, 0.33],
        [0.33, 0.34, 0.33]
    )
end

"""
    dde_derivatives(u, u_lag_1, u_lag_mat, p::DdeModelConfig, habitat_scalar = ones(6)) -> Vector{Float64}

Compute instantaneous rates of change ``\\frac{du_s}{dt}`` for all stages:

```math
\\begin{aligned}
\\frac{du_1}{dt} &= v_1 u_2(t-1) \\frac{K_2}{K_1} - d_{r,1} \\\\
\\frac{du_2}{dt} &= v_2 u_3(t-1) \\frac{K_3}{K_2} - v_1 u_2 - d_{r,2} \\\\
\\frac{du_3}{dt} &= v_3 u_4(t-1) \\frac{K_4}{K_3} - v_2 u_3 - d_{r,3} \\\\
\\frac{du_4}{dt} &= v_4 u_5(t-1) \\frac{K_5}{K_4} - v_3 u_4 - d_{r,4} \\\\
\\frac{du_5}{dt} &= b_5^T u_{\\text{mat}}(t-\\tau) \\frac{K_6}{K_5} - v_4 u_5 - d_{r,5} \\\\
\\frac{du_6}{dt} &= b_6^T u_{\\text{mat}}(t-\\tau) - d_{r,6}
\\end{aligned}
```
where ``d_{r,s} = d_s u_s + d_{2,s} \\frac{u_s^2}{H_s(t)}``.
"""
function dde_derivatives(u::AbstractVector{<:Real},
                         u_lag_1::AbstractVector{<:Real},
                         u_lag_mat::AbstractVector{<:Real},
                         p::DdeModelConfig,
                         habitat_scalar::AbstractVector{<:Real} = ones(p.num_stages))
    K = p.k_carrying_capacity
    d = p.natural_mortality
    d2 = p.density_mortality
    v = p.transition_rates
    b5 = p.reproduction_weights_m
    b6 = p.reproduction_weights_f

    du = zeros(Float64, p.num_stages)

    # Losses to mortality: natural + density-dependent habitat space
    dr = zeros(Float64, p.num_stages)
    for s in 1:(p.num_stages)
        h_val = max(1e-4, habitat_scalar[s])
        dr[s] = d[s] * u[s] + (d2[s] * u[s]^2) / h_val
    end

    # Moult transition influxes from 1-year lags
    tr = [v[1] * u_lag_1[2], v[2] * u_lag_1[3], v[3] * u_lag_1[4], v[4] * u_lag_1[5]]

    # Recruits from maternal lag
    rec_m = dot(b5, u_lag_mat) * (K[6] / K[5])
    rec_f = dot(b6, u_lag_mat)

    # Dynamics equations
    du[1] = tr[1] * (K[2] / K[1]) - dr[1]
    du[2] = tr[2] * (K[3] / K[2]) - tr[1] - dr[2]
    du[3] = tr[3] * (K[4] / K[3]) - tr[2] - dr[3]
    du[4] = tr[4] * (K[5] / K[4]) - tr[3] - dr[4]
    du[5] = rec_m - tr[4] - dr[5]
    du[6] = rec_f - dr[6]

    return du
end

"""
    simulate_dde_system(cfg::DdeModelConfig, tspan::Tuple{Real, Real};
                         dt::Real = 1.0 / 12.0, removals_times = Float64[],
                         removals_amount = Float64[]) -> NamedTuple

Integrate continuous multi-stage delay differential equations with impulsive catch removals:
- Supports intra-annual time steps (e.g. monthly: ``dt = 1/12``).
- Integrates deterministic catch pulses using impulsive state subtractions.
"""
function simulate_dde_system(cfg::DdeModelConfig, tspan::Tuple{Real, Real};
                             dt::Real = 1.0 / 12.0,
                             removals_times::AbstractVector{<:Real} = Float64[],
                             removals_amount::AbstractVector{<:Real} = Float64[])
    t_start, t_end = tspan
    times = collect(t_start:dt:t_end)
    nt = length(times)

    # Storage for states across stages (rows = time steps, columns = stages)
    states = zeros(Float64, nt, cfg.num_stages)

    # Initialize history with equilibrium carrying capacity fractions
    u_init = copy(cfg.k_carrying_capacity)
    states[1, :] .= u_init

    # Discrete step integration with lagged state lookups
    for i in 1:(nt - 1)
        t_curr = times[i]
        u_curr = states[i, :]

        # Look up 1-year lagged state
        t_lag1 = t_curr - 1.0
        idx_lag1 = findlast(t -> t <= t_lag1, times[1:i])
        u_lag1 = isnothing(idx_lag1) ? u_init : states[idx_lag1, :]

        # Look up maternal reproductive lag (e.g., ~8 years)
        u_lag_mat = zeros(Float64, length(cfg.lags))
        for (k, lag_val) in enumerate(cfg.lags)
            t_lag_k = t_curr - lag_val
            idx_k = findlast(t -> t <= t_lag_k, times[1:i])
            u_lag_mat[k] = isnothing(idx_k) ? u_init[6] : states[idx_k, 6]
        end

        # Derivatives
        du = dde_derivatives(u_curr, u_lag1, u_lag_mat, cfg)

        # Forward Euler / Runge-Kutta step
        u_next = max.(1e-6, u_curr .+ dt .* du)

        # Check for pulse fishery removals occurring in this interval
        for (r_idx, r_time) in enumerate(removals_times)
            if t_curr <= r_time < t_curr + dt
                # Remove legal commercial males (stage 1 / M0)
                u_next[1] = max(1e-6, u_next[1] - removals_amount[r_idx])
            end
        end

        states[i + 1, :] .= u_next
    end

    return (time = times, states = states, stage_names = ["M0", "M1", "M2", "M3", "M4", "f_mat"])
end

"""
    bayesian_delay_differential_model(survey_times, survey_data, cfg, tspan;
                                      dt = 0.25,
                                      removals_times = Float64[],
                                      removals_amount = Float64[],
                                      prior_q0 = 0.0,
                                      prior_q1 = 1.0,
                                      obs_cv = 0.25)

Bayesian continuous multi-stage delay differential equation (DDE) population model in `Turing.jl`:
- Integrates continuous size-structured stages (`M0` through `M4` and `f_mat`) with developmental lags.
- Couples pulse fishery removals with continuous natural and density-dependent losses.
- Estimates carrying capacity multiplier ``\\kappa``, maternal recruitment scaling ``\\beta``,
  and observation error ``\\sigma_o``.
- Maps latent state concentrations to observed survey biomass / abundance indices.

# Arguments
- `survey_times`: Time indices (in continuous years) corresponding to rows of `survey_data`.
- `survey_data`: Matrix of observed survey indices `(ntimes, nstages)`.
- `cfg`: Baseline `DdeModelConfig`.
- `tspan`: Continuous integration interval `(t_start, t_end)`.
"""
@model function bayesian_delay_differential_model(survey_times::AbstractVector{<:Real},
                                                  survey_data::AbstractMatrix,
                                                  cfg::DdeModelConfig,
                                                  tspan::Tuple{Real, Real};
                                                  dt::Real = 0.25,
                                                  removals_times::AbstractVector{<:Real} = Float64[],
                                                  removals_amount::AbstractVector{<:Real} = Float64[],
                                                  prior_q0::Real = 0.0,
                                                  prior_q1::Real = 1.0,
                                                  obs_cv::Real = 0.25)
    nt_survey, nstages = size(survey_data)
    nstages == cfg.num_stages || error("survey_data columns must match cfg.num_stages.")

    # Priors
    # Scaling factor for carrying capacity
    k_scale ~ truncated(Normal(1.0, 0.25), 0.2, 5.0)
    # Scaling factor for maternal reproduction lag
    rec_scale ~ truncated(Normal(1.0, 0.25), 0.2, 5.0)
    # Observation error standard deviation
    obs_sd ~ truncated(Normal(obs_cv, 0.05), 0.05, 0.60)

    # Re-scale baseline config
    k_scaled = cfg.k_carrying_capacity .* k_scale
    b5_scaled = cfg.reproduction_weights_m .* rec_scale
    b6_scaled = cfg.reproduction_weights_f .* rec_scale

    cur_cfg = DdeModelConfig(
        cfg.num_stages,
        cfg.lags,
        k_scaled,
        cfg.natural_mortality,
        cfg.density_mortality,
        cfg.transition_rates,
        b5_scaled,
        b6_scaled
    )

    # Forward integrate DDE system
    t_start, t_end = tspan
    times = collect(t_start:dt:t_end)
    nt = length(times)

    states = zeros(Float64, nt, cur_cfg.num_stages)
    u_init = copy(cur_cfg.k_carrying_capacity)
    states[1, :] .= u_init

    for i in 1:(nt - 1)
        t_curr = times[i]
        u_curr = states[i, :]

        # Look up 1-year lagged state
        t_lag1 = t_curr - 1.0
        idx_lag1 = findlast(t -> t <= t_lag1, times[1:i])
        u_lag1 = isnothing(idx_lag1) ? u_init : states[idx_lag1, :]

        # Look up maternal reproductive lag
        u_lag_mat = zeros(Float64, length(cur_cfg.lags))
        for (k_idx, lag_val) in enumerate(cur_cfg.lags)
            t_lag_k = t_curr - lag_val
            idx_k = findlast(t -> t <= t_lag_k, times[1:i])
            u_lag_mat[k_idx] = isnothing(idx_k) ? u_init[6] : states[idx_k, 6]
        end

        du = dde_derivatives(u_curr, u_lag1, u_lag_mat, cur_cfg)
        u_next = max.(1e-6, u_curr .+ dt .* du)

        for (r_idx, r_time) in enumerate(removals_times)
            if t_curr <= r_time < t_curr + dt
                u_next[1] = max(1e-6, u_next[1] - removals_amount[r_idx])
            end
        end

        states[i + 1, :] .= u_next
    end

    # Observation likelihood for survey matches
    for k in 1:nt_survey
        t_target = survey_times[k]
        idx_t = findmin(abs.(times .- t_target))[2]
        for s in 1:nstages
            y_obs = survey_data[k, s]
            if !ismissing(y_obs) && isfinite(y_obs) && y_obs > 0.0
                mu_s = max(1e-6, prior_q0 + prior_q1 * states[idx_t, s])
                y_obs ~ LogNormal(log(mu_s), obs_sd)
            end
        end
    end

    return (states = states, times = times, k_scale = k_scale, rec_scale = rec_scale)
end
