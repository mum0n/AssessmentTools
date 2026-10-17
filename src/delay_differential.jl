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
    load_dde_config(path::AbstractString) -> DdeModelConfig

Read a DDE configuration from a TOML file. Every field of [`DdeModelConfig`](@ref)
is required, and the file is the single place where these parameters are stated; no
value is compiled into the source.

Expected keys, under `[dde]`:

| key | meaning |
|:---|:---|
| `num_stages` | number of biological stages |
| `lags` | constant maturation delays in years |
| `k_carrying_capacity` | per-stage asymptotic abundance bounds ``K_s`` |
| `natural_mortality` | per-stage baseline mortality ``d_s`` |
| `density_mortality` | per-stage density-dependent mortality ``d_{2,s}`` |
| `transition_rates` | per-stage moult transition rates ``v_s`` |
| `reproduction_weights_m` | male recruitment coefficients ``b_5`` |
| `reproduction_weights_f` | female recruitment coefficients ``b_6`` |

Stage order is terminal (stage 1, commercial legal males) to youngest (stage 5, instar 9),
with mature females as stage 6; see [`DdeModelConfig`](@ref).
"""
function load_dde_config(path::AbstractString)
    cfg = Config.parse_file(path)
    d = Config.require_section(cfg, "dde", path)
    where = "$path section [dde]"
    n = Config.load_int(d, "num_stages", where)
    n >= 2 || error("'num_stages' in $where must be at least 2, got $n.")

    k = Config.load_reals(d, "k_carrying_capacity", where)
    dmort = Config.load_reals(d, "natural_mortality", where)
    ddens = Config.load_reals(d, "density_mortality", where)
    lags = Config.load_reals(d, "lags", where)
    v = Config.load_reals(d, "transition_rates", where)
    b5 = Config.load_reals(d, "reproduction_weights_m", where)
    b6 = Config.load_reals(d, "reproduction_weights_f", where)

    for (nm, s) in (("k_carrying_capacity", k), ("natural_mortality", dmort),
                    ("density_mortality", ddens))
        length(s) == n ||
            error("'$nm' in $where must have $n entries (one per stage), got $(length(s)).")
    end
    for x in k
        x > 0 || error("'k_carrying_capacity' in $where must be strictly positive (got $x).")
    end
    for x in union(dmort, ddens)
        x >= 0 || error("Mortality values in $where must be non-negative (got $x).")
    end

    return DdeModelConfig(n, lags, k, dmort, ddens, v, b5, b6)
end

"""
    dde_derivatives(u, u_lag_1, u_lag_mat, p::DdeModelConfig, habitat_scalar = ones(p.num_stages)) -> Vector{Float64}

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

where ``d_{r,s} = d_s u_s + d_{2,s} \\frac{u_s^2}{H_s(t)}`` and ``u_{\\text{mat}}(t-\\tau)``
is the female stage 6 history at the maturation lags ``\\tau \\in (7, 8, 9)``.

# Stage indices

The flux of growth runs from stage 5 to stage 1, so stage 1 is the terminal
(oldest) male class, stage 5 receives recruitment, and stage 6 is the terminal
female class:

| index | stage (`M` label) | description                     |
|:-----:|:------------------|:--------------------------------|
| 1     | `M0`              | commercial legal males (≥ 95 mm) |
| 2     | `M1`              | pre-recruit males (instar 12)   |
| 3     | `M2`              | older immature males (instar 11) |
| 4     | `M3`              | younger immature males (instar 10) |
| 5     | `M4`              | newest male instars (instar 9)  |
| 6     | `f_mat`           | mature females                  |

Impulse removals applied by [`simulate_dde_system`](@ref) should target stage 1,
the legal commercial class, and `extract_annual_recruitment` reports stage 2 by
default, the newly recruitable class.
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
    _hermite_interpolate(t_target::Real, times::AbstractVector{<:Real},
                         states::AbstractMatrix{<:Real},
                         derivatives::AbstractMatrix{<:Real},
                         u_init::AbstractVector{<:Real},
                         idx_upper::Int) -> Vector{Float64}

Continuous cubic Hermite polynomial interpolation of state history at `t_target`.
Given state values `u_0, u_1` and time derivatives `m_0, m_1` at boundary grid points
`t_0, t_1`, the interpolated state is:

```math
u(t) = (2s^3 - 3s^2 + 1)u_0 + (s^3 - 2s^2 + s)h m_0 + (-2s^3 + 3s^2)u_1 + (s^3 - s^2)h m_1
```
where `h = t_1 - t_0` and `s = (t_target - t_0) / h \\in [0, 1]`.

When `t_target < times[1]`, the historical state is initialized to `u_init`.
"""
function _hermite_interpolate(t_target::Real, times::AbstractVector{<:Real},
                              states::AbstractMatrix{<:Real},
                              derivatives::AbstractMatrix{<:Real},
                              u_init::AbstractVector{<:Real},
                              idx_upper::Int)
    if t_target <= times[1] || idx_upper <= 1
        return copy(u_init)
    end

    i0 = idx_upper - 1
    i1 = idx_upper
    t0 = times[i0]
    t1 = times[i1]
    h = t1 - t0
    if h <= 1.0e-12
        return copy(states[i0, :])
    end

    s = (t_target - t0) / h
    s = clamp(s, 0.0, 1.0)
    s2 = s * s
    s3 = s2 * s

    # Cubic Hermite basis functions
    h00 = 2.0 * s3 - 3.0 * s2 + 1.0
    h10 = (s3 - 2.0 * s2 + s) * h
    h01 = -2.0 * s3 + 3.0 * s2
    h11 = (s3 - s2) * h

    n_stages = size(states, 2)
    u_interp = Vector{Float64}(undef, n_stages)
    for j in 1:n_stages
        u0 = states[i0, j]
        u1 = states[i1, j]
        m0 = derivatives[i0, j]
        m1 = derivatives[i1, j]
        u_interp[j] = h00 * u0 + h10 * m0 + h01 * u1 + h11 * m1
    end
    return u_interp
end

"""
    simulate_dde_system(cfg::DdeModelConfig, tspan::Tuple{Real, Real};
                         dt::Real = 1.0 / 12.0, removals_times = Float64[],
                         removals_amount = Float64[]) -> NamedTuple

Integrate continuous multi-stage delay differential equations with impulsive catch removals:
- Supports intra-annual time steps (e.g. monthly: ``dt = 1/12``).
- Continuous cubic Hermite history interpolation ensures accurate phase tracking during
  cohort propagation and mitigates stiffness induced by quadratic density-dependent mortality.
- Integrates deterministic catch pulses using impulsive state subtractions.
"""
function simulate_dde_system(cfg::DdeModelConfig, tspan::Tuple{Real, Real};
                             dt::Real = 1.0 / 12.0,
                             removals_times::AbstractVector{<:Real} = Float64[],
                             removals_amount::AbstractVector{<:Real} = Float64[])
    t_start, t_end = tspan
    times = collect(t_start:dt:t_end)
    nt = length(times)

    # Storage for states and derivatives across stages
    states = zeros(Float64, nt, cfg.num_stages)
    derivs = zeros(Float64, nt, cfg.num_stages)

    # Initialize history with equilibrium carrying capacity fractions
    u_init = copy(cfg.k_carrying_capacity)
    states[1, :] .= u_init

    # Compute initial derivative vector at t = t_start with u_lag = u_init
    derivs[1, :] .= dde_derivatives(u_init, u_init, fill(u_init[6], length(cfg.lags)), cfg)

    # Discrete step integration with continuous cubic Hermite history interpolation
    for i in 1:(nt - 1)
        t_curr = times[i]
        u_curr = states[i, :]

        # Look up 1-year lagged state via continuous Hermite interpolation
        t_lag1 = t_curr - 1.0
        if t_lag1 <= times[1]
            u_lag1 = copy(u_init)
        else
            idx_lag1 = searchsortedlast(view(times, 1:i), t_lag1)
            idx_upper = min(i, idx_lag1 + 1)
            u_lag1 = _hermite_interpolate(t_lag1, times, states, derivs, u_init, idx_upper)
        end

        # Look up maternal reproductive lag (e.g., ~8 years)
        u_lag_mat = zeros(Float64, length(cfg.lags))
        for (k, lag_val) in enumerate(cfg.lags)
            t_lag_k = t_curr - lag_val
            if t_lag_k <= times[1]
                u_lag_mat[k] = u_init[6]
            else
                idx_k = searchsortedlast(view(times, 1:i), t_lag_k)
                idx_upper_k = min(i, idx_k + 1)
                u_interp_k = _hermite_interpolate(t_lag_k, times, states, derivs,
                                                  u_init, idx_upper_k)
                u_lag_mat[k] = u_interp_k[6]
            end
        end

        # Derivatives at current step
        du = dde_derivatives(u_curr, u_lag1, u_lag_mat, cfg)
        derivs[i, :] .= du

        # Forward integration step
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

    # Record final derivatives
    t_last = times[nt]
    t_lag1_end = t_last - 1.0
    u_lag1_end = t_lag1_end <= times[1] ? copy(u_init) :
                 _hermite_interpolate(t_lag1_end, times, states, derivs, u_init,
                                      min(nt, searchsortedlast(times, t_lag1_end) + 1))
    u_lag_mat_end = zeros(Float64, length(cfg.lags))
    for (k, lag_val) in enumerate(cfg.lags)
        t_lag_k = t_last - lag_val
        u_lag_mat_end[k] = t_lag_k <= times[1] ? u_init[6] :
                           _hermite_interpolate(t_lag_k, times, states, derivs, u_init,
                                                min(nt, searchsortedlast(times, t_lag_k) + 1))[6]
    end
    derivs[nt, :] .= dde_derivatives(states[nt, :], u_lag1_end, u_lag_mat_end, cfg)

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
                                                  obs_cv::Real = 0.25,
                                                  prior_k_scale_mu::Real = 1.0,
                                                  prior_k_scale_sd::Real = 0.25,
                                                  prior_rec_scale_mu::Real = 1.0,
                                                  prior_rec_scale_sd::Real = 0.25,
                                                  scale_lower::Real = 0.2,
                                                  scale_upper::Real = 5.0,
                                                  prior_obs_sd_sd::Real = 0.05,
                                                  obs_sd_lower::Real = 0.05,
                                                  obs_sd_upper::Real = 0.60)
    nt_survey, nstages = size(survey_data)
    nstages == cfg.num_stages || error("survey_data columns must match cfg.num_stages.")

    # Priors
    # Scaling factor for carrying capacity
    k_scale ~ truncated(Normal(prior_k_scale_mu, prior_k_scale_sd), scale_lower, scale_upper)
    # Scaling factor for maternal reproduction lag
    rec_scale ~ truncated(Normal(prior_rec_scale_mu, prior_rec_scale_sd),
                          scale_lower, scale_upper)
    # Observation error standard deviation
    obs_sd ~ truncated(Normal(obs_cv, prior_obs_sd_sd), obs_sd_lower, obs_sd_upper)

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
    derivs = zeros(Float64, nt, cur_cfg.num_stages)
    u_init = copy(cur_cfg.k_carrying_capacity)
    states[1, :] .= u_init
    derivs[1, :] .= dde_derivatives(u_init, u_init, fill(u_init[6], length(cur_cfg.lags)), cur_cfg)

    for i in 1:(nt - 1)
        t_curr = times[i]
        u_curr = states[i, :]

        # Look up 1-year lagged state via continuous cubic Hermite interpolation
        t_lag1 = t_curr - 1.0
        if t_lag1 <= times[1]
            u_lag1 = copy(u_init)
        else
            idx_lag1 = searchsortedlast(view(times, 1:i), t_lag1)
            idx_upper = min(i, idx_lag1 + 1)
            u_lag1 = _hermite_interpolate(t_lag1, times, states, derivs, u_init, idx_upper)
        end

        # Look up maternal reproductive lag
        u_lag_mat = zeros(Float64, length(cur_cfg.lags))
        for (k_idx, lag_val) in enumerate(cur_cfg.lags)
            t_lag_k = t_curr - lag_val
            if t_lag_k <= times[1]
                u_lag_mat[k_idx] = u_init[6]
            else
                idx_k = searchsortedlast(view(times, 1:i), t_lag_k)
                idx_upper_k = min(i, idx_k + 1)
                u_lag_mat[k_idx] = _hermite_interpolate(t_lag_k, times, states, derivs,
                                                        u_init, idx_upper_k)[6]
            end
        end

        du = dde_derivatives(u_curr, u_lag1, u_lag_mat, cur_cfg)
        derivs[i, :] .= du
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
