# ------------------------------------------------------------------------------------
# Biomass Dynamics & Discrete Logistic Surplus Production Models.
# Supports state-space surplus production simulation, observation matching,
# survey catchability calibrations, and multi-area data preparation.
# ------------------------------------------------------------------------------------

"""
Result of forward simulation or trajectory reconstruction from a discrete logistic model.

# Fields
- `years::Vector{Int}`: Model time steps / calendar years.
- `biomass::Vector{Float64}`: Unharvested / prefishery biomass trajectory ``B_t``.
- `biomass_post::Vector{Float64}`: Post-fishery biomass trajectory ``B_t - C_t``.
- `removals::Vector{Float64}`: Harvest removals / catches ``C_t``.
- `survey_pred::Vector{Float64}`: Predicted survey relative abundance index ``\\hat{I}_t``.
- `harvest_rate::Vector{Float64}`: Realized harvest fraction ``H_t = C_t / B_t``.
- `surplus_production::Vector{Float64}`: Annual surplus biomass production ``P_t``.
"""
struct LogisticDiscreteResult
    years::Vector{Int}
    biomass::Vector{Float64}
    biomass_post::Vector{Float64}
    removals::Vector{Float64}
    survey_pred::Vector{Float64}
    harvest_rate::Vector{Float64}
    surplus_production::Vector{Float64}
end

"""
    logistic_step(b::Real, r::Real, k::Real, c::Real) -> Real

Single forward time-step of the discrete Schaefer surplus production model:

```math
B_{t+1} = \\max\\left(0, B_t + r B_t \\left(1 - \\frac{B_t}{K}\\right) - C_t\\right)
```

# Inputs
- `b`: Current exploitable biomass ``B_t``.
- `r`: Intrinsic rate of population increase.
- `k`: Environmental carrying capacity ``K``.
- `c`: Harvest removals ``C_t``.
"""
function logistic_step(b::Real, r::Real, k::Real, c::Real)
    k > 0 || error("logistic_step: carrying capacity K must be strictly positive.")
    next_b = b + r * b * (1.0 - b / k) - c
    return next_b > 0.0 ? next_b : 0.0
end

"""
    logistic_step_frac(m::Real, r::Real, k::Real, c::Real) -> Real

Normalised (depletion scale ``m = B / K``) one-step transition of the discrete
logistic surplus production map, clamped below at zero:

```math
m_{t+1} = \\max\\left(0,\\; m_t + r\\,m_t\\left(1 - m_t\\right) - \\frac{C_t}{K}\\right)
```

Dividing [`logistic_step`](@ref) by ``K`` yields exactly this map, so both
formulations are numerically identical. This is the single definition used by the
deterministic simulators and by the Turing state-space model, which keeps the two
implementations consistent by construction.
"""
logistic_step_frac(m::Real, r::Real, k::Real, c::Real) = max(0.0, m + r * m * (1.0 - m) - c / k)

"""
    logistic_map_step(m::Real, r::Real, k::Real, c::Real) -> Real

Normalised map formulation on the unit depletion scale ``m_t = B_t / K``:

```math
m_{t+1} = \\max\\left(0,\\; m_t + r\\,m_t\\left(1 - m_t\\right) - \\frac{C_t}{K}\\right)
```
"""
function logistic_map_step(m::Real, r::Real, k::Real, c::Real)
    k > 0 || error("logistic_map_step: K must be strictly positive.")
    next_m = logistic_step_frac(m, r, k, c)
    return next_m
end

"""
    simulate_logistic_discrete(years::AbstractVector{<:Integer}, removals::AbstractVector{<:Real};
                               r::Real, k::Real, b0::Real = k, q::Real = 1.0,
                               survey_transition::Integer = 0) -> LogisticDiscreteResult

Simulate time-series trajectories under a discrete logistic production process.

# Arguments
- `years`: Vector of assessment years.
- `removals`: Annual catch removals corresponding to `years`.
- `r`: Intrinsic rate of population growth.
- `k`: Carrying capacity ``K``.
- `b0`: Initial biomass at year 1 (defaults to ``K``).
- `q`: Survey catchability scalar (where ``\\hat{I}_t = q \\cdot B_t``).
- `survey_transition`: Year index where survey shifted from pre-fishery (spring) to post-fishery (fall).
  If `0`, assumed post-fishery throughout.
"""
function simulate_logistic_discrete(years::AbstractVector{<:Integer},
                                    removals::AbstractVector{<:Real};
                                    r::Real, k::Real, b0::Real = k, q::Real = 1.0,
                                    survey_transition::Integer = 0)
    nt = length(years)
    length(removals) == nt || error("removals and years lengths must match.")

    b = zeros(Float64, nt)
    b_post = zeros(Float64, nt)
    s_pred = zeros(Float64, nt)
    hr = zeros(Float64, nt)
    p = zeros(Float64, nt)

    b[1] = b0
    for i in 1:nt
        rem = removals[i]
        b_post[i] = max(0.0, b[i] - rem)
        hr[i] = b[i] > 0.0 ? rem / b[i] : 0.0
        p[i] = r * b[i] * (1.0 - b[i] / k)

        # Survey observation expectations based on seasonal survey timing
        if survey_transition == 0
            # Post-fishery (fall)
            s_pred[i] = q * b_post[i]
        else
            if i < survey_transition
                # Spring survey (pre-fishery)
                s_pred[i] = q * b[i]
            elseif i == survey_transition
                # Transition year (average removals)
                avg_rem = i > 1 ? 0.5 * (removals[i - 1] + rem) : rem
                s_pred[i] = q * max(0.0, b[i] - avg_rem)
            else
                # Fall survey (post-fishery)
                s_pred[i] = q * b_post[i]
            end
        end

        if i < nt
            b[i + 1] = max(0.0, b_post[i] + p[i])
        end
    end

    return LogisticDiscreteResult(
        collect(years), b, b_post, collect(removals), s_pred, hr, p
    )
end

"""
    logistic_msy_reference_points(r::Real, k::Real) -> NamedTuple

Compute classic Schaefer maximum sustainable yield (MSY) biological reference points:
- ``B_{\\mathrm{MSY}} = K / 2``
- ``\\mathrm{MSY} = r K / 4``
- ``F_{\\mathrm{MSY}} = r / 2``
- ``U_{\\mathrm{MSY}} = 1 - \\exp(-F_{\\mathrm{MSY}})``
"""
function logistic_msy_reference_points(r::Real, k::Real)
    r > 0 || error("r must be strictly positive.")
    k > 0 || error("k must be strictly positive.")
    bmsy = 0.5 * k
    msy = 0.25 * r * k
    fmsy = 0.5 * r
    umsy = 1.0 - exp(-fmsy)
    return (bmsy = bmsy, msy = msy, fmsy = fmsy, umsy = umsy)
end

"""
    bayesian_logistic_discrete(removals, survey_obs;
                               survey_transition::Integer = 0,
                               prior_r = (0.35, 0.15),
                               prior_k = (100.0, 30.0),
                               prior_q = (1.0, 0.25),
                               prior_m0 = (0.8, 0.2),
                               proc_sd_range = (0.01, 0.40),
                               obs_sd_range = (0.02, 0.50))

Bayesian state-space discrete Schaefer surplus production model implemented in `Turing.jl`:

```math
\\begin{aligned}
m_1 &\\sim \\mathrm{TruncatedNormal}(\\mu_{m0}, \\sigma_{m0}, 0, 1) \\\\
m_{t+1} &\\sim \\mathrm{TruncatedNormal}\\left(m_t + r m_t (1 - m_t) - \\frac{C_t}{K}, \\sigma_p, 0, 1\\right) \\\\
I_t &\\sim \\mathrm{LogNormal}\\left(\\log(q K m_t), \\sigma_o\\right)
\\end{aligned}
```

# Arguments
- `removals`: Annual catch removals vector ``C_t``.
- `survey_obs`: Survey relative abundance index ``I_t`` (supports `missing` and `NaN`).
- `survey_transition`: Year index where survey shifted from pre-fishery (spring) to post-fishery (fall).
  If `0`, assumed post-fishery throughout.
- `prior_r`: Mean and standard deviation for intrinsic growth rate ``r``.
- `prior_k`: Mean and standard deviation for carrying capacity ``K``.
- `prior_q`: Mean and standard deviation for survey catchability scalar ``q``.
- `prior_m0`: Prior for initial relative depletion ``m_0 = B_1 / K``.
- `prior_proc_sd`, `prior_proc_sd_sd`: prior mean and standard deviation for the process
  error standard deviation; `prior_obs_sd`, `prior_obs_sd_sd` likewise for observation error.
- `proc_sd_range`, `obs_sd_range`: admissible bounds for those standard deviations.
- `k_lower`, `k_upper_factor`: admissible bounds for ``K``, as `prior_k[1] * k_upper_factor`.
- `r_lower`, `r_upper`, `q_lower`, `q_upper`, `m_lower`, `m_upper`: admissible bounds for
  ``r``, ``q`` and relative depletion.
- `b_floor`: small value guarding logarithms and divisions against zero.
"""
@model function bayesian_logistic_discrete(removals::AbstractVector{<:Real},
                                           survey_obs::AbstractVector;
                                           survey_transition::Integer = 0,
                                           prior_r::Tuple{Real, Real} = (0.35, 0.15),
                                           prior_k::Tuple{Real, Real} = (100.0, 30.0),
                                           prior_q::Tuple{Real, Real} = (1.0, 0.25),
                                           prior_m0::Tuple{Real, Real} = (0.8, 0.2),
                                           prior_proc_sd::Real = 0.10,
                                           prior_proc_sd_sd::Real = 0.05,
                                           prior_obs_sd::Real = 0.15,
                                           prior_obs_sd_sd::Real = 0.05,
                                           proc_sd_range::Tuple{Real, Real} = (0.01, 0.40),
                                           obs_sd_range::Tuple{Real, Real} = (0.02, 0.50),
                                           k_lower::Real = 1.0,
                                           k_upper_factor::Real = 10.0,
                                           r_lower::Real = 0.01,
                                           r_upper::Real = 1.5,
                                           q_lower::Real = 0.01,
                                           q_upper::Real = 10.0,
                                           m_lower::Real = 0.01,
                                           m_upper::Real = 1.0,
                                           b_floor::Real = 1.0e-4)
    nt = length(removals)
    length(survey_obs) == nt || error("removals and survey_obs lengths must match.")

    # Priors
    k ~ truncated(Normal(prior_k[1], prior_k[2]), k_lower, k_upper_factor * prior_k[1])
    r ~ truncated(Normal(prior_r[1], prior_r[2]), r_lower, r_upper)
    q ~ truncated(Normal(prior_q[1], prior_q[2]), q_lower, q_upper)
    proc_sd ~ truncated(Normal(prior_proc_sd, prior_proc_sd_sd),
                        proc_sd_range[1], proc_sd_range[2])
    obs_sd ~ truncated(Normal(prior_obs_sd, prior_obs_sd_sd),
                       obs_sd_range[1], obs_sd_range[2])

    m = Vector{Float64}(undef, nt)
    m[1] ~ truncated(Normal(prior_m0[1], prior_m0[2]), m_lower, m_upper)

    for i in 1:(nt - 1)
        # State transition: `logistic_step_frac`, the single definition of the
        # surplus production map, clamped to the admissible depletion range.
        mu_safe = max(m_lower, min(m_upper, logistic_step_frac(m[i], r, k, removals[i])))
        m[i + 1] ~ truncated(Normal(mu_safe, proc_sd), m_lower, m_upper)
    end

    # Observation likelihood with seasonal survey timing
    for i in 1:nt
        s_obs = survey_obs[i]
        if !ismissing(s_obs) && isfinite(s_obs) && s_obs > 0.0
            # Expected exploitable biomass according to survey timing
            b_exp = if survey_transition == 0
                max(b_floor, k * m[i] - removals[i])
            else
                if i < survey_transition
                    k * m[i]
                elseif i == survey_transition
                    c_avg = i > 1 ? 0.5 * (removals[i - 1] + removals[i]) : removals[i]
                    max(b_floor, k * m[i] - c_avg)
                else
                    max(b_floor, k * m[i] - removals[i])
                end
            end
            s_pred = max(b_floor, q * b_exp)
            s_obs ~ LogNormal(log(s_pred), obs_sd)
        end
    end

    return (biomass = k .* m, m = m, k = k, r = r, q = q)
end
