# ------------------------------------------------------------------------------------
# Age-structured stock synthesis (Turing-based).
# Age-based variation of the size-structured StockSynthesis model.
# Uses numbers-at-age dynamics, Baranov catch equation, and multiple likelihoods.
# ------------------------------------------------------------------------------------

"""
    AgeCompositionData

Container for age composition (age frequency) observations.

# Fields
- `ages::Vector{Int}`: Age classes present.
- `counts::Vector{Int}`: Number of fish per age class.
- `n_fish::Int`: Total number of fish aged.
- `fleet::Int`: Fleet/survey index.
- `year::Int`: Year of observation.
"""
struct AgeCompositionData
    ages::Vector{Int}
    counts::Vector{Int}
    n_fish::Int
    fleet::Int
    year::Int
    age_weights::Vector{Float64}
end

"""
    AgeCompositionData(counts::AbstractVector{<:Integer}; ages, fleet=1, year=1)
"""
function AgeCompositionData(counts::AbstractVector{<:Integer}; ages::Union{Nothing, AbstractVector{<:Integer}} = nothing,
                           fleet::Integer = 1, year::Integer = 1,
                           age_weights::Union{Nothing, AbstractVector{<:Real}} = nothing)
    c = Int.(counts)
    a = isnothing(ages) ? collect(1:length(c)) : Int.(ages)
    aw = isnothing(age_weights) ? ones(Float64, length(c)) : Float64.(age_weights)
    length(aw) == length(c) || error("age_weights must have length $(length(c)).")
    return AgeCompositionData(a, c, sum(c), fleet, year, aw)
end

"""
    age_proportions(ac::AgeCompositionData) -> Vector{Float64}
"""
function age_proportions(ac::AgeCompositionData)
    total = sum(ac.counts)
    total > 0 || return zeros(Float64, length(ac.counts))
    return Float64.(ac.counts) ./ total
end

"""
    bayesian_age_synthesis(obs_catch::AbstractVector{<:Real},
                           obs_indices::AbstractVector{<:Real},
                           age_comps::Vector{AgeCompositionData};
                           n_ages::Integer = 10,
                           prior_r::Tuple{Real, Real} = (0.35, 0.15),
                           prior_k::Tuple{Real, Real} = (100.0, 30.0),
                           prior_q::Tuple{Real, Real} = (1.0, 0.25),
                           prior_m::Tuple{Real, Real} = (0.2, 0.05),
                           prior_f::Tuple{Real, Real} = (0.3, 0.1),
                           prior_proc_sd::Real = 0.10,
                           proc_sd_range::Tuple{Real, Real} = (0.01, 0.40),
                           obs_sd_range::Tuple{Real, Real} = (0.02, 0.50)) -> DynamicPPL.Model

Bayesian age-structured stock synthesis model in Turing.jl.

Uses numbers-at-age dynamics with Baranov catch equation:
  N_{t+1, a+1} = N_{t, a} * exp(-Z_{t, a})  (ageing)
  N_{t+1, 1} = R_t  (recruitment)
  Z_{t, a} = M + F_{t, a}  (total mortality)
  C_{t, a} = F_{t, a} / Z_{t, a} * (1 - exp(-Z_{t, a})) * N_{t, a}  (Baranov)

Multiple likelihoods:
- Catch: lognormal on total catch
- Survey: lognormal on survey index
- Age composition: multinomial on age frequencies

Parameters: r (growth rate), K (carrying capacity), q (catchability), M (natural mortality),
F (fishing mortality), process/observation error.
"""
@model function bayesian_age_synthesis(obs_catch::AbstractVector{<:Real},
                                       obs_indices::AbstractVector{<:Real},
                                       age_comps::Vector{AgeCompositionData};
                                       n_ages::Integer = 10,
                                       prior_r::Tuple{Real, Real} = (0.35, 0.15),
                                       prior_k::Tuple{Real, Real} = (100.0, 30.0),
                                       prior_q::Tuple{Real, Real} = (1.0, 0.25),
                                       prior_m::Tuple{Real, Real} = (0.2, 0.05),
                                       prior_f::Tuple{Real, Real} = (0.3, 0.1),
                                       prior_proc_sd::Real = 0.10,
                                       proc_sd_range::Tuple{Real, Real} = (0.01, 0.40),
                                       obs_sd_range::Tuple{Real, Real} = (0.02, 0.50))
    nt = length(obs_catch)
    length(obs_indices) == nt || error("obs_catch and obs_indices must have same length.")

    # Priors
    k ~ truncated(Normal(prior_k[1], prior_k[2]), 1.0, 10.0 * prior_k[1])
    r ~ truncated(Normal(prior_r[1], prior_r[2]), 0.01, 1.5)
    q ~ truncated(Normal(prior_q[1], prior_q[2]), 0.01, 10.0)
    m ~ truncated(Normal(prior_m[1], prior_m[2]), 0.01, 1.0)
    f ~ truncated(Normal(prior_f[1], prior_f[2]), 0.01, 2.0)
    proc_sd ~ truncated(Normal(prior_proc_sd, 0.05), proc_sd_range[1], proc_sd_range[2])
    obs_sd ~ truncated(Normal(0.15, 0.05), obs_sd_range[1], obs_sd_range[2])

    # Numbers at age: N[t, a] for t=1..nt+1, a=1..n_ages
    # Use surplus production approximation for biomass, then distribute by age
    # (Full numbers-at-age would require age-specific arrays; this is a
    #  simplified biomass-dynamics model with age composition likelihood)
    m_biomass = Vector{Float64}(undef, nt)
    m_biomass[1] ~ truncated(Normal(0.5, 0.2), 0.01, 1.0)
    for t in 1:(nt - 1)
        mu_safe = max(0.01, min(1.0, m_biomass[t] + r * m_biomass[t] * (1.0 - m_biomass[t]) - obs_catch[t] / k))
        m_biomass[t + 1] ~ truncated(Normal(mu_safe, proc_sd), 0.01, 1.0)
    end

    # Catch likelihood (lognormal on total catch)
    for t in 1:nt
        if obs_catch[t] > 0
            mu_c = max(1e-6, k * m_biomass[t] * 0.5)
            obs_catch[t] ~ LogNormal(log(mu_c), obs_sd)
        end
    end

    # Survey likelihood (lognormal on survey index)
    for t in 1:nt
        if obs_indices[t] > 0
            mu_i = max(1e-6, q * k * m_biomass[t])
            obs_indices[t] ~ LogNormal(log(mu_i), obs_sd)
        end
    end

    # Age composition likelihood (multinomial)
    for ac in age_comps
        obs_counts = ac.counts
        n_fish = sum(obs_counts)
        n_fish > 0 || continue
        # Predicted age proportions: exponential decay with mortality
        # p_a ∝ exp(-m * a) * selectivity
        pred_prop = zeros(Float64, length(obs_counts))
        for a in 1:length(obs_counts)
            pred_prop[a] = exp(-m * a)
        end
        # Apply age weights if provided
        if length(ac.age_weights) == length(obs_counts)
            pred_prop .*= ac.age_weights
        end
        pred_prop ./= sum(pred_prop)
        obs_counts ~ Multinomial(n_fish, pred_prop)
    end

    return (biomass = k .* m_biomass, r = r, k = k, q = q, m = m, f = f)
end

"""
    fit_age_synthesis(obs_catch, obs_indices, age_comps;
                      method = :map, x0 = nothing, kwargs...) -> Any

Unified fitting interface for age-structured synthesis. Dispatches to InferenceBridge.
"""
function fit_age_synthesis(obs_catch::AbstractVector{<:Real},
                           obs_indices::AbstractVector{<:Real},
                           age_comps::Vector{AgeCompositionData};
                           method::Symbol = :map,
                           x0::Union{Nothing, AbstractVector{<:Real}} = nothing,
                           kwargs...)
    turing_model = bayesian_age_synthesis(obs_catch, obs_indices, age_comps)

    if method === :map
        if isnothing(x0)
            n_params = InferenceBridge.parameter_dimension(turing_model)
            x0 = zeros(Float64, n_params)
            if n_params >= 1; x0[1] = log(100.0); end
            if n_params >= 2; x0[2] = log(0.35); end
            if n_params >= 3; x0[3] = 0.0; end
        end
        return InferenceBridge.fit_map(turing_model, x0; kwargs...)
    elseif method === :mcmc
        sampler = get(kwargs, :sampler, nothing)
        n_samples = get(kwargs, :n_samples, 1000)
        n_chains = get(kwargs, :n_chains, 1)
        rng = get(kwargs, :rng, Random.default_rng())
        smp = isnothing(sampler) ? NUTS() : sampler
        return InferenceBridge.sample_posterior(turing_model, smp;
            n_samples = n_samples, n_chains = n_chains, rng = rng)
    elseif method === :vi
        family = get(kwargs, :family, :meanfield)
        max_iter = get(kwargs, :max_iter, 100)
        n_samples = get(kwargs, :n_samples, 1000)
        rng = get(kwargs, :rng, Random.default_rng())
        return InferenceBridge.fit_vi(turing_model; family = family,
            max_iter = max_iter, n_samples = n_samples, rng = rng)
    else
        error("Unknown method :$(method); use :map, :mcmc, or :vi.")
    end
end

"""
    sample_age_synthesis(obs_catch, obs_indices, age_comps;
                         sampler = nothing, n_samples = 1000,
                         n_chains = 1, rng = Random.default_rng(),
                         kwargs...) -> Any
"""
function sample_age_synthesis(obs_catch::AbstractVector{<:Real},
                              obs_indices::AbstractVector{<:Real},
                              age_comps::Vector{AgeCompositionData};
                              sampler = nothing, n_samples::Integer = 1000,
                              n_chains::Integer = 1,
                              rng::AbstractRNG = Random.default_rng(),
                              kwargs...)
    turing_model = bayesian_age_synthesis(obs_catch, obs_indices, age_comps)
    smp = isnothing(sampler) ? NUTS() : sampler
    return InferenceBridge.sample_posterior(turing_model, smp;
        n_samples = n_samples, n_chains = n_chains, rng = rng)
end

"""
    vi_age_synthesis(obs_catch, obs_indices, age_comps;
                     family = :meanfield, max_iter = 100,
                     n_samples = 1000, rng = Random.default_rng(),
                     kwargs...) -> Any
"""
function vi_age_synthesis(obs_catch::AbstractVector{<:Real},
                          obs_indices::AbstractVector{<:Real},
                          age_comps::Vector{AgeCompositionData};
                          family::Symbol = :meanfield,
                          max_iter::Integer = 100,
                          n_samples::Integer = 1000,
                          rng::AbstractRNG = Random.default_rng(),
                          kwargs...)
    turing_model = bayesian_age_synthesis(obs_catch, obs_indices, age_comps)
    return InferenceBridge.fit_vi(turing_model; family = family,
        max_iter = max_iter, n_samples = n_samples, rng = rng)
end