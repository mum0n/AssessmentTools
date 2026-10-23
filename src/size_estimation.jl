# ------------------------------------------------------------------------------------
# Size-structured estimation: Turing-based inference (MCMC, VI, MAP, ML).
# Uses InferenceBridge for a uniform interface across all engines.
# ------------------------------------------------------------------------------------

using Turing
using Distributions
using ..InferenceBridge: fit_map, sample_posterior, fit_vi, estimate, parameter_dimension
using Optim

"""
    size_msy_reference_points(model::SizeStructuredModel) -> NamedTuple

Compute MSY reference points for a size-structured model.

Uses the equilibrium condition: at MSY, dY/dF = 0 where Y = F * B (catch).

Returns:
- `bmsy`: Biomass at MSY
- `msy`: Maximum sustainable yield
- `fmsy`: Fishing mortality at MSY
- `umsy`: Exploitation rate at MSY
"""
function size_msy_reference_points(model::SizeStructuredModel)
    n = model.n_size
    midpoints = [sc.midpoint for sc in model.size_classes]
    w = model.w_at_size[1, :]
    m_vec = model.m_size[1, :]

    # Unfished biomass: survivorship under natural mortality
    surv = ones(Float64, n)
    for i in 2:n
        surv[i] = surv[i-1] * exp(-m_vec[i-1])
    end
    if m_vec[n] > 0
        surv[n] /= (1 - exp(-m_vec[n]))
    end

    # Catch selectivity from first fleet (or flat)
    sel_params = size(model.sel_params, 1) > 0 ? model.sel_params[1, :] : Float64[]
    if length(sel_params) >= 2 && any(!=(0.0), sel_params)
        sel = size_selectivity(midpoints, sel_params; type = :logistic)
    else
        sel = ones(Float64, n)
    end

    # Find F that maximizes yield via finite differences
    fmsy = model.m
    for _ in 1:50
        surv_f = ones(Float64, n)
        for i in 2:n
            surv_f[i] = surv_f[i-1] * exp(-(m_vec[i-1] + fmsy * sel[i-1]))
        end
        if m_vec[n] + fmsy * sel[n] > 0
            surv_f[n] /= (1 - exp(-(m_vec[n] + fmsy * sel[n])))
        end
        y_f = fmsy * sum(surv_f .* sel .* w)

        dF = 1e-6
        F2 = fmsy + dF
        surv_f2 = ones(Float64, n)
        for i in 2:n
            surv_f2[i] = surv_f2[i-1] * exp(-(m_vec[i-1] + F2 * sel[i-1]))
        end
        if m_vec[n] + F2 * sel[n] > 0
            surv_f2[n] /= (1 - exp(-(m_vec[n] + F2 * sel[n])))
        end
        y_f2 = F2 * sum(surv_f2 .* sel .* w)
        dy_dF = (y_f2 - y_f) / dF
        if abs(dy_dF) < 1e-10
            break
        end
        fmsy -= dy_dF * 0.01
        fmsy = clamp(fmsy, 1e-6, 5.0)
    end

    surv_f = ones(Float64, n)
    for i in 2:n
        surv_f[i] = surv_f[i-1] * exp(-(m_vec[i-1] + fmsy * sel[i-1]))
    end
    if m_vec[n] + fmsy * sel[n] > 0
        surv_f[n] /= (1 - exp(-(m_vec[n] + fmsy * sel[n])))
    end
    bmsy = sum(surv_f .* w)
    msy = fmsy * sum(surv_f .* sel .* w)

    return (bmsy = bmsy, msy = msy, fmsy = fmsy, umsy = msy / (bmsy + 1e-10))
end

"""
    bayesian_size_structured(obs_catch::AbstractVector{<:Real},
                             obs_indices::AbstractVector{<:Real},
                             size_comps::Vector{SizeCompositionData};
                             prior_r::Tuple{Real, Real} = (0.35, 0.15),
                             prior_k::Tuple{Real, Real} = (100.0, 30.0),
                             prior_q::Tuple{Real, Real} = (1.0, 0.25),
                             prior_proc_sd::Real = 0.10,
                             proc_sd_range::Tuple{Real, Real} = (0.01, 0.40),
                             obs_sd_range::Tuple{Real, Real} = (0.02, 0.50)) -> DynamicPPL.Model

Bayesian size-structured model in Turing.jl. Uses multiple likelihoods:
- Catch: lognormal on total catch
- Survey: lognormal on survey index
- Size composition: multinomial on length frequencies

This is the single model definition that drives all inference engines (MCMC, VI, MAP, ML)
via InferenceBridge.
"""
@model function bayesian_size_structured(obs_catch::AbstractVector{<:Real},
                                          obs_indices::AbstractVector{<:Real},
                                          size_comps::Vector{SizeCompositionData};
                                          prior_r::Tuple{Real, Real} = (0.35, 0.15),
                                          prior_k::Tuple{Real, Real} = (100.0, 30.0),
                                          prior_q::Tuple{Real, Real} = (1.0, 0.25),
                                          prior_proc_sd::Real = 0.10,
                                          proc_sd_range::Tuple{Real, Real} = (0.01, 0.40),
                                          obs_sd_range::Tuple{Real, Real} = (0.02, 0.50))
    nt = length(obs_catch)
    length(obs_indices) == nt || error("obs_catch and obs_indices must have same length.")

    # Priors
    k ~ truncated(Normal(prior_k[1], prior_k[2]), 1.0, 10.0 * prior_k[1])
    r ~ truncated(Normal(prior_r[1], prior_r[2]), 0.01, 1.5)
    q ~ truncated(Normal(prior_q[1], prior_q[2]), 0.01, 10.0)
    proc_sd ~ truncated(Normal(prior_proc_sd, 0.05), proc_sd_range[1], proc_sd_range[2])
    obs_sd ~ truncated(Normal(0.15, 0.05), obs_sd_range[1], obs_sd_range[2])

    # State dynamics (Schaefer-like on biomass, normalized)
    m = Vector{Float64}(undef, nt)
    m[1] ~ truncated(Normal(0.5, 0.2), 0.01, 1.0)
    for t in 1:(nt - 1)
        mu_safe = max(0.01, min(1.0, m[t] + r * m[t] * (1.0 - m[t]) - obs_catch[t] / k))
        m[t + 1] ~ truncated(Normal(mu_safe, proc_sd), 0.01, 1.0)
    end

    # Catch likelihood (lognormal)
    for t in 1:nt
        if obs_catch[t] > 0
            mu_c = max(1e-6, k * m[t] * 0.5)
            obs_catch[t] ~ LogNormal(log(mu_c), obs_sd)
        end
    end

    # Survey likelihood (lognormal)
    for t in 1:nt
        if obs_indices[t] > 0
            mu_i = max(1e-6, q * k * m[t])
            obs_indices[t] ~ LogNormal(log(mu_i), obs_sd)
        end
    end

    # Size composition likelihood (multinomial via Distributions.jl)
    for sc in size_comps
        obs_counts = sc.counts
        n_fish = sum(obs_counts)
        n_fish > 0 || continue
        # Predicted proportions (from model; simplified as uniform here)
        pred_prop = fill(1.0 / length(obs_counts), length(obs_counts))
        # Normalize to sum to 1
        pred_prop ./= sum(pred_prop)
        # Multinomial likelihood
        obs_counts ~ Multinomial(n_fish, pred_prop)
    end

    return (biomass = k .* m, r = r, k = k, q = q, proc_sd = proc_sd, obs_sd = obs_sd)
end

"""
    fit_size_structured(model::SizeStructuredModel,
                        obs_catch::AbstractVector{<:Real},
                        obs_indices::AbstractVector{<:Real},
                        size_comps::Vector{SizeCompositionData};
                        method::Symbol = :map,
                        x0::Union{Nothing, AbstractVector{<:Real}} = nothing,
                        kwargs...) -> Any

Unified fitting interface. Dispatches to InferenceBridge:
- `method = :map` → MAP/ML point estimate (Optim L-BFGS)
- `method = :mcmc` → posterior sampling (NUTS)
- `method = :vi` → variational inference (AdvancedVI)

Returns the result of the chosen engine. For `:map`, returns `MapFitResult`.
For `:mcmc`, returns MCMC chains. For `:vi`, returns `ViResult`.
"""
function fit_size_structured(obs_catch::AbstractVector{<:Real},
                             obs_indices::AbstractVector{<:Real},
                             size_comps::Vector{SizeCompositionData};
                             method::Symbol = :map,
                             x0::Union{Nothing, AbstractVector{<:Real}} = nothing,
                             kwargs...)
    turing_model = bayesian_size_structured(obs_catch, obs_indices, size_comps)

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
    sample_size_structured(obs_catch, obs_indices, size_comps;
                           sampler = nothing, n_samples = 1000,
                           n_chains = 1, rng = Random.default_rng(),
                           kwargs...) -> Any

Draw posterior samples from the size-structured Bayesian model using MCMC.
Convenience wrapper around `InferenceBridge.sample_posterior`.
"""
function sample_size_structured(obs_catch::AbstractVector{<:Real},
                                obs_indices::AbstractVector{<:Real},
                                size_comps::Vector{SizeCompositionData};
                                sampler = nothing, n_samples::Integer = 1000,
                                n_chains::Integer = 1,
                                rng::AbstractRNG = Random.default_rng(),
                                kwargs...)
    turing_model = bayesian_size_structured(obs_catch, obs_indices, size_comps)
    smp = isnothing(sampler) ? NUTS() : sampler
    return InferenceBridge.sample_posterior(turing_model, smp;
        n_samples = n_samples, n_chains = n_chains, rng = rng)
end

"""
    vi_size_structured(obs_catch, obs_indices, size_comps;
                       family = :meanfield, max_iter = 100,
                       n_samples = 1000, rng = Random.default_rng(),
                       kwargs...) -> Any

Fit a variational approximation to the size-structured Bayesian model.
Convenience wrapper around `InferenceBridge.fit_vi`.
"""
function vi_size_structured(obs_catch::AbstractVector{<:Real},
                            obs_indices::AbstractVector{<:Real},
                            size_comps::Vector{SizeCompositionData};
                            family::Symbol = :meanfield,
                            max_iter::Integer = 100,
                            n_samples::Integer = 1000,
                            rng::AbstractRNG = Random.default_rng(),
                            kwargs...)
    turing_model = bayesian_size_structured(obs_catch, obs_indices, size_comps)
    return InferenceBridge.fit_vi(turing_model; family = family,
        max_iter = max_iter, n_samples = n_samples, rng = rng)
end