using Distributions

# ------------------------------------------------------------------------------------
# Supports multiple observation series (fleets, surveys) with different
# likelihood types per series.
# ------------------------------------------------------------------------------------

"""
    LikelihoodSpec

Specification for a single likelihood component.

# Fields
- `name::String`: Identifier for this likelihood component.
- `type::Symbol`: Likelihood type (:normal, :lognormal, :multinomial, :dirichlet_multinomial).
- `weight::Float64`: Weight in the combined objective (default 1.0).
- `data::Any`: The observed data for this component.
- `predicted::Any`: The model-predicted values for this component.
"""
struct LikelihoodSpec
    name::String
    type::Symbol
    weight::Float64
    data::Any
    predicted::Any
end

"""
    LikelihoodSpec(name, type, data, predicted; weight=1.0) -> LikelihoodSpec
"""
function LikelihoodSpec(name::AbstractString, type::Symbol, data, predicted; weight::Real = 1.0)
    return LikelihoodSpec(String(name), type, Float64(weight), data, predicted)
end

"""
    evaluate_likelihood(spec::LikelihoodSpec) -> Float64

Evaluate the negative log-likelihood for a single likelihood specification.
"""
function evaluate_likelihood(spec::LikelihoodSpec)
    d = spec.data
    p = spec.predicted
    w = spec.weight

    if spec.type == :normal
        # Normal NLL: sum of squared residuals
        resid = Float64.(d) .- Float64.(p)
        nll = 0.5 * sum(resid .^ 2) / (length(resid) > 0 ? length(resid) : 1)
        return w * nll
    elseif spec.type == :lognormal
        # Lognormal NLL
        d_safe = max.(Float64.(d), 1e-10)
        p_safe = max.(Float64.(p), 1e-10)
        resid = log.(d_safe) .- log.(p_safe)
        nll = 0.5 * sum(resid .^ 2)
        return w * nll
    elseif spec.type == :multinomial
        # Multinomial NLL via Distributions.jl
        obs = Int.(round.(Float64.(d)))
        pred = Float64.(p)
        po = sum(obs)
        pp = sum(pred)
        po > 0 || return 0.0
        pp > 0 || return Inf
        pred ./= pp
        mn = Multinomial(po, pred)
        return w * (-logpdf(mn, obs))
    elseif spec.type == :dirichlet_multinomial
        # Dirichlet-multinomial NLL via Distributions.jl
        obs = Int.(round.(Float64.(d)))
        pred = Float64.(p)
        po = sum(obs)
        pp = sum(pred)
        po > 0 || return 0.0
        pp > 0 || return Inf
        pred ./= pp
        alpha = po .* pred
        dm = DirichletMultinomial(po, alpha)
        return w * (-logpdf(dm, obs))
    elseif spec.type == :poisson
        # Poisson NLL
        d_safe = max.(Float64.(d), 0.0)
        p_safe = max.(Float64.(p), 1e-10)
        nll = sum(p_safe .- d_safe .* log.(p_safe))
        return w * nll
    else
        error("Unknown likelihood type: $(spec.type)")
    end
end

"""
    MultiSeriesLikelihood

Container for multiple observation series with different likelihoods.

# Fields
- `specs::Vector{LikelihoodSpec}`: List of likelihood specifications.
"""
struct MultiSeriesLikelihood
    specs::Vector{LikelihoodSpec}
end

"""
    MultiSeriesLikelihood() -> MultiSeriesLikelihood

Empty likelihood container.
"""
function MultiSeriesLikelihood()
    return MultiSeriesLikelihood(LikelihoodSpec[])
end

"""
    add_likelihood!(msl::MultiSeriesLikelihood, spec::LikelihoodSpec)

Add a likelihood specification to the container.
"""
function add_likelihood!(msl::MultiSeriesLikelihood, spec::LikelihoodSpec)
    push!(msl.specs, spec)
    return msl
end

"""
    total_nll(msl::MultiSeriesLikelihood) -> Float64

Compute the total negative log-likelihood across all series.
"""
function total_nll(msl::MultiSeriesLikelihood)
    return sum(evaluate_likelihood, msl.specs)
end

"""
    multi_series_residual(msl::MultiSeriesLikelihood) -> Vector{Float64}

Compute residuals for all series (for optimization).
"""
function multi_series_residual(msl::MultiSeriesLikelihood)
    residuals = Float64[]
    for spec in msl.specs
        d = Float64.(spec.data)
        p = Float64.(spec.predicted)
        if spec.type in (:normal, :lognormal)
            r = d .- p
            append!(residuals, r)
        end
    end
    return residuals
end

"""
    multi_series_likelihood(obs_series::Vector{<:AbstractVector},
                            pred_series::Vector{<:AbstractVector},
                            types::Vector{Symbol};
                            weights::Union{Nothing, Vector{<:Real}} = nothing,
                            names::Union{Nothing, Vector{<:AbstractString}} = nothing) -> MultiSeriesLikelihood

Construct a multi-series likelihood from parallel arrays of observations, predictions,
and likelihood types.
"""
function multi_series_likelihood(obs_series::Vector, pred_series::Vector,
                                 types::Vector{Symbol};
                                 weights::Union{Nothing, Vector{<:Real}} = nothing,
                                 names::Union{Nothing, Vector{<:AbstractString}} = nothing)
    n = length(obs_series)
    length(pred_series) == n || error("obs and pred series must have same length.")
    length(types) == n || error("types must have same length as series.")

    w = isnothing(weights) ? ones(Float64, n) : Float64.(weights)
    nm = isnothing(names) ? ["series_$i" for i in 1:n] : String.(names)

    specs = LikelihoodSpec[]
    for i in 1:n
        push!(specs, LikelihoodSpec(nm[i], types[i], obs_series[i], pred_series[i]; weight = w[i]))
    end
    return MultiSeriesLikelihood(specs)
end

# ------------------------------------------------------------------------------------
# Turing-native multi-series likelihood model
# ------------------------------------------------------------------------------------

using Turing
using Distributions

"""
    bayesian_multi_series(obs_series, pred_series, types;
                          weights = ones(...),
                          names = ...) -> DynamicPPL.Model

Turing model combining multiple observation series with different likelihoods.
Each series uses a likelihood type from `types`:
  :normal, :lognormal, :multinomial, :dirichlet_multinomial, :poisson

Usable with MCMC (NUTS), VI (AdvancedVI), MAP (Optim), or ML via InferenceBridge.
"""
@model function bayesian_multi_series(obs_series::Vector, pred_series::Vector,
                                      types::Vector{Symbol};
                                      weights::Vector{<:Real} = ones(length(obs_series)),
                                      names::Vector{<:AbstractString} = ["series_\$i" for i in 1:length(obs_series)])
    n = length(obs_series)
    length(pred_series) == n || error("obs and pred must have same length.")
    length(types) == n || error("types must have same length.")

    obs_sd ~ truncated(Normal(0.15, 0.05), 0.01, 1.0)

    for i in 1:n
        d = Float64.(obs_series[i])
        p = Float64.(pred_series[i])
        w = Float64(weights[i])
        t = types[i]

        if t == :normal
            for j in eachindex(d)
                d[j] ~ Normal(p[j], obs_sd)
            end
        elseif t == :lognormal
            for j in eachindex(d)
                if d[j] > 0 && p[j] > 0
                    d[j] ~ LogNormal(log(p[j]), obs_sd)
                end
            end
        elseif t == :multinomial
            po = sum(d)
            pp = sum(p)
            if po > 0 && pp > 0
                n_tot = Int(round(po))
                pred_norm = p ./ pp
                d_counts = Int.(round.(d))
                d_counts ~ Multinomial(n_tot, pred_norm)
            end
        elseif t == :dirichlet_multinomial
            po = sum(d)
            pp = sum(p)
            if po > 0 && pp > 0
                n_tot = Int(round(po))
                pred_norm = p ./ pp
                alpha = n_tot .* pred_norm
                d_counts = Int.(round.(d))
                d_counts ~ DirichletMultinomial(n_tot, alpha)
            end
        elseif t == :poisson
            for j in eachindex(d)
                if p[j] > 0
                    d[j] ~ Poisson(p[j])
                end
            end
        else
            error("Unknown likelihood type: \$t")
        end
    end

    return obs_sd
end