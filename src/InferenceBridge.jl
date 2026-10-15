# ------------------------------------------------------------------------------------
# InferenceBridge: one simple Turing model per estimation problem, three engines.
#
# Each estimation problem is written once as a concise Turing `@model`. The same
# object then drives every estimation path, with no likelihood or state recursion
# re-implemented per engine:
#
#   * `fit_map`          - MAP / least-squares point estimation (Optim, L-BFGS)
#   * `sample_posterior` - posterior sampling (AbstractMCMC / NUTS)
#   * `fit_vi`           - variational inference (AdvancedVI)
#
# `fit_map` exposes the model to Optim through `LogDensityProblems.logdensity`,
# which evaluates a Turing model directly; gradients come from `ForwardDiff`.
# Points outside the prior support return `-Inf` (and a zero gradient), which
# `Optim` treats as an impassable barrier. `estimate` dispatches between engines.
# ------------------------------------------------------------------------------------

module InferenceBridge

using DynamicPPL
using LogDensityProblems
using AbstractMCMC
using LinearAlgebra
using ForwardDiff
using Optim
using Random
using Turing

export LogPosterior, condition_model
export ViResult, parameter_dimension, fit_map, sample_posterior, fit_vi, estimate

"""
    LogPosterior(model::DynamicPPL.Model, anchor::AbstractVector)

Callable holding the log posterior of a Turing model. `lp(x)` returns the log
posterior density at the parameter vector `x`, and `gradient!(g, lp, x)` its
gradient.

`Optim` line searches cannot handle an `-Inf` objective, so points outside the
model's prior support return a large finite penalty quadratic in the distance
from `anchor` (usually the starting point). The penalty is always worse than any
admissible value and its gradient points back toward `anchor`, so the optimiser
is guided into the admissible region instead of stalling.
"""
struct LogPosterior
    model::DynamicPPL.Model
    ldf::Any
    anchor::Vector{Float64}
end

function LogPosterior(model::DynamicPPL.Model, anchor::AbstractVector{<:Real})
    LogPosterior(model, DynamicPPL.LogDensityFunction(model),
                 collect(Float64, anchor))
end

const MAX_GRAD_NORM_PENALTY = 1.0e6

@inline function _penalty(lp::LogPosterior, x::AbstractVector)
    d = x .- lp.anchor
    return -MAX_GRAD_NORM_PENALTY .* (1.0 .+ sum(abs2, d))
end

@inline function _safe_logdensity(lp::LogPosterior, x::AbstractVector)
    v = try
        w = LogDensityProblems.logdensity(lp.ldf, x)
        isfinite(w) ? w : oftype(w, -Inf)
    catch
        oftype(x[1], -Inf)
    end
    return isfinite(v) ? v : _penalty(lp, x)
end

@inline (lp::LogPosterior)(x::AbstractVector) = _safe_logdensity(lp, x)

Base.length(lp::LogPosterior) = LogDensityProblems.dimension(lp.ldf)

function gradient!(g::AbstractVector, lp::LogPosterior, x::AbstractVector)
    try
        ForwardDiff.gradient!(g, y -> _safe_logdensity(lp, y), x)
    catch
        fill!(g, 0.0)
    end
    return g
end

"""
    MapFitResult

# Fields
- `x`: parameter vector at the estimate
- `log_posterior`: log posterior at `x`
- `converged`: `Optim` convergence flag
- `iterations`: number of `Optim` iterations used
"""
struct MapFitResult
    x::Vector{Float64}
    log_posterior::Float64
    converged::Bool
    iterations::Int
end

"""
    fit_map(model, x0; iterations::Integer = 2000, g_abstol::Real = 1e-8,
            verbose::Bool = false) -> MapFitResult

Estimate the maximum a posteriori (MAP) point of a Turing `model` with `Optim`
L-BFGS and `ForwardDiff` gradients. `x0` is the starting parameter vector, of
length `parameter_dimension(model)`.
"""
function fit_map(model::DynamicPPL.Model, x0::AbstractVector{<:Real};
                 iterations::Integer = 2000, g_abstol::Real = 1e-8,
                 verbose::Bool = false)
    lp = LogPosterior(model, x0)
    length(x0) == length(lp) ||
        error("Starting vector has length $(length(x0)); the model expects $(length(lp)).")

    fval = y -> -lp(y)
    fgrad!(G, y) = (gradient!(G, lp, y); G .*= -1; G)

    opt = Optim.Options(iterations = iterations, g_abstol = g_abstol,
                        show_trace = verbose)
    res = Optim.optimize(fval, fgrad!, collect(Float64, x0), Optim.LBFGS(), opt)

    if verbose
        println("MAP fit: log posterior = ", Optim.minimum(res),
                ", converged = ", Optim.converged(res),
                ", iterations = ", Optim.iterations(res))
    end

    return MapFitResult(Optim.minimizer(res), -Optim.minimum(res),
                        Optim.converged(res), Optim.iterations(res))
end

"""
    ViResult

# Fields
- `result`: the `Turing.Variational` fit object (variational family and fit state)
- `model`: the fitted Turing model
- `samples`: posterior samples drawn from the fitted variational approximation
"""
struct ViResult
    result::Any
    model::DynamicPPL.Model
    samples::Any
end

"""
    fit_vi(model; family::Symbol = :meanfield, max_iter::Integer = 100,
           n_samples::Integer = 1000, rng = Random.default_rng(),
           verbose::Bool = false) -> ViResult

Fit a variational approximation to a Turing `model` with `AdvancedVI`.
`family` selects the variational family (`:meanfield` for a mean-field Gaussian,
`:fullrank` for a full-rank Gaussian). The returned object also carries
`n_samples` draws from the fitted approximation, which are much faster to obtain
than MCMC draws and are adequate for quick, large-sample posterior summaries.
"""
function fit_vi(model::DynamicPPL.Model;
                family::Symbol = :meanfield, max_iter::Integer = 100,
                n_samples::Integer = 1000,
                rng::AbstractRNG = Random.default_rng(),
                verbose::Bool = false)
    # `Turing.Variational.vi` requires the variational family to be a callable of the
    # form `(rng, ldf) -> q`, so the family is built by passing the RNG into the
    # constructor rather than by calling it beforehand. The linker's own start vector
    # is used as the location so that the initial approximation is inside the region
    # where the log density is finite, which a zero-mean Gaussian often is not.
    ld0 = DynamicPPL.LogDensityFunction(model)
    x_start = collect(Float64, ld0._x)
    # `q_initialize_scale` can only shrink the scale, so the value passed here must be
    # small enough that the initial samples already give finite log-densities. A scale
    # tied to the magnitude of the start vector is used rather than `0.6*I`, so models
    # whose parameters live far from the origin do not start with an infeasible family.
    sigma0 = fill(max(0.02, 0.01 * maximum(abs, x_start, init = 0.0)), length(x_start))
    family_fn = if family === :meanfield
        (rng, prob) -> Turing.Variational.q_meanfield_gaussian(rng, prob;
                                                                location = x_start,
                                                                scale = Diagonal(sigma0))
    elseif family === :fullrank
        (rng, prob) -> Turing.Variational.q_fullrank_gaussian(rng, prob;
                                                              location = x_start,
                                                              scale = LowerTriangular(Diagonal(sigma0)))
    else
        error("Unknown variational family :$(family); use :meanfield or :fullrank.")
    end

    local res
    try
        res = Turing.Variational.vi(model, family_fn, max_iter)
    catch err
        if err isa DomainError
            error("""
fit_vi: the model is not defined on an unconstrained parameter space.

Variational inference samples the whole parameter space, so every parameter must
be unbounded. Declare parameters that are naturally constrained (positive scales,
variances, catchabilities) on a transformed scale, e.g.

    log_sigma ~ Normal(0.0, 1.0)
    sigma = exp(log_sigma)

and use `sigma` in the likelihood. The same requirement applies to NUTS.
""")
        end
        rethrow()
    end

    samples = if n_samples > 0
        # `Turing.Variational.vi(res, n, rng)` returns raw parameter draws from the
        # fitted variational approximation, each as a `DynamicPPL.VarNamedTuple`.
        rand(rng, res, n_samples)
    else
        nothing
    end

    return ViResult(res, model, samples)
end

"""
    sample_posterior(model, sampler; initial_params = nothing,
                     n_samples::Integer = 1000, n_chains::Integer = 1,
                     rng = Random.default_rng())

Draw posterior samples of a Turing `model` with `sampler` (e.g. `NUTS()`).
Returns an `MCMCChains.Chains` object for `n_chains = 1`, and a `Vector` of
chains otherwise. Start values are drawn from the model prior unless
`initial_params` is supplied.
"""
function sample_posterior(model::DynamicPPL.Model, sampler;
                          initial_params = nothing, n_samples::Integer = 1000,
                          n_chains::Integer = 1,
                          rng::AbstractRNG = Random.default_rng(),
                          verbose::Bool = false)
    n_samples >= 1 || error("n_samples must be positive.")
    n_chains >= 1 || error("n_chains must be at least 1.")

    draw_one = function ()
        if isnothing(initial_params)
            return AbstractMCMC.sample(rng, model, sampler, n_samples)
        end
        x0 = collect(Float64, initial_params)
        return AbstractMCMC.sample(rng, model, sampler, n_samples,
                                   initial_params = x0)
    end

    return n_chains == 1 ? draw_one() : [draw_one() for _ in 1:n_chains]
end

"""
    parameter_dimension(model::DynamicPPL.Model) -> Int

Number of free (unconstrained) parameters of a Turing model, as seen by the simulation
and sampling kernels. Use it to size the starting vector required by [`fit_map`](@ref).
"""
function parameter_dimension(model::DynamicPPL.Model)
    return LogDensityProblems.dimension(DynamicPPL.LogDensityFunction(model))
end

"""
    condition_model(model, fixed::NamedTuple) -> Model

Return `model` with the named parameters held at fixed values, e.g.
`condition_model(m, (q = 0.001,))`.
"""
condition_model(model::DynamicPPL.Model, fixed::NamedTuple) =
    DynamicPPL.condition(model; fixed...)

"""
    estimate(model, x0; method::Symbol = :map, sampler = NUTS(),
             family::Symbol = :meanfield, n_samples::Integer = 1000,
             max_iter::Integer = 100, iterations::Integer = 2000,
             rng = Random.default_rng(), verbose::Bool = false)

Single entry point over the three engines. `method` is `:map` for MAP point
estimation ([`fit_map`](@ref)), `:mcmc` for posterior sampling
([`sample_posterior`](@ref)), or `:vi` for variational inference
([`fit_vi`](@ref)). `x0` is used by `:map` only.
"""
function estimate(model::DynamicPPL.Model, x0::AbstractVector{<:Real};
                  method::Symbol = :map, sampler = nothing,
                  family::Symbol = :meanfield, n_samples::Integer = 1000,
                  max_iter::Integer = 100, iterations::Integer = 2000,
                  rng::AbstractRNG = Random.default_rng(), verbose::Bool = false)
    if method === :map
        return fit_map(model, x0; iterations = iterations, verbose = verbose)
    elseif method === :mcmc
        smp = isnothing(sampler) ? NUTS() : sampler
        return sample_posterior(model, smp; n_samples = n_samples, rng = rng,
                                verbose = verbose)
    elseif method === :vi
        return fit_vi(model; family = family, max_iter = max_iter,
                      n_samples = n_samples, rng = rng, verbose = verbose)
    else
        error("Unknown method :$(method); use :map, :mcmc, or :vi.")
    end
end

"""
    sample_prior(model; rng = Random.default_rng())

Draw one prior realisation of `model` (prior predictive check).
"""
sample_prior(model::DynamicPPL.Model; rng::AbstractRNG = Random.default_rng()) =
    rand(rng, model)

end # module InferenceBridge
