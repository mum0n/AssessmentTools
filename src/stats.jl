# ------------------------------------------------------------------------------------
# Statistical densities (negative log scale) and selectivity primitives.
# All densities return negative log-likelihoods including normalising constants.
# ------------------------------------------------------------------------------------

const LOG2PI = log(2π)

"""
    primal(x)

Strip automatic-differentiation duals (recursively) and return the plain value.
"""
primal(x::Real) = x
primal(x::ForwardDiff.Dual) = primal(ForwardDiff.value(x))
primal(x::AbstractArray) = primal.(x)

"""
    plogis(x, μ, σ)

Logistic function `1 / (1 + exp((μ - x)/σ))`, used for maturity- and
selectivity-at-age (σ is a scale, not a standard deviation).
"""
plogis(x, mu, sd) = 1 / (1 + exp((mu - x) / sd))

"""
    dnorm(x, μ, σ)

Normal negative log density evaluated via `Distributions.Normal(μ, σ)`:
`-logpdf(Normal(μ, σ), x)`.
"""
dnorm(x::Real, mu::Real, sd::Real) = -logpdf(Normal(mu, sd), x)

"""
    dnorm(residual::AbstractVector, σ)

Sum of normal negative log densities of `residual` with scalar or vector standard deviation `σ`.
"""
function dnorm(res::AbstractVector, sd::Real)
    dist = Normal(0.0, sd)
    return -sum(r -> logpdf(dist, r), res)
end

function dnorm(res::AbstractVector, sd::AbstractVector)
    length(res) == length(sd) || error("dnorm: residual and sd lengths differ.")
    return -sum(i -> logpdf(Normal(0.0, sd[i]), res[i]), eachindex(res, sd))
end

"""
    dlnorm(x, μ, σ)

Log-normal negative log density evaluated via `Distributions.LogNormal(μ, σ)`:
`-logpdf(LogNormal(μ, σ), x)`.
"""
dlnorm(x::Real, mu::Real, sd::Real) = -logpdf(LogNormal(mu, sd), x)

"""
    dbeta(x, a, b)

Beta negative log density evaluated via `Distributions.Beta(a, b)`:
`-logpdf(Beta(a, b), x)`.
"""
function dbeta(x::Real, a::Real, b::Real)
    0 < primal(x) < 1 || throw(DomainError(x, "dbeta requires 0 < x < 1."))
    return -logpdf(Beta(a, b), x)
end

"""
    dgamma(x, a, b)

Gamma negative log density with shape `a` and rate `b` (scale `θ = 1/b`) evaluated
via `Distributions.Gamma(a, 1/b)`: `-logpdf(Gamma(a, 1/b), x)`.
"""
dgamma(x::Real, a::Real, b::Real) = -logpdf(Gamma(a, 1 / b), x)

"""
    dinvgamma(x, a, b)

Inverse-gamma negative log density with shape `a` and scale `b` evaluated
via `Distributions.InverseGamma(a, b)`: `-logpdf(InverseGamma(a, b), x)`.
"""
dinvgamma(x::Real, a::Real, b::Real) = -logpdf(InverseGamma(a, b), x)

"""
    pool_age_bins(oo, pp, minp) -> (o1, p1, bins)

Pool observed (`oo`) and predicted (`pp`) proportions so that each pooled bin ends in
an age whose observed proportion exceeds `minp` (Richards, Schnute & Olsen 1997,
Appendix A). Proportions `≤ minp` are added to the current bin; those after the last
bin are added to the last bin. `bins[k]` is the age index that closes bin `k`.
"""
function pool_age_bins(oo::AbstractVector, pp::AbstractVector{T}, minp::Real) where {T}
    n = count(>(minp), oo)
    n > 0 || error("All observed proportions in an age-composition row are ≤ " *
                   "min_prop = $minp; the row cannot be pooled.")
    o1 = zeros(eltype(oo), n)
    p1 = zeros(T, n)
    bins = zeros(Int, n)
    k = 1
    for j in eachindex(oo)
        o1[k] += oo[j]
        p1[k] += pp[j]
        if oo[j] > minp
            bins[k] = j
            k < n && (k += 1)
        end
    end
    return o1, p1, bins
end

"""
    dmvlogistic(O, P, minp) -> (nll, ν, τ²)

Multivariate logistic negative log-likelihood for age compositions with the
variance concentrated out (Schnute & Richards 1995; Richards et al. 1997).

For each row `t`, observed `o` and predicted `p` proportions are normalised and
pooled with [`pool_age_bins`](@ref). Residuals are
`ν_{t,a} = log o_{t,a} - log p_{t,a} - mean_a(log o_{t,·} - log p_{t,·})`.
With `D = Σ_t (n_t - 1)` the MLE of the variance is `τ̂² = Σ ν² / D` and the
concentrated likelihood is `D log τ̂²`. Returns the residual matrix `ν` (zeros in
pooled cells) and `τ̂²`.
"""
function dmvlogistic(O::AbstractMatrix, P::AbstractMatrix{T}, minp::Real) where {T}
    nu = zeros(T, size(P))
    df = 0
    for i in axes(O, 1)
        oo = O[i, :] ./ sum(O[i, :])
        pp = P[i, :] ./ sum(P[i, :])
        o1, p1, bins = pool_age_bins(oo, pp, minp)
        r = log.(o1) .- log.(p1)
        r = r .- sum(r) / length(r)
        nu[i, bins] .= r
        df += length(bins) - 1
    end
    df > 0 || error("dmvlogistic: no degrees of freedom (every row has one bin).")
    tau2 = sum(abs2, nu) / df
    return df * log(tau2), nu, primal(tau2)
end

"""
    dmultinom_pooled(O, P, minp; proc_err = 0.009) -> (nll, ν, τ)

Multinomial negative log-likelihood with pooled bins, using an
effective sample size `N = n / (1 + proc_err·n)` where `n = Σ_a O_{t,a}` (raw
counts), following the HCAM process-error adjustment. The likelihood omits the
multinomial constant: `Σ_t Σ_a N o (log o - log p)`. Residuals are Pearson residuals
`(o - p)/√(p(1-p)/N)`; `τ` is the median absolute residual (lower median).
"""
function dmultinom_pooled(O::AbstractMatrix, P::AbstractMatrix{T}, minp::Real;
                          proc_err::Real = 0.009) where {T}
    nu = zeros(T, size(P))
    absres = Float64[]
    nll = zero(T)
    for i in axes(O, 1)
        ntot = sum(O[i, :])
        nsamp = ntot / (1 + proc_err * ntot)
        oo = O[i, :] ./ ntot
        pp = P[i, :] ./ sum(P[i, :])
        o1, p1, bins = pool_age_bins(oo, pp, minp)
        r = (o1 .- p1) ./ sqrt.(p1 .* (1 .- p1) ./ nsamp)
        nu[i, bins] .= r
        append!(absres, abs.(primal(r)))
        nll += sum(nsamp .* o1 .* (log.(o1) .- log.(p1)))
    end
    isempty(absres) && error("dmultinom_pooled: no residuals.")
    w = sort(absres)
    idx = (length(w) + 1) ÷ 2
    return nll, nu, w[idx]
end

"""
    rmvlogistic(p, τ, rng) -> Vector

Random multivariate logistic proportions: `x = log p + ε`, where `ε ~ Normal(0, τ)`.
Returns `exp(x - x̄) / Σ exp(x - x̄)`. All `p` must be positive.
"""
function rmvlogistic(p::AbstractVector, tau::Real, rng::AbstractRNG)
    all(>(0), p) || error("rmvlogistic requires strictly positive proportions.")
    eps = rand(rng, Normal(0.0, tau), length(p))
    x = log.(p) .+ eps
    x .-= sum(x) / length(x)
    e = exp.(x)
    return e ./ sum(e)
end

"""
    eplogis(x, x1, x2, γ)

Exponential-logistic selectivity (Thompson 1994, CJFAS 51:2654) parameterised by the
inflection points `x1` (ascending) and `x2` (descending) and `γ ∈ (0,1)`
(γ → 0 approaches the logistic; larger γ gives stronger doming).
"""
function eplogis(x::AbstractVector, x1, x2, gamma)
    0 < primal(gamma) < 1 || throw(DomainError(gamma, "eplogis requires 0 < γ < 1."))
    t1 = 2 - 4gamma + 2gamma^2
    t3 = 1 + 2gamma - 2gamma^2
    t5 = sqrt(1 + 4gamma - 4gamma^2)
    k1 = log(t1 / (t3 + t5))
    k2 = log(t1 / (t3 - t5))
    beta = (k1 * x2 - x1 * k2) / (k1 - k2)
    alpha = k2 / (x2 - beta)
    c = (1 / (1 - gamma)) * ((1 - gamma) / gamma)^gamma
    return c .* exp.(alpha * gamma .* (beta .- x)) ./ (1 .+ exp.(alpha .* (beta .- x)))
end
