# Mortality estimation via catch curves and Chapman-Robson estimators.

"""
Result of Chapman-Robson survival and total mortality estimation.

# Fields
- `s::Float64`: Estimated annual survival fraction ``S``.
- `z::Float64`: Estimated instantaneous total mortality rate ``Z = -\\log(S)``.
- `var_s::Float64`: Analytical variance of the survival estimator ``\\widehat{\\mathrm{Var}}(S)``.
- `se_s::Float64`: Standard error of survival ``\\mathrm{SE}(S) = \\sqrt{\\widehat{\\mathrm{Var}}(S)}``.
- `se_z::Float64`: Standard error of mortality ``\\mathrm{SE}(Z) \\approx \\frac{\\mathrm{SE}(S)}{S}`` via the delta method.
- `mean_t::Float64`: Coded mean age above recruitment age ``\\bar{T}``.
- `n::Float64`: Total sample size of aged individuals in the descending limb.
"""
struct ChapmanRobsonResult
    s::Float64
    z::Float64
    var_s::Float64
    se_s::Float64
    se_z::Float64
    mean_t::Float64
    n::Float64
end

"""
    chapman_robson(ages::AbstractVector{<:Real}, catches::AbstractVector{<:Real};
                   recruitment_age::Union{Nothing, Real} = nothing) -> ChapmanRobsonResult

Compute the Chapman and Robson (1960) minimum-variance unbiased estimator (MVUE) of
annual survival ``S`` and instantaneous total mortality rate ``Z`` from age composition data.

# Mathematical Formulation
Let ``a_0`` denote the age of full recruitment (the peak of the catch curve), and
let ``i = a - a_0 \\ge 0`` denote the recoded relative age. Given catch numbers ``C_i``
at recoded age ``i`` for ``i \\in \\{0, 1, \\dots, k\\}``:

```math
n = \\sum_{i=0}^k C_i, \\quad T = \\sum_{i=0}^k i C_i, \\quad \\bar{T} = \\frac{T}{n}
```

The minimum-variance unbiased estimator of annual survival ``S`` is:
```math
S = \\frac{\\bar{T}}{1 + \\bar{T} - 1/n}
```

The analytical variance estimator of ``S`` is:
```math
\\widehat{\\mathrm{Var}}(S) = S \\left( S - \\frac{\\bar{T} - 1}{n + \\bar{T} - 1} \\right)
```

The instantaneous mortality rate ``Z`` and its standard error via first-order Taylor
expansion (delta method) are:
```math
Z = -\\log(S), \\quad \\mathrm{SE}(Z) \\approx \\frac{\\mathrm{SE}(S)}{S}
```

# Inputs
- `ages`: Vector of ages.
- `catches`: Vector of observed counts or sample numbers corresponding to `ages`.
- `recruitment_age`: Optional minimum fully recruited age. If `nothing`, defaults to
  the age corresponding to the maximum catch count.

# Outputs
A [`ChapmanRobsonResult`](@ref) containing point estimates and standard errors.

# References
- Chapman, D.G. and D.S. Robson. 1960. The analysis of a catch curve.
  Biometrics 16:354-368.
- Robson, D.S. and D.G. Chapman. 1961. Catch curves and mortality rates.
  Trans. Am. Fish. Soc. 90:181-189.
"""
function chapman_robson(
    ages::AbstractVector{<:Real},
    catches::AbstractVector{<:Real};
    recruitment_age::Union{Nothing, Real} = nothing
)
    length(ages) == length(catches) ||
        throw(DimensionMismatch("ages and catches must have equal lengths"))

    # Determine peak recruitment age if not explicitly supplied
    a0 = if isnothing(recruitment_age)
        max_idx = argmax(catches)
        ages[max_idx]
    else
        recruitment_age
    end

    # Restrict to descending limb where age >= a0 and catch > 0
    valid_mask = (ages .>= a0) .& (catches .> 0)
    sum(valid_mask) >= 2 ||
        error("At least two age classes with positive catch required at or above recruitment age $a0")

    sub_ages = ages[valid_mask]
    sub_catches = catches[valid_mask]

    recoded_ages = Float64.(sub_ages .- a0)
    n = Float64(sum(sub_catches))
    t_sum = sum(recoded_ages .* sub_catches)
    mean_t = t_sum / n

    denom = 1.0 + mean_t - (1.0 / n)
    s = mean_t / denom

    # Variance calculation per Chapman and Robson (1960)
    var_term = (mean_t - 1.0) / (n + mean_t - 1.0)
    var_s = s * (s - var_term)
    if var_s < 0.0
        var_s = 0.0
    end
    se_s = sqrt(var_s)

    z = -log(s)
    se_z = se_s / s

    return ChapmanRobsonResult(s, z, var_s, se_s, se_z, mean_t, n)
end

"""
Result of linear catch curve regression.

# Fields
- `z::Float64`: Estimated instantaneous total mortality rate ``Z = -\\beta_1``.
- `intercept::Float64`: Regression intercept ``\\alpha``.
- `se_z::Float64`: Standard error of ``Z``.
- `r2::Float64`: Coefficient of determination ``R^2``.
- `ages_fitted::Vector{Float64}`: Ages included in the descending limb fit.
- `log_catches::Vector{Float64}`: Log catch values ``\\log(C_a)``.
"""
struct CatchCurveResult
    z::Float64
    intercept::Float64
    se_z::Float64
    r2::Float64
    ages_fitted::Vector{Float64}
    log_catches::Vector{Float64}
end

"""
    catch_curve_linear(ages::AbstractVector{<:Real}, catches::AbstractVector{<:Real};
                       recruitment_age::Union{Nothing, Real} = nothing,
                       max_age::Union{Nothing, Real} = nothing,
                       weighted::Bool = false) -> CatchCurveResult

Fit a linear catch curve regression to the descending limb of an age composition.

# Mathematical Formulation
```math
\\log(C_a) = \\alpha - Z a + \\epsilon_a, \\quad \\epsilon_a \\sim \\mathcal{N}(0, \\sigma^2 / w_a)
```
where ``Z`` is the instantaneous mortality rate. If `weighted = true`, weights are set
proportional to observed sample numbers ``w_a = C_a`` following Maceina & Bettoli (1998)
to reduce the leverage of sparsely sampled older age groups.

# Inputs
- `ages`: Vector of ages.
- `catches`: Vector of observed counts or sample numbers.
- `recruitment_age`: Optional minimum fully recruited age. Defaults to peak catch age.
- `max_age`: Optional upper truncation age.
- `weighted`: Whether to apply catch-weighted least squares (default: `false`).

# Outputs
A [`CatchCurveResult`](@ref) containing point estimate of ``Z``, standard error, and ``R^2``.

# References
- Ricker, W.E. 1975. Computation and interpretation of biological statistics of fish
  populations. Bull. Fish. Res. Board Can. 191:1-382.
- Maceina, M.J. and P.W. Bettoli. 1998. Variation in abundance, growth, and mortality of
  black crappies in Tennessee reservoirs. N. Am. J. Fish. Manage. 18:856-867.
"""
function catch_curve_linear(
    ages::AbstractVector{<:Real},
    catches::AbstractVector{<:Real};
    recruitment_age::Union{Nothing, Real} = nothing,
    max_age::Union{Nothing, Real} = nothing,
    weighted::Bool = false
)
    length(ages) == length(catches) ||
        throw(DimensionMismatch("ages and catches must have equal lengths"))

    a0 = if isnothing(recruitment_age)
        max_idx = argmax(catches)
        ages[max_idx]
    else
        recruitment_age
    end

    mask = (ages .>= a0) .& (catches .> 0)
    if !isnothing(max_age)
        mask = mask .& (ages .<= max_age)
    end
    sum(mask) >= 3 || error("At least 3 observations required for catch curve regression")

    x = Float64.(ages[mask])
    c = Float64.(catches[mask])
    y = log.(c)
    n = length(x)

    w = weighted ? c : ones(Float64, n)
    w_sum = sum(w)

    x_bar = sum(w .* x) / w_sum
    y_bar = sum(w .* y) / w_sum

    ss_xx = sum(w .* (x .- x_bar).^2)
    ss_xy = sum(w .* (x .- x_bar) .* (y .- y_bar))
    ss_yy = sum(w .* (y .- y_bar).^2)

    slope = ss_xy / ss_xx
    intercept = y_bar - slope * x_bar

    # Residual sum of squares
    residuals = y .- (intercept .+ slope .* x)
    rss = sum(w .* residuals.^2)
    df = n - 2
    s2 = rss / df

    se_slope = sqrt(s2 / ss_xx)
    r2 = 1.0 - (rss / ss_yy)

    # Instantaneous total mortality Z is negative slope
    z = -slope
    se_z = se_slope

    return CatchCurveResult(z, intercept, se_z, r2, x, y)
end

"""
    bayesian_catch_curve(ages, catches; recruitment_age = nothing, max_age = nothing,
                         prior_z_mu = 0.40, prior_z_sd = 0.30, z_lower = 0.0, z_upper = 3.0,
                         alpha_sd = 5.0,
                         prior_sigma_mu = 0.20, prior_sigma_sd = 0.10,
                         sigma_lower = 0.01, sigma_upper = 1.0)

Bayesian linear catch curve regression in `Turing.jl`:

```math
\\begin{aligned}
Z &\\sim \\mathrm{TruncatedNormal}(\\mu_Z, \\sigma_Z, Z_{\\rm low}, Z_{\\rm up}) \\\\
\\alpha &\\sim \\mathrm{Normal}(\\overline{\\log C}, \\sigma_\\alpha) \\\\
\\sigma &\\sim \\mathrm{TruncatedNormal}(\\mu_\\sigma, \\sigma_{\\sigma}, \\sigma_{\\rm low}, \\sigma_{\\rm up}) \\\\
\\log(C_a) &\\sim \\mathrm{Normal}(\\alpha - Z \\cdot a, \\sigma)
\\end{aligned}
```

# Arguments
- `ages`: Vector of ages.
- `catches`: Vector of observed counts or sample numbers.
- `recruitment_age`: Optional recruitment age cutoff.
- `max_age`: Optional maximum age cutoff.
- `prior_z_mu`, `prior_z_sd`, `z_lower`, `z_upper`: prior mean, standard deviation and
  admissible bounds for total mortality `Z`.
- `alpha_sd`: standard deviation of the prior on the intercept `alpha`, centred on the
  mean of `log C` over the used ages.
- `prior_sigma_mu`, `prior_sigma_sd`, `sigma_lower`, `sigma_upper`: prior mean, standard
  deviation and admissible bounds for the observation standard deviation.
"""
@model function bayesian_catch_curve(ages::AbstractVector{<:Real},
                                     catches::AbstractVector{<:Real};
                                     recruitment_age::Union{Nothing, Real} = nothing,
                                     max_age::Union{Nothing, Real} = nothing,
                                     prior_z_mu::Real = 0.40,
                                     prior_z_sd::Real = 0.30,
                                     z_lower::Real = 0.0,
                                     z_upper::Real = 3.0,
                                     alpha_sd::Real = 5.0,
                                     prior_sigma_mu::Real = 0.20,
                                     prior_sigma_sd::Real = 0.10,
                                     sigma_lower::Real = 0.01,
                                     sigma_upper::Real = 1.0)
    length(ages) == length(catches) || throw(DimensionMismatch("ages and catches lengths must match."))

    a0 = isnothing(recruitment_age) ? ages[argmax(catches)] : recruitment_age
    mask = (ages .>= a0) .& (catches .> 0)
    if !isnothing(max_age)
        mask = mask .& (ages .<= max_age)
    end
    sum(mask) >= 2 || error("At least 2 observations required for catch curve regression.")

    sub_ages = Float64.(ages[mask])
    log_catches = log.(Float64.(catches[mask]))

    # Priors
    z ~ truncated(Normal(prior_z_mu, prior_z_sd), z_lower, z_upper)
    alpha ~ Normal(mean(log_catches), alpha_sd)
    sigma ~ truncated(Normal(prior_sigma_mu, prior_sigma_sd), sigma_lower, sigma_upper)

    for i in eachindex(sub_ages)
        mu = alpha - z * (sub_ages[i] - sub_ages[1])
        log_catches[i] ~ Normal(mu, sigma)
    end

    return (z = z, alpha = alpha, sigma = sigma)
end
