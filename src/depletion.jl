# Population depletion estimation via Leslie and DeLury methods.

"""
Result of Leslie depletion regression analysis.

# Fields
- `n0::Float64`: Estimated initial abundance ``N_0``.
- `q::Float64`: Estimated catchability coefficient ``q``.
- `se_n0::Float64`: Approximate standard error of initial abundance ``\\mathrm{SE}(N_0)``.
- `se_q::Float64`: Standard error of catchability ``\\mathrm{SE}(q)``.
- `r2::Float64`: Coefficient of determination ``R^2``.
- `cum_catch::Vector{Float64}`: Cumulative catch vector prior to each interval ``K_t``.
- `cpue::Vector{Float64}`: Catch per unit effort ``\\mathrm{CPUE}_t``.
"""
struct LeslieResult
    n0::Float64
    q::Float64
    se_n0::Float64
    se_q::Float64
    r2::Float64
    cum_catch::Vector{Float64}
    cpue::Vector{Float64}
end

"""
Result of DeLury depletion regression analysis.

# Fields
- `n0::Float64`: Estimated initial abundance ``N_0``.
- `q::Float64`: Estimated catchability coefficient ``q``.
- `se_n0::Float64`: Approximate standard error of initial abundance ``\\mathrm{SE}(N_0)``.
- `se_q::Float64`: Standard error of catchability ``\\mathrm{SE}(q)``.
- `r2::Float64`: Coefficient of determination ``R^2``.
- `cum_effort::Vector{Float64}`: Cumulative effort vector prior to midpoint of interval ``E_t``.
- `log_cpue::Vector{Float64}`: Natural logarithm of catch per unit effort ``\\log(\\mathrm{CPUE}_t)``.
"""
struct DeLuryResult
    n0::Float64
    q::Float64
    se_n0::Float64
    se_q::Float64
    r2::Float64
    cum_effort::Vector{Float64}
    log_cpue::Vector{Float64}
end

"""
    leslie_depletion(catch_seq::AbstractVector{<:Real},
                     effort_seq::AbstractVector{<:Real};
                     midpoint_adjustment::Bool = true) -> LeslieResult

Estimate initial population abundance ``N_0`` and catchability ``q`` using Leslie's
depletion method.

# Mathematical Formulation
In a closed population experiencing removal without significant recruitment or natural
mortality over a short period:

```math
\\mathrm{CPUE}_t = \\frac{C_t}{f_t} = q N_t
```
where ``N_t = N_0 - K_t``, and ``K_t`` is the cumulative removals prior to time ``t``:
```math
K_t = \\sum_{i=1}^{t-1} C_i + \\frac{1}{2} C_t
```
(with midpoint adjustment) or ``K_t = \\sum_{i=1}^{t-1} C_i`` (without midpoint adjustment).

Linear regression of ``y_t = \\mathrm{CPUE}_t`` on ``x_t = K_t`` yields:
```math
y_t = \\alpha - \\beta x_t
```
where:
```math
q = \\beta, \\quad N_0 = \\frac{\\alpha}{\\beta}
```
The variance of ``N_0`` is approximated via the delta method:
```math
\\mathrm{Var}(N_0) \\approx \\frac{1}{\\beta^2} \\mathrm{Var}(\\alpha) +
  \\frac{\\alpha^2}{\\beta^4} \\mathrm{Var}(\\beta) -
  2 \\frac{\\alpha}{\\beta^3} \\mathrm{Cov}(\\alpha, \\beta)
```

# Inputs
- `catch_seq`: Sequence of observed catch numbers or weights over consecutive intervals.
- `effort_seq`: Sequence of observed nominal fishing effort.
- `midpoint_adjustment`: If `true` (default), adds half of the current period catch to cumulative catch.

# Outputs
A [`LeslieResult`](@ref) containing point estimates and standard errors.

# References
- Leslie, P.H. and D.H.S. Davis. 1939. An attempt to determine the absolute number of
  rats on a given area. J. Anim. Ecol. 8:94-113.
- Ricker, W.E. 1975. Computation and interpretation of biological statistics of fish
  populations. Bull. Fish. Res. Board Can. 191:1-382.
"""
function leslie_depletion(
    catch_seq::AbstractVector{<:Real},
    effort_seq::AbstractVector{<:Real};
    midpoint_adjustment::Bool = true
)
    n = length(catch_seq)
    length(effort_seq) == n ||
        throw(DimensionMismatch("catch_seq and effort_seq must have identical lengths"))
    n >= 3 || error("Leslie depletion requires at least 3 consecutive observations")

    c = Float64.(catch_seq)
    f = Float64.(effort_seq)
    all(f .> 0) || error("All effort values must be strictly positive")

    cpue = c ./ f

    # Cumulative catch
    k = zeros(Float64, n)
    cum = 0.0
    for t in 1:n
        k[t] = midpoint_adjustment ? (cum + 0.5 * c[t]) : cum
        cum += c[t]
    end

    x = k
    y = cpue

    x_bar = mean(x)
    y_bar = mean(y)

    ss_xx = sum((x .- x_bar).^2)
    ss_xy = sum((x .- x_bar) .* (y .- y_bar))
    ss_yy = sum((y .- y_bar).^2)

    slope = ss_xy / ss_xx
    intercept = y_bar - slope * x_bar

    # If slope is positive, depletion assumption failed
    slope < 0.0 ||
        @warn "Leslie depletion slope is positive ($slope); CPUE increased over time"

    q = -slope
    n0 = intercept / q

    # Residuals & covariance
    residuals = y .- (intercept .+ slope .* x)
    rss = sum(residuals.^2)
    df = n - 2
    s2 = rss / df

    var_slope = s2 / ss_xx
    var_intercept = s2 * (1.0 / n + (x_bar^2 / ss_xx))
    cov_int_slope = -s2 * x_bar / ss_xx

    se_q = sqrt(var_slope)

    # Delta method for SE(N0) = SE(alpha / beta) where beta = -slope
    beta = -slope
    alpha = intercept
    var_beta = var_slope
    var_alpha = var_intercept
    cov_alpha_beta = -cov_int_slope

    var_n0 = (1.0 / beta^2) * var_alpha +
             (alpha^2 / beta^4) * var_beta -
             2.0 * (alpha / beta^3) * cov_alpha_beta

    se_n0 = var_n0 > 0.0 ? sqrt(var_n0) : NaN
    r2 = 1.0 - (rss / ss_yy)

    return LeslieResult(n0, q, se_n0, se_q, r2, x, y)
end

"""
    delury_depletion(catch_seq::AbstractVector{<:Real},
                     effort_seq::AbstractVector{<:Real};
                     midpoint_adjustment::Bool = true) -> DeLuryResult

Estimate initial population abundance ``N_0`` and catchability ``q`` using DeLury's
depletion method.

# Mathematical Formulation
Assuming catchability ``q`` is constant and natural mortality is negligible:

```math
\\log(\\mathrm{CPUE}_t) = \\log(q N_0) - q E_t
```
where ``E_t`` is cumulative effort:
```math
E_t = \\sum_{i=1}^{t-1} f_i + \\frac{1}{2} f_t
```
Linear regression of ``y_t = \\log(\\mathrm{CPUE}_t)`` on ``x_t = E_t`` yields:
```math
y_t = \\alpha - \\beta x_t
```
where:
```math
q = \\beta, \\quad N_0 = \\frac{\\exp(\\alpha)}{q}
```

# Inputs
- `catch_seq`: Sequence of observed catch numbers or weights over consecutive intervals.
- `effort_seq`: Sequence of observed nominal fishing effort.
- `midpoint_adjustment`: If `true` (default), adds half of the current period effort to cumulative effort.

# Outputs
A [`DeLuryResult`](@ref) containing point estimates and standard errors.

# References
- DeLury, D.B. 1947. On the estimation of biological populations.
  Biometrics 3:145-167.
"""
function delury_depletion(
    catch_seq::AbstractVector{<:Real},
    effort_seq::AbstractVector{<:Real};
    midpoint_adjustment::Bool = true
)
    n = length(catch_seq)
    length(effort_seq) == n ||
        throw(DimensionMismatch("catch_seq and effort_seq must have identical lengths"))
    n >= 3 || error("DeLury depletion requires at least 3 consecutive observations")

    c = Float64.(catch_seq)
    f = Float64.(effort_seq)
    all(f .> 0) || error("All effort values must be strictly positive")
    all(c .> 0) || error("All catch values must be strictly positive for log(CPUE)")

    cpue = c ./ f
    log_cpue = log.(cpue)

    e = zeros(Float64, n)
    cum_e = 0.0
    for t in 1:n
        e[t] = midpoint_adjustment ? (cum_e + 0.5 * f[t]) : cum_e
        cum_e += f[t]
    end

    x = e
    y = log_cpue

    x_bar = mean(x)
    y_bar = mean(y)

    ss_xx = sum((x .- x_bar).^2)
    ss_xy = sum((x .- x_bar) .* (y .- y_bar))
    ss_yy = sum((y .- y_bar).^2)

    slope = ss_xy / ss_xx
    intercept = y_bar - slope * x_bar

    slope < 0.0 ||
        @warn "DeLury depletion slope is positive ($slope); log(CPUE) increased with effort"

    q = -slope
    n0 = exp(intercept) / q

    # Residuals & covariance
    residuals = y .- (intercept .+ slope .* x)
    rss = sum(residuals.^2)
    df = n - 2
    s2 = rss / df

    var_slope = s2 / ss_xx
    var_intercept = s2 * (1.0 / n + (x_bar^2 / ss_xx))
    cov_int_slope = -s2 * x_bar / ss_xx

    se_q = sqrt(var_slope)

    # Delta method for SE(N0) = SE(exp(alpha) / beta)
    beta = -slope
    alpha = intercept
    g_alpha = exp(alpha) / beta
    g_beta = -exp(alpha) / (beta^2)

    var_n0 = (g_alpha^2) * var_intercept +
             (g_beta^2) * var_slope +
             2.0 * g_alpha * g_beta * (-cov_int_slope)

    se_n0 = var_n0 > 0.0 ? sqrt(var_n0) : NaN
    r2 = 1.0 - (rss / ss_yy)

    return DeLuryResult(n0, q, se_n0, se_q, r2, x, y)
end

"""
    bayesian_leslie_depletion(catch_seq, effort_seq; midpoint_adjustment = true,
                              prior_n0 = 10000.0, prior_n0_sd = 5000.0)

Bayesian Leslie depletion regression in `Turing.jl`:
- Formulates CPUE decline: ``\\mathrm{CPUE}_t \\sim \\mathrm{Normal}(q(N_0 - K_t), \\sigma)``.
- Estimates initial abundance ``N_0`` and catchability scalar ``q``.
"""
@model function bayesian_leslie_depletion(catch_seq::AbstractVector{<:Real},
                                          effort_seq::AbstractVector{<:Real};
                                          midpoint_adjustment::Bool = true,
                                          prior_n0::Real = 10000.0,
                                          prior_n0_sd::Real = 5000.0,
                                          n0_upper_factor::Real = 10.0,
                                          prior_q_mu::Real = 0.001,
                                          prior_q_sd::Real = 0.001,
                                          q_lower::Real = 1.0e-7,
                                          q_upper::Real = 0.1,
                                          prior_sigma_mu::Real = 0.20,
                                          prior_sigma_sd::Real = 0.10,
                                          sigma_lower::Real = 0.01,
                                          sigma_upper::Real = 2.0,
                                          mu_floor::Real = 1.0e-4)
    n = length(catch_seq)
    length(effort_seq) == n || throw(DimensionMismatch("catch_seq and effort_seq lengths must match."))

    cpue = Float64.(catch_seq) ./ Float64.(effort_seq)
    k_t = zeros(Float64, n)
    cum = 0.0
    for i in 1:n
        k_t[i] = midpoint_adjustment ? cum + 0.5 * Float64(catch_seq[i]) : cum
        cum += Float64(catch_seq[i])
    end

    n0 ~ truncated(Normal(prior_n0, prior_n0_sd), cum, n0_upper_factor * prior_n0)
    q ~ truncated(Normal(prior_q_mu, prior_q_sd), q_lower, q_upper)
    sigma ~ truncated(Normal(prior_sigma_mu, prior_sigma_sd), sigma_lower, sigma_upper)

    for i in 1:n
        mu = max(mu_floor, q * (n0 - k_t[i]))
        cpue[i] ~ Normal(mu, sigma)
    end

    return (n0 = n0, q = q, sigma = sigma)
end

"""
    bayesian_delury_depletion(catch_seq, effort_seq;
                              prior_n0 = 10000.0, prior_n0_sd = 5000.0,
                              n0_upper_factor = 10.0,
                              prior_q_mu = 0.001, prior_q_sd = 0.001,
                              q_lower = 1.0e-7, q_upper = 0.1,
                              prior_sigma_mu = 0.20, prior_sigma_sd = 0.10,
                              sigma_lower = 0.01, sigma_upper = 2.0,
                              mu_floor = 1.0e-4)

Bayesian DeLury depletion regression in `Turing.jl`:
- Formulates log CPUE decline: ``\\log(\\mathrm{CPUE}_t) \\sim \\mathrm{Normal}(\\log(q N_0) - q E_t, \\sigma)``.

Every prior mean, standard deviation and bound is a named keyword, so the prior
specification is user-level rather than compiled into the model body.
"""
@model function bayesian_delury_depletion(catch_seq::AbstractVector{<:Real},
                                          effort_seq::AbstractVector{<:Real};
                                          prior_n0::Real = 10000.0,
                                          prior_n0_sd::Real = 5000.0,
                                          n0_upper_factor::Real = 10.0,
                                          prior_q_mu::Real = 0.001,
                                          prior_q_sd::Real = 0.001,
                                          q_lower::Real = 1.0e-7,
                                          q_upper::Real = 0.1,
                                          prior_sigma_mu::Real = 0.20,
                                          prior_sigma_sd::Real = 0.10,
                                          sigma_lower::Real = 0.01,
                                          sigma_upper::Real = 2.0,
                                          mu_floor::Real = 1.0e-4)
    n = length(catch_seq)
    length(effort_seq) == n || throw(DimensionMismatch("catch_seq and effort_seq lengths must match."))

    log_cpue = log.(Float64.(catch_seq) ./ Float64.(effort_seq))
    e_t = zeros(Float64, n)
    cum = 0.0
    for i in 1:n
        e_t[i] = cum + 0.5 * Float64(effort_seq[i])
        cum += Float64(effort_seq[i])
    end

    n0 ~ truncated(Normal(prior_n0, prior_n0_sd), sum(catch_seq), n0_upper_factor * prior_n0)
    q ~ truncated(Normal(prior_q_mu, prior_q_sd), q_lower, q_upper)
    sigma ~ truncated(Normal(prior_sigma_mu, prior_sigma_sd), sigma_lower, sigma_upper)

    for i in 1:n
        mu = log(max(1e-4, q * n0)) - q * e_t[i]
        log_cpue[i] ~ Normal(mu, sigma)
    end

    return (n0 = n0, q = q, sigma = sigma)
end
