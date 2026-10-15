# ------------------------------------------------------------------------------------
# Virtual Population Analysis (VPA) and Pope's Cohort Analysis.
# Reconstructs historical stock numbers-at-age and fishing mortality rates
# by backward recursion from terminal catch observations.
# ------------------------------------------------------------------------------------

"""
Container for results of cohort reconstruction or Virtual Population Analysis.

# Fields
- `N::Matrix{Float64}`: Reconstructed numbers-at-age (dimension `nages × nyears`).
- `F::Matrix{Float64}`: Estimated instantaneous fishing mortality rates (dimension `nages × nyears`).
- `biomass_total::Vector{Float64}`: Total annual exploitable biomass.
- `biomass::Vector{Float64}`: Annual (vulnerable) biomass or total abundance, \\sum_a N_{a,t}.
- `biomass_ssb::Vector{Float64}`: Annual spawning stock measure. Equal to \\sum_a N_{a,t} w_a m_a
  when `weight_at_age` and `maturity` are given, otherwise total abundance \\sum_a N_{a,t},
  which is a proxy only.
- `catch_pred::Matrix{Float64}`: Model predicted catches via Baranov catch equation.
"""
struct VpaResult
    N::Matrix{Float64}
    F::Matrix{Float64}
    biomass_total::Vector{Float64}
    biomass_ssb::Vector{Float64}
    catch_pred::Matrix{Float64}
end

"""
    _spawning_abundance(N::AbstractMatrix{<:Real}, weight_at_age, maturity) -> Vector{Float64}

Annual spawning stock measure from a numbers-at-age matrix `N` (`nages × nyears`).

``SSB_t = \\sum_a N_{a,t}\\, w_a\\, m_a``

where `w_a` is weight at age and `m_a` maturity (spawning fraction) at age. When
either argument is `nothing` the corresponding factor is 1, so the result is the
total abundance ``\\sum_a N_{a,t}``; that is a proxy for SSB, not SSB itself.
Both vectors, when given, must have length equal to `size(N, 1)`.
"""
function _spawning_abundance(N::AbstractMatrix{<:Real},
                             weight_at_age::Union{Nothing, AbstractVector{<:Real}},
                             maturity::Union{Nothing, AbstractVector{<:Real}})
    na = size(N, 1)
    wa = isnothing(weight_at_age) ? ones(Float64, na) : collect(Float64, weight_at_age)
    ma = isnothing(maturity) ? ones(Float64, na) : collect(Float64, maturity)
    length(wa) == na || error("weight_at_age must have $na elements.")
    length(ma) == na || error("maturity must have $na elements.")
    return [sum(N[a, t] * wa[a] * ma[a] for a in 1:na) for t in axes(N, 2)]
end

"""
    popes_cohort_analysis(catches::AbstractMatrix{<:Real}, M;
                          terminal_f::AbstractVector{<:Real},
                          plus_group_f::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
                          plus_group_age::Integer = size(catches, 1),
                          weight_at_age = nothing,
                          maturity = nothing) -> VpaResult

Compute historical numbers-at-age and fishing mortalities using Pope's (1972) cohort analysis:

```math
N_{a,t} = N_{a+1, t+1} \\exp(M_{a,t}) + C_{a,t} \\exp\\left(\\frac{M_{a,t}}{2}\\right)
```

# Arguments
- `catches`: Matrix of catch in numbers of dimension `nages × nyears`.
- `M`: Instantaneous natural mortality (scalar, vector per age, or matrix `nages × nyears`).
- `terminal_f`: Vector of terminal year fishing mortalities across ages (length `nages`).
- `plus_group_f`: Fishing mortality assumed for the oldest age class across years.
  If `nothing`, defaults to terminal F of the plus group.
- `plus_group_age`: Row index designating the plus-group (defaults to `nages`).
- `weight_at_age`: Optional weight at age; enables proper spawning biomass in `vpa.biomass_ssb`.
- `maturity`: Optional maturity (spawning fraction) at age; enables proper SSB.
"""
function popes_cohort_analysis(catches::AbstractMatrix{<:Real}, M;
                               terminal_f::AbstractVector{<:Real},
                               plus_group_f::Union{Nothing, Real, AbstractVector{<:Real}} = nothing,
                               plus_group_age::Integer = size(catches, 1),
                               weight_at_age::Union{Nothing, AbstractVector{<:Real}} = nothing,
                               maturity::Union{Nothing, AbstractVector{<:Real}} = nothing)
    na, ny = size(catches)
    length(terminal_f) == na || error("terminal_f length must match number of ages ($na).")

    # Expand M into full na x ny matrix
    M_mat = zeros(Float64, na, ny)
    if isa(M, Real)
        M_mat .= Float64(M)
    elseif isa(M, AbstractVector)
        length(M) == na || error("M vector length must match number of ages ($na).")
        for j in 1:ny
            M_mat[:, j] .= M
        end
    elseif isa(M, AbstractMatrix)
        size(M) == (na, ny) || error("M matrix size must match catches ($(na) × $(ny)).")
        M_mat .= M
    else
        error("Unsupported format for natural mortality M.")
    end

    N = zeros(Float64, na, ny)
    F = zeros(Float64, na, ny)

    # Initialize terminal year N and F
    F[:, ny] .= terminal_f
    for a in 1:na
        fa = F[a, ny]
        ma = M_mat[a, ny]
        za = fa + ma
        # Baranov inversion in terminal year:
        if fa > 0.0
            N[a, ny] = catches[a, ny] / ((fa / za) * (1.0 - exp(-za)))
        else
            N[a, ny] = 0.0
        end
    end

    # Initialize plus-group F and N across all years
    f_plus = if isnothing(plus_group_f)
        fill(terminal_f[plus_group_age], ny)
    elseif isa(plus_group_f, Real)
        fill(Float64(plus_group_f), ny)
    else
        collect(Float64.(plus_group_f))
    end

    for t in 1:(ny - 1)
        F[plus_group_age, t] = f_plus[t]
        fa = f_plus[t]
        ma = M_mat[plus_group_age, t]
        za = fa + ma
        if fa > 0.0
            N[plus_group_age, t] = catches[plus_group_age, t] / ((fa / za) * (1.0 - exp(-za)))
        end
    end

    # Backward recursion via Pope's approximation
    for a in reverse(1:(plus_group_age - 1))
        for t in reverse(1:(ny - 1))
            m_at = M_mat[a, t]
            N[a, t] = N[a + 1, t + 1] * exp(m_at) + catches[a, t] * exp(0.5 * m_at)

            # Compute realized F from survival ratio
            if N[a, t] > 0.0 && N[a + 1, t + 1] > 0.0
                F[a, t] = max(0.0, log(N[a, t] / N[a + 1, t + 1]) - m_at)
            else
                F[a, t] = 0.0
            end
        end
    end

    # Predict catches from Baranov equation
    c_pred = zeros(Float64, na, ny)
    for t in 1:ny
        for a in 1:na
            za = F[a, t] + M_mat[a, t]
            if za > 0.0
                c_pred[a, t] = (F[a, t] / za) * (1.0 - exp(-za)) * N[a, t]
            end
        end
    end

    b_tot = vec(sum(N, dims = 1))
    b_ssb = _spawning_abundance(N, weight_at_age, maturity)

    return VpaResult(N, F, b_tot, b_ssb, c_pred)
end

"""
    bayesian_cohort_analysis(catches, survey_index, M;
                             survey_ages = 1:size(catches, 1),
                             prior_terminal_f = 0.30,
                             prior_f_sd = 0.20,
                             obs_cv = 0.25)

Bayesian state-space cohort analysis formulated in `Turing.jl`:
- Reconstructs cohort dynamics using Pope's backward approximation.
- Estimates terminal fishing mortalities ``F_{a, T}`` and age-specific survey catchability ``q_a``.
- Computes likelihood of observed survey indices
  ``A_{a, t} \\sim \\mathrm{LogNormal}(\\log(q_a N_{a, t}), \\sigma_o)``.

Catch is an input to the reconstruction and is treated as known without error, which
is the standard form of ADAPT and Pope cohort analysis; the only observation
likelihood is on the survey index. The matching deterministic estimators
([`adapt_vpa`](@ref), [`popes_cohort_analysis`](@ref)) make the same assumption, so
adding a catch likelihood here would double-count information already used to
condition the reconstruction.

# Arguments
- `catches`: Matrix of catch counts of size `(nages, nyears)`.
- `survey_index`: Matrix of relative abundance indices of size `(nages, nyears)` (supports missing/NaN).
- `M`: Instantaneous natural mortality (scalar, vector, or matrix).
"""
@model function bayesian_cohort_analysis(catches::AbstractMatrix{<:Real},
                                         survey_index::AbstractMatrix,
                                         M;
                                         survey_ages::AbstractVector{<:Integer} = 1:size(catches, 1),
                                         prior_terminal_f::Real = 0.30,
                                         prior_f_sd::Real = 0.20,
                                         obs_cv::Real = 0.25)
    na, ny = size(catches)
    size(survey_index) == (na, ny) || error("catches and survey_index must have identical dimensions.")

    # Expand natural mortality matrix
    M_mat = zeros(Float64, na, ny)
    if isa(M, Real)
        M_mat .= Float64(M)
    elseif isa(M, AbstractVector)
        for j in 1:ny
            M_mat[:, j] .= M
        end
    elseif isa(M, AbstractMatrix)
        M_mat .= M
    end

    # Priors on terminal year F across ages
    ft = Vector{Float64}(undef, na)
    for a in 1:na
        ft[a] ~ truncated(Normal(prior_terminal_f, prior_f_sd), 0.01, 2.5)
    end

    # Priors on survey catchabilities q_a for each age class
    q = Vector{Float64}(undef, na)
    for a in 1:na
        q[a] ~ truncated(Normal(0.001, 0.0005), 1e-6, 0.1)
    end

    obs_sd ~ truncated(Normal(obs_cv, 0.05), 0.05, 0.60)

    # Pope's cohort backward reconstruction
    N = zeros(Float64, na, ny)
    F = zeros(Float64, na, ny)

    F[:, ny] .= ft
    for a in 1:na
        za = ft[a] + M_mat[a, ny]
        if ft[a] > 0.0
            N[a, ny] = catches[a, ny] / ((ft[a] / za) * (1.0 - exp(-za)))
        else
            N[a, ny] = 1e-4
        end
    end

    # Backward recursion across cohorts
    for a in reverse(1:(na - 1))
        for t in reverse(1:(ny - 1))
            m_at = M_mat[a, t]
            N[a, t] = N[a + 1, t + 1] * exp(m_at) + catches[a, t] * exp(0.5 * m_at)
            if N[a, t] > 0.0 && N[a + 1, t + 1] > 0.0
                F[a, t] = max(0.0, log(N[a, t] / N[a + 1, t + 1]) - m_at)
            else
                F[a, t] = 0.0
            end
        end
    end

    # Observation likelihood on survey index
    for a in survey_ages
        for t in 1:ny
            s_val = survey_index[a, t]
            if !ismissing(s_val) && isfinite(s_val) && s_val > 0.0
                mu_s = max(1e-6, q[a] * N[a, t])
                s_val ~ LogNormal(log(mu_s), obs_sd)
            end
        end
    end

    return (N = N, F = F, q = q, ft = ft)
end

"""
    AdaptResult

Output of the SEQUENCE conditional Maximum Likelihood (cMLE) calibration used by
[`adapt_vpa`](@ref).

# Fields
- `vpa::VpaResult`: reconstructed population matrices `N`, `F` and biomasses
- `q::Vector{Float64}`: age-specific survey catchability estimates ``q_a``
- `residuals::Matrix{Float64}`: residual matrix ``\\log(I_{a,t}) - \\log(q_a N_{a,t})``
- `terminal_f::Vector{Float64}`: recovered terminal-year fishing mortality rates
- `terminal_n::Vector{Float64}`: recovered terminal-year abundances
- `ssq::Float64`: weighted residual sum of squares at the optimum
- `converged::Bool`: whether `Optim` reported convergence
- `iterations::Int`: number of `Optim` iterations used
- `residual_variance::Float64`: estimated observation error variance in log space,
  `ssq / (n - 1)` where `n` is the number of residuals
"""
struct AdaptResult
    vpa::VpaResult
    q::Vector{Float64}
    residuals::Matrix{Float64}
    terminal_f::Vector{Float64}
    terminal_n::Vector{Float64}
    ssq::Float64
    converged::Bool
    iterations::Int
    residual_variance::Float64
end

"""
    adapt_vpa(catches::AbstractMatrix{<:Real}, survey_index::AbstractMatrix, M;
              survey_ages = 1:size(catches, 1),
              init_terminal_f = fill(0.3, size(catches, 1)),
              method = :pope) -> AdaptResult

ADAPT Virtual Population Analysis (Gavaris 1988, Conser & Powers 1990):
Calibrates terminal year stock numbers ``N_{a, T}`` or fishing mortalities ``F_{a, T}``
by minimizing the sum of squared log residuals against survey relative abundance indices:

```math
\\min_{\\mathbf{F}_T} \\sum_{a} \\sum_{t} \\left( \\log(I_{a,t}) - \\log(q_a N_{a,t}) \\right)^2
```
where conditional maximum likelihood survey catchability is analytically evaluated as:
```math
\\log(q_a) = \\frac{1}{n_a} \\sum_{t} \\left( \\log(I_{a,t}) - \\log(N_{a,t}) \\right)
```

# Arguments
- `catches`: Catch-at-age matrix `(nages, nyears)`.
- `survey_index`: Survey relative abundance index matrix `(nages, nyears)` (supports missing/NaN/zero).
- `M`: Instantaneous natural mortality rate (scalar, vector, or matrix).
- `survey_ages`: Age class indices included in survey calibration.
- `init_terminal_f`: Initial guess for terminal year fishing mortalities.
- `method`: VPA engine (`:pope` for Pope's cohort analysis, `:standard` for Baranov VPA).
"""
function adapt_vpa(catches::AbstractMatrix{<:Real},
                   survey_index::AbstractMatrix,
                   M;
                   survey_ages::AbstractVector{<:Integer} = 1:size(catches, 1),
                   init_terminal_f::AbstractVector{<:Real} = fill(0.3, size(catches, 1)),
                   method::Symbol = :pope)
    na, ny = size(catches)
    size(survey_index) == (na, ny) || error("catches and survey_index dimensions must match.")
    length(init_terminal_f) == na || error("init_terminal_f length must equal number of ages ($na).")

    # Objective function: evaluate SSQ over log survey residuals
    function objective(ft_log::AbstractVector{T}) where {T}
        ft = exp.(ft_log)
        vpa_res = if method == :pope
            popes_cohort_analysis(catches, M; terminal_f = ft)
        else
            standard_vpa(catches, M; terminal_f = ft)
        end
        N_curr = vpa_res.N

        ssq = zero(T)
        for a in survey_ages
            diffs = T[]
            for t in 1:ny
                s_val = survey_index[a, t]
                if !ismissing(s_val) && isfinite(s_val) && s_val > 0.0 && N_curr[a, t] > 0.0
                    push!(diffs, log(s_val) - log(N_curr[a, t]))
                end
            end
            if !isempty(diffs)
                mean_diff = sum(diffs) / length(diffs)
                for d in diffs
                    ssq += (d - mean_diff)^2
                end
            end
        end
        return ssq
    end

    x0 = log.(max.(1e-4, init_terminal_f))
    opt = Optim.optimize(objective, x0, Optim.BFGS(), Optim.Options(iterations = 300, g_tol = 1e-6))
    best_ft = exp.(Optim.minimizer(opt))

    final_vpa = if method == :pope
        popes_cohort_analysis(catches, M; terminal_f = best_ft)
    else
        standard_vpa(catches, M; terminal_f = best_ft)
    end

    # Calculate final q and residuals
    q_vec = zeros(Float64, na)
    res_mat = fill(NaN, na, ny)
    for a in survey_ages
        diffs = Float64[]
        for t in 1:ny
            s_val = survey_index[a, t]
            if !ismissing(s_val) && isfinite(s_val) && s_val > 0.0 && final_vpa.N[a, t] > 0.0
                push!(diffs, log(s_val) - log(final_vpa.N[a, t]))
            end
        end
        if !isempty(diffs)
            q_vec[a] = exp(mean(diffs))
            for t in 1:ny
                s_val = survey_index[a, t]
                if !ismissing(s_val) && isfinite(s_val) && s_val > 0.0 && final_vpa.N[a, t] > 0.0
                    res_mat[a, t] = log(s_val) - log(q_vec[a] * final_vpa.N[a, t])
                end
            end
        end
    end

    fin_res = filter(isfinite, res_mat)

    return AdaptResult(
        final_vpa,
        q_vec,
        res_mat,
        best_ft,
        final_vpa.N[:, end],
        Optim.minimum(opt),
        Optim.converged(opt),
        Optim.iterations(opt),
        length(fin_res) > 1 ? sum(abs2, fin_res) / (length(fin_res) - 1) : 0.0
    )
end

"""
    bayesian_adapt_vpa(catches, survey_index, M;
                       survey_ages = 1:size(catches, 1),
                       prior_terminal_f = fill(0.30, size(catches, 1)),
                       prior_f_sd = 0.25,
                       obs_cv = 0.20)

Bayesian ADAPT Virtual Population Analysis in `Turing.jl`:
- Formulates ADAPT tuning with informative lognormal priors on terminal fishing mortalities.
- Integrates full backward cohort reconstruction.
- Evaluates lognormal observation likelihood for age-specific survey abundance indices.
"""
@model function bayesian_adapt_vpa(catches::AbstractMatrix{<:Real},
                                    survey_index::AbstractMatrix,
                                    M;
                                    survey_ages::AbstractVector{<:Integer} = 1:size(catches, 1),
                                    prior_terminal_f::AbstractVector{<:Real} = fill(0.30, size(catches, 1)),
                                    prior_f_sd::Real = 0.25,
                                    obs_cv::Real = 0.20,
                                    prior_q_mu::Real = 0.001,
                                    prior_q_sd::Real = 0.0005,
                                    q_lower::Real = 1.0e-6,
                                    q_upper::Real = 0.1,
                                    prior_obs_sd_sd::Real = 0.05,
                                    obs_sd_lower::Real = 0.05,
                                    obs_sd_upper::Real = 0.60,
                                    mu_floor::Real = 1.0e-6)
    na, ny = size(catches)
    size(survey_index) == (na, ny) || error("catches and survey_index dimensions must match.")

    # Prior on terminal F
    ft = Vector{Float64}(undef, na)
    for a in 1:na
        ft[a] ~ truncated(Normal(prior_terminal_f[a], prior_f_sd), 0.01, 2.5)
    end

    # Prior on survey catchabilities
    q = Vector{Float64}(undef, na)
    for a in 1:na
        q[a] ~ truncated(Normal(prior_q_mu, prior_q_sd), q_lower, q_upper)
    end

    obs_sd ~ truncated(Normal(obs_cv, prior_obs_sd_sd), obs_sd_lower, obs_sd_upper)

    # Expand M
    M_mat = zeros(Float64, na, ny)
    if isa(M, Real)
        M_mat .= Float64(M)
    elseif isa(M, AbstractVector)
        for j in 1:ny
            M_mat[:, j] .= M
        end
    elseif isa(M, AbstractMatrix)
        M_mat .= M
    end

    # Terminal year N
    N = zeros(Float64, na, ny)
    F = zeros(Float64, na, ny)
    F[:, ny] .= ft
    for a in 1:na
        za = ft[a] + M_mat[a, ny]
        if ft[a] > 0.0
            N[a, ny] = catches[a, ny] / ((ft[a] / za) * (1.0 - exp(-za)))
        else
            N[a, ny] = 1e-4
        end
    end

    # Pope's recursion
    for a in reverse(1:(na - 1))
        for t in reverse(1:(ny - 1))
            m_at = M_mat[a, t]
            N[a, t] = N[a + 1, t + 1] * exp(m_at) + catches[a, t] * exp(0.5 * m_at)
            if N[a, t] > 0.0 && N[a + 1, t + 1] > 0.0
                F[a, t] = max(0.0, log(N[a, t] / N[a + 1, t + 1]) - m_at)
            else
                F[a, t] = 0.0
            end
        end
    end

    # Likelihood
    for a in survey_ages
        for t in 1:ny
            s_val = survey_index[a, t]
            if !ismissing(s_val) && isfinite(s_val) && s_val > 0.0
                mu_s = max(1e-6, q[a] * N[a, t])
                s_val ~ LogNormal(log(mu_s), obs_sd)
            end
        end
    end

    return (N = N, F = F, q = q, ft = ft, ssb = vec(sum(N, dims = 1)))
end
"""
    standard_vpa(catches::AbstractMatrix{<:Real}, M;
                 terminal_f::AbstractVector{<:Real},
                 tol::Real = 1e-8,
                 plus_group_age::Integer = size(catches, 1),
                 weight_at_age = nothing,
                 maturity = nothing) -> VpaResult

Exact Virtual Population Analysis using the Gulland / Baranov catch equation. Each
age-by-year cell is inverted for `F` with [`Solvers.solve_scalar_root`](@ref), which
is driven by Pope's approximation as the starting value, so no iteration limit or
derivative has to be maintained at this site.

```math
\\frac{C_{a,t}}{N_{a+1,t+1}} = \\frac{F_{a,t}}{F_{a,t} + M_{a,t}} \\frac{1 - \\exp(-(F_{a,t} + M_{a,t}))}{\\exp(-(F_{a,t} + M_{a,t}))}
```

`tol` is the root tolerance. `weight_at_age` and `maturity` are optional and enable a
proper spawning biomass in `vpa.biomass_ssb`; see [`popes_cohort_analysis`](@ref).
"""
function standard_vpa(catches::AbstractMatrix{<:Real}, M;
                      terminal_f::AbstractVector{<:Real},
                      tol::Real = 1.0e-8,
                      plus_group_age::Integer = size(catches, 1),
                      weight_at_age::Union{Nothing, AbstractVector{<:Real}} = nothing,
                      maturity::Union{Nothing, AbstractVector{<:Real}} = nothing)
    na, ny = size(catches)
    length(terminal_f) == na || error("terminal_f length must match number of ages ($na).")

    M_mat = zeros(Float64, na, ny)
    if isa(M, Real)
        M_mat .= Float64(M)
    elseif isa(M, AbstractVector)
        for j in 1:ny
            M_mat[:, j] .= M
        end
    else
        M_mat .= M
    end

    N = zeros(Float64, na, ny)
    F = zeros(Float64, na, ny)

    # Terminal year
    F[:, ny] .= terminal_f
    for a in 1:na
        fa = F[a, ny]
        za = fa + M_mat[a, ny]
        if fa > 0.0
            N[a, ny] = catches[a, ny] / ((fa / za) * (1.0 - exp(-za)))
        end
    end

    # Oldest age class (plus-group) assumption across historical years: F_{na, t} = terminal_f[na]
    for t in 1:(ny - 1)
        F[na, t] = terminal_f[na]
        fa = terminal_f[na]
        za = fa + M_mat[na, t]
        if fa > 0.0
            N[na, t] = catches[na, t] / ((fa / za) * (1.0 - exp(-za)))
        end
    end

    # Backward recursion across cohorts
    for a in reverse(1:(na - 1))
        for t in reverse(1:(ny - 1))
            n_next = N[a + 1, t + 1]
            c_val = catches[a, t]
            m_val = M_mat[a, t]

            if n_next <= 0.0 || c_val <= 0.0
                N[a, t] = 0.0
                F[a, t] = 0.0
                continue
            end

            # Target ratio: C / N_{t+1}
            target_ratio = c_val / n_next

            # Initial guess via Pope's approximation, then solve the exact Baranov
            # catch equation for F with the package's shared scalar root solver.
            f_guess = log((n_next * exp(m_val) + c_val * exp(0.5 * m_val)) / n_next) - m_val

            # Residual: (F/Z)(1 - exp(-Z))/exp(-Z) - target, zero at the answer.
            residual = let mv = m_val, tr = target_ratio
                f -> begin
                    z = f + mv
                    ez = exp(-z)
                    return (f / z) * ((1.0 - ez) / ez) - tr
                end
            end
            f_cur = Solvers.solve_scalar_root(residual, max(0.01, f_guess);
                                               lower = 0.0, upper = 20.0, tol = tol)

            F[a, t] = f_cur
            z_final = f_cur + m_val
            N[a, t] = catches[a, t] / ((f_cur / z_final) * (1.0 - exp(-z_final)))
        end
    end

    c_pred = zeros(Float64, na, ny)
    for t in 1:ny
        for a in 1:na
            za = F[a, t] + M_mat[a, t]
            if za > 0.0
                c_pred[a, t] = (F[a, t] / za) * (1.0 - exp(-za)) * N[a, t]
            end
        end
    end

    return VpaResult(N, F, vec(sum(N, dims = 1)),
                     _spawning_abundance(N, weight_at_age, maturity), c_pred)
end
