# ------------------------------------------------------------------------------------
# Retrospective Analysis & Mohn's Rho Diagnostics for Stock Assessment Models
# Supports sequential retrospective peeling, bias metrics, and diagnostic reporting.
# ------------------------------------------------------------------------------------

"""
    mohn_rho(terminal_peeled::AbstractVector{<:Real}, terminal_base::AbstractVector{<:Real}) -> Float64

Calculate Mohn's rho (ρ), the standard fisheries diagnostic metric for retrospective bias
resulting from sequential truncation (peeling) of observation years (Mohn 1999):

```math
\\rho = \\frac{1}{P} \\sum_{p=1}^P \\frac{\\hat{X}_{T-p, p} - \\hat{X}_{T-p, 0}}{\\hat{X}_{T-p, 0}}
```

# Arguments
- `terminal_peeled`: Vector of terminal-year estimates from each retrospective peel ``p = 1, \\dots, P``.
- `terminal_base`: Vector of corresponding estimates for those same years from the unpeeled base model.

# Output
Scalar ``\\rho``. Values in ``[-0.15, +0.20]`` typically indicate acceptable retrospective consistency (Hurtado-Ferro et al. 2015).
"""
function mohn_rho(terminal_peeled::AbstractVector{<:Real},
                  terminal_base::AbstractVector{<:Real})
    length(terminal_peeled) == length(terminal_base) ||
        error("terminal_peeled and terminal_base must have equal lengths.")
    n = length(terminal_peeled)
    n == 0 && return 0.0

    total_rel = 0.0
    valid_count = 0
    for i in 1:n
        b = Float64(terminal_base[i])
        p = Float64(terminal_peeled[i])
        if abs(b) > 1e-6
            total_rel += (p - b) / b
            valid_count += 1
        end
    end
    return valid_count > 0 ? (total_rel / valid_count) : 0.0
end

"""
Result of retrospective peel analysis for a stock assessment model.

# Fields
- `peel_labels::Vector{String}`: Labels for peels (e.g. `["Peel 0 (Base)", "Peel -1", ...]`).
- `peel_years::Vector{Vector{Int}}`: Assessment years corresponding to each peel.
- `biomass_trajectories::Vector{Vector{Float64}}`: Reconstructed biomass time series for each peel.
- `fishing_mortalities::Vector{Vector{Float64}}`: Reconstructed fishing mortality series for each peel.
- `terminal_years::Vector{Int}`: Calendar year of the terminal assessment for each peel.
- `terminal_biomass_peeled::Vector{Float64}`: Terminal biomass estimated by each truncated model.
- `terminal_biomass_base::Vector{Float64}`: Biomass in that same terminal year estimated by the full base model.
- `rho_biomass::Float64`: Mohn's rho calculated for stock biomass.
- `rho_f::Float64`: Mohn's rho calculated for fishing mortality.
"""
struct RetrospectiveResult
    peel_labels::Vector{String}
    peel_years::Vector{Vector{Int}}
    biomass_trajectories::Vector{Vector{Float64}}
    fishing_mortalities::Vector{Vector{Float64}}
    terminal_years::Vector{Int}
    terminal_biomass_peeled::Vector{Float64}
    terminal_biomass_base::Vector{Float64}
    rho_biomass::Float64
    rho_f::Float64
end

"""
    retrospective_analysis(years, removals;
                           n_peels = 5,
                           r = 0.35, k = 100.0, b0 = 80.0, q = 1.0,
                           survey_transition = 0) -> RetrospectiveResult

Perform sequential retrospective analysis for a discrete surplus production model:
- Re-runs simulation / estimation sequentially dropping up to `n_peels` terminal years.
- Extracts terminal year estimates for each peel and compares them to the unpeeled base model.
- Computes Mohn's rho for exploitable biomass.
"""
function retrospective_analysis(years::AbstractVector{<:Integer},
                                removals::AbstractVector{<:Real};
                                n_peels::Integer = 5,
                                r::Real = 0.35,
                                k::Real = 100.0,
                                b0::Real = 80.0,
                                q::Real = 1.0,
                                survey_transition::Integer = 0)
    nt = length(years)
    n_peels < nt - 2 || error("n_peels must be at least 2 years less than total assessment length.")

    # Base run (Peel 0)
    base_res = simulate_logistic_discrete(years, removals; r = r, k = k, b0 = b0,
                                          q = q, survey_transition = survey_transition)

    labels = ["Base (0)"]
    peel_yrs = [collect(years)]
    b_trajs = [base_res.biomass]
    f_trajs = [base_res.harvest_rate]

    term_yrs = Int[]
    term_b_peeled = Float64[]
    term_b_base = Float64[]
    term_f_peeled = Float64[]
    term_f_base = Float64[]

    for p in 1:n_peels
        sub_len = nt - p
        sub_years = years[1:sub_len]
        sub_removals = removals[1:sub_len]

        peel_res = simulate_logistic_discrete(sub_years, sub_removals; r = r, k = k, b0 = b0,
                                              q = q, survey_transition = survey_transition)

        push!(labels, "Peel -$p")
        push!(peel_yrs, collect(sub_years))
        push!(b_trajs, peel_res.biomass)
        push!(f_trajs, peel_res.harvest_rate)

        t_yr = sub_years[end]
        push!(term_yrs, t_yr)
        push!(term_b_peeled, peel_res.biomass[end])
        push!(term_b_base, base_res.biomass[sub_len])
        push!(term_f_peeled, peel_res.harvest_rate[end])
        push!(term_f_base, base_res.harvest_rate[sub_len])
    end

    rho_b = mohn_rho(term_b_peeled, term_b_base)
    rho_f = mohn_rho(term_f_peeled, term_f_base)

    return RetrospectiveResult(
        labels, peel_yrs, b_trajs, f_trajs, term_yrs,
        term_b_peeled, term_b_base, rho_b, rho_f
    )
end

"""
    retrospective_analysis(catches::AbstractMatrix{<:Real}, survey_index::AbstractMatrix, M;
                           n_peels = 4,
                           survey_ages = 1:size(catches, 1)) -> RetrospectiveResult

Perform sequential retrospective analysis for ADAPT Virtual Population Analysis:
- Sequentially deletes terminal years of catch and survey matrices.
- Calibrates terminal fishing mortalities and reconstructs population trajectories.
- Evaluates Mohn's rho on spawning stock biomass and average fishing mortality.
"""
function retrospective_analysis(catches::AbstractMatrix{<:Real},
                                survey_index::AbstractMatrix,
                                M;
                                n_peels::Integer = 4,
                                survey_ages::AbstractVector{<:Integer} = 1:size(catches, 1))
    na, ny = size(catches)
    n_peels < ny - 2 || error("n_peels must be less than $(ny - 2).")

    # Base ADAPT run
    base_adapt = adapt_vpa(catches, survey_index, M; survey_ages = survey_ages)
    base_ssb = base_adapt.vpa.biomass_ssb
    base_f = vec(mean(base_adapt.vpa.F, dims = 1))

    labels = ["Base (0)"]
    peel_yrs = [collect(1:ny)]
    b_trajs = [base_ssb]
    f_trajs = [base_f]

    term_yrs = Int[]
    term_b_peeled = Float64[]
    term_b_base = Float64[]
    term_f_peeled = Float64[]
    term_f_base = Float64[]

    for p in 1:n_peels
        sub_ny = ny - p
        sub_c = catches[:, 1:sub_ny]
        sub_s = survey_index[:, 1:sub_ny]

        sub_adapt = adapt_vpa(sub_c, sub_s, M; survey_ages = survey_ages)
        sub_ssb = sub_adapt.vpa.biomass_ssb
        sub_f = vec(mean(sub_adapt.vpa.F, dims = 1))

        push!(labels, "Peel -$p")
        push!(peel_yrs, collect(1:sub_ny))
        push!(b_trajs, sub_ssb)
        push!(f_trajs, sub_f)

        push!(term_yrs, sub_ny)
        push!(term_b_peeled, sub_ssb[end])
        push!(term_b_base, base_ssb[sub_ny])
        push!(term_f_peeled, sub_f[end])
        push!(term_f_base, base_f[sub_ny])
    end

    rho_b = mohn_rho(term_b_peeled, term_b_base)
    rho_f = mohn_rho(term_f_peeled, term_f_base)

    return RetrospectiveResult(
        labels, peel_yrs, b_trajs, f_trajs, term_yrs,
        term_b_peeled, term_b_base, rho_b, rho_f
    )
end
