# ------------------------------------------------------------------------------------
# Size-structured population dynamics for Stock Synthesis.
# ------------------------------------------------------------------------------------

using Distributions
using SpecialFunctions

"""
    SizeClass

A size (length) class in the size-structured model.

# Fields
- `midpoint::Float64`: Midpoint of the size bin [length units].
- `lower::Float64`: Lower bound of the size bin.
- `upper::Float64`: Upper bound of the size bin.
- `width::Float64`: Width of the size bin.
"""
struct SizeClass
    midpoint::Float64
    lower::Float64
    upper::Float64
    width::Float64
end

"""
    GrowthParameters

von Bertalanffy growth parameters for size-structured dynamics.

# Fields
- `linf::Float64`: Asymptotic length L_inf.
- `k::Float64`: Growth rate coefficient.
- `t0::Float64`: Theoretical age at zero length.
- `cv::Float64`: Coefficient of variation of length-at-age (for size transition).
"""
struct GrowthParameters
    linf::Float64
    k::Float64
    t0::Float64
    cv::Float64
end

"""
    GrowthParameters(linf, k; t0 = 0.0, cv = 0.1) -> GrowthParameters

Construct growth parameters with defaults.
"""
function GrowthParameters(linf::Real, k::Real; t0::Real = 0.0, cv::Real = 0.1)
    linf > 0 || error("linf must be positive.")
    k > 0 || error("k must be positive.")
    0 < cv < 1 || error("cv must be in (0, 1).")
    return GrowthParameters(Float64(linf), Float64(k), Float64(t0), Float64(cv))
end

"""
    SizeStructuredModel

Container for a size-structured stock assessment model.

# Fields
- `n_size::Int`: Number of size classes.
- `size_classes::Vector{SizeClass}`: Size class definitions.
- `growth::GrowthParameters`: von Bertalanffy growth parameters.
- `n_years::Int`: Number of years.
- `n_fleets::Int`: Number of fishing fleets.
- `n_surveys::Int`: Number of survey series.
- `m::Float64`: Natural mortality rate.
- `m_size::Vector{Float64}`: Size-specific natural mortality (optional).
- `w_at_size::Vector{Float64}`: Weight at size (midpoint).
- `mat_at_size::Vector{Float64}`: Maturity at size.
- `fec_at_size::Vector{Float64}`: Fecundity at size.
- `size_weights::Matrix{Float64}`: Weighting factor per size group (default 1.0), for factors unrelated to size selectivity (e.g., habitat differences).
- `sel_params::Matrix{Float64}`: Catch selectivity parameters (n_fleets x n_sel_params).
- `survey_sel_params::Matrix{Float64}`: Survey selectivity parameters (n_surveys x n_sel_params).
- `q::Vector{Float64}`: Survey catchabilities.
"""
struct SizeStructuredModel
    n_size::Int
    n_sexes::Int
    size_classes::Vector{SizeClass}
    growth::GrowthParameters
    n_years::Int
    n_fleets::Int
    n_surveys::Int
    m::Float64
    m_size::Matrix{Float64}          # (n_sexes x n_size)
    w_at_size::Matrix{Float64}       # (n_sexes x n_size)
    mat_at_size::Matrix{Float64}     # (n_sexes x n_size) — sex-specific maturity oogives
    fec_at_size::Matrix{Float64}     # (n_sexes x n_size)
    size_weights::Matrix{Float64}     # (n_sexes x n_size) — weighting factor per size group (default 1.0)
    sel_params::Matrix{Float64}      # (n_fleets x n_sel_params) — catch selectivity
    survey_sel_params::Array{Float64,3}  # (n_surveys x n_sexes x n_sel_params) — sex-specific survey selectivity
    q::Vector{Float64}              # (n_surveys)
end

"""
    make_size_classes(min_size, max_size, n_classes) -> Vector{SizeClass}

Create evenly spaced size classes.
"""
function make_size_classes(min_size::Real, max_size::Real, n_classes::Integer)
    n_classes >= 2 || error("Need at least 2 size classes.")
    min_size < max_size || error("min_size must be less than max_size.")
    edges = collect(range(min_size, max_size, length = n_classes + 1))
    classes = SizeClass[]
    for i in 1:n_classes
        push!(classes, SizeClass(
            (edges[i] + edges[i+1]) / 2,
            edges[i],
            edges[i+1],
            edges[i+1] - edges[i]
        ))
    end
    return classes
end

"""
    von_bertalanffy_length(age, growth::GrowthParameters) -> Float64

von Bertalanffy length-at-age: L(a) = L_inf * (1 - exp(-k * (a - t0))).
"""
function von_bertalanffy_length(age::Real, growth::GrowthParameters)
    return growth.linf * (1.0 - exp(-growth.k * (age - growth.t0)))
end

"""
    size_transition_matrix(growth::GrowthParameters, size_classes::Vector{SizeClass},
                           dt::Real = 1.0) -> Matrix{Float64}

Compute the size transition (growth) matrix G where G[i, j] is the probability
that an individual in size class j grows to size class i in one time step.

Uses a normal distribution around the expected growth increment with CV from
growth parameters.
"""
function size_transition_matrix(growth::GrowthParameters, size_classes::Vector{SizeClass},
                                dt::Real = 1.0)
    n = length(size_classes)
    G = zeros(Float64, n, n)
    midpoints = [sc.midpoint for sc in size_classes]

    for j in 1:n
        # Expected size next year for an individual currently at midpoint j
        # Use inverse von Bertalanffy: find age at current size, add dt, get new size
        Lj = size_classes[j].midpoint
        # age at current size: a = t0 - log(1 - L/Linf) / k
        if Lj >= growth.linf
            # Already at asymptotic size; stay
            G[n, j] = 1.0
            continue
        end
        age_j = growth.t0 - log(1.0 - Lj / growth.linf) / growth.k
        age_next = age_j + dt
        L_next = von_bertalanffy_length(age_next, growth)
        L_next = min(L_next, growth.linf * 0.99)

        # Growth increment std
        sd_growth = growth.cv * (growth.linf - Lj)

        # Distribute probability across size classes using normal CDF
        for i in 1:n
            lo = size_classes[i].lower
            hi = size_classes[i].upper
            if sd_growth > 0
                dist = Normal(L_next, sd_growth)
                G[i, j] = cdf(dist, hi) - cdf(dist, lo)
            else
                G[i, j] = (lo <= L_next < hi) ? 1.0 : 0.0
            end
        end
        # Normalize row (column j sums to 1)
        col_sum = sum(G[:, j])
        if col_sum > 0
            G[:, j] ./= col_sum
        else
            G[n, j] = 1.0  # fallback: stay in largest class
        end
    end
    return G
end

"""
    size_selectivity(size_midpoints, params::AbstractVector{<:Real};
                     type::Symbol = :logistic) -> Vector{Float64}

Compute selectivity at size for a given parameter vector.

Types:
- `:logistic`: params = [a50, slope] (size at 50% selectivity, slope)
- `:double_normal`: params = [peak, asc, desc, ...] (simplified)
- `:knife_edge`: params = [min_size] (all sizes >= min_size selected)
"""
function size_selectivity(size_midpoints::AbstractVector{<:Real},
                          params::AbstractVector{<:Real};
                          type::Symbol = :logistic)
    n = length(size_midpoints)
    sel = zeros(Float64, n)
    if type == :logistic
        length(params) >= 2 || error("Logistic selectivity requires [a50, slope].")
        a50, slope = params[1], params[2]
        for i in 1:n
            sel[i] = 1.0 / (1.0 + exp(-slope * (size_midpoints[i] - a50)))
        end
    elseif type == :knife_edge
        length(params) >= 1 || error("Knife-edge selectivity requires [min_size].")
        for i in 1:n
            sel[i] = size_midpoints[i] >= params[1] ? 1.0 : 0.0
        end
    elseif type == :double_normal
        length(params) >= 4 || error("Double-normal requires [peak, asc_sd, desc_sd, plateau].")
        peak, asc_sd, desc_sd, plateau = params[1], params[2], params[3], params[4]
        for i in 1:n
            x = size_midpoints[i]
            if x <= peak
                sel[i] = exp(-0.5 * ((x - peak) / asc_sd)^2)
            else
                sel[i] = exp(-0.5 * ((x - peak) / desc_sd)^2) * plateau
            end
        end
    else
        error("Unknown selectivity type: $type")
    end
    return sel
end

"""
    SizeStructuredModel constructor from parameters.
"""
function SizeStructuredModel(size_classes::Vector{SizeClass},
                             growth::GrowthParameters,
                             n_years::Integer, n_fleets::Integer, n_surveys::Integer;
                             n_sexes::Integer = 1,
                             m::Real = 0.2,
                             m_size::Union{Nothing, AbstractVecOrMat{<:Real}} = nothing,
                             w_at_size::Union{Nothing, AbstractVecOrMat{<:Real}} = nothing,
                             mat_at_size::Union{Nothing, AbstractVecOrMat{<:Real}} = nothing,
                             fec_at_size::Union{Nothing, AbstractVecOrMat{<:Real}} = nothing,
                             size_weights::Union{Nothing, AbstractVecOrMat{<:Real}} = nothing,
                             n_sel_params::Integer = 2,
                             survey_n_sel_params::Integer = 2,
                             q::Union{Nothing, AbstractVector{<:Real}} = nothing,
                             survey_sel_params::Union{Nothing, AbstractArray{<:Real,3}} = nothing,
                             catch_sel_params::Union{Nothing, AbstractMatrix{<:Real}} = nothing)
    n = length(size_classes)

    # Helper: convert 1D input to (n_sexes x n) matrix
    to_sex_matrix(v, default_fn) = if isnothing(v)
        default_fn()
    elseif ndims(v) == 1
        repeat(Float64.(v)', n_sexes, 1)
    else
        Float64.(v)
    end

    m_size_mat = to_sex_matrix(m_size, () -> fill(Float64(m), n_sexes, n))
    w_mat = to_sex_matrix(w_at_size, () -> repeat([sc.midpoint^3 * 1e-5 for sc in size_classes]', n_sexes, 1))
    mat_mat = to_sex_matrix(mat_at_size, () -> repeat([sc.midpoint > 20.0 ? 1.0 : 0.0 for sc in size_classes]', n_sexes, 1))
    fec_mat = if isnothing(fec_at_size)
        w_mat .* mat_mat
    elseif ndims(fec_at_size) == 1
        repeat(Float64.(fec_at_size)', n_sexes, 1)
    else
        Float64.(fec_at_size)
    end

    # Size weights: (n_sexes x n_size), defaulting to 1.0
    sw_mat = if isnothing(size_weights)
        ones(Float64, n_sexes, n)
    elseif ndims(size_weights) == 1
        repeat(Float64.(size_weights)', n_sexes, 1)
    else
        Float64.(size_weights)
    end

    # Catch selectivity: (n_fleets x n_sel_params)
    sel = isnothing(catch_sel_params) ? zeros(Float64, n_fleets, n_sel_params) : Float64.(catch_sel_params)

    # Survey selectivity: (n_surveys x n_sexes x n_sel_params) — sex-specific
    if isnothing(survey_sel_params)
        ssp = zeros(Float64, n_surveys, n_sexes, survey_n_sel_params)
        for s in 1:n_surveys, sx in 1:n_sexes
            if survey_n_sel_params >= 2
                ssp[s, sx, 1] = size_classes[1].midpoint
                ssp[s, sx, 2] = 0.0
            end
        end
    else
        ssp = Float64.(survey_sel_params)
        size(ssp) == (n_surveys, n_sexes, survey_n_sel_params) ||
            error("survey_sel_params must be (n_surveys, n_sexes, survey_n_sel_params)")
    end

    q_vec = isnothing(q) ? ones(Float64, n_surveys) : Float64.(q)
    return SizeStructuredModel(n, n_sexes, size_classes, growth, n_years, n_fleets, n_surveys,
                               Float64(m), m_size_mat, w_mat, mat_mat, fec_mat, sw_mat,
                               sel, ssp, q_vec)
end