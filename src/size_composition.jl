# ------------------------------------------------------------------------------------
# Size composition (length frequency) handling for Stock Synthesis.
# Inputs are typically individual fish sizes (lengths); this module handles
# binning, expansion, and likelihood computation.
# ------------------------------------------------------------------------------------

"""
    SizeCompositionData

Container for size composition (length frequency) observations.

# Fields
- `lengths::Vector{Float64}`: Individual fish lengths (raw measurements).
- `counts::Vector{Int}`: Number of fish per length bin (if pre-binned).
- `bin_edges::Vector{Float64}`: Bin edges for length classes.
- `n_fish::Int`: Total number of fish measured.
- `fleet::Int`: Fleet/survey index.
- `year::Int`: Year of observation.
"""
struct SizeCompositionData
    lengths::Vector{Float64}
    counts::Vector{Int}
    bin_edges::Vector{Float64}
    n_fish::Int
    fleet::Int
    year::Int
end

"""
    SizeCompositionData(lengths::AbstractVector{<:Real}; fleet=1, year=1) -> SizeCompositionData

Construct from individual length measurements. Bins are computed automatically.
"""
function SizeCompositionData(lengths::AbstractVector{<:Real}; fleet::Integer = 1, year::Integer = 1,
                            n_bins::Integer = 20)
    ls = Float64.(lengths)
    edges = collect(range(minimum(ls), maximum(ls), length = n_bins + 1))
    counts = zeros(Int, n_bins)
    for l in ls
        for i in 1:n_bins
            if edges[i] <= l < edges[i+1] || (i == n_bins && l == edges[end])
                counts[i] += 1
                break
            end
        end
    end
    return SizeCompositionData(ls, counts, edges, length(ls), fleet, year)
end

"""
    SizeCompositionData(counts::AbstractVector{<:Integer}, bin_edges::AbstractVector{<:Real};
                       fleet=1, year=1) -> SizeCompositionData

Construct from pre-binned counts.
"""
function SizeCompositionData(counts::AbstractVector{<:Integer}, bin_edges::AbstractVector{<:Real};
                            fleet::Integer = 1, year::Integer = 1)
    ls = Float64[]
    for (i, c) in enumerate(counts)
        mid = (bin_edges[i] + bin_edges[i+1]) / 2
        append!(ls, fill(mid, c))
    end
    return SizeCompositionData(ls, Int.(counts), Float64.(bin_edges), sum(counts), fleet, year)
end

"""
    bin_lengths(lengths, bin_edges) -> Vector{Int}

Bin individual length measurements into size classes defined by `bin_edges`.
"""
function bin_lengths(lengths::AbstractVector{<:Real}, bin_edges::AbstractVector{<:Real})
    n_bins = length(bin_edges) - 1
    counts = zeros(Int, n_bins)
    for l in lengths
        for i in 1:n_bins
            if bin_edges[i] <= l < bin_edges[i+1] || (i == n_bins && l == bin_edges[end])
                counts[i] += 1
                break
            end
        end
    end
    return counts
end

"""
    size_proportions(sc::SizeCompositionData) -> Vector{Float64}

Convert counts to proportions (summing to 1).
"""
function size_proportions(sc::SizeCompositionData)
    total = sum(sc.counts)
    total > 0 || return zeros(Float64, length(sc.counts))
    return Float64.(sc.counts) ./ total
end

"""
    size_proportions(counts::AbstractVector{<:Real}) -> Vector{Float64}

Convert counts to proportions.
"""
function size_proportions(counts::AbstractVector{<:Real})
    total = sum(counts)
    total > 0 || return zeros(Float64, length(counts))
    return Float64.(counts) ./ total
end

"""
    effective_sample_size(counts::AbstractVector{<:Real}; proc_err::Real = 0.009) -> Float64

Compute effective sample size with process error adjustment (HCAM method):
N_eff = N / (1 + proc_err * N) where N = sum(counts).
"""
function effective_sample_size(counts::AbstractVector{<:Real}; proc_err::Real = 0.009)
    ntot = sum(counts)
    return ntot / (1.0 + proc_err * ntot)
end

"""
    dirichlet_multinomial_nll(obs, pred, n_eff) -> Float64

Dirichlet-multinomial NLL via Distributions.jl.
"""
function dirichlet_multinomial_nll(obs::AbstractVector{<:Real}, pred::AbstractVector{<:Real},
                                   n_eff::Real)
    o = Float64.(obs)
    p = Float64.(pred)
    po = sum(o)
    pp = sum(p)
    po > 0 || return 0.0
    pp > 0 || return Inf
    o ./= po
    p ./= pp
    n = n_eff > 0 ? n_eff : po
    alpha = n .* p
    obs_counts = Int.(round.(o .* n))
    dm = DirichletMultinomial(Int(n), alpha)
    return -logpdf(dm, obs_counts)
end

using Distributions

"""
    multinomial_nll(obs, pred; n_eff=0.0) -> Float64

Multinomial negative log-likelihood. Delegates to Distributions.jl.
"""
function multinomial_nll(obs::AbstractVector{<:Real}, pred::AbstractVector{<:Real};
                         n_eff::Real = 0.0)
    o = Float64.(obs)
    p = Float64.(pred)
    po = sum(o)
    pp = sum(p)
    po > 0 || return 0.0
    pp > 0 || return Inf
    o ./= po
    p ./= pp
    n = n_eff > 0 ? n_eff : po
    # Convert proportions back to counts for Multinomial
    obs_counts = Int.(round.(o .* n))
    pred_prop = p
    mn = Multinomial(Int(n), pred_prop)
    return -logpdf(mn, obs_counts)
end

"""
    size_composition_residual(obs::AbstractVector{<:Real}, pred::AbstractVector{<:Real};
                             likelihood::Symbol = :multinomial,
                             n_eff::Real = 0.0, proc_err::Real = 0.009) -> Float64

Compute negative log-likelihood for size composition data.

Likelihood types:
- `:multinomial` — standard multinomial
- `:dirichlet_multinomial` — overdispersed multinomial
- `:lognormal` — lognormal on proportions
"""
function size_composition_residual(obs::AbstractVector{<:Real}, pred::AbstractVector{<:Real};
                                    likelihood::Symbol = :multinomial,
                                    n_eff::Real = 0.0, proc_err::Real = 0.009)
    if n_eff == 0.0
        n_eff = effective_sample_size(obs; proc_err = proc_err)
    end

    if likelihood == :multinomial
        return multinomial_nll(obs, pred; n_eff = n_eff)
    elseif likelihood == :dirichlet_multinomial
        return dirichlet_multinomial_nll(obs, pred, n_eff)
    elseif likelihood == :lognormal
        o = Float64.(obs); p = Float64.(pred)
        po = sum(o); pp = sum(p)
        po > 0 || return 0.0
        pp > 0 || return Inf
        o ./= po; p ./= pp
        nll = 0.0
        for i in eachindex(o)
            if o[i] > 0 && p[i] > 0
                nll += 0.5 * (log(o[i]) - log(p[i]))^2
            end
        end
        return nll
    else
        error("Unknown likelihood type: $likelihood. Use :multinomial, :dirichlet_multinomial, or :lognormal.")
    end
end