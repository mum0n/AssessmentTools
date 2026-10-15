# ------------------------------------------------------------------------------------
# Stage-structured (Leslie / Lefkovitch) matrix population models.
#
# A single projection matrix is built once from named parameters; the deterministic
# projection, the demographic analysis (growth rate, stable stage distribution,
# reproductive value, net reproductive rate, generation time and elasticities) and the
# Bayesian state-space model all consume that same matrix, so the three views cannot
# diverge. Elasticities and eigenvalues use `LinearAlgebra.eigen`; nothing is
# reimplemented locally.
#
# The framework follows Caswell (2001), "Matrix Population Models"
# (ISBN 0-87893-096-5) for construction, analysis and sensitivity calculation.
# ------------------------------------------------------------------------------------

module StageMatrix

using LinearAlgebra
using Statistics
using Distributions
using Turing
using ..Config
using ..Solvers
using ..InferenceBridge: estimate

export StageMatrixConfig, StageMatrixResult, load_stage_matrix_config
export build_projection_matrix, project_stage_matrix
export net_reproductive_rate, generation_time, stable_stage_distribution
export reproductive_value, stage_elasticities
export bayesian_stage_matrix_model

"""
    StageMatrixConfig

Fully user-specified stage-structured population configuration. Every field is read
from a configuration file via [`load_stage_matrix_config`](@ref); nothing is defaulted
in code.

# Fields
- `n_stages::Int`: number of life stages, ordered youngest to oldest.
- `fertility::Vector{Float64}`: expected offspring produced per capita per time step by
  each stage. Only the last stages are normally non-zero.
- `survival::Vector{Float64}`: probability of surviving the step within each stage.
- `retention::Vector{Float64}`: probability of staying in the same stage given survival.
- `transition::Vector{Float64}`: probability of advancing one stage given survival.
  `retention[k] + transition[k] <= 1`.
- `stage_weights::Vector{Float64}`: weight (biomass) per individual in each stage.
- `maturity::Vector{Float64}`: spawning-age fraction of each stage, used only for
  reporting a maturation-weighted index, not for fertility (fertility is already
  per capita).
- `catchability::Vector{Float64}`: survey catchability per stage.
- `initial_abundance::Vector{Float64}`: starting stage abundances.
- `n_years::Int`: number of time steps.
"""
struct StageMatrixConfig
    n_stages::Int
    fertility::Vector{Float64}
    survival::Vector{Float64}
    retention::Vector{Float64}
    transition::Vector{Float64}
    stage_weights::Vector{Float64}
    maturity::Vector{Float64}
    catchability::Vector{Float64}
    initial_abundance::Vector{Float64}
    n_years::Int
end

"""
    load_stage_matrix_config(path::AbstractString) -> StageMatrixConfig

Read a stage-matrix configuration from a TOML file. Every key under `[stages]` is
required; a missing key raises rather than silently defaulting, so the file fully
states the model.

Expected keys:

| key | meaning |
|:---|:---|
| `n_stages` | number of stages |
| `fertility` | per-capita offspring per stage, youngest to oldest |
| `survival` | within-stage survival per stage |
| `retention` | probability of staying put given survival |
| `transition` | probability of advancing given survival |
| `stage_weights` | biomass per individual per stage |
| `maturity` | maturing fraction per stage |
| `catchability` | survey catchability per stage |
| `initial_abundance` | starting abundance per stage |
| `n_years` | number of projection steps |
"""
function load_stage_matrix_config(path::AbstractString)
    cfg = Config.parse_file(path)
    d = Config.require_section(cfg, "stages", path)
    where = "$path section [stages]"

    n = Config.load_int(d, "n_stages", where)
    n >= 2 || error("'n_stages' in $where must be at least 2, got $n.")

    fertility = Config.load_reals(d, "fertility", where)
    survival = Config.load_reals(d, "survival", where)
    retention = Config.load_reals(d, "retention", where)
    transition = Config.load_reals(d, "transition", where)
    weights = Config.load_reals(d, "stage_weights", where)
    mat = Config.load_reals(d, "maturity", where)
    q = Config.load_reals(d, "catchability", where)
    n0 = Config.load_reals(d, "initial_abundance", where)
    ny = Config.load_int(d, "n_years", where)

    for (nm, v) in (("fertility", fertility), ("survival", survival),
                    ("retention", retention), ("transition", transition),
                    ("stage_weights", weights), ("maturity", mat),
                    ("catchability", q), ("initial_abundance", n0))
        length(v) == n ||
            error("'$nm' in $where must have $n entries (one per stage), got $(length(v)).")
    end
    for (nm, v) in (("survival", survival), ("retention", retention),
                    ("transition", transition), ("maturity", mat))
        for x in v
            0 <= x <= 1 ||
                error("'$nm' in $where must lie in [0, 1] (got $x).")
        end
    end
    for k in 1:n
        retention[k] + transition[k] <= 1 + 1.0e-12 ||
            error("'retention[$k]'+'transition[$k]' in $where exceeds 1 (got $(retention[k] + transition[k])).")
    end
    ny >= 1 || error("'n_years' in $where must be at least 1, got $ny.")

    return StageMatrixConfig(n, fertility, survival, retention, transition,
                             weights, mat, q, n0, ny)
end

"""
    build_projection_matrix(cfg::StageMatrixConfig) -> Matrix{Float64}

Assemble the Lefkovitch projection matrix ``A`` with `n_stages` rows:

```math
A_{1,k} = f_k, \\qquad A_{k,k} = s_k\\,r_k, \\qquad A_{k+1,k} = s_k\\,g_k
```

where `f` is fertility, `s` survival, `r` retention and `g` transition. Sub-diagonal
entries are the advancing fluxes; the first row is reproduction into the youngest stage.
This is the single construction site: the deterministic projection, the demographic
analysis and the Bayesian model all use the matrix built here.
"""
function build_projection_matrix(cfg::StageMatrixConfig)
    n = cfg.n_stages
    A = zeros(Float64, n, n)
    A[1, :] .= cfg.fertility
    for k in 1:n
        A[k, k] = cfg.survival[k] * cfg.retention[k]
        if k < n
            A[k + 1, k] = cfg.survival[k] * cfg.transition[k]
        end
    end
    return A
end

"""
    StageMatrixResult

Output of [`project_stage_matrix`](@ref).

# Fields
- `abundance::Matrix{Float64}`: abundances, `n_stages × (n_years + 1)`, column 1 being
  the initial state.
- `biomass::Vector{Float64}`: total biomass per time step.
- `spawning_biomass::Vector{Float64}`: maturation-weighted biomass per time step.
- `projection_matrix::Matrix{Float64}`: the matrix used.
- `growth_rate::Float64`: dominant eigenvalue `lambda` of the projection matrix.
"""
struct StageMatrixResult
    abundance::Matrix{Float64}
    biomass::Vector{Float64}
    spawning_biomass::Vector{Float64}
    projection_matrix::Matrix{Float64}
    growth_rate::Float64
end

"""
    project_stage_matrix(cfg::StageMatrixConfig; A = build_projection_matrix(cfg)) -> StageMatrixResult

Iterate ``N_{t+1} = A N_t`` for `cfg.n_years` steps and report stage abundances,
biomass and maturation-weighted biomass. Eigen-decomposition of `A` is delegated to
`LinearAlgebra.eigen`; the growth rate is its dominant eigenvalue.
"""
function project_stage_matrix(cfg::StageMatrixConfig;
                              A::AbstractMatrix = build_projection_matrix(cfg))
    n = cfg.n_stages
    size(A) == (n, n) || error("Projection matrix must be $n x $n, got $(size(A)).")
    N = zeros(Float64, n, cfg.n_years + 1)
    N[:, 1] .= cfg.initial_abundance
    for t in 1:cfg.n_years
        N[:, t + 1] .= A * N[:, t]
    end
    biomass = [sum(N[:, t] .* cfg.stage_weights) for t in axes(N, 2)]
    spawning = [sum(N[:, t] .* cfg.stage_weights .* cfg.maturity) for t in axes(N, 2)]
    lambda = _dominant_eigenvalue(A)
    return StageMatrixResult(N, biomass, spawning, Matrix(A), lambda)
end

"""
    _dominant_eigenvalue(A) -> Float64

Largest-magnitude eigenvalue of a square matrix, via `LinearAlgebra.eigen`. For a
non-negative irreducible matrix this is the asymptotic growth rate `lambda`.
"""
function _dominant_eigenvalue(A::AbstractMatrix)
    ev = eigvals(A)
    return real(ev[argmax(abs.(ev))])
end

"""
    _logistic(x) -> Float64

Standard logistic transform, used to map unbounded sampled values onto `(0, 1)` for
probabilities. Written once and used by every probability parameter in the model.
Accepts arrays so broadcasting works, which `ForwardDiff` duals require.
"""
_logistic(x) = 1.0 ./ (1.0 .+ exp.(-x))

"""
    net_reproductive_rate(A) -> Float64

Net reproductive rate ``R_0``: the expected lifetime number of offspring produced by
one individual entering the first (youngest) stage, per Caswell (2001) ch. 4.

The life-cycle graph is split into reproduction ``F`` (the first row of `A`) and
within-generation transfer ``T`` (everything else). The expected time an individual
spends in each stage before dying, ``\\phi = (I - T)^{-1} e_1``, is `LinearAlgebra`'s
fundamental matrix applied to the entry vector, and

```math
R_0 = \\mathbf{f}^{T}\\phi
```

accumulates fertility over that residency. For a classical Leslie matrix this reduces to
the familiar ``\\sum_j f_j \\prod_{i<j} s_i``. With stage retention, ``\\phi`` correctly
counts time spent in a stage over multiple steps.
"""
function net_reproductive_rate(A::AbstractMatrix)
    n = size(A, 1)
    size(A, 2) == n || error("net_reproductive_rate requires a square matrix.")
    n >= 2 || error("net_reproductive_rate requires at least 2 stages.")

    f = A[1, :]
    T = copy(A)
    T[1, :] .= 0.0
    entry = zeros(Float64, n)
    entry[1] = 1.0
    residency = (I - T) \ entry
    return dot(f, residency)
end

"""
    generation_time(A) -> Float64

Mean generation time ``T = \\log R_0 / \\log \\lambda``, the average age of mothers at
birth, computed from [`net_reproductive_rate`](@ref) and the dominant eigenvalue.
Positive when the population is growing and `R_0` and `lambda` exceed 1.
"""
function generation_time(A::AbstractMatrix)
    r0 = net_reproductive_rate(A)
    lambda = _dominant_eigenvalue(A)
    (r0 > 0 && lambda > 0) || error("Cannot compute generation time for R0=$r0, lambda=$lambda.")
    return log(r0) / log(lambda)
end

"""
    stable_stage_distribution(A) -> Vector{Float64}

Right eigenvector of `A` for the dominant eigenvalue, normalised to sum to 1: the
asymptotic fraction of the population in each stage.
"""
function stable_stage_distribution(A::AbstractMatrix)
    lambda = _dominant_eigenvalue(A)
    F = eigen(A)
    # Pick the eigenvector whose eigenvalue matches the dominant one. `argmin` on the
    # absolute difference is required: `argmax` would select the least-matching vector.
    i = argmin(abs.(real.(F.values) .- lambda))
    v = real.(F.vectors[:, i])
    v .= abs.(v)
    v ./= sum(v)
    return v
end

"""
    reproductive_value(A) -> Vector{Float64}

Left eigenvector of `A` for the dominant eigenvalue, normalised so `w[1] = 1`: the
reproductive value of each stage, i.e. its expected contribution to population growth
relative to a newborn. `LinearAlgebra.eigen` returns the left eigenvectors as the
conjugated transpose of the vector matrix, so row `i` of `F.vectors'` is a left
eigenvector.
"""
function reproductive_value(A::AbstractMatrix)
    size(A, 1) == size(A, 2) || error("reproductive_value requires a square matrix.")
    lambda = _dominant_eigenvalue(A)

    # Left eigenvectors of A are the eigenvectors of A' = adjoint(A), conjugated and
    # transposed back. LinearAlgebra.eigen supplies these directly.
    FL = eigen(adjoint(A))
    j = argmin(abs.(real.(FL.values) .- lambda))
    w = real.(conj.(FL.vectors[:, j]))

    # Normalise so v' w = 1 with v the stable stage distribution (Caswell 2001, ch. 5),
    # then set the youngest stage to 1 so values are relative to a newborn.
    v = stable_stage_distribution(A)
    denom = dot(v, w)
    isapprox(denom, 0.0; atol = 1.0e-14) &&
        error("Left and right eigenvectors are orthogonal; cannot normalise (lambda = $lambda).")
    w ./= denom
    w ./= w[1]
    return w
end

"""
    stage_elasticities(A) -> Matrix{Float64}

Elasticities of `lambda` to each matrix element, per Caswell (2001) eq. 5.34:

```math
e_{ij} = \\frac{A_{ij}}{\\lambda}\\frac{v_i w_j}{v^T w}
```

Each entry is the proportional change in `lambda` produced by a proportional change in
`A[i, j]`; elasticities sum to 1. Negative or complex parts are dropped because `A` is
non-negative. `eigen` and the inner products come from `LinearAlgebra`.
"""
function stage_elasticities(A::AbstractMatrix)
    lambda = _dominant_eigenvalue(A)
    w = stable_stage_distribution(A)
    v = reproductive_value(A)
    v = v ./ v[1] * 1.0
    denom = sum(w .* v)
    isapprox(denom, 0.0; atol = 1.0e-12) &&
        error("Cannot normalise elasticity by w'v when it is zero.")
    E = zeros(Float64, size(A))
    @inbounds for j in axes(A, 2), i in axes(A, 1)
        E[i, j] = (A[i, j] / lambda) * (v[i] * w[j]) / denom
    end
    return E
end

# ------------------------------------------------------------------------------------
# Bayesian state-space version.
# ------------------------------------------------------------------------------------

"""
    bayesian_stage_matrix_model(survey_index, stage_biomass, weight_per_ind,
                                maturity, catchability, n_stages, n_years;
                                prior_fertility_mu = 0.0,
                                prior_fertility_sd = 1.0,
                                prior_survival_mu = 0.8, prior_survival_sd = 0.15,
                                prior_retention_mu = 0.5, prior_retention_sd = 0.3,
                                prior_shift_mu = 0.5, prior_shift_sd = 0.3,
                                prior_proc_sd = 0.08, proc_sd_sd = 0.04,
                                proc_sd_lower = 0.01, proc_sd_upper = 0.30,
                                prior_obs_sd = 0.15, prior_obs_sd_sd = 0.05,
                                obs_sd_lower = 0.02, obs_sd_upper = 0.50,
                                proc_sd_scale = 1.0)

Bayesian Lefkovitch stage-classified projection model in `Turing.jl`.

- **State.** ``N_{t+1} = A N_t`` where the matrix is rebuilt from the estimated
  fertility, survival, retention and advancement-shift parameters, so the sampled
  parameters are the same ones the deterministic machinery consumes.
- **Process model.** Log-space random walk on the abundance vector:
  ``\\log N_{t+1} \\sim \\mathrm{Normal}(\\log A N_t,\\, \\sigma_p\\,\\|N_t\\|)``,
  allowing demographic stochasticity and missing drivers.
- **Observation model.** Lognormal survey index
  ``I_t = q^T (N_t \\odot w^T) \\, e^{\\epsilon}`` with ``\\epsilon \\sim N(0, \\sigma_o)``.
- **Priors.** Every prior mean, standard deviation and bound is a named keyword, so the
  prior specification is user-level rather than compiled into the model body.

# Arguments
- `survey_index`: observed index per step (missing or non-positive entries are skipped).
- `stage_biomass`: per-stage biomass weight (used for the state's numeric scale).
- `weight_per_ind`: per-stage weight per individual (for the survey mapping).
- `maturity`: per-stage fractional maturity; combined with `weight_per_ind` when
  `survey_type = :spawning`.
- `catchability`: survey catchability per stage.
- `n_stages`, `n_years`: dimensions.
- `survey_type`: `:vulnerable` for a total vulnerable-biomass index, `:spawning` for a
  maturity-weighted (spawning biomass) index. This decides which stage weights the
  observation model applies, so `maturity` is used whenever the index is spawning
  biomass rather than being a silently ignored argument.
"""
@model function bayesian_stage_matrix_model(survey_index::AbstractVector,
                                            stage_biomass::AbstractVector,
                                            weight_per_ind::AbstractVector,
                                            maturity::AbstractVector,
                                            catchability::AbstractVector,
                                            n_stages::Integer,
                                            n_years::Integer;
                                            survey_type::Symbol = :vulnerable,
                                            prior_fertility_mu::Real = 0.0,
                                            prior_fertility_sd::Real = 1.0,
                                            prior_survival_mu::Real = 0.8,
                                            prior_survival_sd::Real = 0.15,
                                            prior_retention_mu::Real = 0.5,
                                            prior_retention_sd::Real = 0.3,
                                            prior_shift_mu::Real = 0.5,
                                            prior_shift_sd::Real = 0.3,
                                            prior_proc_sd::Real = 0.08,
                                            proc_sd_sd::Real = 0.04,
                                            proc_sd_lower::Real = 0.01,
                                            proc_sd_upper::Real = 0.30,
                                            prior_obs_sd::Real = 0.15,
                                            prior_obs_sd_sd::Real = 0.05,
                                            obs_sd_lower::Real = 0.02,
                                            obs_sd_upper::Real = 0.50,
                                            proc_sd_scale::Real = 1.0)
    nt = length(survey_index)
    nt == n_years + 1 || error("survey_index length must be n_years + 1, got $nt for $n_years years.")

    # --- priors on the life-history parameters --------------------------------
    # fertility: positive, log-linked
    log_fertility ~ filldist(LogNormal(0.0, 1.0), n_stages)
    fert = exp.(log_fertility)
    # survival, retention and advancement: unit probabilities
    p_surv ~ filldist(Normal(prior_survival_mu, prior_survival_sd), n_stages)
    survival = 0.001 .+ 0.998 .* _logistic.(p_surv)
    p_ret ~ filldist(Normal(prior_retention_mu, prior_retention_sd), n_stages)
    retention = 0.001 .+ 0.498 .* _logistic.(p_ret)
    p_shift ~ filldist(Normal(prior_shift_mu, prior_shift_sd), n_stages)
    shift = 0.001 .+ 0.498 .* _logistic.(p_shift)
    # enforce retention + advancement <= 1 by renormalising
    adv = shift ./ (retention .+ shift .+ 1.0e-9) .* (1.0 .- retention)
    proc_sd ~ truncated(Normal(prior_proc_sd, proc_sd_sd), proc_sd_lower, proc_sd_upper)
    obs_sd ~ truncated(Normal(prior_obs_sd, prior_obs_sd_sd), obs_sd_lower, obs_sd_upper)

    # --- build the projection matrix the same way the deterministic code does ----
    # The matrix element type follows the sampling types so that automatic
    # differentiation through the matrix product is preserved.
    El = promote_type(eltype(fert), eltype(survival), Float64)
    A = Matrix{El}(undef, n_stages, n_stages)
    fill!(A, zero(El))
    A[1, :] = fert'
    for k in 1:n_stages
        A[k, k] = survival[k] * retention[k]
        if k < n_stages
            A[k + 1, k] = survival[k] * adv[k]
        end
    end

    # --- state-space recursion --------------------------------------------------
    # The state type follows the sampling types so differentiation is preserved.
    ElS = promote_type(eltype(stage_biomass), eltype(A))
    N = Matrix{ElS}(undef, n_stages, n_years + 1)
    for k in 1:n_stages
        N[k, 1] = stage_biomass[k]
    end

    # Each time step gets its own slice of `log_actual`, so no variable is
    # re-declared inside the loop.
    log_actual = Matrix{ElS}(undef, n_stages, n_years)
    for t in 1:n_years
        expected = A * N[:, t]
        log_expected = log.(expected)
        proc = (proc_sd_scale * proc_sd)^2
        log_actual[:, t] ~ MvNormal(Vector(log_expected), proc * I)
        @views N[:, t + 1] = exp.(log_actual[:, t])
    end

    # --- observation model ------------------------------------------------------
    # The observation scale must be strictly positive; the linker unconstrained space
    # can supply negative values during a line search, which LogNormal rejects.
    # `_logistic` maps any real input onto (0, 1) without a branch, so the scale is
    # finite and positive for every sampled `obs_sd`.
    scale_obs = 0.010 .+ 0.490 .* _logistic(obs_sd)
    # Stage weights applied to the observed quantity: a vulnerable-biomass index uses
    # `weight_per_ind`, a spawning-biomass index weights it by `maturity`.
    obs_weight = survey_type === :spawning ? weight_per_ind .* maturity : weight_per_ind
    for i in 1:(n_years + 1)
        obs = survey_index[i]
        if !ismissing(obs) && isfinite(obs) && obs > 0.0
            # μ is the expected index. The log of a possibly non-positive or non-finite
            # mean is avoided by taking the absolute value plus a small floor, which
            # keeps the term finite and differentiable for every sampled state.
            mu = sum(N[:, i] .* obs_weight .* catchability)
            survey_index[i] ~ LogNormal(log(abs(mu) + 1.0e-9), scale_obs)
        end
    end

    # The dominant eigenvalue is reported as a diagnostic. During optimisation and
    # variational optimisation the matrix can contain non-finite entries, where LAPACK
    # raises on the eigen-decomposition; it is then reported as `Inf` so that the model
    # evaluation proceeds and the likelihood terms alone determine the objective.
    growth = try
        _dominant_eigenvalue(A)
    catch
        Inf
    end

    return (abundance = N, projection_matrix = A, fertility = fert,
            survival = survival, retention = retention, advancement = adv,
            growth_rate = growth)
end

end # module StageMatrix
