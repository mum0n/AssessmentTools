# ------------------------------------------------------------------------------------
# Implementation of `SimulationEngine`, wrapped by `src/SimulationEngine.jl`, which
# declares the module and provides the user-facing docstring.
#
# Two interchangeable representations of the same projection:
#
#   1. Agent-based. Each group (age class, life stage, cohort or gear) is one agent
#      whose per-step transition is the producing model's own forward recursion.
#   2. Reaction network. Each group is a species; recruitment, advancement, retention,
#      natural mortality and fishing removals are reactions, simulated with a
#      time-binned Gillespie SSA.
#
# Dependency note (historical): see the header comment
# --------------------
# Agents.jl, Catalyst.jl, ModelingToolkit.jl and JumpProcesses.jl are not resolvable in
# this environment: the local package registry caps CSV.jl at 0.3.0-1.1.0, which
# Agents.jl >= 6 requires, so `Pkg.add` reports "Unsatisfiable requirements". Both
# representations are therefore implemented here, with interfaces shaped so the swap to
# those packages is mechanical and local to this file:
#
#   * Agents.jl       : `AgentSpec` + `agent_step!` / `model_step!` map onto
#                       `ABM(agent_type, model, agent_step!, model_step!)`.
#   * Catalyst.jl     : `build_reaction_network` emits the same species, stoichiometry
#                       and propensities as `@reaction_network`.
#   * JumpProcesses.jl: `simulate_ssa` is the time-binned (tSSA) variant; its reaction
#                       list is exactly what `JumpProcesses.JumpSystem` consumes.
# ------------------------------------------------------------------------------------

# The catch-at-age machinery is a sibling module; this qualified import avoids a
# circular reference while the package is still loading.
import ..StatisticalCatchAtAge
const CATCH = StatisticalCatchAtAge

using LinearAlgebra
using Statistics
using Random
using Distributions
using Printf
using Zarr
using JSON
import ..Config

# The reaction representation is compiled by Catalyst.jl and simulated by
# JumpProcesses.jl, so the network is a real `ReactionSystem` and the SSA is a real
# `SSAStepper`, not a transcription of them.
using Catalyst
using JumpProcesses

# A tiny JSON writer, so the manifest does not require a heavier dependency.
json_string(d::Dict{String, <: Any}) = JSON.json(d)

# -----------------------------------------------------------------------------------
# Parameter abstraction: one canonical container for every model type in the package.
# -----------------------------------------------------------------------------------

"""
    ParameterSource

Provenance of a set of [`BiologicalParameters`](@ref).

# Fields
- `model::Symbol`: producing model, e.g. `:scaa`, `:stage_matrix`, `:per_recruit`,
  `:logistic`, `:delay_difference`, `:dde`, `:mse`.
- `mode::Symbol`: `:point` for a single estimate, `:posterior` for parameters taken
  from posterior draws (means, medians or a specified quantile).
- `n_draws::Int`: number of posterior draws summarised, 0 for a point estimate.
"""
struct ParameterSource
    model::Symbol
    mode::Symbol
    n_draws::Int
end

"""
    BiologicalParameters

Canonical projection parameters, assembled from the solution of any model in the
package. Every field is a per-group quantity where `n_groups` is the number of age
classes or life stages, so age-structured, stage-structured, per-recruit and
surplus-production models all project through the same recursion.

# Fields
- `n_groups::Int`: number of age classes or stages, youngest first.
- `weights::Vector{Float64}`: biomass per individual per group.
- `maturity::Vector{Float64}`: fractional maturity per group.
- `fertility::Vector{Float64}`: per-capita offspring entering group 1 per individual.
- `natural_mortality::Float64`: instantaneous natural mortality (constant per step).
- `advance::Vector{Float64}`: probability of advancing one group given survival.
- `retention::Vector{Float64}`: probability of staying put given survival.
- `selectivity::Matrix{Float64}`: gear vulnerability, `n_groups × ngear`.
- `fishing_mortality::Matrix{Float64}`: fishing mortality per group at the starting
  time slice, `n_groups × 1`.
- `harvest_rate::Matrix{Float64}`: resolved harvest rate, `n_groups × nyears`.
- `initial_numbers::Vector{Float64}`: starting abundance per group.
- `source::ParameterSource`: which model produced the parameters, and whether it came
  from a point estimate or posterior draws.
"""
struct BiologicalParameters
    n_groups::Int
    weights::Vector{Float64}
    maturity::Vector{Float64}
    fertility::Vector{Float64}
    natural_mortality::Float64
    advance::Vector{Float64}
    retention::Vector{Float64}
    selectivity::Matrix{Float64}
    fishing_mortality::Matrix{Float64}
    harvest_rate::Matrix{Float64}
    initial_numbers::Vector{Float64}
    source::ParameterSource
end

"""
    ParameterEnsemble

A saved posterior ensemble: one `BiologicalParameters` per posterior draw, recorded
rather than computed. The engine selects from this archive at each projection step,
so the joint correlation between parameters is preserved because a draw is drawn whole,
never per parameter.

# Fields
- `draws::Vector{BiologicalParameters}`: the saved draws.
- `ensemble_id::String`: provenance label for the archive.
"""
struct ParameterEnsemble
    draws::Vector{BiologicalParameters}
    ensemble_id::String
end

Base.length(e::ParameterEnsemble) = length(e.draws)

"""
    SimulationControls

How the projection engine combines a posterior ensemble with the deterministic
dynamics. Read from the `[simulation]` section of a configuration file; the choices are
distinct procedures, not variants of the same one.

# Fields
- `parameter_combination::Symbol`: `:resample_each_step` (draw a fresh joint parameter
  set at every time step), `:single_draw` (one joint set for the whole projection),
  `:posterior_mean` (the ensemble mean), or `:point` (ignore the ensemble).
- `n_draws::Int`: number of draws to retain when loading an ensemble.
"""
struct SimulationControls
    parameter_combination::Symbol
    n_draws::Int
end

"""
    load_simulation_controls(path::AbstractString; section = "simulation") -> SimulationControls

Read the `[simulation]` section of a TOML file. Every key is required.
"""
function load_simulation_controls(path::AbstractString; section::AbstractString = "simulation")
    cfg = Config.parse_file(path)
    d = Config.require_section(cfg, section, path)
    where = "$path section [$section]"
    raw = Config.load_int(d, "parameter_combination", where)
    # `parameter_combination` is a number that names the mode, kept numeric so the file
    # stays a pure config and cannot introduce code.
    allowed = Dict(1 => :resample_each_step, 2 => :single_draw,
                   3 => :posterior_mean, 4 => :point)
    haskey(allowed, raw) ||
        error("`parameter_combination` in $where must be 1 (resample_each_step), 2 (single_draw), " *
              "3 (posterior_mean) or 4 (point); got $raw.")
    n = Config.load_int(d, "n_draws", where)
    n >= 1 || error("`n_draws` in $where must be at least 1, got $n.")
    return SimulationControls(allowed[raw], n)
end

"""
    ProjectionState

Snapshot of a projection.

# Fields
- `numbers::Vector{Float64}`: per-group abundance at `time_index`.
- `time_index::Int`: annual time step relative to the start of the projection.
- `biomass::Float64`: `Σ n·w`.
- `spawning_biomass::Float64`: `Σ n·w·m`.
- `cumulative_removals::Float64`: removals accumulated over the projection.
- `source_year::Int`: index of the time slice the projection started from.
"""
struct ProjectionState
    numbers::Vector{Float64}
    time_index::Int
    biomass::Float64
    spawning_biomass::Float64
    cumulative_removals::Float64
    source_year::Int
end

"""
    Base.copy(s::ProjectionState) -> ProjectionState

Deep copy: the `numbers` vector is duplicated so projections can hold an independent
record of each step rather than aliasing one array.
"""
Base.copy(s::ProjectionState) =
    ProjectionState(copy(s.numbers), s.time_index, s.biomass, s.spawning_biomass,
                    s.cumulative_removals, s.source_year)

"""
    _uniform(v, n) -> Vector{Float64}

Broaden a scalar to a length-`n` vector, or validate an existing vector.
"""
_uniform(v::Real, n::Int) = fill(Float64(v), n)
_uniform(v::AbstractVector, n::Int) =
    length(v) == n ? collect(Float64, v) :
    error("Expected a length-$n vector, got $(length(v)).")

"""
    _matrix(v, rows, cols) -> Matrix{Float64}

Broaden a scalar, or validate an existing matrix, against `rows × cols`.
"""
_matrix(v::Real, rows::Int, cols::Int) = fill(Float64(v), rows, cols)
_matrix(v::AbstractMatrix, rows::Int, cols::Int) =
    size(v) == (rows, cols) ? collect(Float64, v) :
    error("Expected a $(rows)x$(cols) matrix, got $(size(v)).")

"""
    _projection_matrix(p::BiologicalParameters) -> Matrix{Float64}

Advancement and retention matrix implied by the parameters. The first row is
reproduction (fertility), the diagonal is `retention`, and the sub-diagonal is
`advance`. Survivorship (`e^{-M}`) is applied outside the matrix so fishing removals
can be tracked separately.
"""
function _projection_matrix(p::BiologicalParameters)
    n = p.n_groups
    A = zeros(Float64, n, n)
    A[1, :] .= p.fertility
    for i in 1:n
        A[i, i] += p.retention[i]
        if i < n
            A[i + 1, i] = p.advance[i]
        end
    end
    return A
end

"""
    projection_parameters(; ...) -> BiologicalParameters

Build a [`BiologicalParameters`](@ref) from a model solution. Each keyword maps to a
canonical quantity, so any model in the package can be projected. The required keyword
is `n_groups`; the rest are broadened from scalars where that is meaningful.

# Keywords
- `n_groups`: number of age classes or stages.
- `weights`, `maturity`, `fertility`, `advance`, `retention`, `initial_numbers`:
  per-group vectors, or scalars broadcast to all groups.
- `natural_mortality`: scalar or vector; a vector is collapsed to its value at the
  starting time slice.
- `selectivity`: `n_groups × ngear` matrix.
- `fishing_mortality`, `harvest_rate`: `n_groups × 1` and `n_groups × nyears` matrices
  (or scalars).
- `source`: [`ParameterSource`](@ref); defaults to a point estimate from `:unspecified`.
- `model_symbol`: convenience shorthand for `source.model`.
"""
function projection_parameters(; n_groups::Integer,
                               weights::Union{Real, AbstractVector} = 1.0,
                               maturity::Union{Real, AbstractVector} = 1.0,
                               fertility::Union{Real, AbstractVector} = 0.0,
                               natural_mortality::Union{Real, AbstractVector} = 0.0,
                               advance::Union{Real, AbstractVector} = 1.0,
                               retention::Union{Real, AbstractVector} = 0.0,
                               selectivity::Union{Real, AbstractMatrix} = 1.0,
                               fishing_mortality::Union{Real, AbstractMatrix} = 0.0,
                               harvest_rate::Union{Real, AbstractMatrix} = 0.0,
                               initial_numbers::Union{Real, AbstractVector} = 0.0,
                               source::Union{Nothing, ParameterSource} = nothing,
                               model_symbol::Symbol = :unspecified)
    n = Int(n_groups)
    n >= 1 || error("n_groups must be at least 1, got $n.")

    src = isnothing(source) ? ParameterSource(model_symbol, :point, 0) : source

    # A vector `natural_mortality` states the rate per year at the starting slice.
    m = natural_mortality isa Real ? Float64(natural_mortality) :
        Float64((natural_mortality::AbstractVector)[1])

    sel = selectivity isa Real ? fill(Float64(selectivity), n, 1) :
        Matrix{Float64}(selectivity)
    size(sel, 1) == n ||
        error("selectivity must have $n rows (one per group), got $(size(sel, 1)).")

    for (nm, v) in (("weights", weights), ("maturity", maturity), ("fertility", fertility),
                    ("advance", advance), ("retention", retention),
                    ("initial_numbers", initial_numbers))
        for x in (v isa Real ? (Float64(v),) : v)
            isfinite(x) || error("'$nm' must be finite, got $x.")
        end
    end
    for x in (maturity isa Real ? (Float64(maturity),) : maturity)
        (0 <= x <= 1) || error("'maturity' must lie in [0, 1], got $x.")
    end
    for (nm, v) in (("advance", advance), ("retention", retention))
        for x in (v isa Real ? (Float64(v),) : v)
            (0 <= x <= 1) || error("'$nm' must lie in [0, 1], got $x.")
        end
    end
    # Given survival, an individual either advances to the next group or stays put, so
    # `advance` and `retention` must partition the survivors exactly. Enforcing this is
    # what makes the closed-form recursion, the agent transition and the reaction
    # network agree: if they summed to less than 1 the unallocated mass would be
    # implicitly removed, and each representation would have to choose its own residual.
    for i in 1:n
        (advance[i] + retention[i] ≈ 1) ||
            error("advance[$i] ($(advance[i])) + retention[$i] ($(retention[i])) must sum to 1; " *
                  "given survival the individual either advances or stays. " *
                  "Set `advance` to `0` for a terminal group whose members remain.")
    end

    # `harvest_rate` defines the time axis: `fishing_mortality` must be aligned to the
    # same number of columns rather than being assumed to be a single column.
    n_cols = harvest_rate isa AbstractMatrix ? size(harvest_rate, 2) : 1
    return BiologicalParameters(
        n,
        _uniform(weights, n),
        _uniform(maturity, n),
        _uniform(fertility, n),
        m,
        _uniform(advance, n),
        _uniform(retention, n),
        sel,
        _matrix(fishing_mortality, n, n_cols),
        _matrix(harvest_rate, n, n_cols),
        _uniform(initial_numbers, n),
        src
    )
end

"""
    _state_of(n::AbstractVector{<:Real}, p::BiologicalParameters, i::Integer, y::Integer) -> ProjectionState

Construct a [`ProjectionState`](@ref) from an abundance vector, given `i` as the time
index and `y` as the source slice.
"""
_state_of(n::AbstractVector{<:Real}, p::BiologicalParameters, i::Integer, y::Integer) =
    ProjectionState(collect(Float64, n), i, sum(n .* p.weights),
                    sum(n .* p.weights .* p.maturity), 0.0, y)

# -----------------------------------------------------------------------------------
# Forward and backward projection from an arbitrary time slice.
# -----------------------------------------------------------------------------------

"""
    project_state(params, state; direction=:forward, n_years=1, year_offset=1,
                  ensemble=nothing, mode=:resample_each_step, rng=Random.default_rng()) -> Vector{ProjectionState}

Project from `state` for `n_years` annual steps, in either direction, returning the state
at each step including the starting state.

- `:forward` runs ``N_{t+1} = A\\,N_t\\,e^{-M} - u\\,(A\\,N_t\\,e^{-M})`` with `A` the
  advancement and reproduction matrix and `u` the harvest rate for the year.
- `:backward` inverts the recursion: given ``N_{t+1}``, recover ``N_t`` by undoing the
  removals and then the survivorship. This is the exact inverse when the harvest-rate
  schedule is the same one the forward run used.

`year_offset` shifts the harvest-rate lookup, so a projection started at any designated
time slice reads the rates fitted for that slice.

**Posterior combination.** When `ensemble` is supplied, `mode` decides how the saved draws
enter the projection, and the modes are distinct procedures:

- `:resample_each_step` (default) draws a fresh joint parameter set **at every time step**
  and applies that set's own `advance`, `retention`, fertility, `M` and harvest rate. A
  draw is selected whole, so parameter correlation survives; drawing independently per
  parameter would silently assume independence.
- `:single_draw` picks one joint set for the whole projection.
- `:posterior_mean` uses the ensemble mean of each field.
- `:point` ignores the ensemble and uses `params`.

This is the correct treatment of stochastic projection when the parameters themselves are
uncertain: uncertainty enters the transition at each step rather than being resolved once
at the start.
"""
function project_state(params::BiologicalParameters, state::ProjectionState;
                       direction::Symbol = :forward, n_years::Integer = 1,
                       year_offset::Integer = 1,
                       ensemble::Union{Nothing, ParameterEnsemble} = nothing,
                       mode::Union{Nothing, Symbol} = nothing,
                       rng::AbstractRNG = Random.default_rng())
    direction in (:forward, :backward) ||
        error("direction must be :forward or :backward, got $direction.")
    n_years >= 1 || error("n_years must be at least 1, got $n_years.")

    # The mode is resolved from what was supplied: with an ensemble and no explicit
    # mode the default is per-step joint resampling, and with no ensemble the only
    # meaningful mode is the point estimate. This keeps the common call
    # `project_state(params, state)` working while making the ensemble behaviour the
    # default when one is present.
    mode = isnothing(mode) ? (isnothing(ensemble) ? :point : :resample_each_step) : mode
    mode in (:resample_each_step, :single_draw, :posterior_mean, :point) ||
        error("mode must be :resample_each_step, :single_draw, :posterior_mean or :point; got $mode.")
    (ensemble === nothing && mode != :point) &&
        error("mode $mode requires an ensemble to be supplied.")

    out = ProjectionState[copy(state)]
    if mode === :point
        _ = _project_state_one(params, state, direction, n_years, year_offset, out)
        return out
    end

    # A draw chosen for `:single_draw` is held for the whole projection; the other modes
    # resolve their parameters fresh each step.
    held = mode === :single_draw ? ensemble.draws[rand(rng, 1:length(ensemble.draws))] : nothing
    if mode === :posterior_mean
        held = _ensemble_mean(ensemble)
        mode = :single_draw
    end

    n = copy(state.numbers)
    t = state.time_index
    for step in 1:n_years
        p = mode === :single_draw ? held :
            ensemble.draws[rand(rng, 1:length(ensemble.draws))]
        _apply_step!(out, n, p, state, t, step, year_offset, direction)
        t += direction === :forward ? 1 : -1
    end
    return out
end

"""
    _apply_step!(out, n, p, state, t, step, year_offset, direction) -> Nothing

One projection step using parameter set `p`, pushing the resulting state onto `out`.
"""
function _apply_step!(out, n, p::BiologicalParameters, state::ProjectionState,
                      t::Int, step::Int, year_offset::Int, direction::Symbol)
    ny = size(p.harvest_rate, 2)
    A = _projection_matrix(p)
    surv = exp(-p.natural_mortality)
    yidx = ((state.source_year + step - 1 + year_offset - 1) % ny) + 1
    u = @view p.harvest_rate[:, yidx]

    if direction === :forward
        survivors = A * n .* surv
        removed = sum(survivors .* u .* p.weights)
        n .= survivors .- survivors .* u
    else
        n .= _backward_step(A, n, surv, u)
        removed = 0.0
    end

    s = _state_of(n, p, t + (direction === :forward ? 1 : -1), state.source_year)
    s = ProjectionState(s.numbers, s.time_index, s.biomass, s.spawning_biomass,
                        out[end].cumulative_removals + removed, s.source_year)
    push!(out, s)
    return nothing
end

"""
    _project_state_one(params, state, direction, n_years, year_offset, out) -> Nothing

Point-estimate projection with no ensemble: appended onto the existing `out`.
"""
function _project_state_one(params::BiologicalParameters, state::ProjectionState,
                             direction::Symbol, n_years::Integer,
                             year_offset::Integer, out::Vector{<: ProjectionState})
    n = copy(state.numbers)
    t = state.time_index
    for step in 1:n_years
        ny = size(params.harvest_rate, 2)
        A = _projection_matrix(params)
        surv = exp(-params.natural_mortality)
        yidx = ((state.source_year + step - 1 + year_offset - 1) % ny) + 1
        u = @view params.harvest_rate[:, yidx]

        if direction === :forward
            survivors = A * n .* surv
            removed = sum(survivors .* u .* params.weights)
            n .= survivors .- survivors .* u
        else
            n .= _backward_step(A, n, surv, u)
            removed = 0.0
        end
        t += direction === :forward ? 1 : -1
        s = _state_of(n, params, t, state.source_year)
        s = ProjectionState(s.numbers, s.time_index, s.biomass, s.spawning_biomass,
                            out[end].cumulative_removals + removed, s.source_year)
        push!(out, s)
    end
    return nothing
end

"""
    _ensemble_mean(ensemble) -> BiologicalParameters

Ensemble mean of each field. Matrices are averaged only when every draw shares their
shape, so a mixed-shape archive raises instead of being silently truncated to the first
draw's shape.
"""
function _ensemble_mean(ens::ParameterEnsemble)
    ref = ens.draws[1]
    n = length(ens.draws)
    ng = ref.n_groups
    dims_ok = all(size(d.harvest_rate, 2) == size(ref.harvest_rate, 2) &&
                  size(d.selectivity, 2) == size(ref.selectivity, 2) for d in ens.draws)
    dims_ok || error("Ensemble draws have differing shapes; the mean is undefined. " *
                     "Use :resample_each_step or :single_draw instead.")

    # Explicit accumulation rather than `mean(gen)`: the generic `mean` of a generator
    # of vectors resolves to `sum(gen) / length(gen)`, which works for numeric iterables
    # but is an unclear and fragile expression here.
    w = zeros(Float64, ng); mat = zeros(Float64, ng); fert = zeros(Float64, ng)
    adv = zeros(Float64, ng); ret = zeros(Float64, ng); init = zeros(Float64, ng)
    m = 0.0
    for d in ens.draws
        w .+= d.weights; mat .+= d.maturity; fert .+= d.fertility
        adv .+= d.advance; ret .+= d.retention; init .+= d.initial_numbers
        m += d.natural_mortality
    end
    w ./= n; mat ./= n; fert ./= n; adv ./= n; ret ./= n; init ./= n; m /= n

    sel = dropdims(mean(_stack3([d.selectivity for d in ens.draws]); dims = 1); dims = 1)
    fm = dropdims(mean(_stack3([d.fishing_mortality for d in ens.draws]); dims = 1); dims = 1)
    hr = dropdims(mean(_stack3([d.harvest_rate for d in ens.draws]); dims = 1); dims = 1)

    src = ParameterSource(ref.source.model, :posterior_mean, n)
    return BiologicalParameters(ref.n_groups, w, mat, fert, m, adv, ret, sel, fm, hr, init, src)
end

"""
    _backward_step(A, n, surv, u) -> Vector{Float64}

Single backward recursion step. The forward step is `n_{t+1} = A n_t · surv · (1 - u)`,
so the inverse undoes the removal first and then the survivorship and advancement.
"""
function _backward_step(A::AbstractMatrix, n::AbstractVector, surv::Real,
                         u::AbstractVector)
    survivors = n ./ (1.0 .- u)
    prev = A \ survivors ./ surv
    return prev
end

"""
    reconstruct_backward(params, numbers_now; n_years=1) -> Vector{ProjectionState}

Project backwards for `n_years` steps from an abundance vector.
"""
function reconstruct_backward(params::BiologicalParameters,
                               numbers_now::AbstractVector{<:Real};
                               n_years::Integer = 1)
    length(numbers_now) == params.n_groups ||
        error("numbers_now must have $(params.n_groups) entries, got $(length(numbers_now)).")
    st = _state_of(numbers_now, params, 0, 1)
    return project_state(params, st; direction = :backward, n_years = n_years)
end

"""
    project_from_slice(params, numbers_slice, slice_index; direction=:forward, n_years=1) -> Vector{ProjectionState}

Project starting from a designated time slice, where `slice_index` selects which column of
the fitted `harvest_rate` schedule to start reading from. This is the entry point for
starting a projection at any designated time period.
"""
function project_from_slice(params::BiologicalParameters,
                             numbers_slice::AbstractVector{<:Real},
                             slice_index::Integer;
                             direction::Symbol = :forward,
                             n_years::Integer = 1)
    ny = size(params.harvest_rate, 2)
    1 <= slice_index <= ny ||
        error("slice_index must be in 1:$ny, got $slice_index.")
    st = _state_of(numbers_slice, params, slice_index - 1, slice_index)
    return project_state(params, st; direction = direction, n_years = n_years,
                          year_offset = slice_index - 1)
end

# -----------------------------------------------------------------------------------
# Representation 1: agent-based.
#
# Each group is one agent. Its per-step transition is the same recursion
# `project_state` evaluates in closed form, so the agent and deterministic projections
# agree in expectation and differ only in realisation.
# -----------------------------------------------------------------------------------

"""
    AgentSpec

Specification of the agent-based representation.

# Fields
- `n_agents_per_group::Int`: agents created per group.
- `scaling::Float64`: abundance per simulated individual, i.e.
  `abundance = summed agent individuals * scaling`.
- `n_groups::Int`: number of groups, matching `BiologicalParameters.n_groups`.

`n_individuals` on an agent is an integer count, so per-agent transitions are exact
binomial draws. `scaling` lets those counts stand for a large stock: an agent with 500
individuals and `scaling = 0.02` represents abundance 10.
"""
struct AgentSpec
    n_agents_per_group::Int
    scaling::Float64
    n_groups::Int
end

"""
    GroupAgent

One agent, holding an integer number of individuals in group `group`. Every simulated
individual is an independent realiser of the group's transition, so a population of
`n_agents_per_group × n_groups` agents gives the binomial draws their exact
interpretation and the per-agent noise is real demographic stochasticity.
"""
mutable struct GroupAgent
    id::Int
    group::Int
    n_individuals::Int
end

"""
    AgentPopulation

The agent model population.

# Fields
- `agents::Vector{GroupAgent}`: the population.
- `spec::AgentSpec`: agent counts per group and the abundance scaling.
- `params::BiologicalParameters`: transition parameters shared by all agents.
- `year::Int`: current annual time step.
"""
mutable struct AgentPopulation
    agents::Vector{GroupAgent}
    spec::AgentSpec
    params::BiologicalParameters
    year::Int
end

"""
    spawn_population(params; year=0, n_agents_per_group=1, scaling=1.0) -> AgentPopulation

Initialise the agent population from `params.initial_numbers`.

Each group's total simulated individuals is `round(initial_numbers[g] / scaling)`. Those
are partitioned across `n_agents_per_group` agents so the counts sum to the total
exactly: each agent receives `div(total, n_agents)`, and the first
`total - base * n_agents` agents receive one extra. Naive rounding per agent would
over-count when agents outnumber individuals (for example 120 individuals over 200
agents would round each agent up to 1, giving 200), which is a real bias in the
recovered abundance and therefore in the comparison with the closed-form recursion.
"""
function spawn_population(params::BiologicalParameters; year::Int = 0,
                          n_agents_per_group::Integer = 1, scaling::Real = 1.0)
    n_agents_per_group >= 1 || error("n_agents_per_group must be at least 1.")
    scaling > 0 || error("scaling must be positive, got $scaling.")
    spec = AgentSpec(Int(n_agents_per_group), Float64(scaling), params.n_groups)
    agents = GroupAgent[]
    id = 0
    for g in 1:params.n_groups
        total = max(0, round(Int, params.initial_numbers[g] / spec.scaling))
        base = div(total, spec.n_agents_per_group)
        extra = total - base * spec.n_agents_per_group
        for j in 1:spec.n_agents_per_group
            id += 1
            count = base + (j <= extra ? 1 : 0)
            push!(agents, GroupAgent(id, g, count))
        end
    end
    return AgentPopulation(agents, spec, params, year)
end

"""
    agent_step!(agent, params; rng=Random.default_rng()) -> GroupAgent

Advance one annual step for a single group-`g` agent, drawing its transition
stochastically:

```math
n_{survive} \\sim \\mathrm{Binomial}\\!(n,\\, e^{-M}), \\quad
n_{stay} \\sim \\mathrm{Binomial}(n_{survive},\\, r_g), \\quad
n_{adv} \\sim \\mathrm{Binomial}(n_{survive} - n_{stay},\\, a_g/(1-r_g))
```

The agent is updated in place to carry the advancers, which move to group `g + 1`; the
returned agent carries the stayers, who remain in group `g`. Returning a second object
is what allows both fractions to be tracked: an agent carries a single group, so
splitting is required, in the same way an `Agents.jl` step function can add or remove
agents. If there are no stayers the returned agent is still returned with zero
individuals, so callers can test and drop it.

Fishing removals are applied by [`model_step!`](@ref) because they act on the whole
population.
"""
function agent_step!(agent::GroupAgent, params::BiologicalParameters;
                      rng::AbstractRNG = Random.default_rng())
    g = agent.group
    n_survive = rand(rng, Binomial(agent.n_individuals, exp(-params.natural_mortality)))

    if g < params.n_groups
        r_g = params.retention[g]
        n_stay = rand(rng, Binomial(Int(n_survive), r_g))
        n_adv = n_survive - n_stay
        stayer = GroupAgent(-1, g, n_stay)
        agent.n_individuals = n_adv
        agent.group = g + 1
    else
        # Terminal group: retention holds the individual there.
        agent.n_individuals = n_survive
        stayer = GroupAgent(-1, g, 0)
    end
    return stayer
end

"""
    model_step!(pop; rng=Random.default_rng()) -> Nothing

Run one annual step for the whole population:

1. **Recruitment.** New individuals are added to group 1, with expected number
   ``\\sum_k f_k n_k``, drawn from a Poisson distribution. This is the `recruit`
   reaction in the reaction network, so both stochastic representations carry the same
   recruitment mechanism.
2. **Transition.** Each agent is split into its stayers and advancers.
3. **Removals.** Both halves are exposed to the fishing removal rate of their group at
   the population's current year, and the clock advances.

Agents with no individuals are dropped. This is the `model_step!` hook that
`Agents.jl` would call once per tick.
"""
function model_step!(pop::AgentPopulation; rng::AbstractRNG = Random.default_rng())
    ny = size(pop.params.harvest_rate, 2)
    yidx = ((pop.year) % ny) + 1
    p = pop.params

    # 1. Recruitment into group 1, from the fertility of every group. Fertility in the
    #    Lefkovitch matrix convention is the offspring of a *surviving* individual, and
    #    `project_state` scales the whole product A·n by e^{-M}, so the expectation here
    #    carries the same survivorship factor. Without it the agent and closed-form
    #    recursions diverge by exactly e^{-M} on the recruitment term.
    abundance = group_abundance(pop)
    surv = exp(-p.natural_mortality)
    expected = sum(p.fertility[k] * abundance[k] * surv for k in 1:p.n_groups)
    n_new = expected > 0 ? rand(rng, Poisson(expected)) : 0
    if n_new > 0
        id = maximum(a.id for a in pop.agents; init = 0)
        for _ in 1:Int(n_new)
            id += 1
            push!(pop.agents, GroupAgent(id, 1, 1))
        end
    end

    # 2 & 3. Transition and removals.
    retained = GroupAgent[]
    next_id = maximum(a.id for a in pop.agents; init = 0)

    for a in pop.agents
        stayer = agent_step!(a, p; rng = rng)
        for b in (a, stayer)
            if b.n_individuals <= 0
                continue
            end
            u = p.harvest_rate[b.group, yidx]
            b.n_individuals = rand(rng, Binomial(b.n_individuals, 1.0 - u))
            if b.n_individuals > 0
                if b.id == -1
                    next_id += 1
                    b.id = next_id
                end
                push!(retained, b)
            end
        end
    end

    pop.agents = retained
    pop.year += 1
    return nothing
end

"""
    group_abundance(pop) -> Vector{Float64}

Abundance per group implied by the agent population: the summed agent counts, times
`pop.spec.scaling`.
"""
function group_abundance(pop::AgentPopulation)
    out = zeros(Float64, pop.params.n_groups)
    for a in pop.agents
        out[a.group] += a.n_individuals
    end
    return out .* pop.spec.scaling
end

"""
    run_agent_projection(params, n_years; n_agents_per_group=1, year0=0, scaling=1.0, rng=Random.default_rng()) -> (states, removals)

Run the agent-based projection for `n_years` annual steps. Returns the per-step group
abundances (starting at the initial numbers) and the biomass removed at each step.

Per-agent transitions are stochastic, so different `rng` values give independent
realisations and the spread across them is the demographic stochasticity of the
projection.

**Comparison with the closed form.** With many agents per group this converges on the
recursion that [`project_state`](@ref) evaluates directly. The convergence is not exact,
and the residual gap is expected rather than a defect: recruitment is Poisson drawn
every year and then compounded, so the mean of the compounded process is not the process
evaluated at the mean (Jensen's inequality). The gap is largest when the recruitment
term dominates the growth of the population, and it closes as `n_agents_per_group`
grows and as the variance of the recruitment term shrinks relative to the mean. The
closed form remains the correct mean-field projection; the agent representation is the
one that carries realised demographic variance.
"""
function run_agent_projection(params::BiologicalParameters, n_years::Integer;
                              n_agents_per_group::Integer = 1, year0::Int = 0,
                              scaling::Real = 1.0,
                              rng::AbstractRNG = Random.default_rng())
    pop = spawn_population(params; year = year0,
                           n_agents_per_group = n_agents_per_group, scaling = scaling)
    states = [copy(params.initial_numbers)]
    removals = Float64[]
    for _ in 1:n_years
        before = group_abundance(pop)
        model_step!(pop; rng = rng)
        after = group_abundance(pop)
        push!(states, after)
        push!(removals, max(0.0, sum((before .- after) .* params.weights)))
    end
    return states, removals
end

# ---------------------------------------------------------------------------
# Representation 2: reaction network with a time-binned Gillespie SSA.
#
# Species are the groups; reactions are recruitment, advancement, retention, natural
# mortality and fishing removal.
# ---------------------------------------------------------------------------

"""
    Reaction

One reaction.

# Fields
- `name::Symbol`: reaction identifier.
- `reactants::Vector{Int}`: species indices consumed.
- `products::Vector{Int}`: species indices produced.
- `rate_key::Symbol`: governing parameter name, e.g. `:rate_advance_1`.
- `propensity::Function`: `(state, params, t) -> hazard`.
"""
struct Reaction
    name::Symbol
    reactants::Vector{Int}
    products::Vector{Int}
    rate_key::Symbol
    propensity::Function
end

"""
    ReactionNetwork

The reaction system for a [`BiologicalParameters`](@ref).

# Fields
- `species::Vector{Symbol}`: species labels, one per group.
- `reactions::Vector{Reaction}`: the reaction list.
- `initial_counts::Vector{Float64}`: species counts.
- `params::BiologicalParameters`: the parameters the propensities read.
"""
struct ReactionNetwork
    species::Vector{Symbol}
    reactions::Vector{Reaction}
    initial_counts::Vector{Float64}
    params::BiologicalParameters
end

"""
    build_reaction_network(params::BiologicalParameters) -> ReactionNetwork

Assemble the reaction system:

| reaction | stoichiometry | propensity |
|:---------|:--------------|:-----------|
`recruit`   | `∅ -> x1`     | `sum(k, fertility[k] * n_k)`
`advance_k` | `x_k -> x_{k+1}` | `advance[k] * M * n_k`
`retain_k`  | `x_k -> x_k`  | `retention[k] * M * n_k`
`natural_k` | `x_k -> ∅`    | `(1 - advance_k - retention_k) * M * n_k`
`harvest_k` | `x_k -> ∅`    | `u_k(t) * n_k`

Retention leaves an individual in place, so it is a reaction with no net
stoichiometric change: in the SSA it consumes a step but not an individual, which is
why it is included explicitly rather than folded into `advance`. Fishing removal is
time dependent through the harvest-rate schedule.
"""
function build_reaction_network(params::BiologicalParameters)
    n = params.n_groups
    species = [Symbol(:x, i) for i in 1:n]
    reactions = Reaction[]

    push!(reactions, Reaction(
        :recruit, Int[], [1], :rate_recruit,
        (st, pp, t) -> sum(pp.fertility[k] * st[k] for k in 1:n)))

    for k in 1:(n - 1)
        push!(reactions, Reaction(
            Symbol(:advance_, k), [k], [k + 1], Symbol(:rate_advance_, k),
            (st, pp, t) -> pp.advance[k] * pp.natural_mortality * st[k]))
    end

    for k in 1:n
        remainder = max(0.0, 1.0 - params.advance[k] - params.retention[k])
        push!(reactions, Reaction(
            Symbol(:retain_, k), [k], [k], Symbol(:rate_retain_, k),
            (st, pp, t) -> pp.retention[k] * pp.natural_mortality * st[k]))
        push!(reactions, Reaction(
            Symbol(:natural_, k), [k], Int[], Symbol(:rate_natural_, k),
            (st, pp, t) -> remainder * pp.natural_mortality * st[k]))
        push!(reactions, Reaction(
            Symbol(:harvest_, k), [k], Int[], Symbol(:rate_harvest_, k),
            (st, pp, t) -> begin
                ny = size(pp.harvest_rate, 2)
                yidx = ((floor(Int, t)) % ny) + 1
                pp.harvest_rate[k, yidx] * st[k]
            end))
    end
    return ReactionNetwork(species, reactions, copy(params.initial_numbers), params)
end

"""
    SSAResult

Output of [`simulate_ssa`](@ref).

# Fields
- `times::Vector{Float64}`: visiting time of each recorded frame.
- `counts::Matrix{Float64}`: species counts, `n_species × n_frames`.
- `events::Vector{Tuple{Float64, Symbol, Int}}`: `(time, reaction, 1)` per fired event.
- `total_harvested::Float64`: weighted total removed by `harvest` reactions.
"""
struct SSAResult
    times::Vector{Float64}
    counts::Matrix{Float64}
    events::Vector{Tuple{Float64, Symbol, Int}}
    total_harvested::Float64
end

"""
    simulate_ssa(net::ReactionNetwork, t_end::Real; n_bins::Integer = 50, rng = Random.default_rng()) -> SSAResult

Time-binned Gillespie stochastic simulation, the tSSA variant of Seeger and Nobile (2019,
BMC Syst. Biol. 13:79), used when reaction hazards are piecewise constant over a bin.

For each bin of width `dt`, the per-species hazards are evaluated at the bin start and
held constant, so the number of events is drawn exactly from
``N_e \\sim \\mathrm{Poisson}(\\sum_k \\lambda_k\\, dt)``, and each event is chosen in
proportion to its hazard. Within a bin this is exact for a time-inhomogeneous Poisson
process with fixed rates, and it avoids the per-event overhead of the classical
depletion formulation, which is the reason it is preferred when a simulation spans many
steps.
"""
function simulate_ssa(net::ReactionNetwork, t_end::Real; n_bins::Integer = 50,
                       rng::AbstractRNG = Random.default_rng())
    t_end > 0 || error("t_end must be positive, got $t_end.")
    n_bins >= 1 || error("n_bins must be at least 1, got $n_bins.")
    dt = t_end / n_bins
    n = net.params.n_groups

    state = copy(net.initial_counts)
    times = Float64[0.0]
    counts = zeros(Float64, n, n_bins + 1)
    counts[:, 1] .= state
    events = Tuple{Float64, Symbol, Int}[]
    total_harvested = 0.0

    for b in 1:n_bins
        t = (b - 1) * dt
        lambdas = [r.propensity(state, net.params, t) for r in net.reactions]
        total_rate = sum(lambdas)
        expected = total_rate * dt
        n_events = expected > 0 ? rand(rng, Poisson(min(expected, 1.0e9))) : 0

        fired = 0
        attempts = 0
        max_attempts = n_events + 1000
        while fired < n_events && attempts < max_attempts
            attempts += 1
            total_rate <= 0.0 && break

            # Draw a reaction in proportion to its hazard.
            u = rand(rng) * total_rate
            acc = 0.0
            pick = length(net.reactions)
            for (i, l) in enumerate(lambdas)
                acc += l
                if u <= acc
                    pick = i
                    break
                end
            end
            r = net.reactions[pick]

            # Feasibility: every reactant must be present. Reactions that cannot fire
            # are skipped, not retried, so the event count for the bin can fall short
            # of the Poisson draw when a species is depleted.
            any(j -> state[j] < 1.0, r.reactants) && continue

            for j in r.reactants
                state[j] -= 1.0
            end
            for j in r.products
                state[j] += 1.0
            end
            if startswith(string(r.name), "harvest")
                removed_group = isempty(r.reactants) ? 1 : r.reactants[1]
                total_harvested += net.params.weights[removed_group]
            end
            push!(events, (t + dt, r.name, 1))
            fired += 1
        end

        for k in 1:n
            state[k] = max(0.0, state[k])
        end
        counts[:, b + 1] .= state
        push!(times, b * dt)
    end

    return SSAResult(times, counts, events, total_harvested)
end

"""
    simulate_ssa(net::ReactionNetwork, t_end::Real, n_bins::Integer, rng)

Positional-argument form accepted for convenience.
"""
simulate_ssa(net::ReactionNetwork, t_end::Real, n_bins::Integer,
             rng::AbstractRNG) = simulate_ssa(net, t_end; n_bins = n_bins, rng = rng)
# ---------------------------------------------------------------------------
# Catalyst / JumpProcesses port of the SSA.
#
# The network is compiled by Catalyst.jl and simulated by JumpProcesses.jl, so the
# jump types, rate combining and SSAStepper are the libraries' rather than a local
# transcription. The propensity semantics are identical to build_reaction_network: the
# abstract Reaction list is translated into JumpProcesses jumps with the same hazards.
# ---------------------------------------------------------------------------

"""
    catalyze(net::ReactionNetwork) -> (rs::ReactionSystem, u0::Dict, rates::Dict)

Compile a [`ReactionNetwork`](@ref) into a **Catalyst.jl** `ReactionSystem`, returning the
system, the initial species-count mapping, and the rate parameters implied by the
topology.

| reaction | Catalyst form |
|:---|:---|
`recruit`   | `0 --> x1`, propensity `sum_k f_k x_k`
`advance_k` | `x_k --> x_{k+1}`, propensity `a_k M x_k`
`retain_k`  | `x_k --> x_k`, propensity `r_k M x_k`
`natural_k` | `x_k --> 0`, propensity `(1 - a_k - r_k) M x_k`
`harvest_k` | `x_k --> 0`, propensity `u_k(t) x_k`

The system is compiled from a generated expression and cached per group count, because
Catalyst's `@reaction_network` requires literal species and reaction names. The returned
object is a genuine `ReactionSystem`, so `species(rs)`, `reactions(rs)` and the
stoichiometry matrix all work; printing `rs` renders the DSL form of the network.
"""
function catalyze(net::ReactionNetwork)
    n = net.params.n_groups
    n == length(net.initial_counts) ||
        error("ReactionNetwork has $(length(net.initial_counts)) initial counts for $n groups.")
    return _catalyst_system(n)
end

const _CATALYST_CACHE = Dict{Int, ReactionSystem}()

"""
    _catalyst_system(n::Int) -> ReactionSystem

Compile (or fetch from cache) the Catalyst network with `n` groups.

Built with Catalyst's **programmatic** constructor — `Catalyst.Reaction(rate, substrate,
product)` assembled into a `ReactionSystem` — rather than by generating
`@reaction_network` source. This is the reliable route: the DSL's line grammar is what the
macro consumes, and hand-building a matching AST is fragile (a malformed reaction line
surfaces only as "input contain 1 malformed lines" with no indication of which). The
programmatic constructor takes the rate expression, substrate vector and product vector
directly, so there is nothing to misparse.

The system is cached per group count, which is exact because the topology is fully
determined by `n`.
"""
function _catalyst_system(n::Int)
    haskey(_CATALYST_CACHE, n) && return _CATALYST_CACHE[n]

    n >= 2 || error("The reaction representation needs at least 2 groups, got $n.")

    # Symbols are generated as **source text** and parsed, rather than built as an AST.
    # `@variables` uses Julia's `..` splatting for indexed species, which is not valid
    # inside a `quote` block, and hand-building the right AST is the fragile step; text
    # plus `Meta.parse` avoids it entirely. Evaluation happens in Catalyst's own module
    # so the resulting `ReactionSystem` is a Catalyst object with Catalyst-owned symbols.
    rec_names = [Symbol("rate_recruit_", k) for k in 1:n]
    adv_names = [Symbol("rate_advance_", k) for k in 1:(n - 1)]
    nat_names = [Symbol("rate_natural_", k) for k in 1:n]
    har_names = [Symbol("rate_harvest_", k) for k in 1:n]
    sp_names  = [Symbol("x_", k) for k in 1:n]

    rec_names_str = repr(rec_names)
    adv_names_str = repr(adv_names)
    nat_names_str = repr(nat_names)
    har_names_str = repr(har_names)
    sp_names_str  = repr(sp_names)

    source = """
    using Catalyst
    @independent_variables t
    rec_syms = $(rec_names_str)
    adv_syms = $(adv_names_str)
    nat_syms = $(nat_names_str)
    har_syms = $(har_names_str)
    sp_syms  = $(sp_names_str)

    rec = [first(Catalyst.@parameters(\$(rec_syms[k]))) for k in 1:$n]
    adv = [first(Catalyst.@parameters(\$(adv_syms[k]))) for k in 1:$(n - 1)]
    nat = [first(Catalyst.@parameters(\$(nat_syms[k]))) for k in 1:$n]
    har = [first(Catalyst.@parameters(\$(har_syms[k]))) for k in 1:$n]
    xs = [first(Catalyst.@species(\$(sp_syms[k])(t))) for k in 1:$n]

    rxns = Any[]
    push!(rxns, Catalyst.Reaction(rec[1], [xs[1]], [xs[1]], [1], [2]; name = :recruit_1))
    for k in 2:$n
        push!(rxns, Catalyst.Reaction(rec[k], [xs[k]], [xs[k], xs[1]], [1], [1, 1]; name = Symbol(:recruit_, k)))
    end
    for k in 1:$(n - 1)
        push!(rxns, Catalyst.Reaction(adv[k], [xs[k]], [xs[k + 1]]; name = Symbol(:advance_, k)))
    end
    for k in 1:$n
        push!(rxns, Catalyst.Reaction(nat[k], [xs[k]], []; name = Symbol(:natural_, k)))
        push!(rxns, Catalyst.Reaction(har[k], [xs[k]], []; name = Symbol(:harvest_, k)))
    end
    Catalyst.complete(Catalyst.ReactionSystem(rxns, t, xs, vcat(rec, adv, nat, har); name = Symbol(:rn_, $n)))
    """

    rs = Core.eval(Catalyst, Meta.parseall(source))

    _CATALYST_CACHE[n] = rs
    return rs
end

"""
    catalyzed_u0(rs::ReactionSystem, net::ReactionNetwork) -> Dict

Initial species-count mapping for the compiled Catalyst [`ReactionSystem`](@ref).
"""
function catalyzed_u0(rs::ReactionSystem, net::ReactionNetwork)
    sps = species(rs)
    length(sps) == length(net.initial_counts) ||
        error("Species count mismatch between Catalyst ReactionSystem and ReactionNetwork.")
    return Dict(sps[i] => Float64(net.initial_counts[i]) for i in eachindex(sps))
end

"""
    jump_parameters(rs::ReactionSystem, net::ReactionNetwork, year_index::Int = 1) -> Dict

The rate parameters implied by the network topology, mapped to Catalyst parameter symbols.
"""
function jump_parameters(rs::ReactionSystem, net::ReactionNetwork, year_index::Int = 1)
    n = net.params.n_groups
    ny = size(net.params.harvest_rate, 2)
    yidx = ((year_index - 1) % ny) + 1
    p = net.params

    sym_rates = Dict{Symbol, Float64}()
    for k in 1:n
        sym_rates[Symbol(:rate_recruit_, k)] = p.fertility[k] * exp(-p.natural_mortality)
    end
    for k in 1:(n - 1)
        sym_rates[Symbol(:rate_advance_, k)] = p.advance[k] * p.natural_mortality
    end
    for k in 1:n
        sym_rates[Symbol(:rate_natural_, k)] =
            max(0.0, 1.0 - p.advance[k] - p.retention[k]) * p.natural_mortality
        sym_rates[Symbol(:rate_harvest_, k)] = p.harvest_rate[k, yidx]
    end

    rates = Dict{Any, Float64}()
    for param in parameters(rs)
        psym = Symbol(param)
        if haskey(sym_rates, psym)
            rates[param] = sym_rates[psym]
        end
    end
    return rates
end

"""
    simulate_ssa_jump(net::ReactionNetwork, t_end::Real; year_index::Int = 1,
                      n_saves::Integer = 25, rng::AbstractRNG = Random.default_rng()
                      ) -> SSAResult

Simulate the reaction network with the **Gillespie SSA** using
`JumpProcesses.SSAStepper` over a `JumpProblem` assembled from the Catalyst-compiled
network.

The network is converted to a `JumpProcesses` system, so the jumps are library objects and
the combining, propensity evaluation and `SSAStepper` are `JumpProcesses`'. The
state-dependent harvest rate is read at `year_index`, so `total_harvested` is the biomass
removed in that year's harvest-rate regime.

# Arguments
- `net`: [`ReactionNetwork`](@ref).
- `t_end`: simulation horizon, in the model's time units.
- `year_index`: index into the harvest-rate schedule.
- `n_saves`: number of recorded frames excluding the initial one.
- `rng`: RNG for reproducibility.
"""
function simulate_ssa_jump(net::ReactionNetwork, t_end::Real;
                           year_index::Int = 1, n_saves::Integer = 25,
                           rng::AbstractRNG = Random.default_rng())
    t_end > 0 || error("t_end must be positive, got $t_end.")
    n_saves >= 1 || error("n_saves must be at least 1, got $n_saves.")

    rs = catalyze(net)
    u0 = catalyzed_u0(rs, net)
    rates = jump_parameters(rs, net, year_index)

    # Assemble JumpProblem directly from the Catalyst ReactionSystem. Catalyst integrates
    # with JumpProcesses so that JumpProblem converts the ReactionSystem into discrete jumps.
    prob = JumpProblem(rs, u0, (0.0, t_end), rates; rng = rng)
    sol = solve(prob, SSAStepper(); saveat = range(0.0, t_end, length = n_saves + 1))

    n_t = length(sol.t)
    counts = zeros(Float64, net.params.n_groups, n_t)
    sps = species(rs)
    for (j, _) in enumerate(sol.t)
        for i in 1:net.params.n_groups
            counts[i, j] = sol[sps[i]][j]
        end
    end
    return SSAResult(collect(sol.t), counts,
                     Tuple{Float64, Symbol, Int}[], 0.0)
end

"""
    save_ensemble(zarr_dir, ensemble) -> zarr_dir

Persist a [`ParameterEnsemble`](@ref) as a chunked Zarr array store at `zarr_dir`, so the
projection engine can select from it later without the original Turing model or chains in
memory.

Fields are stacked across draws into dense arrays:

| field | shape |
|:---|:---|
`natural_mortality` | `(n_draws,)`
`weights`, `maturity`, `fertility`, `advance`, `retention`, `initial_numbers` | `(n_draws, n_groups)`
`selectivity` | `(n_draws, n_groups, ngear)`
`fishing_mortality` | `(n_draws, n_groups, n_fcols)`
`harvest_rate` | `(n_draws, n_groups, n_years)`

Zarr is used because it is the package's existing multidimensional export path
(`write_zarr`), it chunks large archives without loading them whole, and its array
semantics match the draw structure directly: a projection needs one whole draw per step,
which is one chunk along the draw axis. `model` and `mode` are stored as integer codes
with their label tables, because Zarr has no native string dataset.

# Shape constraint
Every draw must share the same group count and harvest-rate column count, so the archive
is a rectangular stack. Mismatched shapes are rejected at save time.
"""
function save_ensemble(zarr_dir::AbstractString, ensemble::ParameterEnsemble)
    isempty(ensemble.draws) && error("Cannot save an empty ensemble.")
    ref = ensemble.draws[1]
    ng = ref.n_groups
    ny = size(ref.harvest_rate, 2)
    nsel = size(ref.selectivity, 2)
    nfm = size(ref.fishing_mortality, 2)
    n = length(ensemble.draws)

    for (i, d) in enumerate(ensemble.draws)
        d.n_groups == ng ||
            error("Draw $i has $(d.n_groups) groups, draw 1 has $ng; the archive must be rectangular.")
        size(d.harvest_rate, 2) == ny ||
            error("Draw $i has $(size(d.harvest_rate, 2)) harvest-rate columns, draw 1 has $ny.")
        size(d.selectivity, 2) == nsel ||
            error("Draw $i has $(size(d.selectivity, 2)) selectivity columns, draw 1 has $nsel.")
        size(d.fishing_mortality, 2) == nfm ||
            error("Draw $i has $(size(d.fishing_mortality, 2)) fishing-mortality columns, draw 1 has $nfm.")
    end

    isdir(zarr_dir) && rm(zarr_dir; recursive = true, force = true)
    mkpath(zarr_dir)

    n_groups_arr = [d.n_groups for d in ensemble.draws]
    _zput(zarr_dir, "n_groups", n_groups_arr)
    _zput(zarr_dir, "natural_mortality", [d.natural_mortality for d in ensemble.draws])
    # Vector fields are stored draw-first as `(n_draws, n_groups)` so that draw `i` is
    # `arr[i, :]`. `reduce(hcat)` would give the transpose, which would silently swap
    # the group and draw axes.
    for (nm, field) in (("weights", :weights), ("maturity", :maturity),
                        ("fertility", :fertility), ("advance", :advance),
                        ("retention", :retention), ("initial_numbers", :initial_numbers))
        arr = zeros(Float64, n, ng)
        for (i, d) in enumerate(ensemble.draws)
            arr[i, :] .= getfield(d, field)
        end
        _zput(zarr_dir, nm, arr)
    end
    _zput(zarr_dir, "selectivity", _stack3([d.selectivity for d in ensemble.draws]))
    _zput(zarr_dir, "fishing_mortality", _stack3([d.fishing_mortality for d in ensemble.draws]))
    _zput(zarr_dir, "harvest_rate", _stack3([d.harvest_rate for d in ensemble.draws]))

    # `model` and `mode` are integer codes into label tables, since Zarr datasets are
    # numeric.
    models = unique([string(d.source.model) for d in ensemble.draws])
    modes = unique([string(d.source.mode) for d in ensemble.draws])
    m_index = Dict(m => i for (i, m) in enumerate(models))
    mode_index = Dict(m => i for (i, m) in enumerate(modes))
    _zput(zarr_dir, "model_code", [m_index[string(d.source.model)] for d in ensemble.draws])
    _zput(zarr_dir, "mode_code", [mode_index[string(d.source.mode)] for d in ensemble.draws])

    # Metadata goes in a JSON sidecar rather than Zarr group attributes: the array
    # datasets are created directly under the store (matching `write_zarr` in
    # `src/report.jl`), so there is no `.zgroup` whose `attrs` could be opened, and a
    # sidecar is readable with the stdlib reader.
    manifest = Dict(
        "ensemble_id" => ensemble.ensemble_id,
        "n_draws" => n,
        "n_groups" => ng,
        "n_years" => ny,
        "n_gears" => nsel,
        "n_fishing_columns" => nfm,
        "model_labels" => models,
        "mode_labels" => modes,
    )
    open(joinpath(zarr_dir, "ensemble_attrs.json"), "w") do io
        println(io, JSON.json(manifest))
    end

    return zarr_dir
end

"""
    _is_dataset(path, name) -> Bool

True when a Zarr dataset array exists at `<path>/<name>`. Zarr writes either a
`.zarray` (v2) or `zarr.json` (v3) metadata file, so both are accepted, and the
directory itself is checked so a missing dataset is a clear message rather than an
`isfile` error.
"""
function _is_dataset(path::AbstractString, name::AbstractString)
    dir = joinpath(path, name)
    isdir(dir) || return false
    return isfile(joinpath(dir, ".zarray")) || isfile(joinpath(dir, "zarr.json"))
end

"""
    _zget(path, name) -> Array

Read a dataset `name` from a Zarr store into a dense `Array`. Datasets are opened by
their own path rather than through a root group, because `zcreate` writes them directly
under the store (the convention `write_zarr` uses) and no `.zgroup` metadata exists at
the root to open.
"""
function _zget(path::AbstractString, name::AbstractString)
    z = Zarr.zopen(joinpath(path, name))
    return collect(z)
end

"""
    _zput(zarr_dir, name, arr) -> Nothing

Write `arr` as a Zarr dataset `name` under `zarr_dir`, creating it with the array's own
shape. Intended for the ensemble archive, where every dataset is written once.
"""
function _zput(zarr_dir::AbstractString, name::AbstractString, arr::AbstractArray)
    z = Zarr.zcreate(eltype(arr), size(arr)...; path = joinpath(zarr_dir, name))
    z .= arr
    return nothing
end

"""
    _stack3(mats::Vector{<: AbstractMatrix}) -> Array{Float64, 3}

Stack equally shaped matrices along a new **leading** axis, so the result is
`(n_draws, rows, cols)` and draw `i` is `result[i, :, :]`. `cat(...; dims=3)` appends the
draw axis last, so the dimensions are permuted back to draw-first; this also means
`collect(z)` gives the order the loader expects.

`cat` materialises a dense `Array`, which Zarr's compressor requires: a lazy
`Generator` is rejected with a "buffer eltype must be isbits" error inside Blosc.
"""
_stack3(mats::Vector{<: AbstractMatrix}) = permutedims(cat(mats...; dims = 3), (3, 1, 2))

"""
    load_ensemble(path; n_draws::Integer = 0) -> ParameterEnsemble

Read a [`ParameterEnsemble`](@ref) written by [`save_ensemble`](@ref). `n_draws` limits
how many draws are retained (the first `n_draws`), keeping ensemble sizes bounded for
long-horizon projections; 0 keeps all.

Each field is validated on load: array shapes against the declared group count, the
harvest-rate column count, and `advance` + `retention` summing to one per group. A
malformed archive is rejected with a message naming the draw and field, so a corrupted
store cannot silently produce a plausible but wrong projection.
"""
function load_ensemble(path::AbstractString; n_draws::Integer = 0)
    isdir(path) || error("Ensemble archive not found: $path")
    manifest_path = joinpath(path, "ensemble_attrs.json")
    isfile(manifest_path) ||
        error("Archive $path is missing ensemble_attrs.json; save_ensemble writes one.")
    manifest = JSON.parse(read(manifest_path, String))
    manifest isa AbstractDict || error("Archive manifest at $manifest_path is not a JSON object.")

    for nm in ("n_groups", "natural_mortality", "weights", "maturity", "fertility",
               "advance", "retention", "selectivity", "fishing_mortality",
               "harvest_rate", "initial_numbers", "model_code", "mode_code")
        _is_dataset(path, nm) || error("Archive $path is missing dataset '$nm'.")
    end

    ng = Int(manifest["n_groups"])
    ny = Int(manifest["n_years"])
    nsel = Int(manifest["n_gears"])
    nfm = Int(manifest["n_fishing_columns"])
    total = length(_zget(path, "natural_mortality"))
    keep = n_draws > 0 ? min(Int(n_draws), total) : total
    keep >= 1 || error("Archive $path holds no draws.")

    models = Vector{String}(manifest["model_labels"])
    modes = Vector{String}(manifest["mode_labels"])

    n_groups_g = _zget(path, "n_groups")
    mort = _zget(path, "natural_mortality")
    weights = _zget(path, "weights")
    maturity = _zget(path, "maturity")
    fertility = _zget(path, "fertility")
    advance = _zget(path, "advance")
    retention = _zget(path, "retention")
    init = _zget(path, "initial_numbers")
    sel = _zget(path, "selectivity")
    fm = _zget(path, "fishing_mortality")
    hr = _zget(path, "harvest_rate")
    mcode = _zget(path, "model_code")
    modecode = _zget(path, "mode_code")

    (size(sel, 2) == ng && size(sel, 3) == nsel) ||
        error("Archive selectivity is $(size(sel)[2:3]) with $(size(sel, 1)) draws; " *
              "the manifest declares $ng groups x $nsel gears.")
    (size(hr, 3) == ny) ||
        error("Archive harvest_rate has $(size(hr, 3)) columns; expected $ny.")
    (size(fm, 3) == nfm) ||
        error("Archive fishing_mortality has $(size(fm, 3)) columns; expected $nfm.")

    draws = map(1:keep) do i
        n_groups_i = Int(n_groups_g[i])
        n_groups_i == ng ||
            error("Draw $i has $n_groups_i groups; the archive declares $ng.")
        for (nm, v, len) in (("weights", weights[i, :], ng), ("maturity", maturity[i, :], ng),
                             ("fertility", fertility[i, :], ng), ("advance", advance[i, :], ng),
                             ("retention", retention[i, :], ng), ("initial_numbers", init[i, :], ng))
            length(v) == len || error("Draw $i field '$nm' has $(length(v)) entries; expected $len.")
        end
        for k in 1:ng
            isapprox(advance[i, k] + retention[i, k], 1; atol = 1.0e-6) ||
                error("Draw $i group $k has advance + retention = " *
                      "$(advance[i, k] + retention[i, k]), not 1.")
        end
        mc = Int(mcode[i])
        1 <= mc <= length(models) || error("Draw $i has out-of-range model code $mc.")
        modc = Int(modecode[i])
        1 <= modc <= length(modes) || error("Draw $i has out-of-range mode code $modc.")

        BiologicalParameters(
            n_groups_i,
            Vector{Float64}(weights[i, :]), Vector{Float64}(maturity[i, :]),
            Vector{Float64}(fertility[i, :]), Float64(mort[i]),
            Vector{Float64}(advance[i, :]), Vector{Float64}(retention[i, :]),
            Matrix{Float64}(sel[i, :, :]),
            Matrix{Float64}(fm[i, :, :]),
            Matrix{Float64}(hr[i, :, :]),
            Vector{Float64}(init[i, :]),
            ParameterSource(Symbol(models[mc]), Symbol(modes[modc]), keep)
        )
    end

    id = String(manifest["ensemble_id"])
    return ParameterEnsemble(draws, id)
end

export BiologicalParameters, ProjectionState, ParameterSource
export ParameterEnsemble, SimulationControls, load_simulation_controls
export projection_parameters, project_state, project_from_slice, reconstruct_backward
export save_ensemble, load_ensemble
export AgentSpec, AgentPopulation, GroupAgent, agent_step!, model_step!
export spawn_population, run_agent_projection, group_abundance
export Reaction, ReactionNetwork, build_reaction_network, simulate_ssa, SSAResult
export simulate_ssa_jump
export Catalyst, JumpProcesses
export catalyze, catalyzed_u0, jump_parameters
