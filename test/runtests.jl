# ==============================================================================
# AssessmentTools Comprehensive Test Suite Orchestrator
# ==============================================================================

import Pkg

const RUNNING_UNDER_PKG_TEST = basename(dirname(Base.active_project())) == "test"
if !RUNNING_UNDER_PKG_TEST
    Pkg.activate(joinpath(@__DIR__, ".."); io = devnull)
    Pkg.instantiate(; io = devnull)
end

# Usage:
#   julia --project=. test/runtests.jl                # Run every segment
#   julia --project=. test/runtests.jl models         # Run single-species models (Silver Hake, Cusk, SCAA)
#   julia --project=. test/runtests.jl dynamics       # Run empirical & surplus production dynamics
#   julia --project=. test/runtests.jl cohort         # Run VPA & Pope's cohort analysis
#   julia --project=. test/runtests.jl simulation     # Run DDE, continuous biomass & SimulationEngine
#
# Under `Pkg.test()` the whole suite runs; pass segments as test_args:
#   Pkg.test(test_args = ["models", "dynamics"])
# ==============================================================================

using Test
using AssessmentTools

const AVAILABLE_SEGMENTS = Dict{Symbol, @NamedTuple{file::String, desc::String}}(
    :models => (
        file = "test_models.jl",
        desc = "Single-Species Stock Models (Silver Hake, Cusk), SCAA & Reference Points"
    ),
    :dynamics => (
        file = "test_dynamics.jl",
        desc = "Empirical Assessments, Depletion, Surplus Production & HCR"
    ),
    :cohort => (
        file = "test_cohort.jl",
        desc = "Cohort Analysis (Pope's, Exact VPA, ADAPT) & Retrospective Peels"
    ),
    :simulation => (
        file = "test_simulation.jl",
        desc = "Continuous DDE, Continuous Biomass & SimulationEngine (SSA/Agents)"
    ),
    :multigear_ad => (
        file = "test_multigear_ad.jl",
        desc = "Multi-Gear Baranov Jacobian & AD Compatibility"
    )
)

"""
    parse_requested_segments(args::Vector{String}) -> Vector{Symbol}

Resolves CLI segment arguments to segment keys.
"""
function parse_requested_segments(args::Vector{String})::Vector{Symbol}
    if isempty(args)
        return collect(keys(AVAILABLE_SEGMENTS))
    end

    selected = Symbol[]
    for a in args
        key = Symbol(lowercase(strip(a)))
        if haskey(AVAILABLE_SEGMENTS, key)
            push!(selected, key)
        else
            avail = sort(string.(collect(keys(AVAILABLE_SEGMENTS))))
            @warn "Unrecognized test segment: '$a'. Available segments: $(join(avail, ", "))"
        end
    end

    return isempty(selected) ? sort!(collect(keys(AVAILABLE_SEGMENTS))) : unique(selected)
end

requested_segments = parse_requested_segments(ARGS)

@testset "AssessmentTools Comprehensive Suite" begin
    for seg_key in requested_segments
        info = AVAILABLE_SEGMENTS[seg_key]
        println("\n" * "="^80)
        println(">>> Running Segment: $(info.desc) ($(info.file))")
        println("="^80)
        include(joinpath(@__DIR__, info.file))
    end
end
