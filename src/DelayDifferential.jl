# ------------------------------------------------------------------------------------
# Module for Continuous Multi-Stage Delay Differential Equation Models
# ------------------------------------------------------------------------------------

"""
    AssessmentTools.DelayDifferential

Continuous size- and stage-structured delay differential population dynamics:
- Multi-stage coupled DDE systems with developmental and maternal reproductive delay lags.
- Density-dependent natural and habitat-constrained mortality.
- Intra-annual seasonal time steps with pulse catch removals.
"""
module DelayDifferential

using LinearAlgebra
using Statistics
using Distributions
using Turing
using ..Config

export DdeModelConfig, load_dde_config
export dde_derivatives, simulate_dde_system
export bayesian_delay_differential_model
export DelayDifferenceConfig, DelayDifferenceResult
export delay_difference_step, simulate_delay_difference, extract_annual_recruitment
export bayesian_delay_difference_model

include("delay_differential.jl")
include("delay_difference.jl")

end # module DelayDifferential
