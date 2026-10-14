# ------------------------------------------------------------------------------------
# Module for Biomass Dynamics and Discrete Surplus Production Models
# ------------------------------------------------------------------------------------

"""
    AssessmentTools.BiomassDynamics

Discrete-time and continuous surplus production models:
- Schaefer discrete logistic models with seasonal survey transitions.
- Multi-area biomass dynamics simulations.
- Classical MSY biological reference point computations (Bmsy, Fmsy, MSY).
"""
module BiomassDynamics

using LinearAlgebra
using Statistics
using Distributions
using Turing

export LogisticDiscreteResult
export logistic_step, logistic_map_step, simulate_logistic_discrete
export logistic_msy_reference_points
export bayesian_logistic_discrete

include("biomass_dynamics.jl")

end # module BiomassDynamics
