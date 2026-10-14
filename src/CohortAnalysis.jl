# ------------------------------------------------------------------------------------
# Module for Cohort Analysis & Virtual Population Analysis (VPA)
# ------------------------------------------------------------------------------------

"""
    AssessmentTools.CohortAnalysis

Cohort reconstruction and Virtual Population Analysis:
- Pope's (1972) backward cohort reconstruction approximation.
- Exact Gulland/Baranov VPA with Newton-Raphson cohort convergence.
- Spawning stock biomass and fishing mortality estimation.
"""
module CohortAnalysis

using LinearAlgebra
using Statistics
using Distributions
using Turing
using Optim

export VpaResult, AdaptResult
export popes_cohort_analysis, standard_vpa, adapt_vpa
export bayesian_cohort_analysis, bayesian_adapt_vpa

include("cohort_analysis.jl")

end # module CohortAnalysis
