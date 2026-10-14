# Module wrapper for empirical fisheries assessment methods

"""
    AssessmentTools.Empirical

Empirical assessment methods for direct data diagnostics without complex likelihood models:
- Chapman-Robson (1960) MVUE total mortality and survival estimation.
- Weighted and unweighted linear catch curve regressions.
- Leslie and DeLury depletion estimators for initial abundance and catchability.
- Age-Length Key (ALK) conditional transition matrices and length frequency expansions.
"""
module Empirical

using LinearAlgebra
using Statistics
using Distributions
using Turing

export ChapmanRobsonResult, CatchCurveResult, chapman_robson, catch_curve_linear
export bayesian_catch_curve
export LeslieResult, DeLuryResult, leslie_depletion, delury_depletion
export bayesian_leslie_depletion, bayesian_delury_depletion
export AlkResult, compute_alk_matrix, apply_age_length_key

include("catch_curve.jl")
include("depletion.jl")
include("age_length_key.jl")

end # module Empirical
