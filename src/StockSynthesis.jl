# ------------------------------------------------------------------------------------
# Module for Stock Synthesis: Size-Structured Stock Assessment
#
# Extends the age-structured StatisticalCatchAtAge framework with:
# - Size- (length-) structured population dynamics
# - Multiple observation likelihoods (multinomial, Dirichlet-multinomial,
#   lognormal, normal) for catch, survey, and composition data
# - Multiple observation series (surveys, fleets, tagging)
# - Size composition (length frequency) handling
# ------------------------------------------------------------------------------------

"""
    AssessmentTools.StockSynthesis

Size-structured and age-structured stock assessment framework (Stock Synthesis style):
- Length-based and age-based population dynamics
- Multiple observation likelihoods for catch, survey indices, and compositions
- Multi-fleet, multi-survey data integration
- Size/age composition handling from individual measurements
- MSY reference points for size-structured models
- Turing-based inference (MCMC, VI, MAP, ML) via InferenceBridge
"""
module StockSynthesis

using LinearAlgebra
using Statistics
using Distributions
using Turing
using ForwardDiff
using Optim
using Random

export SizeStructuredModel, SizeClass, GrowthParameters
export size_transition_matrix, size_selectivity
export SizeCompositionData, bin_lengths, size_proportions, effective_sample_size
export size_composition_residual, AgeCompositionData, age_proportions
export LikelihoodSpec, MultiSeriesLikelihood, evaluate_likelihood, total_nll
export add_likelihood!, multi_series_likelihood, bayesian_multi_series
export fit_size_structured, sample_size_structured, vi_size_structured
export size_msy_reference_points
export bayesian_size_structured
export bayesian_age_synthesis, fit_age_synthesis, sample_age_synthesis, vi_age_synthesis

include("size_structure.jl")
include("size_composition.jl")
include("multilike.jl")
include("size_estimation.jl")
include("age_synthesis.jl")

end # module StockSynthesis