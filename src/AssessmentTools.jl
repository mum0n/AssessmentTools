"""
    AssessmentTools

A modular fisheries stock assessment and management analysis toolkit in native Julia.

# Submodules
- `StatisticalCatchAtAge`: Full age-structured statistical stock synthesis framework
  with automatic differentiation, phased optimization, and reference points.
- `BiomassDynamics`: Surplus production and discrete logistic models with seasonal
  survey calibrations and classical Schaefer MSY reference points.
- `CohortAnalysis`: Backward cohort reconstruction, Pope's cohort analysis, and exact
  virtual population analysis (VPA).
- `DelayDifferential`: Multi-stage continuous delay differential equation (DDE) models
  with reproductive lag dynamics and intra-annual seasonal resolution.
- `Empirical`: Standalone non-parametric and regression-based methods including catch curves,
  Chapman-Robson survival estimators, Leslie/DeLury depletion models, and Age-Length Keys.
- `Management`: Harvest Control Rules (HCR) and interannual TAC change stabilizers.
"""
module AssessmentTools

# Submodules are included cleanly so each domain is self-contained:
include("StatisticalCatchAtAge.jl")
include("BiomassDynamics.jl")
include("CohortAnalysis.jl")
include("DelayDifferential.jl")
include("Empirical.jl")
include("Management.jl")

using LinearAlgebra
using Statistics
using UnicodePlots

using .StatisticalCatchAtAge
using .BiomassDynamics
using .CohortAnalysis
using .DelayDifferential
using .Empirical
using .Management

# Re-export submodules for explicit namespaced use
export StatisticalCatchAtAge
export BiomassDynamics
export CohortAnalysis
export DelayDifferential
export Empirical
export Management

# Re-export core StatisticalCatchAtAge API
export StockData, StockControl, ProjectionControl, StockModel
export PhaseContext, ParameterLayout, ObjectiveComponents
export FitResult, MSYEquilibrium, DecisionMetrics
export load_dataset, load_parameters, write_report, write_parquet, write_zarr
export plogis, calc_selectivities, calc_total_mortality, calc_numbers_at_age
export calc_fishery_observations, calc_age_proportions
export calc_survey_observations, calc_stock_recruitment, objective_function
export solve_baranov_f, calc_reference_points, calc_equilibrium_point
export fit, project_population, decision_table, simulate_data
export bayesian_statistical_catch_at_age

# Re-export BiomassDynamics methods
export LogisticDiscreteResult
export logistic_step, logistic_map_step, simulate_logistic_discrete
export logistic_msy_reference_points
export bayesian_logistic_discrete

# Re-export CohortAnalysis methods
export VpaResult, AdaptResult, popes_cohort_analysis, standard_vpa, adapt_vpa
export bayesian_cohort_analysis, bayesian_adapt_vpa

# Re-export DelayDifferential & DelayDifference methods
export DdeModelConfig, default_snowcrab_dde_config, dde_derivatives, simulate_dde_system
export bayesian_delay_differential_model
export DelayDifferenceConfig, DelayDifferenceResult
export delay_difference_step, simulate_delay_difference, extract_annual_recruitment
export bayesian_delay_difference_model

# Re-export Empirical methods
export ChapmanRobsonResult, CatchCurveResult, chapman_robson, catch_curve_linear
export bayesian_catch_curve
export LeslieResult, DeLuryResult, leslie_depletion, delury_depletion
export bayesian_leslie_depletion, bayesian_delury_depletion
export AlkResult, compute_alk_matrix, apply_age_length_key

# Re-export Management methods
export HockeyStickHCR, eval_hcr, apply_tac_stabilizer
export MseResult, simulate_mse, bayesian_mse_feedback

# Diagnostics & Visualization
include("retrospective.jl")
include("visualization.jl")

export mohn_rho, RetrospectiveResult, retrospective_analysis
export plot_trajectory, plot_trajectory!, plot_state_variables, plot_retrospective
export plot_fishery_predictions, plot_fishery_footprint

end # module AssessmentTools
