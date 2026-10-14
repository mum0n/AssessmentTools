# ------------------------------------------------------------------------------------
# Main Statistical Catch-at-Age Module
# ------------------------------------------------------------------------------------

"""
    AssessmentTools.StatisticalCatchAtAge

Core statistical catch-at-age forward simulation and estimation framework:
- Parameter layout, bounded transformations, and estimation phases.
- Baranov multi-fleet catch equations, age compositions, and survey indexing.
- Multi-gear MSY equilibrium reference points and Baranov fishing mortality solver.
- Phased ForwardDiff AD optimization via L-BFGS.
- Decision tables, projections, and simulation engine.
- High-performance storage export (Apache Parquet and chunked Zarr).
"""
module StatisticalCatchAtAge

using LinearAlgebra
using Statistics
using Random
using Printf
using SpecialFunctions
using ForwardDiff
using Optim
using TOML
using Distributions
using Turing
using Interpolations
using DataInterpolations
using Parquet2
using Zarr

# Primary model definitions
export StockData, StockControl, ProjectionControl, StockModel
export PhaseContext, ParameterLayout, ObjectiveComponents
export FitResult, MSYEquilibrium, DecisionMetrics

# File I/O & Storage
export load_dataset, load_parameters
export write_report, write_parquet, write_zarr

# Core modeling & dynamics
export plogis, calc_selectivities, calc_total_mortality, calc_numbers_at_age
export calc_fishery_observations, calc_age_proportions
export calc_survey_observations, calc_stock_recruitment
export objective_function
export bayesian_statistical_catch_at_age

# Reference points & optimization
export solve_baranov_f, calc_reference_points, calc_equilibrium_point
export fit, project_population, decision_table, simulate_data

# Include component sources
include("stats.jl")
include("splines.jl")
include("io.jl")
include("model.jl")
include("selectivity.jl")
include("dynamics.jl")
include("objective.jl")
include("baranov_msy.jl")
include("simulation_projection.jl")
include("estimation.jl")
include("report.jl")
include("bayesian_scaa.jl")

end # module StatisticalCatchAtAge
