# AssessmentTools.jl

A modular fisheries stock assessment and management analysis toolkit in native Julia.

## Overview

`AssessmentTools.jl` provides structured, high-performance analytical tools for fisheries stock assessment, diagnostic modeling, and harvest control simulations:

- **Statistical Catch-at-Age (`StatisticalCatchAtAge`)**: Integrated forward-simulating age-structured population dynamics with multi-fleet Baranov catch equations, flexible selectivity formulations, exact automatic differentiation via `ForwardDiff.jl`, phased L-BFGS numerical optimization, and multi-fleet MSY equilibrium reference points.
- **Biomass Dynamics (`BiomassDynamics`)**: Continuous and discrete-time surplus production models (Schaefer logistic models, discrete maps) with seasonal survey timing calibrations and classical biological reference points ($B_{\text{MSY}}$, $F_{\text{MSY}}$, $\text{MSY}$).
- **Cohort Analysis (`CohortAnalysis`)**: Backward cohort reconstruction techniques including Pope's (1972) approximation and exact Virtual Population Analysis (VPA) with Newton-Raphson convergence.
- **Delay Differential Dynamics (`DelayDifferential`)**: Multi-stage continuous delay differential equation (DDE) systems capturing development lags, maternal reproductive delays, habitat-crowding constraints, and pulse fishery removals (Choi 2024).
- **Empirical Diagnostics (`Empirical`)**: Rapid non-parametric and linear regression diagnostics including Chapman-Robson minimum-variance unbiased mortality estimators, weighted catch curves, Leslie and DeLury depletion estimators, and Age-Length Key (ALK) expansion matrices.
- **Harvest Management (`Management`)**: Precautionary Harvest Control Rules (piecewise-linear hockey-stick HCRs) and quota change stabilizers for simulation and Management Strategy Evaluation (MSE).
- **Modern Data & Storage**: Pure CSV, TOML, Apache Parquet, and chunked Zarr array storage without binary legacy dependencies.

For complete mathematical definitions, architectural design, and API contracts across all submodules, refer to [AssessmentToolsAPI.md](file:///c:/home/jae/projects/AssessmentTools/AssessmentToolsAPI.md).

## Quick Start

```julia
using AssessmentTools

# 1. Statistical Catch-at-Age Assessment
model = StockModel("data/silverhake")
result = fit(model; verbose = true)
ref_points = calc_reference_points(model, result.x)
risk_table = decision_table(model, result.x)

# 2. Discrete Logistic Surplus Production
using AssessmentTools.BiomassDynamics
sim_bio = simulate_logistic_discrete(2000:2020, fill(10.0, 21); r = 0.35, k = 100.0)
msy_ref = logistic_msy_reference_points(0.35, 100.0)

# 3. Cohort Analysis & VPA
using AssessmentTools.CohortAnalysis
pope_res = popes_cohort_analysis(catches_matrix, 0.2; terminal_f = [0.2, 0.3, 0.4])
vpa_res = standard_vpa(catches_matrix, 0.2; terminal_f = [0.2, 0.3, 0.4])
adapt_res = adapt_vpa(catches_matrix, survey_matrix, 0.2)

# 4. Multi-Stage Delay Differential Equations (DDE)
using AssessmentTools.DelayDifferential
dde_cfg = default_snowcrab_dde_config()
dde_sim = simulate_dde_system(dde_cfg, (0.0, 10.0); dt = 1.0 / 12.0)

# 5. Empirical Diagnostics & Management (MSE)
using AssessmentTools.Empirical
using AssessmentTools.Management
cr = chapman_robson(ages, catches)
hcr = HockeyStickHCR(1000.0, 2500.0, 0.05, 0.30)
mse_res = simulate_mse(2020:2035, hcr; r = 0.35, k = 100.0)

# 6. Retrospective Analysis & Visualizations
retro_bio = retrospective_analysis(2000:2020, catches; n_peels = 4)
plot_retrospective(retro_bio)
plot_state_variables(sim_bio)
plot_fishery_predictions(years, ensemble_trajectories; obs_time=obs_years, obs_values=obs_survey)
plot_fishery_footprint(years, biomass_fished, biomass_unfished)

# 7. High-Performance Columnar and Multidimensional Storage
write_parquet("trajectories.parquet", model, result.x)
write_zarr("state.zarr", model, result.x)
```

## Interactive Pluto Workflows

Seven comprehensive interactive workflow notebooks are located in [`notebooks/`](file:///c:/home/jae/projects/AssessmentTools/notebooks/):

1. **Silver Hake Workflow**: [`notebooks/silver_hake_workflow.jl`](file:///c:/home/jae/projects/AssessmentTools/notebooks/silver_hake_workflow.jl)
   - Multi-gear age-structured synthesis with phased automatic differentiation and forward projections.
2. **Cusk Assessment Workflow**: [`notebooks/cusk_workflow.jl`](file:///c:/home/jae/projects/AssessmentTools/notebooks/cusk_workflow.jl)
   - 4-fleet model with time-varying catchability random walks and Baranov equilibrium reference points.
3. **Biomass Dynamics Workflow**: [`notebooks/biomass_dynamics_workflow.jl`](file:///c:/home/jae/projects/AssessmentTools/notebooks/biomass_dynamics_workflow.jl)
   - Ingested Scotian Shelf snow crab biomass data, Schaefer surplus production simulations, and MSY calculations.
4. **Cohort Analysis Workflow**: [`notebooks/cohort_analysis_workflow.jl`](file:///c:/home/jae/projects/AssessmentTools/notebooks/cohort_analysis_workflow.jl)
   - Reconstructing historical gadoid numbers-at-age and fishing mortality via Pope's approximation and exact Newton-Raphson VPA.
5. **Delay Differential Models Workflow**: [`notebooks/delay_differential_workflow.jl`](file:///c:/home/jae/projects/AssessmentTools/notebooks/delay_differential_workflow.jl)
   - Continuous 6-stage developmental and maternal lag DDE simulations with seasonal pulse removals (Choi 2024).
6. **Delay Difference Dynamics Workflow**: [`notebooks/delay_difference_workflow.jl`](file:///c:/home/jae/projects/AssessmentTools/notebooks/delay_difference_workflow.jl)
   - Deriso-Schnute delay-difference biomass dynamics driven by male recruitment ($M_1$) extracted from the continuous DDE system.
7. **Empirical Assessment & HCR Methods**: [`notebooks/empirical_and_hcr_methods.jl`](file:///c:/home/jae/projects/AssessmentTools/notebooks/empirical_and_hcr_methods.jl)
   - Chapman-Robson survival, linear catch curves, Leslie/DeLury depletion, Age-Length Keys, and hockey-stick HCR stabilizers.

To launch any interactive session:
```julia
import Pluto
Pluto.run(notebook = "notebooks/biomass_dynamics_workflow.jl")
```
