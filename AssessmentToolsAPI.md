# AssessmentTools.jl API Specification

`AssessmentTools.jl` is a modular fisheries stock assessment and management toolkit developed in Julia. This document details the software architecture, core data structures, mathematical definitions, expected data schemas, and API contracts across all submodules.

---

## 1. Package Architecture and Submodules

The package exposes six specialized functional namespaces:

```
AssessmentTools
├── StatisticalCatchAtAge  # Age-structured synthesis, forward dynamics, phased AD estimation, MSY reference points
├── BiomassDynamics        # Discrete logistic / Schaefer surplus production, seasonal survey calibrations
├── CohortAnalysis         # Pope's cohort analysis, exact Gulland/Baranov VPA back-reconstruction
├── DelayDifferential      # Continuous size- and stage-structured DDE models with reproductive lags
├── Empirical              # Catch curves, Chapman-Robson mortality, Leslie/DeLury depletion, Age-Length Keys
└── Management             # Harvest Control Rules (HCR), limit/target reference points, TAC stabilization
```

Top-level convenience exports re-export primary symbols from each submodule.

---

## 2. StatisticalCatchAtAge Submodule

### 2.1 Core Types and Structures

#### `StockData`
Encapsulates historical observations, biological parameters, and fleet dimensions:
- `syr::Int`, `nyr::Int`: Start and terminal years of the assessment horizon.
- `sage::Int`, `nage::Int`: Minimum and plus-group age classes.
- `ngear::Int`: Number of fishing gears/fleets.
- `allocation::Vector{Float64}`: Catch allocation fractions across fleets ($\sum a_k = 1$).
- `catch_type::Vector{Int}`: Observation model for catches ($1$ = biomass, $2$ = numbers).
- `obs_ct::Matrix{Float64}`: Observed catch matrix of dimension `ngear × ny`, where $ny = nyr - syr + 1$.
- `surveys::Vector{SurveySeries}`: Collection of relative abundance index series.
- `agecomps::Vector{AgeCompSeries}`: Collection of age-composition data series.
- `a::Float64`, `b::Float64`: Length-weight allometric power parameters ($w = a \cdot l^b$).
- `linf::Float64`, `vonb_k::Float64`, `to::Float64`: von Bertalanffy growth parameters.
- `ah::Float64`, `gh::Float64`: Maturity-at-age logistic inflection and scale.

#### `SurveySeries`
Represents an individual abundance index:
- `year::Vector{Int}`: Survey observation years.
- `index::Vector{Float64}`: Observed index values.
- `gear::Vector{Int}`: Associated gear index for selectivity mapping.
- `weight::Vector{Float64}`: Relative precision weights.
- `timing::Vector{Float64}`: Annual fraction of total mortality elapsed before survey sampling ($0 \le \text{timing} \le 1$).
- `survey_type::Int`: Index observation target ($1$ = vulnerable numbers, $2$ = vulnerable biomass, $3$ = spawning biomass).

#### `AgeCompSeries`
Represents composition sample observations:
- `year::Vector{Int}`: Sampling years.
- `gear::Vector{Int}`: Associated gear index.
- `sage::Int`, `nage::Int`: Age boundaries for the series.
- `obs::Matrix{Float64}`: Observed age-composition matrix (rows = years, columns = ages).

#### `StockControl`
Model configuration and estimation controls:
- `theta_control::Matrix{Float64}`: $7 \times 7$ matrix specifying initial values, bounds, prior distribution types, and estimation phases for leading parameters:
  1. $\log R_0$: Unfished equilibrium recruitment
  2. $h$: Steepness of the stock-recruitment curve
  3. $\log M$: Natural mortality rate
  4. $\log \bar{R}$: Average recruitment
  5. $\log R_{\text{init}}$: Initial recruitment scalar
  6. $\rho$: Variance proportion parameter
  7. $\vartheta$: Precision parameter
- `isel_type::Vector{Int}`: Selectivity formulation flags for each gear:
  - `1`: Logistic with small offset ($10^{-10}$)
  - `2`: Independent age-specific parameters with flat plus-group
  - `3`: Clamped cubic spline over age nodes on $[0, 1]$
  - `4`: Time-varying cubic spline per year
  - `5`: 2D Bicubic spline over age $\times$ year nodes
  - `6`: Exact logistic without offset
  - `7`: Logistic over standardized weight-at-age
  - `8`: Logistic plus standardized weight deviations
  - `11`: Length-based logistic
  - `12`: Length-based cubic spline
- `sel_control::Vector{Matrix{Float64}}`: Bounded parameters and estimation phases for selectivity.
- `misc::StockMiscControl`: Miscellaneous tuning flags (recruitment model type, penalty weights, M deviation nodes).

#### `StockModel`
Main unified model container:
- Constructed directly from a directory: `StockModel("data/silverhake")`.
- Constructed from components: `StockModel(data, control, projection)`.
- Contains parameter layout, precomputed growth and maturity schedules, empirical weight-at-age matrices, and initial parameter vector `x0`.

### 2.2 Forward Dynamics Equations

#### Recruitment Models ($R_t$)
Beverton-Holt ($rec\_type = 1$):
```math
R_t = \frac{s_0 \cdot S_{t-a_0}}{1 + \beta \cdot S_{t-a_0}} \cdot \exp(\delta_t - 0.5 \tau^2)
```
Ricker ($rec\_type = 2$):
```math
R_t = s_0 \cdot S_{t-a_0} \exp(-\beta \cdot S_{t-a_0}) \cdot \exp(\delta_t - 0.5 \tau^2)
```
where $S_t$ is spawning stock biomass, $s_0 = \kappa / \phi_B$, $\kappa$ is Goodyear's compensation ratio, and $\phi_B$ is unfished spawning biomass per recruit.

#### Baranov Catch Equation
Catch in numbers $C_{k,t,a}$ for gear $k$, year $t$, and age $a$:
```math
C_{k,t,a} = \frac{F_{k,t,a}}{Z_{t,a}} \left( 1 - \exp(-Z_{t,a}) \right) N_{t,a}
```
where $Z_{t,a} = M_{t,a} + \sum_k F_{k,t,a}$ and $F_{k,t,a} = f_{k,t} \cdot v_{k,t,a}$.

#### Multi-Gear Baranov Solver
`solve_baranov_f(ct, m, V, na, wa; max_iter=50, tol=1e-12, max_f=5.0) -> Vector{Float64}`
Solves for instantaneous fishing mortalities $f_k$ given observed total catch weights $C_k$, selectivity matrix $V$, numbers $N$, and weight-at-age $w$ using Newton-Raphson iterations.

### 2.3 Objective Function and Likelihoods

`objective_function(model, x, ctx) -> (f_total, details::ObjectiveComponents)`

Components evaluated:
- **Catches**: Log-normal or normal negative log-likelihood across gears.
- **Surveys**: Lognormal likelihood with analytical concentrated catchability $\hat{q}_k$ or state-space random walk penalties.
- **Age Compositions**: Concentrated multivariate logistic (`dmvlogistic`) or Dirichlet/multinomial with Richards et al. (1997) bin pooling.
- **Recruitment Deviations**: Lognormal process error penalty $\sum \delta_t^2 / (2 \tau^2) + n \log \tau$.
- **Selectivity Regularization**: Second-difference smoothness penalties and doming penalties.
- **Prior Distributions**: Normal (`dnorm`), Log-normal (`dlnorm`), Beta (`dbeta`), and Gamma (`dgamma`) evaluated via `Distributions.jl`.

### 2.4 Biological Reference Points

`calc_reference_points(model, mort, log_sel, theta) -> MSYEquilibrium`
Evaluates equilibrium states across a range of fishing mortality multipliers to identify maximum sustainable yield:
- `bo`: Unfished equilibrium spawning biomass ($B_0$)
- `bmsy`: Spawning biomass at MSY ($B_{\text{MSY}}$)
- `msy`: Fleet-specific and aggregate maximum sustainable yield
- `fmsy`: Fleet-specific instantaneous fishing mortality at MSY ($F_{\text{MSY}}$)
- `umsy`: Harvest rate at MSY ($U_{\text{MSY}} = 1 - \exp(-\sum F_{\text{MSY}})$)

### 2.5 Optimization and Fitting

`fit(model; max_phase, iterations, g_abstol, verbose) -> FitResult`
Executes multi-phase estimation. In each phase $1 \le ph \le \text{max\_phase}$, inactive parameters are held fixed while active parameters are optimized using `Optim.jl` (L-BFGS) with exact automatic differentiation gradients computed via `ForwardDiff.jl`.

### 2.6 Bayesian Statistical Catch-at-Age

`bayesian_statistical_catch_at_age(model; prior_log_ro_sd, prior_h_sd, prior_m_sd, ...)`
Full Bayesian state-space forward simulation in `Turing.jl`:
- Samples leading biological parameters $\theta = (\log R_0, h, M, \log\bar{R}, \log R_{\text{init}}, \rho, \vartheta)$.
- Samples annual recruitment deviations $\delta_t \sim \mathrm{Normal}(0, \tau)$ and log fishing mortalities.
- Evaluates the joint posterior across commercial catches, relative survey indices, and stock-recruitment relationships.

### 2.7 Persistence and Storage

- `write_report(io_or_path, model, x)`: Produces detailed diagnostic text report files.
- `write_parquet(filepath, model, x)`: Exports annual trajectory time series (biomass, spawning biomass, recruitment, catches, average mortality) to Apache Parquet format.
- `write_zarr(zarr_dir, model, x)`: Serializes full multidimensional arrays ($N_{t,a}$, $F_{t,a}$) and state trajectories to a chunked Zarr directory store.

---

## 3. BiomassDynamics Submodule

Surplus production models and aggregated biomass dynamics.

### 3.1 Discrete Logistic Equations

`simulate_logistic_discrete(years, removals; r, k, b0, q, survey_transition) -> LogisticDiscreteResult`
Simulates time series under the discrete Schaefer surplus production model:
```math
B_{t+1} = \max\left(0, B_t + r B_t \left(1 - \frac{B_t}{K}\right) - C_t\right)
```
Accounts for pre-fishery (spring), transition, or post-fishery (fall) survey observation regimes.

`logistic_msy_reference_points(r, k) -> NamedTuple`
Computes classical Schaefer MSY reference points:
```math
B_{\text{MSY}} = \frac{K}{2}, \quad \text{MSY} = \frac{r K}{4}, \quad F_{\text{MSY}} = \frac{r}{2}, \quad U_{\text{MSY}} = 1 - \exp(-F_{\text{MSY}})
```

### 3.2 Bayesian State-Space Biomass Dynamics

`bayesian_logistic_discrete(removals, survey_obs; survey_transition=0, ...)`
Bayesian state-space discrete Schaefer surplus production model in `Turing.jl`:
- Samples intrinsic growth rate $r$, carrying capacity $K$, survey catchability $q$, and initial depletion $m_0 = B_1 / K$.
- Propagates latent depletion $m_t$ with process error $\sigma_p$ and seasonal survey timing (spring, transition, or fall).
- Observes survey indices $I_t \sim \mathrm{LogNormal}(\log(q K m_t), \sigma_o)$.

---

## 4. CohortAnalysis Submodule

Historical cohort back-calculation and Virtual Population Analysis.

### 4.1 Pope's Cohort Analysis

`popes_cohort_analysis(catches, M; terminal_f, plus_group_f, plus_group_age) -> VpaResult`
Back-calculates numbers-at-age and fishing mortality rates using Pope's (1972) mid-year catch approximation:
```math
N_{a,t} = N_{a+1, t+1} \exp(M_{a,t}) + C_{a,t} \exp\left(\frac{M_{a,t}}{2}\right)
```

### 4.2 Exact Baranov VPA

`standard_vpa(catches, M; terminal_f, tol, max_iter) -> VpaResult`
Inverts the continuous Baranov catch equation for every cohort cell $(a, t)$ using Newton-Raphson iterations:
```math
\frac{C_{a,t}}{N_{a+1,t+1}} = \frac{F_{a,t}}{F_{a,t} + M_{a,t}} \frac{1 - \exp(-(F_{a,t} + M_{a,t}))}{\exp(-(F_{a,t} + M_{a,t}))}
```

### 4.3 Bayesian State-Space Cohort Analysis

`bayesian_cohort_analysis(catches, survey_index, M; survey_ages, prior_terminal_f, prior_f_sd, obs_cv)`
Bayesian cohort analysis in `Turing.jl`:
- Formulates backward cohort transitions using Pope's mid-year approximation.
- Samples terminal fishing mortalities $F_{a, T}$ and age-specific survey catchabilities $q_a$.
- Evaluates lognormal likelihoods for observed survey indices $I_{a, t} \sim \mathrm{LogNormal}(\log(q_a N_{a, t}), \sigma_o)$.

---

## 5. DelayDifferential Submodule

Continuous stage- and sex-structured delay differential equation systems (Choi 2024).

### 5.1 Model Structure & Lags

`default_snowcrab_dde_config() -> DdeModelConfig`
Standard parameterization for 6 life-history stages:
1. Sub-legal instar 9 males ($M_0$)
2. Instar 10 males ($M_1$)
3. Instar 11 males ($M_2$)
4. Commercial recruits ($M_3$)
5. Legal commercial males ($M_4$)
6. Mature females ($f_{\text{mat}}$)

`simulate_dde_system(cfg, tspan; dt, removals_times, removals_amount) -> NamedTuple`
Integrates continuous coupled delay differential equations across developmental lags $\tau \in \{7, 8, 9\}$ years with pulse fishery removals and density-dependent habitat-crowding mortality $d_2 u^2 / H(t)$.

### 5.2 Deriso-Schnute Delay-Difference Dynamics

`DelayDifferenceConfig(s, alpha, rho, wk)`
Encapsulates Ford-Brody growth coefficients ($\rho$, $\alpha = w_k - \rho w_{k-1}$), recruitment entry weight $w_k$, and natural annual survival $s = \exp(-M)$.

`extract_annual_recruitment(dde_simulation; stage_index=2, target_month=9.0) -> (years, recruits)`
Extracts annual incoming recruitment counts from a continuous multi-stage DDE simulation at a designated seasonal survey month (e.g., stage $M_1$ entering the fishable class).

`simulate_delay_difference(years, catches_biomass, recruits, cfg; b0, n0) -> DelayDifferenceResult`
Forward-simulates fishable biomass $B_t$ and numbers $N_t$ driven by external recruitment trajectories:
```math
N_{t+1} = s (N_t - C_t) + R_{t+1}
```
```math
B_{t+1} = s \alpha (N_t - C_t) + s \rho (B_t - Y_t) + w_k R_{t+1}
```

### 5.3 Bayesian Delay Differential & Delay Difference Models

`bayesian_delay_differential_model(survey_times, survey_data, cfg, tspan; dt=0.25, ...)`
Bayesian continuous multi-stage DDE model in `Turing.jl`:
- Integrates continuous size-structured stages ($M_0$ through $M_4$ and $f_{\text{mat}}$) with developmental delays.
- Samples carrying capacity multiplier $\kappa$, maternal recruitment scale $\beta$, and observation error $\sigma_o$.
- Evaluates lognormal likelihoods against survey observations.

`bayesian_delay_difference_model(catches_biomass, recruits, obs_survey, cfg; obs_cv=0.20)`
Bayesian state-space Deriso-Schnute delay-difference model in `Turing.jl`:
- Samples initial biomass $B_0$, annual natural survival $s$, survey catchability $q$, and process error $\sigma_p$.
- Propagates latent biomass $B_t$ and numbers $N_t$ driven by external recruitment $R_t$.
- Evaluates survey lognormal observation likelihood.

---

## 6. Empirical Submodule

Standalone non-parametric and linear regression diagnostics.

### 6.1 Catch Curves and Chapman-Robson

`chapman_robson(ages, catches; recruitment_age=nothing) -> ChapmanRobsonResult`
Computes the Chapman and Robson (1960) minimum-variance unbiased estimator (MVUE) of annual survival $S$ and total mortality $Z = -\log(S)$:
```math
S = \frac{\bar{T}}{1 + \bar{T} - 1/n}, \quad \widehat{\mathrm{Var}}(S) = S \left( S - \frac{\bar{T} - 1}{n + \bar{T} - 1} \right)
```
where $\bar{T}$ is the coded mean age above recruitment age $a_0$. Standard error of $Z$ is derived via the delta method: $\mathrm{SE}(Z) \approx \mathrm{SE}(S) / S$.

`catch_curve_linear(ages, catches; recruitment_age=nothing, weighted=true) -> CatchCurveResult`
Fits linear regression $\log(C_a) = \alpha - Z \cdot a$. When `weighted=true`, employs Maceina & Bettoli (1998) weights $w_a = C_a / \sum C$.

`bayesian_catch_curve(ages, catches; recruitment_age=nothing, max_age=nothing)`
Bayesian catch curve regression in `Turing.jl`:
- Samples total instantaneous mortality $Z$, log-intercept $\alpha$, and residual dispersion $\sigma$.
- Observes descending limb $\log(C_a) \sim \mathrm{Normal}(\alpha - Z(a - a_0), \sigma)$.

### 6.2 Depletion Estimators

`leslie_depletion(catch_per_effort, cumulative_catch) -> LeslieResult`
Fits Leslie & Davis (1939) model:
```math
\mathrm{CPUE}_t = q N_0 - q K_t
```
where $K_t$ is cumulative catch prior to period $t$, yielding initial abundance $N_0$ and catchability $q$.

`delury_depletion(catch_per_effort, cumulative_effort) -> DeLuryResult`
Fits DeLury (1947) model:
```math
\log(\mathrm{CPUE}_t) = \log(q N_0) - q E_t
```
where $E_t$ is cumulative fishing effort prior to period $t$.

`bayesian_leslie_depletion(catch_seq, effort_seq; midpoint_adjustment=true, ...)`
Bayesian Leslie depletion in `Turing.jl` sampling initial abundance $N_0$, catchability scalar $q$, and dispersion $\sigma$.

`bayesian_delury_depletion(catch_seq, effort_seq; ...)`
Bayesian DeLury depletion in `Turing.jl` sampling $N_0$, $q$, and dispersion $\sigma$.

### 6.3 Age-Length Keys

`compute_alk_matrix(sample_lengths, sample_ages, length_bins, age_bins) -> AlkResult`
Constructs the conditional probability matrix $P(a \mid l)$ from aged subsamples:
```math
P(a \mid l) = \frac{n_{l,a}}{\sum_{a'} n_{l,a'}}
```

`apply_age_length_key(alk_res, total_length_frequencies) -> Vector{Float64}`
Expands bulk length-frequency samples into total estimated age composition numbers:
```math
N_a = \sum_l N_l \cdot P(a \mid l)
```

---

## 7. Management Submodule

### 7.1 Harvest Control Rules

`HockeyStickHCR(blim, bpa, flow, ftarget)`
Defines a precautionary piecewise-linear harvest control rule:
```math
F(B) = \begin{cases}
F_{\text{low}}, & B \le B_{\text{lim}} \\
F_{\text{low}} + (F_{\text{target}} - F_{\text{low}}) \frac{B - B_{\text{lim}}}{B_{\text{pa}} - B_{\text{lim}}}, & B_{\text{lim}} < B < B_{\text{pa}} \\
F_{\text{target}}, & B \ge B_{\text{pa}}
\end{cases}
```

`eval_hcr(hcr::HockeyStickHCR, biomass::Real) -> Float64`
Evaluates the recommended target fishing mortality rate for observed biomass $B$.

### 7.2 Quota Stabilization

`apply_tac_stabilizer(target_tac, previous_tac; max_decrease=0.15, max_increase=0.15) -> Float64`
Applies asymmetric or symmetric change bounds to prevent interannual TAC volatility:
```math
\mathrm{TAC}_{\text{rec}} = \max\left( \mathrm{TAC}_{t-1} (1 - \Delta_{\text{dec}}), \min\left( \mathrm{TAC}_{t-1} (1 + \Delta_{\text{inc}}), \mathrm{TAC}^* \right) \right)
```

### 7.3 Management Strategy Evaluation (MSE)

`simulate_mse(years, hcr; r=0.35, k=100.0, b0=80.0, obs_cv=0.15, proc_cv=0.05, tac_constraint=0.15) -> MseResult`
Simulates closed-loop feedback management incorporating operating model stochastic dynamics, observation assessment errors, HCR evaluations, and TAC constraint implementations.

`bayesian_mse_feedback(observed_biomass, catches, hcr; prior_r, prior_k, obs_cv)`
Bayesian state-space feedback model in `Turing.jl` sampling biological capacity $r, K$, process error, and observation likelihood given HCR actions.

---

## 8. Retrospective Analysis & Mohn's Rho Diagnostics

`mohn_rho(terminal_peeled, terminal_base) -> Float64`
Computes Mohn's rho (ρ) across retrospective truncation peels:
```math
\rho = \frac{1}{P} \sum_{p=1}^P \frac{\hat{X}_{T-p, p} - \hat{X}_{T-p, 0}}{\hat{X}_{T-p, 0}}
```

`retrospective_analysis(years, removals; n_peels=5, ...) -> RetrospectiveResult`
Automates retrospective peeling for discrete surplus production models.

`retrospective_analysis(catches, survey_index, M; n_peels=4, ...) -> RetrospectiveResult`
Automates retrospective peeling for ADAPT Virtual Population Analysis.

---

## 9. Visualizations

Terminal and notebook plotting built on `UnicodePlots`:
- `plot_trajectory(times, values; title, xlabel, ylabel, color)`: Single time-series plot.
- `plot_trajectory!(plot_obj, times, values; name, color)`: Multi-series overlay.
- `plot_state_variables(model_result)`: Dispatches automated diagnostic visualization for `LogisticDiscreteResult`, `VpaResult`, `AdaptResult`, `DelayDifferenceResult`, and `MseResult`.
- `plot_retrospective(retro_res; target=:biomass)`: Renders retrospective peel trajectories with Mohn's $\rho$.
- `plot_fishery_predictions(times, sims; obs_time, obs_values, title, xlabel, ylabel, show_samples)`: Replicates `model_fishery` ensemble forward projection traces displaying mean trajectory, sample realizations, and observed survey data points.
- `plot_fishery_footprint(times, b_fished, b_unfished; title, xlabel, ylabel)`: Calculates and visualizes the relative fishery depletion footprint $\frac{B_t^{\text{unfished}} - B_t^{\text{fished}}}{B_t^{\text{unfished}}} \times 100\%$.

---

## 10. Input Data Schema and Expectations

Datasets must follow the native modular directory layout:

```
data/
├── silverhake/           # Age-structured catch, survey, and age comps (CSV/TOML)
├── cusk/                 # 4-fleet time-varying catchability gadoid dataset (CSV/TOML)
└── snowcrab/             # Multi-area biomass and size-structured survey data
    ├── biodyn_biomass.csv
    ├── biodyn_biomass.parquet
    ├── biodyn_number_size_struct.csv
    ├── biodyn_number_size_struct.parquet
    └── snowcrab.zarr/
```

All data loaders strictly ingest standard CSV, TOML, Parquet, and Zarr formats without binary dependencies.
