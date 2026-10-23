# StockSynthesis.jl — Size-Structured Stock Assessment

## Overview

`StockSynthesis` is a size-structured extension of the age-structured `StatisticalCatchAtAge`
framework. It follows the Stock Synthesis (SS3) paradigm (Methot & Wetzel 2013) but
implemented natively in Julia with multiple likelihoods, size compositions, and
multi-series data integration.

## Mathematical Underpinnings

### 1. Size-Structured Population Dynamics

The population is divided into `n` size (length) classes. The state vector `N_t` (length `n`)
represents the number of individuals in each size class at time `t`.

**Growth (von Bertalanffy):**
```
L(a) = L_inf * (1 - exp(-k * (a - t0)))
```
where `L_inf` is asymptotic length, `k` is growth rate, `t0` is theoretical age at zero length.

**Size transition matrix G** (`n x n`): `G[i,j]` = probability that an individual in
size class `j` grows to size class `i` in one time step. Computed from the von
Bertalanffy growth curve with a normal distribution around the expected growth increment.

**Natural mortality:** Can be size-dependent (`m_size[i]`) or constant (`m`).

**Population dynamics:**
```
N_{t+1} = G * (N_t * exp(-M)) - C_t
```
where `M` is natural mortality, `C_t` is catch by size class.

### 2. Selectivity and Maturity

Size-based selectivity functions for both catch and survey, with sex-specific support:
- **Catch selectivity**: `sel_params` (n_fleets × n_sel_params) — controls fishing mortality by size
- **Survey selectivity**: `survey_sel_params` (n_surveys × n_sexes × n_sel_params) — sex-specific survey catchability by size
- **Maturity oogives**: `mat_at_size` (n_sexes × n_size) — sex-specific maturity schedules
- **Weight at size**: `w_at_size` (n_sexes × n_size) — sex-specific weight
- **Fecundity at size**: `fec_at_size` (n_sexes × n_size) — sex-specific fecundity
- **Natural mortality**: `m_size` (n_sexes × n_size) — sex-specific mortality
- **Logistic:** `sel(L) = 1 / (1 + exp(-slope * (L - a50)))`
- **Knife-edge:** `sel(L) = 1 if L >= min_size else 0`
- **Double-normal:** dome-shaped selectivity with ascending and descending limbs

### 3. Multiple Likelihoods

The framework supports multiple observation series, each with its own likelihood:

| Likelihood | Use Case | Formula |
|---|---|---|
| Normal | Continuous indices | `0.5 * sum((obs - pred)^2) / sigma^2` |
| Lognormal | Positive continuous (catch, CPUE) | `0.5 * sum((log(obs) - log(pred))^2)` |
| Multinomial | Size/age compositions | `-sum(n_i * log(p_i))` |
| Dirichlet-multinomial | Overdispersed compositions | Adjusted for process error |
| Poisson | Count data | `sum(pred - obs * log(pred))` |

### 4. Size Composition Handling

Individual fish lengths are binned into size classes. The framework handles:
- Raw individual measurements → binned counts
- Pre-binned counts → proportions
- Effective sample size adjustment (HCAM method): `N_eff = N / (1 + proc_err * N)`
- Multiple composition likelihoods (multinomial, Dirichlet-multinomial, lognormal)

### 5. Multi-Series Integration (Turing-Native)

Multiple observation series with different likelihood types are combined into a single
Turing model (`bayesian_multi_series`). This model can be used with MCMC, VI, MAP,
or ML via InferenceBridge:

```julia
obs_series = [[10.0, 12.0, 15.0], [50.0, 55.0, 60.0], counts]
pred_series = [[9.5, 11.5, 14.0], [48.0, 53.0, 58.0], pred_counts]
types = [:lognormal, :lognormal, :multinomial]

model = bayesian_multi_series(obs_series, pred_series, types)
# MAP: fit_map(model, x0)
# MCMC: sample(model, NUTS(), 1000)
# VI: fit_vi(model)
```

An imperative `MultiSeriesLikelihood` container is also available for quick NLL
computation without a full Turing model.

### 6. MSY Reference Points

For size-structured models, MSY is computed via equilibrium analysis:
- At equilibrium: `B_{t+1} = B_t` (steady state)
- MSY: maximum of `Y = F * B` over F
- Fmsy: F that maximizes yield
- Bmsy: biomass at Fmsy
- Umsy: exploitation rate at MSY = MSY / Bmsy

### 7. Bayesian Model

The Bayesian size-structured model uses Turing.jl with:
- State-space formulation on biomass (Schaefer-like)
- Multiple likelihoods for catch, survey, and size composition
- Priors on r, K, q, process error, observation error
- Truncated normals for bounded parameters

## Age-Structured Variant

An age-based variation (`bayesian_age_synthesis`) is also available, using
numbers-at-age dynamics with the Baranov catch equation:

- `N_{t+1, a+1} = N_{t, a} * exp(-Z_{t, a})` (ageing)
- `N_{t+1, 1} = R_t` (recruitment)
- `Z_{t, a} = M + F_{t, a}` (total mortality)
- `C_{t, a} = F_{t, a} / Z_{t, a} * (1 - exp(-Z_{t, a})) * N_{t, a}` (Baranov)

Multiple likelihoods:
- Catch: lognormal on total catch
- Survey: lognormal on survey index
- Age composition: multinomial on age frequencies

Usage:
```julia
ac = AgeCompositionData([10, 20, 15, 5]; ages = [1, 2, 3, 4])
model = bayesian_age_synthesis(obs_catch, obs_indices, [ac])
# MAP: fit_age_synthesis(obs_catch, obs_indices, [ac]; method = :map)
# MCMC: sample_age_synthesis(obs_catch, obs_indices, [ac])
# VI: vi_age_synthesis(obs_catch, obs_indices, [ac])
```

1. **von Bertalanffy growth** is used for size transition. Alternative growth models (Gompertz, logistic) are not implemented.
2. **Size classes are uniform** in width. Irregular binning is handled by binning individual measurements.
3. **Natural mortality** can be size-dependent but defaults to constant.
4. **Selectivity** is size-based; age-based selectivity is handled by the StatisticalCatchAtAge module.
5. **Single species** — no multi-species interactions.
6. **Equilibrium recruitment** — MSY computations assume steady-state conditions.
7. **Size composition** assumes random sampling of the catch/survey.
8. **Sex-specific support**: survey selectivity, maturity oogives, weight, fecundity, and natural mortality can all be specified per sex via `n_sexes`. Defaults to single sex (n_sexes=1) for backward compatibility.

## Workflow

1. **Define size classes**: `make_size_classes(min, max, n)`
2. **Set growth parameters**: `GrowthParameters(linf, k, t0, cv)`
3. **Build model**: `SizeStructuredModel(size_classes, growth, n_years, n_fleets, n_surveys)`
4. **Prepare data**: Bin individual lengths into `SizeCompositionData`
5. **Fit (MAP/ML)**: `fit_size_structured(obs_catch, obs_indices, size_comps; method = :map)`
6. **Fit (MCMC)**: `fit_size_structured(obs_catch, obs_indices, size_comps; method = :mcmc)`
7. **Fit (VI)**: `fit_size_structured(obs_catch, obs_indices, size_comps; method = :vi)`
8. **Or use Turing directly**: `bayesian_size_structured(obs_catch, obs_indices, size_comps)` → `sample`, `optimize`, `vi`
9. **Reference points**: `size_msy_reference_points(model)`
10. **Bayesian**: `bayesian_size_structured(obs_catch, obs_indices, size_comps)`

## References

- Methot, R. D. & Wetzel, C. R. (2013). Stock Synthesis: A biological and statistical framework for fish stock assessment and fishery management. *Fisheries Research*, 142, 86-99.
- von Bertalanffy, L. (1938). A quantitative theory of organic growth. *Human Biology*, 10, 181-213.
- Fournier, D. A. et al. (1998). MULTIFAN-CL: A length-based, age-structured model for fisheries stock assessment. *Canadian Journal of Fisheries and Aquatic Sciences*, 55, 2105-2116.
- Haddon, M. (2011). *Modelling and Quantitative Methods in Fisheries* (2nd ed.). Chapman and Hall/CRC.
- Quinn, T. J. & Deriso, R. B. (1999). *Quantitative Fish Dynamics*. Oxford University Press.
- Maunder, M. N. & Punt, A. E. (2013). A review of integrated analysis in fisheries stock assessment. *Fisheries Research*, 142, 61-74.
- Thorson, J. T. & Minto, C. (2015). Mixed-effects models for fish growth. *Canadian Journal of Fisheries and Aquatic Sciences*, 72, 1-13.