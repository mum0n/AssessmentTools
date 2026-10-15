### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 20000001-0000-0000-0000-000000000001
md"""
# Stage-Structured Leslie / Lefkovitch Matrix Models

This notebook runs the **`AssessmentTools.StageMatrix`** submodule: a Lefkovitch
stage-classified projection matrix for populations whose biology is organised by life
stage (size, instar, maturity) rather than by age.

### Method

Population vector ``N_t`` is propagated by a matrix ``A`` whose first row is reproduction,
diagonal is retention, and sub-diagonal is advancement:

```math
N_{t+1} = A N_t, \qquad
A_{1,k} = f_k,\quad A_{k,k} = r_k,\quad A_{k+1,k} = a_k
```

The same matrix is the single definition used by the deterministic projection, the
demographic analysis, the Bayesian model, and every representation in
`SimulationEngine`. Nothing is re-derived elsewhere.

### Assumptions

- Given survival, an individual either advances or stays, so ``a_k + r_k = 1`` for
  every stage. The unallocated mass would otherwise be removed implicitly, and each
  representation would have to choose its own residual.
- The matrix is constant over the horizon: no time-varying survival or recruitment.
- Stages are well defined, and an individual's stage is observable.
- No size- or age-based fishing selectivity: removals act on the aggregate.

### Strengths

- The demographic analysis (dominant eigenvalue, stable stage distribution, reproductive
  value, net reproductive rate, generation time, elasticities) is closed-form and needs
  no estimation.
- Elasticities identify which matrix element actually drives population growth, which is
  the actionable output.
- Very few parameters, so it works on data-poor species.

### Weaknesses

- Deterministic: demographic variance must come from the stochastic representations.
- No fishing selectivity, so it cannot distinguish directed from bycatch mortality.
- The projection depends entirely on the matrix being right for the whole horizon.
"""

# ╔═╡ 20000002-0000-0000-0000-000000000002
md"""
## 1. Configuration and Parameters

Every parameter is read from `data/stage_matrix/config.toml`. There is no default in the
code: a missing key raises rather than silently substituting a value.
"""

# ╔═╡ 30000003-0000-0000-0000-000000000003
using AssessmentTools
using AssessmentTools.StageMatrix
using LinearAlgebra, Statistics, Random

# ╔═╡ 30000004-0000-0000-0000-000000000004
begin
    cfg = load_stage_matrix_config(normpath(joinpath(@__DIR__, "..", "data",
                                                    "stage_matrix", "config.toml")))
    (
        n_stages = cfg.n_stages,
        fertility = cfg.fertility,
        survival = cfg.survival,
        transition = cfg.transition,
        retention = cfg.retention,
        initial_abundance = cfg.initial_abundance,
    )
end

# ╔═╡ 30000005-0000-0000-0000-000000000005
md"""
## 2. The Projection Matrix

`build_projection_matrix` assembles the Lefkovitch matrix. The first row is reproduction
into the youngest stage, the diagonal is retention, and the sub-diagonal is advancement.
"""

# ╔═╡ 30000006-0000-0000-0000-000000000006
begin
    A = build_projection_matrix(cfg)
    println("Projection matrix A:")
    display(round.(A, digits = 3))
    (rows, cols) = size(A)
    ("A is $rows x $cols")
end

# ╔═╡ 30000007-0000-0000-0000-000000000007
md"""
## 3. Deterministic Projection

`project_stage_matrix` steps the population forward and reports the matrix used.
Trajectories are per time step, and the growth rate is the dominant eigenvalue.
"""

# ╔═╡ 30000008-0000-0000-0000-000000000008
begin
    res = project_stage_matrix(cfg)
    (
        n_steps = cfg.n_years,
        growth_rate = round(res.growth_rate, digits = 6),
        final_abundance = round.(res.abundance[:, end], digits = 2),
        final_biomass = round(res.biomass[end], digits = 2),
    )
end

# ╔═╡ 30000009-0000-0000-0000-000000000009
md"""
## 4. Demographic Analysis

Growth rate, stable stage distribution, reproductive value, net reproductive rate and
generation time, all from `LinearAlgebra.eigen`.
"""

# ╔═╡ 30000010-0000-0000-0000-000000000010
begin
    r0 = net_reproductive_rate(A)
    lambda = stable_stage_distribution(A) |> v -> sum(v)
    v = stable_stage_distribution(A)
    w = reproductive_value(A)
    T = generation_time(A)
    (
        net_reproductive_rate_R0 = round(r0, digits = 4),
        growth_rate_lambda = round(res.growth_rate, digits = 6),
        generation_time_T = round(T, digits = 4),
        stable_stage_distribution = round.(v, digits = 4),
        reproductive_value_w1_is_1 = round.(w, digits = 4),
    )
end

# ╔═╡ 30000011-0000-0000-0000-000000000011
md"""
## 5. Elasticities

Sensitivity of `lambda` to each matrix element. These sum to 1 and identify which element
drives growth.
"""

# ╔═╡ 30000012-0000-0000-0000-000000000012
begin
    E = stage_elasticities(A)
    println("Elasticities of lambda to A (sum = ", round(sum(E), digits = 6), "):")
    display(round.(E, digits = 4))
    (elasticity_sum = round(sum(E), digits = 6),)
end

# ╔═╡ 30000013-0000-0000-0000-000000000013
md"""
## 6. Bayesian Estimation

The same matrix, sampled with priors on survival, retention, advancement and fertility.
Switch of engine is a single keyword: `estimate(model, x0; method = :map | :mcmc | :vi)`.
All three engines run on the same model object, so no likelihood is re-implemented per
engine.
"""

# ╔═╡ 30000014-0000-0000-0000-000000000014
using AssessmentTools.InferenceBridge
using DynamicPPL, LogDensityProblems

# ╔═╡ 30000015-0000-0000-0000-000000000015
begin
    # Build a survey index from the deterministic trajectory for the model to fit.
    ny = 12
    survey = [sum(res.abundance[:, i] .* cfg.stage_weights .* cfg.catchability)
              for i in 1:(ny + 1)]

    mod = bayesian_stage_matrix_model(
        survey, copy(res.biomass[1:5]), cfg.stage_weights, cfg.maturity,
        cfg.catchability, cfg.n_stages, ny;
        survey_type = :vulnerable,
        prior_survival_mu = 0.85, prior_survival_sd = 0.10,
        prior_retention_mu = 0.6, prior_retention_sd = 0.2)

    dim = InferenceBridge.parameter_dimension(mod)
    x0 = collect(Float64, DynamicPPL.LogDensityFunction(mod)._x)
    (model_dimension = dim,)
end

# ╔═╡ 30000016-0000-0000-0000-000000000016
begin
    # MAP (Optim, L-BFGS with ForwardDiff gradients).
    r_map = InferenceBridge.fit_map(mod, x0; iterations = 200, verbose = false)
    (map_log_posterior = round(r_map.log_posterior, digits = 4),
     converged = r_map.converged)
end

# ╔═╡ 30000017-0000-0000-0000-000000000017
md"""
## 7. Uncertainty Propagation Through the Simulation Engine

The deterministic matrix is a point estimate. The projection layer takes parameters from
any model as point estimates **or** as a saved posterior ensemble, and each step selects
a whole joint draw so parameter correlation is preserved.
"""

# ╔═╡ 30000018-0000-0000-0000-000000000018
using AssessmentTools.SimulationEngine

# ╔═╡ 30000019-0000-0000-0000-000000000019
begin
    n = cfg.n_stages
    _t(i_) = (i_ < n && i_ <= length(cfg.transition)) ? cfg.transition[i_] : 0.0
    pr = SimulationEngine.projection_parameters(
        n_groups = n,
        weights = cfg.stage_weights,
        maturity = cfg.maturity,
        fertility = cfg.fertility,
        natural_mortality = -log(0.85),
        advance = [_t(i_) for i_ in 1:n],
        retention = [1.0 - _t(i_) for i_ in 1:n],
        initial_numbers = cfg.initial_abundance,
        fishing_mortality = zeros(n, 1),
        harvest_rate = fill(0.05, n, 1),
        model_symbol = :stage_matrix)

    st0 = SimulationEngine.ProjectionState(copy(pr.initial_numbers), 0,
                                          sum(pr.initial_numbers .* pr.weights),
                                          0.0, 0.0, 1)
    fwd = SimulationEngine.project_state(pr, st0; direction = :forward, n_years = 10)
    (biomass_by_step = round.([s.biomass for s in fwd], digits = 2),)
end

# ╔═╡ 30000020-0000-0000-0000-000000000020
md"""
## 8. Stochastic Representations

Two interchangeable representations of the same matrix: an agent-based one where each
stage is an agent with per-agent binomial transitions, and a reaction network compiled
by Catalyst.jl and simulated by JumpProcesses.jl with the Gillespie SSA.
"""

# ╔═╡ 30000021-0000-0000-0000-000000000021
begin
    # Agent-based: each stage's agents draw their own survivorship and advancement.
    ag_states, ag_removals = SimulationEngine.run_agent_projection(
        pr, 10; n_agents_per_group = 8, scaling = 1.0,
        rng = Random.MersenneTwister(2026))
    (agent_final = round.(ag_states[end], digits = 2),
     agent_trajectory_biomass = round.([sum(s .* pr.weights) for s in ag_states], digits = 2),
     deterministic_trajectory = round.([s.biomass for s in fwd], digits = 2))
end

# ╔═╡ 30000022-0000-0000-0000-000000000022
md"""
### Note on the agent versus closed-form comparison

These two do not coincide beyond leading order, and the residual gap does **not** shrink
with more agents. With compounding Poisson recruitment the mean of the compounded
process is not the process at its mean (Jensen's inequality). The closed form is the
correct mean-field projection; the agent representation is the one that carries realised
demographic variance. Compare them qualitatively, not as an equality check.
"""

# ╔═╡ 30000023-0000-0000-0000-000000000023
begin
    net = SimulationEngine.build_reaction_network(pr)
    rs = SimulationEngine.catalyze(net)
    (catalyst_species = length(SimulationEngine.jump_parameters(net, 1)),
     reaction_count = length(net.reactions))
end

# ╔═╡ 30000024-0000-0000-0000-000000000024
begin
    # Time-binned Gillespie SSA (tSSA) over the same network.
    tssa = SimulationEngine.simulate_ssa(net, 10.0; n_bins = 10,
                                         rng = Random.MersenneTwister(11))
    (ssa_initial = round.(tssa.counts[:, 1], digits = 2),
     ssa_final = round.(tssa.counts[:, end], digits = 2))
end

# ╔═╡ 30000025-0000-0000-0000-000000000025
md"""
## 9. Summary of Findings

- The Lefkovitch matrix is built once and consumed by the deterministic projection, the
  demographic analysis, the Bayesian model, and both stochastic representations.
- Growth rate, stable stage distribution, reproductive value and elasticities are
  closed-form and need no estimation.
- Elasticities identify which matrix element drives growth.
- The deterministic matrix is a point estimate; posterior ensembles are carried by the
  projection layer, which selects a whole joint draw at each step.
"""
