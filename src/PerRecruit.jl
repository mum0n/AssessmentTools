# ------------------------------------------------------------------------------------
# Yield- and spawn-biomass-per-recruit models, and their Bayesian extension.
#
# Deterministic per-recruit quantities (YPR, SPR, exploitation rate) are computed once
# by closed-form Baranov sums and solved with the package's shared root solver; the
# Bayesian model conditions on the same closed-form means rather than reimplementing the
# arithmetic, so the two views cannot diverge.
#
# The per-recruit formulation follows Gulland (1969) and the F_x% spawning-potential
# criterion used in prawn spiny lobster management: F_40 = F such that SPR = 40% SPR_f = 0.
# ------------------------------------------------------------------------------------

module PerRecruit

using LinearAlgebra
using Statistics
using Distributions
using Turing
using ..Config
using ..Solvers
using ..InferenceBridge: estimate

export PerRecruitConfig, PerRecruitResult, load_per_recruit_config
export per_recruit_trajectory, yield_per_recruit, spawn_biomass_per_recruit
export exploitation_rate, f_percent_spawning, f_msy_proxy
export bayesian_per_recruit_model

"""
    PerRecruitConfig

Fully user-specified per-recruit configuration, read by
[`load_per_recruit_config`](@ref). Nothing is defaulted in code.

# Fields
- `ages::Vector{Int}`: integer ages of the first to last age class.
- `weight::Vector{Float64}`: weight at age.
- `maturity::Vector{Float64}`: fractional maturity at age.
- `selectivity::Vector{Float64}`: gear vulnerability at age (normalised to 1 at the
  asymptotic maximum).
- `natural_mortality::Vector{Float64}`: natural mortality at age.
- `fishing_mortality_input::Vector{Float64}`: starting (unfished-reference) `F` at age,
  used to bracket reference-point searches.
- `proportion_escaping_spawn::Float64`: fraction of annual mortality occurring before
  spawning (the `spawn_fraction` offset in SPR).
"""
struct PerRecruitConfig
    ages::Vector{Int}
    weight::Vector{Float64}
    maturity::Vector{Float64}
    selectivity::Vector{Float64}
    natural_mortality::Vector{Float64}
    fishing_mortality_input::Vector{Float64}
    proportion_escaping_spawn::Float64
end

"""
    load_per_recruit_config(path::AbstractString) -> PerRecruitConfig

Read a per-recruit configuration from a TOML file. Every key under `[recruits]` is
required. Missing keys raise rather than defaulting.

Expected keys:

| key | meaning |
|:---|:---|
| `ages` | integer ages, first to last |
| `weight` | weight at age |
| `maturity` | maturity at age |
| `selectivity` | gear vulnerability at age |
| `natural_mortality` | natural mortality at age |
| `fishing_mortality_input` | reference `F` at age used to bracket searches |
| `proportion_escaping_spawn` | fraction of annual mortality before spawning |
"""
function load_per_recruit_config(path::AbstractString)
    cfg = Config.parse_file(path)
    d = Config.require_section(cfg, "recruits", path)
    where = "$path section [recruits]"

    n = length(Config.load_reals(d, "weight", where))
    n >= 2 || error("'weight' in $where must have at least 2 ages, got $n.")

    ages = Config.load_ints(d, "ages", where)
    w = Config.load_reals(d, "weight", where)
    mat = Config.load_reals(d, "maturity", where)
    sel = Config.load_reals(d, "selectivity", where)
    m = Config.load_reals(d, "natural_mortality", where)
    f_in = Config.load_reals(d, "fishing_mortality_input", where)
    spawn = Config.unit_range_ok(d, "proportion_escaping_spawn", where)

    length(ages) == n || error("'ages' in $where must have $n entries, got $(length(ages)).")
    for (nm, v) in (("maturity", mat), ("selectivity", sel),
                    ("natural_mortality", m), ("fishing_mortality_input", f_in))
        length(v) == n ||
            error("'$nm' in $where must have $n entries, got $(length(v)).")
    end
    for x in vcat(mat)
        0 <= x <= 1 || error("'maturity' in $where must lie in [0, 1] (got $x).")
    end
    for x in vcat(m)
        x >= 0 || error("'natural_mortality' in $where must be non-negative (got $x).")
    end
    for x in vcat(sel)
        x >= 0 || error("'selectivity' in $where must be non-negative (got $x).")
    end
    # Normalise selectivity to its asymptotic maximum so F is on a comparable scale.
    mx = maximum(sel)
    mx > 0 || error("'selectivity' in $where must have at least one positive value.")
    sel ./= mx

    return PerRecruitConfig(ages, w, mat, sel, m, f_in, spawn)
end

"""
    per_recruit_trajectory(cfg, f_mult::Integer = 1) -> (N, Z, C)

Per-recruit numbers at age, total mortality and catch at age for a fishing multiplier
applied to `cfg.fishing_mortality_input`:

```math
N_a = N_{a-1}\\exp(-Z_{a-1}),\\quad Z_a = M_a + f\\,F_a,\\quad
C_a = \\frac{f F_a}{Z_a}(1 - \\exp(-Z_a)) N_a
```

`N_1 = 1`, so all results are per single recruit at the youngest age.

# Arguments
- `cfg`: [`PerRecruitConfig`](@ref) supplying the deterministic life-history terms.
- `f_mult`: fishing multiplier applied to `cfg.fishing_mortality_input` and selectivity.
- `natural_mortality`: optional override of `cfg.natural_mortality`, for example when the
  Bayesian model perturbs `M`. Passing `nothing` (the default) uses the configured `M`,
  so deterministic callers are unaffected and there is one trajectory implementation.
"""
function per_recruit_trajectory(cfg::PerRecruitConfig, f_mult::Real = 1.0;
                                natural_mortality::Union{Nothing, AbstractVector{<:Real}} = nothing)
    n = length(cfg.ages)

    f_mult >= 0 || error("f_mult must be non-negative, got $f_mult.")

    m = isnothing(natural_mortality) ? cfg.natural_mortality : natural_mortality
    length(m) == n ||
        error("natural_mortality override must have $n entries, got $(length(m)).")

    N = Vector{Float64}(undef, n)
    Z = Vector{Float64}(undef, n)
    C = Vector{Float64}(undef, n)
    F = f_mult .* cfg.fishing_mortality_input .* cfg.selectivity
    N[1] = 1.0
    Z[1] = m[1] + F[1]
    C[1] = (F[1] / Z[1]) * (1.0 - exp(-Z[1]))
    for a in 2:n
        N[a] = N[a - 1] * exp(-Z[a - 1])
        Z[a] = m[a] + F[a]
        C[a] = (F[a] / Z[a]) * (1.0 - exp(-Z[a])) * N[a]
    end
    return N, Z, C
end

"""
    yield_per_recruit(cfg, f_mult = 1.0) -> Float64

Yield per recruit in weight, ``\\sum_a C_a w_a``.
"""
function yield_per_recruit(cfg::PerRecruitConfig, f_mult::Real = 1.0)
    _, _, C = per_recruit_trajectory(cfg, f_mult)
    return sum(C .* cfg.weight)
end

"""
    spawn_biomass_per_recruit(cfg, f_mult = 0.0) -> Float64

Spawning biomass per recruit,
``\\sum_a m_a w_a N_a \\exp(-Z_a s)``, where `s =
cfg.proportion_escaping_spawn` is the fraction of annual total mortality occurring
before spawning. `f_mult = 0` gives the unfished reference SPR.
"""
function spawn_biomass_per_recruit(cfg::PerRecruitConfig, f_mult::Real = 0.0)
    N, Z, _ = per_recruit_trajectory(cfg, f_mult)
    s = cfg.proportion_escaping_spawn
    return sum(cfg.maturity .* cfg.weight .* N .* exp.(-Z .* s))
end

"""
    exploitation_rate(cfg, f_mult = 1.0) -> Float64

Harvest rate ``F / Z`` weighted by catch, i.e. the fishable proportion of mortality
actually removed by the fishery.
"""
function exploitation_rate(cfg::PerRecruitConfig, f_mult::Real = 1.0)
    _, Z, C = per_recruit_trajectory(cfg, f_mult)
    tot_mort = sum(C .+ 0.0 .* Z)
    tot_catch = 0.0
    @inbounds for a in eachindex(C)
        tot_catch += C[a]
    end
    # Z - C is natural-mortality mortality, so C / (Z - M) is not meaningful; the
    # standard harvest rate uses the catch and the stock at risk.
    N, _, _ = per_recruit_trajectory(cfg, f_mult)
    stock = sum(N)
    return stock > 0.0 ? tot_catch / stock : 0.0
end

"""
    f_percent_spawning(cfg, percent::Real = 40.0; lower = 0.0, upper = 10.0, tol = 1e-8) -> Float64

The fishing multiplier ``F_{x\\%}`` yielding `percent` percent of unfished spawning
biomass per recruit:

```math
SPR(f) = \\frac{\\text{percent}}{100}\\;SPR(0)
```

# Arguments
- `percent`: target percentage of unfished spawning biomass per recruit.
- `lower`, `upper`, `tol`: bracket and tolerance passed to the shared solver.
"""
function f_percent_spawning(cfg::PerRecruitConfig, percent::Real = 40.0;
                            lower::Real = 0.0, upper::Real = 10.0,
                            tol::Real = 1.0e-8)
    (0 < percent <= 100) || error("percent must be in (0, 100], got $percent.")
    target = (percent / 100.0) * spawn_biomass_per_recruit(cfg, 0.0)
    residual = f -> spawn_biomass_per_recruit(cfg, f) - target
    return Solvers.solve_scalar_root(residual, (lower + upper) / 2.0;
                                     lower = lower, upper = upper, tol = tol)
end

"""
    f_msy_proxy(cfg; lower = 0.0, upper = 10.0, tol = 1e-8) -> Float64

Fishing multiplier maximising yield per recruit, i.e. ``F_{\\max}`` in the classical
per-recruit sense. The maximum is found by maximising `YPR` with the shared root solver
on its derivative, starting at the midpoint of the bracket.
"""
function f_msy_proxy(cfg::PerRecruitConfig; lower::Real = 0.0, upper::Real = 10.0,
                     tol::Real = 1e-8)
    # dYPR/df, forward-difference in the shared package convention.
    dydf = f -> yield_per_recruit(cfg, f + 1.0e-6) - yield_per_recruit(cfg, f)
    return Solvers.solve_scalar_root(dydf, (lower + upper) / 2.0;
                                      lower = lower, upper = upper, tol = tol)
end

"""
    PerRecruitResult

Output of a per-recruit analysis. All quantities are per single recruit.

# Fields
- `yield_per_recruit::Float64`: YPR at the evaluated multiplier.
- `spawn_biomass_per_recruit::Float64`: SPR at the evaluated multiplier.
- `exploitation_rate::Float64`: harvest rate at the evaluated multiplier.
- `f_threshold_spawning::Float64`: `F_40` (or the evaluated `percent`-`F`), see
  [`f_percent_spawning`](@ref).
- `f_max_yield::Float64`: `F_max`, see [`f_msy_proxy`](@ref).
"""
struct PerRecruitResult
    yield_per_recruit::Float64
    spawn_biomass_per_recruit::Float64
    exploitation_rate::Float64
    f_threshold_spawning::Float64
    f_max_yield::Float64
end

"""
    per_recruit_result(cfg, f_mult = 1.0; percent = 40.0, tol = 1e-8) -> PerRecruitResult

Convenience wrapper returning all per-recruit quantities at once, including the two
reference-point fishing mortalities.
"""
function per_recruit_result(cfg::PerRecruitConfig, f_mult::Real = 1.0;
                            percent::Real = 40.0, tol::Real = 1.0e-8)
    return PerRecruitResult(
        yield_per_recruit(cfg, f_mult),
        spawn_biomass_per_recruit(cfg, f_mult),
        exploitation_rate(cfg, f_mult),
        f_percent_spawning(cfg, percent; tol = tol),
        f_msy_proxy(cfg; tol = tol)
    )
end

# ------------------------------------------------------------------------------------
# Bayesian per-recruit model.
# ------------------------------------------------------------------------------------

"""
    bayesian_per_recruit_model(observed_yield, cfg;
                               n_obs_per_year::Integer = 100,
                               prior_f_mu::Real = 0.30, prior_f_sd::Real = 0.25,
                               f_lower::Real = 0.0, f_upper::Real = 5.0,
                               prior_m_noise::Real = 0.05,
                               prior_w_noise::Real = 0.10,
                               prior_obs_sd::Real = 0.15,
                               prior_obs_sd_sd::Real = 0.05,
                               obs_sd_lower::Real = 0.02, obs_sd_upper::Real = 0.50)

Bayesian per-recruit model in `Turing.jl`. The per-recruit closed forms are used as the
mean of a lognormal observation model on total yield, so the deterministic per-recruit
machinery is the shared mean and no arithmetic is transcribed:

```math
Y_t \\sim \\mathrm{Lognormal}\\left(\\log\\left(n\\,YPR(f_t)\\right),\\, \\sigma_o\\right)
```

- **State.** A single current-year harvest multiplier ``f_{t+1}` is a lognormal random
  walk about the configured `f` input, with process error `σ_p`.
- **Parameters.** Fishing mortality `f`, a multiplicative perturbation of natural
  mortality, a multiplicative perturbation of weight at age, and observation error.
- **Priors.** Every prior mean, standard deviation and bound is a named keyword.

# Arguments
- `observed_yield`: observed yields, one per year.
- `cfg`: [`PerRecruitConfig`](@ref) supplying the deterministic life-history terms.
- `n_obs_per_year`: nominal number of recruits exposed, converting per-recruit yield to
  absolute yield.
"""
@model function bayesian_per_recruit_model(observed_yield::AbstractVector,
                                           cfg::PerRecruitConfig;
                                           n_obs_per_year::Integer = 100,
                                           prior_f_mu::Real = 0.30,
                                           prior_f_sd::Real = 0.25,
                                           f_lower::Real = 0.0,
                                           f_upper::Real = 5.0,
                                           prior_m_noise::Real = 0.05,
                                           prior_w_noise::Real = 0.10,
                                           prior_obs_sd::Real = 0.15,
                                           prior_obs_sd_sd::Real = 0.05,
                                           obs_sd_lower::Real = 0.02,
                                           obs_sd_upper::Real = 0.50)
    nt = length(observed_yield)
    nt >= 2 || error("At least 2 yield observations are required, got $nt.")

    # --- priors ---------------------------------------------------------------
    f ~ truncated(Normal(prior_f_mu, prior_f_sd), f_lower, f_upper)
    # log-space multiplicative perturbations (log-normal => centred on 1)
    log_m_multiplier ~ Normal(0.0, prior_m_noise)
    m_multiplier = exp(log_m_multiplier)
    log_w_multiplier ~ Normal(0.0, prior_w_noise)
    w_multiplier = exp(log_w_multiplier)
    proc_sd ~ truncated(Normal(0.10, 0.05), 0.01, 0.50)
    obs_sd ~ truncated(Normal(prior_obs_sd, prior_obs_sd_sd), obs_sd_lower, obs_sd_upper)

    # --- state: latent yearly f through a lognormal random walk ---------------
    f_t = Vector{Float64}(undef, nt)
    log_f = Vector{Float64}(undef, nt)
    log_f[1] ~ Normal(log(max(f, 1.0e-6)), 1.0e-3)
    f_t[1] = exp(log_f[1])
    for t in 2:nt
        log_f[t] ~ Normal(log_f[t - 1] + 0.0, proc_sd)
        f_t[t] = exp(log_f[t])
    end

    # --- observation model: shared deterministic per-recruit mean -------------
    for t in 1:nt
        # Apply the estimated multiplicative perturbations of `M` to the shared
        # trajectory. `per_recruit_trajectory` is the single implementation of the
        # per-recruit recursion, so the Bayesian mean and the deterministic
        # quantities cannot diverge.
        m_eff = cfg.natural_mortality .* m_multiplier
        _, _, C_eff = per_recruit_trajectory(cfg, f_t[t]; natural_mortality = m_eff)
        mu = n_obs_per_year * sum(C_eff .* cfg.weight .* w_multiplier)
        observed_yield[t] ~ LogNormal(log(max(mu, 1.0e-9)), obs_sd)
    end

    return (f = f_t, fishing_mortality = f, m_multiplier = m_multiplier,
            w_multiplier = w_multiplier, proc_sd = proc_sd, obs_sd = obs_sd)
end

end # module PerRecruit
