# Precautionary Harvest Control Rules (HCR) and Feedback Simulation.

"""
Precautionary Harvest Control Rule parameters.

# Fields
- `blim::Float64`: Biomass limit threshold ``B_{\\mathrm{lim}}``. Below this level,
  fishing mortality is reduced to ``F_{\\mathrm{low}}`` or zero to avoid recruitment overfishing.
- `bpa::Float64`: Precautionary biomass threshold ``B_{\\mathrm{pa}}`` (or ``B_{\\mathrm{trigger}}``).
  Above this level, target fishing mortality ``F_{\\mathrm{target}}`` is maintained.
- `flow::Float64`: Residual or minimum fishing mortality ``F_{\\mathrm{low}} \\ge 0`` when ``B \\le B_{\\mathrm{lim}}``.
- `ftarget::Float64`: Target fishing mortality ``F_{\\mathrm{target}}`` (e.g. ``F_{\\mathrm{msy}}`` or ``F_{\\mathrm{pa}}``).
"""
struct HockeyStickHCR
    blim::Float64
    bpa::Float64
    flow::Float64
    ftarget::Float64

    function HockeyStickHCR(blim::Real, bpa::Real, flow::Real, ftarget::Real)
        blim < bpa || error("blim must be strictly less than bpa")
        flow >= 0.0 || error("flow must be non-negative")
        ftarget >= flow || error("ftarget must be greater than or equal to flow")
        new(Float64(blim), Float64(bpa), Float64(flow), Float64(ftarget))
    end
end

"""
    eval_hcr(hcr::HockeyStickHCR, ssb::Real) -> Float64

Evaluate a standard piecewise-linear hockey-stick harvest control rule at observed
or estimated spawning stock biomass ``B``.

# Mathematical Formulation
```math
F(B) = \\begin{cases}
F_{\\mathrm{low}} & B \\le B_{\\mathrm{lim}} \\\\
F_{\\mathrm{low}} + (F_{\\mathrm{target}} - F_{\\mathrm{low}}) \\frac{B - B_{\\mathrm{lim}}}{B_{\\mathrm{pa}} - B_{\\mathrm{lim}}} & B_{\\mathrm{lim}} < B < B_{\\mathrm{pa}} \\\\
F_{\\mathrm{target}} & B \\ge B_{\\mathrm{pa}}
\\end{cases}
```

# Inputs
- `hcr`: Configured `HockeyStickHCR` instance.
- `ssb`: Current spawning biomass or proxy abundance index.

# Outputs
Prescribed target fishing mortality rate ``F``.
"""
function eval_hcr(hcr::HockeyStickHCR, ssb::Real)
    b = Float64(ssb)
    if b <= hcr.blim
        return hcr.flow
    elseif b >= hcr.bpa
        return hcr.ftarget
    else
        slope = (hcr.ftarget - hcr.flow) / (hcr.bpa - hcr.blim)
        return hcr.flow + slope * (b - hcr.blim)
    end
end

"""
    apply_tac_stabilizer(tac_target::Real, tac_prev::Real;
                         max_decrease::Real = 0.15, max_increase::Real = 0.15,
                         biomass::Union{Nothing, Real} = nothing,
                         blim::Union{Nothing, Real} = nothing) -> Float64

Constrain interannual TAC fluctuations to prevent severe socioeconomic volatility, with
an **emergency override** adhering to the Precautionary Approach:

# Mathematical Formulation
```math
\\mathrm{TAC}_t = \\begin{cases}
\\min\\big(\\mathrm{TAC}_{t-1}(1 + \\delta_{\\mathrm{inc}}), \\; \\mathrm{TAC}_{\\mathrm{target}}\\big) & \\text{if } B_t < B_{\\mathrm{lim}} \\text{ (emergency override)} \\\\[6pt]
\\max\\Big(\\mathrm{TAC}_{t-1}(1 - \\delta_{\\mathrm{dec}}), \\;
           \\min\\big(\\mathrm{TAC}_{t-1}(1 + \\delta_{\\mathrm{inc}}), \\;
                      \\mathrm{TAC}_{\\mathrm{target}}\\big)\\Big) & \\text{otherwise}
\\end{cases}
```

When a stock declines below the limit reference point (``B < B_{\\mathrm{lim}}``), capping downward
quota adjustments at a fixed annual percentage perpetuates overfishing and delays rebuilding.
The emergency override suspends the downward constraint (``\\delta_{\\mathrm{dec}}`` floor), allowing
the quota to drop immediately to the target prescribed by the Harvest Control Rule.

# Inputs
- `tac_target`: Unconstrained TAC resulting directly from the HCR.
- `tac_prev`: Catch or TAC realized in the preceding year.
- `max_decrease`: Maximum proportional decrease allowed (e.g. 0.15 for -15%).
- `max_increase`: Maximum proportional increase allowed (e.g. 0.15 for +15%).
- `biomass`: Current perceived or estimated biomass ``B_t``.
- `blim`: Limit reference point ``B_{\\mathrm{lim}}``. When `biomass < blim`, downward capping is suspended.

# Outputs
Constrained harvest recommendation.
"""
function apply_tac_stabilizer(
    tac_target::Real,
    tac_prev::Real;
    max_decrease::Real = 0.15,
    max_increase::Real = 0.15,
    biomass::Union{Nothing, Real} = nothing,
    blim::Union{Nothing, Real} = nothing
)
    if tac_prev <= 0.0
        return Float64(tac_target)
    end

    # Precautionary emergency override: when B < Blim, suspend downward constraint
    override_decrease = !isnothing(biomass) && !isnothing(blim) && (biomass < blim)

    lower_bound = override_decrease ? 0.0 : tac_prev * (1.0 - max_decrease)
    upper_bound = tac_prev * (1.0 + max_increase)

    if tac_target < lower_bound
        return Float64(lower_bound)
    elseif tac_target > upper_bound
        return Float64(upper_bound)
    else
        return Float64(tac_target)
    end
end

"""
Result of a closed-loop Management Strategy Evaluation (MSE) feedback simulation.

# Fields
- `years::Vector{Int}`: Simulation horizon years.
- `true_biomass::Vector{Float64}`: Operating model true latent biomass trajectory.
- `observed_biomass::Vector{Float64}`: Assessment/survey perceived biomass trajectory.
- `catches::Vector{Float64}`: Realized catch removals.
- `harvest_rate::Vector{Float64}`: Realized harvest rates ``C_t / B_t``.
"""
struct MseResult
    years::Vector{Int}
    true_biomass::Vector{Float64}
    observed_biomass::Vector{Float64}
    catches::Vector{Float64}
    harvest_rate::Vector{Float64}
end

"""
    simulate_mse(years, hcr; rng = Random.default_rng(),
                 r = 0.35, k = 100.0, b0 = 80.0,
                 obs_cv = 0.15, proc_cv = 0.05,
                 tac_constraint = 0.15,
                 emergency_override = true) -> MseResult

Deterministic/stochastic closed-loop Management Strategy Evaluation (MSE) simulation:
- Simulates an underlying Schaefer operating model with process error.
- Applies observation error to generate assessment biomass estimates ``\\hat{B}_t``.
- Evaluates the Harvest Control Rule (HCR) and interannual TAC stabilizer.
- If `emergency_override = true`, suspends downward TAC stabilization whenever
  ``\\hat{B}_t < B_{\\mathrm{lim}}`` in compliance with the Precautionary Approach.
- Feeds realized catch back into the population dynamics.
"""
function simulate_mse(years::AbstractVector{<:Integer},
                      hcr::HockeyStickHCR;
                      rng::Random.AbstractRNG = Random.default_rng(),
                      r::Real = 0.35,
                      k::Real = 100.0,
                      b0::Real = 80.0,
                      obs_cv::Real = 0.15,
                      proc_cv::Real = 0.05,
                      tac_constraint::Real = 0.15,
                      emergency_override::Bool = true)
    ny = length(years)
    b_true = zeros(Float64, ny)
    b_obs = zeros(Float64, ny)
    c_real = zeros(Float64, ny)
    hr = zeros(Float64, ny)

    b_true[1] = b0
    prev_tac = 0.0

    for t in 1:ny
        # Observation process with assessment error
        b_obs[t] = max(1e-4, b_true[t] * exp(randn(rng) * obs_cv - 0.5 * obs_cv^2))

        # Management procedure: HCR evaluation
        f_prescribed = eval_hcr(hcr, b_obs[t])
        unconstrained_tac = f_prescribed * b_obs[t]
        tac = apply_tac_stabilizer(unconstrained_tac, prev_tac;
                                   max_decrease = tac_constraint,
                                   max_increase = tac_constraint,
                                   biomass = emergency_override ? b_obs[t] : nothing,
                                   blim = emergency_override ? hcr.blim : nothing)
        prev_tac = tac

        # Implementation process: catch removal
        catch_t = min(0.90 * b_true[t], tac)
        c_real[t] = catch_t
        hr[t] = b_true[t] > 0.0 ? catch_t / b_true[t] : 0.0

        if t < ny
            # Operating model transition with process error
            p_t = r * b_true[t] * (1.0 - b_true[t] / k)
            dev_p = exp(randn(rng) * proc_cv - 0.5 * proc_cv^2)
            b_true[t + 1] = max(1e-4, (b_true[t] - catch_t + p_t) * dev_p)
        end
    end

    return MseResult(collect(years), b_true, b_obs, c_real, hr)
end

"""
    bayesian_mse_feedback(observed_biomass, catches, hcr;
                          prior_r = (0.35, 0.15),
                          prior_k = (100.0, 30.0),
                          obs_cv = 0.15,
                          prior_proc_sd = 0.08,
                          prior_proc_sd_sd = 0.04,
                          prior_obs_sd_sd = 0.05,
                          k_lower = 10.0,
                          k_upper_factor = 5.0,
                          r_lower = 0.05,
                          r_upper = 1.20,
                          proc_sd_lower = 0.01,
                          proc_sd_upper = 0.30,
                          obs_sd_lower = 0.02,
                          obs_sd_upper = 0.50,
                          prior_b0_sd = 15.0,
                          b0_lower = 5.0,
                          proc_sd_scale = 1.0,
                          b_floor = 1.0e-4)

Bayesian state-space feedback model for Management Strategy Evaluation in `Turing.jl`:
- Formulates joint posterior over latent operating model states given HCR catches and biomass observations.
- Estimates biological productivity ``r, K`` and process/observation errors.
- Evaluates two likelihoods: the assessment biomass index
  ``\\hat{B}_t \\sim \\mathrm{LogNormal}(\\log B_t, \\sigma_o)``, and implementation error on the
  catch against the target the harvest control rule prescribes applied to the latent stock,
  ``C_t \\sim \\mathrm{LogNormal}(\\log(f_t B_t), \\sigma_i)``.

Every prior mean, standard deviation and bound is a named keyword argument, so the prior
specification is user-level rather than compiled into the model body.

# Arguments
- `observed_biomass`: assessment or survey biomass estimates.
- `catches`: realized removals.
- `hcr::HockeyStickHCR`: the harvest control rule in force, which sets the fishing rate for
  each year from the assessment biomass.
"""
@model function bayesian_mse_feedback(observed_biomass::AbstractVector{<:Real},
                                      catches::AbstractVector{<:Real},
                                      hcr::HockeyStickHCR;
                                      prior_r::Tuple{Real, Real} = (0.35, 0.15),
                                      prior_k::Tuple{Real, Real} = (100.0, 30.0),
                                      obs_cv::Real = 0.15,
                                      prior_proc_sd::Real = 0.08,
                                      prior_proc_sd_sd::Real = 0.04,
                                      prior_obs_sd_sd::Real = 0.05,
                                      k_lower::Real = 10.0,
                                      k_upper_factor::Real = 5.0,
                                      r_lower::Real = 0.05,
                                      r_upper::Real = 1.20,
                                      proc_sd_lower::Real = 0.01,
                                      proc_sd_upper::Real = 0.30,
                                      obs_sd_lower::Real = 0.02,
                                      obs_sd_upper::Real = 0.50,
                                      prior_b0_sd::Real = 15.0,
                                      b0_lower::Real = 5.0,
                                      proc_sd_scale::Real = 1.0,
                                      b_floor::Real = 1.0e-4,
                                      prior_impl_sd::Real = 0.10,
                                      prior_impl_sd_sd::Real = 0.04,
                                      impl_sd_lower::Real = 0.02,
                                      impl_sd_upper::Real = 0.50)
    nt = length(observed_biomass)
    length(catches) == nt || error("observed_biomass and catches lengths must match.")

    k ~ truncated(Normal(prior_k[1], prior_k[2]), k_lower, k_upper_factor * prior_k[1])
    r ~ truncated(Normal(prior_r[1], prior_r[2]), r_lower, r_upper)
    proc_sd ~ truncated(Normal(prior_proc_sd, prior_proc_sd_sd),
                        proc_sd_lower, proc_sd_upper)
    obs_sd ~ truncated(Normal(obs_cv, prior_obs_sd_sd), obs_sd_lower, obs_sd_upper)
    # Implementation error: the deviation between the catch the HCR prescribes and
    # the catch actually removed. It is a separate process from observation error on
    # biomass, so it gets its own parameter.
    impl_sd ~ truncated(Normal(prior_impl_sd, prior_impl_sd_sd),
                        impl_sd_lower, impl_sd_upper)

    b = Vector{Float64}(undef, nt)
    b[1] ~ truncated(Normal(observed_biomass[1], prior_b0_sd), b0_lower, k)

    for t in 1:(nt - 1)
        esc = max(b_floor, b[t] - catches[t])
        mu_next = esc + r * esc * (1.0 - esc / k)
        b[t + 1] ~ truncated(Normal(mu_next, proc_sd_scale * proc_sd * b[t]),
                             b_floor, 2.0 * k)
    end

    # Assessment biomass index: lognormal observation error on the latent stock.
    for t in 1:nt
        observed_biomass[t] ~ LogNormal(log(max(b_floor, b[t])), obs_sd)
    end

    # Management performance: realized catch against the catch the HCR prescribes for
    # the biomass seen by the assessment. The HCR acts on the assessment biomass in
    # closed-loop simulation, so it is evaluated on `observed_biomass[t]`, and the
    # prescribed rate is applied to the latent stock to obtain a target catch.
    for t in 1:nt
        f_hcr = eval_hcr(hcr, observed_biomass[t])
        target_catch = max(b_floor, f_hcr * b[t])
        catches[t] ~ LogNormal(log(target_catch), impl_sd)
    end

    return (biomass = b, r = r, k = k, proc_sd = proc_sd, obs_sd = obs_sd,
            impl_sd = impl_sd)
end
