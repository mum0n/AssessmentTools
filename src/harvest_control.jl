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
                         max_decrease::Real = 0.15, max_increase::Real = 0.15) -> Float64

Constrain interannual TAC fluctuations to prevent severe socioeconomic volatility.

# Mathematical Formulation
```math
\\mathrm{TAC}_t = \\max\\Big(\\mathrm{TAC}_{t-1}(1 - \\delta_{\\mathrm{dec}}), \\;
                       \\min\\big(\\mathrm{TAC}_{t-1}(1 + \\delta_{\\mathrm{inc}}), \\;
                                  \\mathrm{TAC}_{\\mathrm{target}}\\big)\\Big)
```

# Inputs
- `tac_target`: Unconstrained TAC resulting directly from the HCR.
- `tac_prev`: Catch or TAC realized in the preceding year.
- `max_decrease`: Maximum proportional decrease allowed (e.g. 0.15 for -15%).
- `max_increase`: Maximum proportional increase allowed (e.g. 0.15 for +15%).

# Outputs
Constrained harvest recommendation.
"""
function apply_tac_stabilizer(
    tac_target::Real,
    tac_prev::Real;
    max_decrease::Real = 0.15,
    max_increase::Real = 0.15
)
    if tac_prev <= 0.0
        return Float64(tac_target)
    end
    lower_bound = tac_prev * (1.0 - max_decrease)
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
    simulate_mse(years::AbstractVector{<:Integer}, hcr::HockeyStickHCR;
                 r::Real = 0.35, k::Real = 100.0, b0::Real = 80.0,
                 obs_cv::Real = 0.15, proc_cv::Real = 0.05,
                 tac_constraint::Real = 0.15) -> MseResult

Deterministic/stochastic closed-loop Management Strategy Evaluation (MSE) simulation:
- Simulates an underlying Schaefer operating model with process error.
- Applies observation error to generate assessment biomass estimates ``\\hat{B}_t``.
- Evaluates the Harvest Control Rule (HCR) and interannual TAC stabilizer.
- Feeds realized catch back into the population dynamics.
"""
function simulate_mse(years::AbstractVector{<:Integer},
                      hcr::HockeyStickHCR;
                      r::Real = 0.35,
                      k::Real = 100.0,
                      b0::Real = 80.0,
                      obs_cv::Real = 0.15,
                      proc_cv::Real = 0.05,
                      tac_constraint::Real = 0.15)
    ny = length(years)
    b_true = zeros(Float64, ny)
    b_obs = zeros(Float64, ny)
    c_real = zeros(Float64, ny)
    hr = zeros(Float64, ny)

    b_true[1] = b0
    prev_tac = 0.0

    for t in 1:ny
        # Observation process with assessment error
        b_obs[t] = max(1e-4, b_true[t] * exp(randn() * obs_cv - 0.5 * obs_cv^2))

        # Management procedure: HCR evaluation
        f_prescribed = eval_hcr(hcr, b_obs[t])
        unconstrained_tac = f_prescribed * b_obs[t]
        tac = apply_tac_stabilizer(unconstrained_tac, prev_tac;
                                   max_decrease = tac_constraint,
                                   max_increase = tac_constraint)
        prev_tac = tac

        # Implementation process: catch removal
        catch_t = min(0.90 * b_true[t], tac)
        c_real[t] = catch_t
        hr[t] = b_true[t] > 0.0 ? catch_t / b_true[t] : 0.0

        if t < ny
            # Operating model transition with process error
            p_t = r * b_true[t] * (1.0 - b_true[t] / k)
            dev_p = exp(randn() * proc_cv - 0.5 * proc_cv^2)
            b_true[t + 1] = max(1e-4, (b_true[t] - catch_t + p_t) * dev_p)
        end
    end

    return MseResult(collect(years), b_true, b_obs, c_real, hr)
end

"""
    bayesian_mse_feedback(observed_biomass, catches, hcr;
                          prior_r = (0.35, 0.15),
                          prior_k = (100.0, 30.0),
                          obs_cv = 0.15)

Bayesian state-space feedback model for Management Strategy Evaluation in `Turing.jl`:
- Formulates joint posterior over latent operating model states given HCR catches and biomass observations.
- Estimates biological productivity ``r, K`` and process/observation errors.
"""
@model function bayesian_mse_feedback(observed_biomass::AbstractVector{<:Real},
                                      catches::AbstractVector{<:Real},
                                      hcr::HockeyStickHCR;
                                      prior_r::Tuple{Real, Real} = (0.35, 0.15),
                                      prior_k::Tuple{Real, Real} = (100.0, 30.0),
                                      obs_cv::Real = 0.15)
    nt = length(observed_biomass)
    length(catches) == nt || error("observed_biomass and catches lengths must match.")

    k ~ truncated(Normal(prior_k[1], prior_k[2]), 10.0, 5.0 * prior_k[1])
    r ~ truncated(Normal(prior_r[1], prior_r[2]), 0.05, 1.20)
    proc_sd ~ truncated(Normal(0.08, 0.04), 0.01, 0.30)
    obs_sd ~ truncated(Normal(obs_cv, 0.05), 0.02, 0.50)

    b = Vector{Float64}(undef, nt)
    b[1] ~ truncated(Normal(observed_biomass[1], 15.0), 5.0, k)

    for t in 1:(nt - 1)
        esc = max(1e-4, b[t] - catches[t])
        mu_next = esc + r * esc * (1.0 - esc / k)
        b[t + 1] ~ truncated(Normal(mu_next, proc_sd * b[t]), 1e-4, 2.0 * k)
    end

    for t in 1:nt
        observed_biomass[t] ~ LogNormal(log(max(1e-4, b[t])), obs_sd)
    end

    return (biomass = b, r = r, k = k, proc_sd = proc_sd, obs_sd = obs_sd)
end
