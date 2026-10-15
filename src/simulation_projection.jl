# ------------------------------------------------------------------------------------
# Forward simulation and projection under TAC options for decision tables.
# ------------------------------------------------------------------------------------

"""
    DecisionMetrics

Monte Carlo performance statistics for one Total Allowable Catch (`tac`) option,
evaluated by [`project_population`](@ref). All `p_*` fields are probabilities in
`[0, 1]` estimated over `n_sim` projection realisations.

Threshold values are not part of this struct: they come from
[`DecisionControls`](@ref). The numbered suffix on each field identifies the
position of the corresponding threshold in the configured vector, so
`p_sb_lt_bo_1` uses `sb_bo_thresholds[1]`.

# Fields
- `tac::Float64`: evaluated total allowable catch
- `p_sb_decline::Float64`: `P(SB_{T+1} < SB_T)`
- `p_sb_lt_bo_1`, `p_sb_lt_bo_2`: `P(SB_{T+1} < f B₀)` for the configured fractions `f`
- `p_sb_lt_bmsy_1`, `p_sb_lt_bmsy_2`: `P(SB_{T+1} < f B_MSY)`
- `p_u_gt_umsy_1`, `p_u_gt_umsy_2`, `p_u_gt_umsy_3`: `P(U_{T+1} > f U_MSY)`
- `p_u_age_gt_limit`: `P(U_{age_floor+, T+1} > age3_exploitation_limit)`
- `dsb5_rate::Float64`: mean of `log(SB_{T+1}/SB_T)` over realisations; the average
  rate of change in spawning biomass, not a probability
"""
struct DecisionMetrics
    tac::Float64
    p_sb_decline::Float64
    p_sb_lt_bo_1::Float64
    p_sb_lt_bo_2::Float64
    p_sb_lt_bmsy_1::Float64
    p_sb_lt_bmsy_2::Float64
    p_u_gt_umsy_1::Float64
    p_u_gt_umsy_2::Float64
    p_u_gt_umsy_3::Float64
    p_u_age_gt_limit::Float64
    dsb5_rate::Float64
end

"""
    project_population(model, x, tac; n_sim = 1000, rng = Random.default_rng()) -> DecisionMetrics

Project one year forward under constant total allowable catch `tac` and estimate the
decision-table probability metrics by Monte Carlo.

The deterministic components — terminal selectivity, mean life-history schedules,
the fishing mortality that yields `tac`, and the reference points — are held fixed.
The stochastic component is the recruitment deviation of the projection year, drawn
from the stock-recruitment process error ``\\epsilon \\sim \\mathrm{Normal}(0, \\tau)``
with ``\\tau`` the process error standard deviation already estimated by the model:

```math
R_{T+1} = \\bar{R}\\, e^{\\epsilon},\\qquad \\epsilon \\sim N(0, \\tau)
```

Among the `3` or more ages that this affects, the resulting spawning biomass ``SB_{T+1}``
is computed each realisation, and the `p_*` fields are the sample proportions of the
corresponding events. `dsb5_rate` is the mean log change ``\\log(SB_{T+1}/SB_T)``.
"""
function project_population(m::StockModel, x::AbstractVector{<:Real}, tac::Real;
                            n_sim::Integer = 1000, rng::AbstractRNG = Random.default_rng())
    n_sim >= 1 || error("n_sim must be at least 1.")
    p = unpack(m.layout, x)
    ctx = PhaseContext(m.layout.max_phase, true)
    lead = leading_parameters(m, p.theta)
    log_sel = calc_selectivities(m, p.sel_par)
    mort = calc_total_mortality(m, log_sel, p.log_ft_pars, p.log_m_nodes, lead.M, ctx)
    pop = calc_numbers_at_age(m, lead, mort, p.log_rec_devs, p.init_log_rec_devs)
    ref = calc_reference_points(m, mort, log_sel)

    ny = m.ny
    na = m.nages
    sf = m.control.misc.spawn_fraction
    tau = lead.tau
    m_M = mort.m_bar

    SB_nyr = sum(pop.N[ny, :] .* exp.(-mort.Z[ny, :] .* sf) .* m.fec[ny, :])
    bo = ref.bo
    bmsy = ref.bmsy
    umsy = ref.umsy

    # Gear allocations and terminal selectivity
    va_bar = zeros(Float64, ngear(m), na)
    p_ct = zeros(Float64, ngear(m))
    for k in 1:ngear(m)
        va_bar[k, :] .= exp.(log_sel[k][end, :])
        p_ct[k] = m.data.allocation[k] * tac
    end

    # Mean fishing mortality that yields `tac`, solved on the projected age structure
    p_ft = solve_baranov_f(p_ct, m_M, va_bar, pop.N[ny + 1, :], m.avg_wt)
    p_Z = fill(m_M, na)
    for k in 1:ngear(m)
        p_Z .+= p_ft[k] .* va_bar[k, :]
    end

    sb_anchor = sum(pop.N[ny + 1, :] .* exp.(-p_Z .* sf) .* m.avg_fec)

    # Age floor for the age-3+ exploitation rate. Ages in this model run from
    # `m.ages`, so the floor is defined by age, not by row index.
    dc = m.control.decision
    age_rows = findall(i -> m.ages[i] >= dc.age_floor, 1:na)
    b_age = sum(pop.N[ny + 1, age_rows] .* exp.(-mort.M_tot[ny, age_rows]) .* m.avg_wt[age_rows])

    eps = dc.epsilon
    counts = Dict{Symbol, Any}(
        :decline => 0, :lt_bo => zeros(Int, length(dc.sb_bo_thresholds)),
        :lt_bmsy => zeros(Int, length(dc.sb_bmsy_thresholds)),
        :gt_umsy => zeros(Int, length(dc.u_umsy_thresholds)),
        :gt_u_floor => 0
    )
    dsb5_sum = 0.0

    for _ in 1:n_sim
        multiplier = tau > 0.0 ? exp(rand(rng, Normal(0.0, tau))) : 1.0
        sb = sb_anchor * multiplier
        u = tac / (sb + eps)
        u_age = tac / (b_age * multiplier + eps)

        counts[:decline] += (sb < SB_nyr)
        for (i, t) in enumerate(dc.sb_bo_thresholds)
            counts[:lt_bo][i] += (sb < t * bo)
        end
        for (i, t) in enumerate(dc.sb_bmsy_thresholds)
            counts[:lt_bmsy][i] += (sb < t * bmsy)
        end
        for (i, t) in enumerate(dc.u_umsy_thresholds)
            counts[:gt_umsy][i] += (u > t * umsy + eps)
        end
        counts[:gt_u_floor] += (u_age > dc.age3_exploitation_limit)
        dsb5_sum += log(sb / (SB_nyr + eps))
    end

    return DecisionMetrics(
        Float64(tac),
        counts[:decline] / n_sim,
        counts[:lt_bo][1] / n_sim, counts[:lt_bo][2] / n_sim,
        counts[:lt_bmsy][1] / n_sim, counts[:lt_bmsy][2] / n_sim,
        counts[:gt_umsy][1] / n_sim, counts[:gt_umsy][2] / n_sim,
        counts[:gt_umsy][3] / n_sim,
        counts[:gt_u_floor] / n_sim, dsb5_sum / n_sim
    )
end

"""
    decision_table(model, x; n_sim = 1000, rng = Random.default_rng()) -> Vector{DecisionMetrics}

Evaluates projected risk metrics across all TAC options defined in the model's
projection control table. Thresholds come from the `[decision]` section of the
configuration file.
"""
function decision_table(m::StockModel, x::AbstractVector{<:Real};
                         n_sim::Integer = 1000,
                         rng::AbstractRNG = Random.default_rng())
    return [project_population(m, x, tac; n_sim = n_sim, rng = rng)
            for tac in m.projection.tac]
end

"""
    simulate_data(model, seed=999; error=true) -> StockModel

Generates a simulated dataset with known truth.
When `error=false` (seed=000), exact data without observation or process error are simulated.
"""
function simulate_data(m::StockModel, seed::Integer = 999; error::Bool = true)
    rng = Random.MersenneTwister(seed)
    p = unpack(m.layout, m.x0)
    ctx = PhaseContext(m.layout.max_phase, true)
    lead = leading_parameters(m, p.theta)
    log_sel = calc_selectivities(m, p.sel_par)

    sig = error ? (lead.rho / lead.varphi) : 0.0
    tau = error ? ((1.0 - lead.rho) / lead.varphi) : 0.0

    wt = error && tau > 0.0 ? rand(rng, Normal(0.0, tau), m.ny) : zeros(m.ny)

    # Initial population
    N = zeros(Float64, m.ny + 1, m.nages)
    lx = survivorship(lead.M, m.nages)
    if m.control.misc.unfished_start
        N[1, :] .= lead.ro .* lx
    else
        N[1, 1] = exp(lead.log_avgrec + p.log_rec_devs[1])
        N[1, 2:end] .= exp.(lead.log_recinit .+ p.init_log_rec_devs) .* lx[2:end]
    end

    for t in 2:m.ny
        N[t, 1] = exp(lead.log_avgrec + p.log_rec_devs[t])
    end
    N[m.ny + 1, 1] = exp(lead.log_avgrec)

    # Forward dynamics
    sim_obs_ct = copy(m.data.obs_ct)
    d3C = [zeros(Float64, m.ny, m.nages) for _ in 1:ngear(m)]

    for t in 1:m.ny
        va = zeros(Float64, ngear(m), m.nages)
        for k in 1:ngear(m)
            va[k, :] .= exp.(log_sel[k][t, :])
        end
        oct = m.data.obs_ct[:, t]
        ft_t = solve_baranov_f(oct, lead.M, va, N[t, :], m.wt_obs[t, :])

        Zt = fill(lead.M, m.nages)
        for k in 1:ngear(m)
            Zt .+= ft_t[k] .* va[k, :]
        end

        # Dynamics update
        if t >= m.sage
            et = sum(N[t - m.sage + 1, :] .* exp.(-Zt .* m.control.misc.spawn_fraction) .* m.fec[t - m.sage + 1, :])
            phie = sum(lx .* m.avg_fec)
            so = lead.kappa / phie
            beta = sr_beta(lead.kappa, lead.ro * phie, m.control.misc.rec_type)
            rt = stock_recruit(et, so, beta, m.control.misc.rec_type)
            N[t + 1, 1] = rt * exp(wt[t] - 0.5 * tau^2)
        end

        N[t + 1, 2:end] .= N[t, 1:(end - 1)] .* exp.(-Zt[1:(end - 1)])
        N[t + 1, end] += N[t, end] * exp(-Zt[end])

        for k in 1:ngear(m)
            sel_k = va[k, :]
            d3C[k][t, :] .= (ft_t[k] .* sel_k ./ Zt) .* (1.0 .- exp.(-Zt)) .* N[t, :]
            sim_obs_ct[k, t] = sum(d3C[k][t, :] .* m.wt_obs[t, :])
        end
    end

    # Simulated survey data
    sim_surveys = SurveySeries[]
    for (k, s) in enumerate(m.data.surveys)
        sd_s = sig ./ s.weight
        eps = error && sig > 0.0 ? [rand(rng, Normal(0.0, sd_s[i])) for i in eachindex(sd_s)] :
                                   zeros(length(s.index))
        sim_it = zeros(Float64, length(s.index))
        for (i, yr) in enumerate(s.year)
            t = yrow(m, yr)
            sel_g = exp.(log_sel[s.gear[i]][t, :])
            Np = N[t, :] .* exp.(-lead.M * s.timing[i])
            units = survey_units(s.survey_type, Np, sel_g, m.wt_obs[t, :], m.fec[t, :], 0.0)
            sim_it[i] = units * exp(eps[i])
        end
        push!(sim_surveys, SurveySeries(copy(s.year), sim_it, copy(s.gear), copy(s.weight), copy(s.timing), s.survey_type))
    end

    # Simulated age compositions
    sim_agecomps = AgeCompSeries[]
    for (k, ac) in enumerate(m.data.agecomps)
        sim_obs = zeros(Float64, size(ac.obs))
        for (i, yr) in enumerate(ac.year)
            t = yrow(m, yr)
            g = ac.gear[i]
            pa = d3C[g][t, (ac.sage - m.sage + 1):(ac.nage - m.sage + 1)]
            pa ./= sum(pa)
            sim_obs[i, :] .= error ? rmvlogistic(pa, 0.3, rng) : pa
        end
        push!(sim_agecomps, AgeCompSeries(copy(ac.year), copy(ac.gear), ac.sage, ac.nage, sim_obs))
    end

    sim_data = StockData(
        m.data.syr, m.data.nyr, m.data.sage, m.data.nage, m.data.ngear,
        copy(m.data.allocation), copy(m.data.catch_type), m.data.fixed_m,
        m.data.linf, m.data.vonbk, m.data.to, m.data.a, m.data.b,
        m.data.ah, m.data.gh, sim_obs_ct, sim_surveys, sim_agecomps, copy(m.data.wt_input)
    )

    return StockModel(sim_data, m.control, m.projection; retro_yrs = m.retro_yrs)
end
