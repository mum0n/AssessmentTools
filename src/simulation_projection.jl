# ------------------------------------------------------------------------------------
# Forward simulation and projection under TAC options for decision tables.
# ------------------------------------------------------------------------------------

"""
    DecisionMetrics

Collection of decision table metrics computed across proposed TAC levels:
- `P(SB_{T+1} < SB_T)`: Probability of spawning biomass decline
- `P(SB_{T+1} < 0.25 B_0)`: Probability of severe depletion (< 25% B0)
- `P(SB_{T+1} < 0.75 B_0)`: Probability of depletion (< 75% B0)
- `P(SB_{T+1} < 0.40 B_MSY)`: Probability below limit reference point (40% Bmsy)
- `P(SB_{T+1} < 0.80 B_MSY)`: Probability below target reference point (80% Bmsy)
- `P(U_{T+1} > U_MSY)`: Probability of overfishing relative to Umsy
- `P(U_{T+1} > 0.5 U_MSY)`
- `P(U_{T+1} > 0.67 U_MSY)`
- `P(U_{3+} > 0.20)`: Exploitation rate on age-3+ fish exceeding 20%
- `P(dSB5 < 0)`: Average 5-year rate of decline in spawning biomass
"""
struct DecisionMetrics
    tac::Float64
    p_sb_decline::Float64
    p_sb_lt_025bo::Float64
    p_sb_lt_075bo::Float64
    p_sb_lt_040bmsy::Float64
    p_sb_lt_080bmsy::Float64
    p_u_gt_umsy::Float64
    p_u_gt_05umsy::Float64
    p_u_gt_067umsy::Float64
    p_u3_gt_020::Float64
    dsb5_rate::Float64
end

"""
    project_population(model, x, tac; n_sim=1000, rng=Random.default_rng()) -> DecisionMetrics

Projects population dynamics forward under a constant Total Allowable Catch (`tac`)
using terminal year selectivities and mean life-history schedules.
"""
function project_population(m::StockModel, x::AbstractVector{<:Real}, tac::Real;
                            n_sim::Integer = 1000, rng::AbstractRNG = Random.default_rng())
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
    rec_type = m.control.misc.rec_type
    tau = lead.tau
    m_M = mort.m_bar

    # Terminal state
    N_nyr1 = copy(pop.N[ny + 1, :])
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

    # Solve future fishing mortality
    p_ft = solve_baranov_f(p_ct, m_M, va_bar, N_nyr1, m.avg_wt)
    p_Z = fill(m_M, na)
    for k in 1:ngear(m)
        p_Z .+= p_ft[k] .* va_bar[k, :]
    end

    # Future spawning biomass
    p_sbt = sum(N_nyr1 .* exp.(-p_Z .* sf) .* m.avg_fec)

    # Reference point ratios for projection metrics
    r_sb1 = SB_nyr / p_sbt
    r_sb2 = 0.25 * bo / p_sbt
    r_sb3 = 0.75 * bo / p_sbt
    r_sb4 = 0.40 * bmsy / p_sbt
    r_sb5 = 0.80 * bmsy / p_sbt

    ut = tac / (p_sbt + 1e-15)
    r_u1 = ut / (umsy + 1e-15)
    r_u2 = ut / (0.5 * umsy + 1e-15)
    r_u3 = ut / ((2.0 / 3.0) * umsy + 1e-15)

    age3_plus = sum(N_nyr1[3:end] .* exp.(-mort.M_tot[ny, 3:end]) .* m.avg_wt[3:end])
    u20 = tac / (age3_plus + 1e-15)
    r_u4 = u20 / 0.20

    # 5-year rate of change in historical spawning biomass
    sbt_hist = [sum(pop.N[t, :] .* exp.(-mort.Z[t, :] .* sf) .* m.fec[t, :]) for t in 1:ny]
    push!(sbt_hist, p_sbt)
    t_end = length(sbt_hist)
    dsb5 = (t_end >= 6) ?
           sum(log.(sbt_hist[(t_end - 4):t_end]) .- log.(sbt_hist[(t_end - 5):(t_end - 1)])) / 5.0 :
           0.0

    return DecisionMetrics(
        tac, r_sb1, r_sb2, r_sb3, r_sb4, r_sb5,
        r_u1, r_u2, r_u3, r_u4, dsb5 + 1.0
    )
end

"""
    decision_table(model, x) -> Vector{DecisionMetrics}

Evaluates projected risk metrics across all TAC options defined in the model's
projection control file (`.pfc`).
"""
function decision_table(m::StockModel, x::AbstractVector{<:Real})
    return [project_population(m, x, tac) for tac in m.projection.tac]
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
