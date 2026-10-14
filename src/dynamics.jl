# ------------------------------------------------------------------------------------
# Population dynamics and observation models.
# Indexing: row t of year-by-age matrices is year syr + t - 1; column j is age
# sage + j - 1. N has ny + 1 rows (the last is the start of year nyr + 1).
# ------------------------------------------------------------------------------------

"""
    recruitment_kappa(h, rec_type)

Goodyear compensation ratio from steepness `h`: Beverton–Holt `κ = 4h/(1-h)`,
Ricker `κ = (5h)^{5/4}` (Myers et al. 1999; Martell et al. 2008).
"""
function recruitment_kappa(h, rec_type::Integer)
    rec_type == 1 && return 4h / (1 - h)
    rec_type == 2 && return (5h)^1.25
    error("Unknown recruitment model $rec_type.")
end

"""
    stock_recruit(s, so, beta, rec_type)

Expected recruits from spawning biomass `s`: Beverton–Holt `R = s₀ s / (1 + β s)`,
Ricker `R = s₀ s exp(-β s)`. With `φ_B` the unfished spawning biomass per recruit and
`B₀ = R₀ φ_B`: `s₀ = κ/φ_B`; `β = (κ-1)/B₀` (BH) or `β = log κ / B₀` (Ricker).
"""
function stock_recruit(s, so, beta, rec_type::Integer)
    rec_type == 1 && return so * s / (1 + beta * s)
    rec_type == 2 && return so * s * exp(-beta * s)
    error("Unknown recruitment model $rec_type.")
end

"""
    sr_beta(kappa, bo, rec_type)

Density-dependent stock–recruit parameter `β` given `κ` and unfished spawning
biomass `B₀` (see [`stock_recruit`](@ref)).
"""
sr_beta(kappa, bo, rec_type::Integer) = rec_type == 1 ? (kappa - 1) / bo : log(kappa) / bo

"""
    leading_parameters(model, theta) -> NamedTuple

Transform `θ = (log R₀, h, log M, log R̄, log R_init, ρ, ϑ)` into model quantities.
Observation and process error sd: `σ = ρ/ϑ`, `τ = (1-ρ)/ϑ` (total precision `ϑ`,
proportion `ρ` of variance due to observation error).
"""
function leading_parameters(m::StockModel, theta::AbstractVector)
    rec_type = m.control.misc.rec_type
    ro = exp(theta[1])
    h = theta[2]
    rho, varphi = theta[6], theta[7]
    return (ro = ro, h = h, M = exp(theta[3]), log_avgrec = theta[4],
            log_recinit = theta[5], rho = rho, varphi = varphi,
            kappa = recruitment_kappa(h, rec_type), sig = rho / varphi,
            tau = (1 - rho) / varphi)
end

"""
    calc_total_mortality(model, log_sel, log_ft_pars, log_m_nodes, M, ctx)

Fishing, natural and total mortality (calcTotalMortality).

- `F_{k,t} = exp(log_ft)` for years with positive catch, else 0;
- `F_{t,a} = Σ_k F_{k,t} v_{k,t,a}`, excluding spawn-on-kelp gears (catch type 3);
- `M_{t,a} = M`, or, when the M nodes are active, the random walk
  `M_t = M_{t-1} exp(δ_t)` with `δ_t` interpolated by a clamped cubic spline through
  the nodes at positions `(t - syr - 1)/(nyr - syr)`, `t = syr+1, …, nyr`;
- `Z = M + F`, `S = exp(-Z)`; `M̄` is the mean of `M` over `pf_cntrl[1]:pf_cntrl[2]`.
"""
function calc_total_mortality(m::StockModel, log_sel, log_ft_pars, log_m_nodes, M,
                              ctx::PhaseContext)
    T = promote_type(eltype(log_sel[1]), eltype(log_ft_pars), typeof(M),
                     eltype(log_m_nodes))
    ng, ny, na = ngear(m), m.ny, m.nages
    ft = zeros(T, ng, ny)
    F = zeros(T, ny, na)
    for k in 1:ng, t in 1:ny
        idx = m.ft_index[k, t]
        idx == 0 && continue
        ft[k, t] = exp(log_ft_pars[idx])
        if m.data.catch_type[k] != 3
            F[t, :] .+= ft[k, t] .* exp.(log_sel[k][t, :])
        end
    end
    M_tot = fill(T(M), ny, na)
    log_m_devs = zeros(T, ny)
    misc = m.control.misc
    if misc.n_m_nodes > 0 && is_active(misc.m_dev_phase, ctx)
        s = CubicSpline(unit_nodes(misc.n_m_nodes), log_m_nodes)
        log_m_devs[2:ny] .= s(collect(0:(ny - 2)) ./ (ny - 1))
        for t in 2:ny
            M_tot[t, :] .= M_tot[t - 1, :] .* exp(log_m_devs[t])
        end
    end
    rows = yrow(m, m.pf_cntrl[1]):yrow(m, m.pf_cntrl[2])
    m_bar = sum(M_tot[rows, :]) / length(M_tot[rows, :])
    Z = M_tot .+ F
    return (ft = ft, F = F, M_tot = M_tot, log_m_devs = log_m_devs, m_bar = m_bar, Z = Z,
            S = exp.(-Z))
end

"""
    survivorship(M, nages; plus_group = true) -> Vector

Survivorship `l_1 = 1`, `l_j = l_{j-1} exp(-M_{j-1})`, with the plus group
`l_A ← l_A / (1 - exp(-M_A))`. `M` is a scalar or an age vector.
"""
function survivorship(M, nages::Integer; plus_group::Bool = true)
    Mv = M isa AbstractVector ? M : fill(M, nages)
    lx = Vector{eltype(Mv)}(undef, nages)
    lx[1] = one(eltype(Mv))
    for j in 2:nages
        lx[j] = lx[j - 1] * exp(-Mv[j - 1])
    end
    plus_group && (lx[end] /= 1 - exp(-Mv[end]))
    return lx
end

"""
    calc_numbers_at_age(model, lead, mort, log_rec_devs, init_log_rec_devs)

Numbers-at-age (calcNumbersAtAge). Initial survivorship uses the mean `M` of the
first year. If `unfished_start`, `N_{syr} = R₀ l`; otherwise
`N_{syr,sage} = R̄ e^{ω_{syr}}`, `N_{syr,a} = R_init e^{ω⁰_a} l_a`. Recruits
`N_{t,sage} = R̄ e^{ω_t}`, `N_{nyr+1,sage} = R̄`. Ageing:
`N_{t+1,a+1} = N_{t,a} S_{t,a} + 10⁻¹⁰` with the plus group accumulating
`N_{t,A} S_{t,A}`.
"""
function calc_numbers_at_age(m::StockModel, lead, mort, log_rec_devs, init_log_rec_devs)
    T = promote_type(eltype(mort.S), eltype(log_rec_devs), eltype(init_log_rec_devs),
                     typeof(lead.ro), typeof(lead.log_avgrec))
    ny, na = m.ny, m.nages
    S = mort.S
    lx = survivorship(sum(mort.M_tot[1, :]) / na, na)
    N = zeros(T, ny + 1, na)
    log_rt = zeros(T, ny)
    if m.control.misc.unfished_start
        log_rt[1] = log(lead.ro)
        N[1, :] .= lead.ro .* lx
    else
        log_rt[1] = lead.log_avgrec + log_rec_devs[1]
        N[1, 1] = exp(log_rt[1])
        N[1, 2:na] .= exp.(lead.log_recinit .+ init_log_rec_devs) .* lx[2:na]
    end
    for t in 2:ny
        log_rt[t] = lead.log_avgrec + log_rec_devs[t]
        N[t, 1] = exp(log_rt[t])
    end
    N[ny + 1, 1] = exp(lead.log_avgrec)
    for t in 1:ny
        N[t + 1, 2:na] .= N[t, 1:(na - 1)] .* S[t, 1:(na - 1)] .+ 1e-10
        N[t + 1, na] += N[t, na] * S[t, na]
    end
    return (N = N, log_rt = log_rt)
end

"""
    baranov_catch_at_age(f_va, Z, N)

Baranov catch equation `C_a = (F_a / Z_a)(1 - e^{-Z_a}) N_a` (Baranov 1918).
"""
baranov_catch_at_age(f_va, Z, N) = f_va ./ Z .* (1 .- exp.(-Z)) .* N

"""
    calc_fishery_observations(model, log_sel, mort, N)

Predicted catch-at-age `Ĉ_{k,t,a}` and catch `Ĉ_{k,t}` (calcFisheryObservations).
For years with catch: catch type 1 is biomass `Σ_a Ĉ w`, type 2 numbers `Σ_a Ĉ`, type 3
(spawn on kelp) `(1 - e^{-F_k}) SSB_t` with `SSB_t = Σ_a N e^{-ρ_s Z} f_{t,a}`.
Residuals `η = log C - log Ĉ`. For years without catch the catch-at-age is computed
with unit F (expected proportions for survey age compositions).
"""
function calc_fishery_observations(m::StockModel, log_sel, mort, N)
    T = promote_type(eltype(N), eltype(mort.Z))
    ng, ny = ngear(m), m.ny
    sf = m.control.misc.spawn_fraction
    Chat = [zeros(T, ny, m.nages) for _ in 1:ng]
    ct = zeros(T, ng, ny)
    eta = zeros(T, ng, ny)
    for t in 1:ny, k in 1:ng
        va = exp.(log_sel[k][t, :])
        Zt = mort.Z[t, :]
        obs = m.data.obs_ct[k, t]
        if obs > 0
            Chat[k][t, :] .= baranov_catch_at_age(mort.ft[k, t] .* va, Zt, N[t, :])
            ctype = m.data.catch_type[k]
            if ctype == 1
                ct[k, t] = sum(Chat[k][t, :] .* m.wt_obs[t, :])
            elseif ctype == 2
                ct[k, t] = sum(Chat[k][t, :])
            else
                ssb = sum(N[t, :] .* exp.(-Zt .* sf) .* m.fec[t, :])
                ct[k, t] = (1 - exp(-mort.ft[k, t])) * ssb
            end
            eta[k, t] = log(obs) - log(ct[k, t])
        else
            Chat[k][t, :] .= baranov_catch_at_age(va, Zt, N[t, :])
        end
    end
    return (Chat = Chat, ct = ct, eta = eta)
end

"""
    calc_age_proportions(model, Chat) -> Vector{Matrix}

Predicted proportions-at-age for each age-composition series and each year within the
model horizon: `p_{t,a} = Ĉ_{g,t,a} / Σ_a Ĉ_{g,t,a}` over the series ages, where `g`
is the gear recorded in the data.
"""
function calc_age_proportions(m::StockModel, Chat)
    T = eltype(Chat[1])
    out = Matrix{T}[]
    for (k, a) in enumerate(m.data.agecomps)
        rows = m.agecomp_rows[k]
        cols = (a.sage - m.sage + 1):(a.nage - m.sage + 1)
        P = zeros(T, length(rows), length(cols))
        for (ii, r) in enumerate(rows)
            g = a.gear[r]
            g <= ngear(m) || error("Age-composition gear index $g exceeds ngear.")
            v = Chat[g][yrow(m, a.year[r]), cols]
            P[ii, :] .= v ./ sum(v)
        end
        push!(out, P)
    end
    return out
end

"""
    survey_units(survey_type, Np, va, w, fec, f_roe)

Vulnerable units observed by a survey from numbers `Np` at the time of the survey:
type 1 `Σ Np v`, type 2 `Σ Np v w`, type 3 `Σ Np f · e^{-F_roe}` (spawning biomass net
of spawn-on-kelp removals).
"""
function survey_units(survey_type::Integer, Np, va, w, fec, f_roe)
    survey_type == 1 && return sum(Np .* va)
    survey_type == 2 && return sum(Np .* va .* w)
    survey_type == 3 && return sum(Np .* fec) * exp(-f_roe)
    error("Unknown survey type $survey_type.")
end

"""
    calc_survey_observations(model, log_sel, mort, N)

Relative abundance predictions (calcSurveyObservations). With
`V_j = survey_units(…)` at `N e^{-Z τ_j}` (τ = survey timing), the conditional MLE of
`log q` gives `z_j = log I_j - log V_j`, `ε_j = z_j - z̄`, `q = e^{z̄}`, `Î = qV`
(Ludwig & Walters 1994; Walters & Ludwig 1994). With `q_prior = 2` (random walk in
q) the residuals are the centred first differences of `z`, and `q_j = e^{z_j}`.
"""
function calc_survey_observations(m::StockModel, log_sel, mort, N)
    T = promote_type(eltype(N), eltype(mort.Z))
    roe = findall(==(3), m.data.catch_type)
    epsilon, pit, qt = Vector{T}[], Vector{T}[], Vector{T}[]
    q = zeros(T, length(m.data.surveys))
    for (i, s) in enumerate(m.data.surveys)
        rows = m.survey_rows[i]
        nx = length(rows)
        V = zeros(T, nx)
        for (jj, r) in enumerate(rows)
            t = yrow(m, s.year[r])
            g = s.gear[r]
            Np = N[t, :] .* exp.(-mort.Z[t, :] .* s.timing[r])
            f_roe = isempty(roe) ? zero(T) : sum(mort.ft[roe, t])
            V[jj] = survey_units(s.survey_type, Np, exp.(log_sel[g][t, :]),
                                 m.wt_obs[t, :], m.fec[t, :], f_roe)
        end
        zt = log.(s.index[rows]) .- log.(V)
        zbar = sum(zt) / nx
        q[i] = exp(zbar)
        if m.control.q_prior[i] == 2
            fd = diff(zt)
            eps = zeros(T, nx)
            eps[1:(nx - 1)] .= fd .- sum(fd) / length(fd)
            qi = exp.(zt)
        else
            eps = zt .- zbar
            qi = fill(q[i], nx)
        end
        push!(epsilon, eps)
        push!(qt, qi)
        push!(pit, V .* qi)
    end
    return (epsilon = epsilon, pit = pit, qt = qt, q = q)
end

"""
    calc_stock_recruitment(model, lead, mort, N, log_rt)

Spawning biomass and stock–recruitment residuals (calcStockRecruitment).

- `M̃_a` = mean over years of `M_{t,a}`; `l_a` survivorship under `M̃`;
  `l^w_a = l_a e^{-ρ_s M̃_a}` (ρ_s = spawn fraction); `φ_B = Σ l^w_a f̄_a`;
- `SSB_t = Σ_a N_{t,a} e^{-ρ_s Z_{t,a}} f_{t,a} · Π_{roe k} e^{-F_{k,t}}`,
  `SSB_{nyr+1} = Σ_a N_{nyr+1,a} e^{-M_{nyr,a}} f_{nyr+1,a}`;
- `S̄B₀ = R₀ φ_B`; predicted `R̂_t` from `SSB_{t-sage}` (see [`stock_recruit`](@ref));
- `δ_t = log R_t - log R̂_t + τ²/2`, `t = syr+sage, …, nyr`.
"""
function calc_stock_recruitment(m::StockModel, lead, mort, N, log_rt)
    T = promote_type(eltype(N), typeof(lead.ro), typeof(lead.kappa))
    ny, na, sage = m.ny, m.nages, m.sage
    misc = m.control.misc
    sf = misc.spawn_fraction
    ma = vec(sum(mort.M_tot; dims = 1)) ./ ny
    lx = survivorship(ma, na; plus_group = false)
    lw = lx .* exp.(-ma .* sf)
    lw[na] /= 1 - exp(-ma[na])
    phib = sum(lw .* m.fa_bar)
    roe = findall(==(3), m.data.catch_type)
    sbt = zeros(T, ny + 1)
    for t in 1:ny
        sbt[t] = sum(N[t, :] .* exp.(-mort.Z[t, :] .* sf) .* m.fec[t, :])
        for k in roe
            sbt[t] *= exp(-mort.ft[k, t])
        end
    end
    sbt[ny + 1] = sum(N[ny + 1, :] .* exp.(-mort.M_tot[ny, :]) .* m.fec[ny + 1, :])
    sbo = lead.ro * phib
    so = lead.kappa / phib
    beta = sr_beta(lead.kappa, sbo, misc.rec_type)
    st = sbt[1:(ny - sage)]
    pred = stock_recruit.(st, so, beta, misc.rec_type)
    rt = exp.(log_rt[(sage + 1):ny])
    delta = log.(rt) .- log.(pred) .+ 0.5 * lead.tau^2
    return (sbt = sbt, sbo = sbo, phib = phib, so = so, beta = beta, rt = rt,
            rt_pred = pred, delta = delta)
end
