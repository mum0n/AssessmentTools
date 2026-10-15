# ------------------------------------------------------------------------------------
# Objective function components, likelihoods, priors, and penalty terms.
# ------------------------------------------------------------------------------------

"""
    ObjectiveComponents

Container holding individual negative log-likelihoods, priors, penalties, and
total objective value including catch, survey, age composition, recruitment, and
selectivity penalties.
"""
struct ObjectiveComponents{T}
    f::T
    nll_catch::Vector{T}
    nll_survey::Vector{T}
    nll_agecomp::Vector{T}
    nll_rec::T
    nll_sel_diff2::Vector{T}
    nll_sel_dome::Vector{T}
    priors_theta::Vector{T}
    priors_q::Vector{T}
    pen_fbar::T
    pen_mdevs::T
    pen_rdevs::T
    pen_init_rdevs::T
    pen_sel_zero_mean::Vector{T}
    age_tau2::Vector{Float64}
    A_nu::Vector{Matrix{T}}
end

"""
    objective_function(model, x, ctx=PhaseContext(model.layout.max_phase, true)) -> (f, details)

Evaluate the integrated statistical catch-at-age objective function given the full
parameter vector `x` and phase context `ctx`.

Negative log-likelihoods and penalties evaluated:
1. Catch observations: normal error on log catch residuals `log(obs_ct) - log(ct)`.
2. Survey relative abundance indices: normal error on centred residuals `ε`.
3. Age compositions: multivariate logistic or multinomial with bin pooling.
4. Stock-recruitment residuals: normal error on `δ_t` with std `τ = (1-ρ)/ϑ`.
5. Selectivity penalties: 2nd difference curvature and dome-shaped penalties.
6. Selectivity zero-mean constraints for non-parametric and spline options.
7. Priors on leading parameters θ and survey catchabilities q.
8. Regularisation penalties on mean log F, recruitment deviations, and M random walks.
"""
function objective_function(m::StockModel, x::AbstractVector{T}, ctx::PhaseContext) where {T}
    p = unpack(m.layout, x)
    lead = leading_parameters(m, p.theta)
    log_sel = calc_selectivities(m, p.sel_par)
    mort = calc_total_mortality(m, log_sel, p.log_ft_pars, p.log_m_nodes, lead.M, ctx)
    pop = calc_numbers_at_age(m, lead, mort, p.log_rec_devs, p.init_log_rec_devs)
    fish = calc_fishery_observations(m, log_sel, mort, pop.N)
    Ahat = calc_age_proportions(m, fish.Chat)
    surv = calc_survey_observations(m, log_sel, mort, pop.N)
    sr = calc_stock_recruitment(m, lead, mort, pop.N, pop.log_rt)

    ng = ngear(m)
    misc = m.control.misc

    # 1. Catch negative log-likelihood (by gear)
    sig_c = ctx.last_phase ? misc.sig_c_last : misc.sig_c_early
    nll_catch = zeros(T, ng)
    if is_active(m.layout.log_ft_pars.phase[1], ctx)
        for k in 1:ng
            for t in 1:m.ny
                if m.data.obs_ct[k, t] > 0
                    nll_catch[k] += dnorm(fish.eta[k, t], 0.0, sig_c)
                end
            end
        end
    end

    # 2. Survey abundance index negative log-likelihood
    nll_survey = zeros(T, length(m.data.surveys))
    for (k, s) in enumerate(m.data.surveys)
        sig_k = (lead.rho / lead.varphi) ./ s.weight[m.survey_rows[k]]
        nll_survey[k] = dnorm(surv.epsilon[k], sig_k)
    end

    # 3. Age-composition likelihood
    nll_agecomp = zeros(T, length(m.data.agecomps))
    age_tau2 = zeros(Float64, length(m.data.agecomps))
    A_nu = Vector{Matrix{T}}(undef, length(m.data.agecomps))
    for (k, ac) in enumerate(m.data.agecomps)
        rows = m.agecomp_rows[k]
        if !isempty(rows)
            O = ac.obs[rows, :]
            P = Ahat[k]
            if misc.agecomp_like == 1
                nll_agecomp[k], A_nu[k], age_tau2[k] = dmvlogistic(O, P, misc.min_prop)
            else
                nll_agecomp[k], A_nu[k], age_tau2[k] =
                    dmultinom_pooled(O, P, misc.min_prop;
                                     proc_err = m.control.penalties.agecomp_process_error)
            end
        else
            A_nu[k] = zeros(T, 0, size(ac.obs, 2))
        end
    end

    # 4. Stock-recruitment relationship
    nll_rec = zero(T)
    if is_active(m.layout.theta.phase[1], ctx)
        nll_rec = dnorm(sr.delta, lead.tau)
    end

    # 5. Selectivity smoothness and dome penalties
    nll_sel_diff2 = zeros(T, ng)
    nll_sel_dome = zeros(T, ng)
    for k in 1:ng
        t = m.control.isel_type[k]
        if is_active(m.control.sel_phz[k], ctx) && !(t in (1, 6, 7, 8, 11))
            w2 = m.control.sel_2nd_diff_wt[k] / m.nages
            wdome = m.control.sel_dome_wt[k]
            for i in 1:m.ny
                sel_row = log_sel[k][i, :]
                df2 = diff(diff(sel_row))
                nll_sel_diff2[k] += w2 * sum(abs2, df2)
                for j in 1:(m.nages - 1)
                    if sel_row[j] > sel_row[j + 1]
                        nll_sel_dome[k] += wdome * abs2(sel_row[j] - sel_row[j + 1])
                    end
                end
            end
        end
    end

    # 6. Zero-mean penalties for selectivity parameters
    pen_sel_zero_mean = zeros(T, ng)
    w_zero = m.control.penalties.sel_zero_mean_weight
    for k in 1:ng
        t = m.control.isel_type[k]
        if is_active(m.control.sel_phz[k], ctx) && !(t in (1, 6, 7, 8, 11))
            P = p.sel_par[k]
            if t == 5
                # Bicubic: ensure columns sum to zero
                for col in eachcol(P)
                    pen_sel_zero_mean[k] += w_zero * abs2(sum(col) / length(col))
                end
            elseif t in (3, 4, 12)
                # Ensure rows sum to zero
                for row in eachrow(P)
                    pen_sel_zero_mean[k] += w_zero * abs2(sum(row) / length(row))
                end
            end
        end
    end

    # 7. Priors for parameters from control file
    priors_theta = zeros(T, 7)
    tc = m.control.theta_control
    for i in 1:7
        if is_active(m.layout.theta.phase[i], ctx)
            val = p.theta[i]
            ptype = Int(tc[i, 5])
            p1 = tc[i, 6]
            p2 = tc[i, 7]
            if ptype == 1
                priors_theta[i] = dnorm(val, p1, p2)
            elseif ptype == 2
                priors_theta[i] = dlnorm(val, p1, p2)
            elseif ptype == 3
                lb = tc[i, 2]
                ub = tc[i, 3]
                priors_theta[i] = dbeta((val - lb) / (ub - lb), p1, p2)
            elseif ptype == 4
                priors_theta[i] = dgamma(val, p1, p2)
            else
                priors_theta[i] = 1.0 / (tc[i, 3] - tc[i, 2])
            end
        end
    end

    # Survey q priors
    priors_q = zeros(T, length(m.data.surveys))
    for (i, qval) in enumerate(surv.q)
        if m.control.q_prior[i] == 1
            priors_q[i] = dnorm(log(qval), m.control.q_mu[i], m.control.q_sd[i])
        end
    end

    # 8. Regularisation penalties
    pen_fbar = zero(T)
    if !isempty(p.log_ft_pars)
        log_fbar = sum(p.log_ft_pars) / length(p.log_ft_pars)
        sd_fbar = ctx.last_phase ? misc.fbar_sd_last : misc.fbar_sd_early
        pen_fbar = dnorm(log_fbar, log(misc.fbar), sd_fbar)
    end

    pen_rdevs = zero(T)
    pen_init_rdevs = zero(T)
    if is_active(m.layout.log_rec_devs.phase[1], ctx)
        pen_cfg = m.control.penalties
        if ctx.last_phase
            pen_rdevs = dnorm(p.log_rec_devs, pen_cfg.rec_dev_sd)
            pen_init_rdevs = dnorm(p.init_log_rec_devs, pen_cfg.rec_dev_sd)
        else
            pen_rdevs = pen_cfg.rec_dev_penalty_weight * sum(abs2, p.log_rec_devs)
            pen_init_rdevs = pen_cfg.rec_dev_penalty_weight * sum(abs2, p.init_log_rec_devs)
        end
    end

    pen_mdevs = zero(T)
    if misc.n_m_nodes > 0 && is_active(misc.m_dev_phase, ctx)
        fd_m = diff(mort.log_m_devs[2:end])
        w_mnode = m.control.penalties.m_dev_node_weight
        pen_mdevs = dnorm(fd_m, misc.m_dev_sd) + w_mnode * sum(abs2, p.log_m_nodes)
    end

    f_total = sum(nll_catch) + sum(nll_survey) + sum(nll_agecomp) + nll_rec +
              sum(nll_sel_diff2) + sum(nll_sel_dome) + sum(pen_sel_zero_mean) +
              sum(priors_theta) + sum(priors_q) + pen_fbar + pen_rdevs + pen_init_rdevs +
              pen_mdevs

    details = ObjectiveComponents(
        f_total, nll_catch, nll_survey, nll_agecomp, nll_rec,
        nll_sel_diff2, nll_sel_dome, priors_theta, priors_q,
        pen_fbar, pen_mdevs, pen_rdevs, pen_init_rdevs, pen_sel_zero_mean,
        age_tau2, A_nu
    )

    return f_total, details
end

(m::StockModel)(x::AbstractVector, ctx::PhaseContext = PhaseContext(m.layout.max_phase, true)) =
    first(objective_function(m, x, ctx))
