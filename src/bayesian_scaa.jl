# ------------------------------------------------------------------------------------
# Bayesian Statistical Catch-at-Age (SCAA) in Turing.jl
# Provides full MCMC and HMC inference over the integrated SCAA forward simulation.
# ------------------------------------------------------------------------------------

"""
    bayesian_statistical_catch_at_age(model;
                                      prior_log_ro_sd = 2.0,
                                      prior_h_sd = 0.15,
                                      prior_m_sd = 0.10,
                                      prior_rec_sd = 0.60,
                                      prior_ft_sd = 0.80)

Bayesian state-space formulation of the statistical catch-at-age forward model:
- Samples primary population parameters ``\\log(R_0), h, M, \\bar{F}``.
- Integrates annual log recruitment deviations ``\\delta_t \\sim \\mathrm{Normal}(0, \\tau)``.
- Samples gear-specific fishing mortality deviations and selectivity parameters.
- Evaluates the joint posterior over commercial catch observations, relative survey indices,
  age composition multinomial/logistic likelihoods, and stock-recruitment relationships.

# Arguments
- `model`: A configured [`StockModel`](@ref) containing observed survey, catch, and age-comp data.
- `prior_log_ro_sd`: Standard deviation for prior on log unfished recruitment ``\\log(R_0)``.
- `prior_h_sd`: Standard deviation for steepness ``h`` centered at control initial value.
- `prior_m_sd`: Standard deviation for natural mortality ``M``.
- `prior_rec_sd`: Standard deviation for annual recruitment deviations ``\\delta_t``.
- `prior_ft_sd`: Standard deviation for log fishing mortality deviations.
"""
@model function bayesian_statistical_catch_at_age(model::StockModel;
                                                  prior_log_ro_sd::Real = 2.0,
                                                  prior_h_sd::Real = 0.15,
                                                  prior_m_sd::Real = 0.10,
                                                  prior_rec_sd::Real = 0.60,
                                                  prior_ft_sd::Real = 0.80)
    L = model.layout
    tc = model.control.theta_control

    # Priors on leading parameters θ from model.control.theta_control
    # Columns of theta_control: [initial, lower, upper, phase, prior_type, p1, p2]
    # Prior types:
    # 0 = Uniform(lower, upper)
    # 1 = Normal(p1, p2)
    # 2 = LogNormal(p1, p2)
    # 3 = Beta(p1, p2) scaled to [lower, upper] via LocationScale
    # 4 = Gamma(p1, p2)
    # θ[1]: log(R0)
    log_ro ~ tc[1, 5] == 1.0 ? Normal(tc[1, 6], tc[1, 7]) :
             tc[1, 5] == 2.0 ? LogNormal(tc[1, 6], tc[1, 7]) :
             tc[1, 5] == 4.0 ? Gamma(tc[1, 6], tc[1, 7]) :
             Uniform(tc[1, 2], tc[1, 3])

    # θ[2]: steepness h
    h ~ tc[2, 5] == 3.0 ? LocationScale(tc[2, 2], tc[2, 3] - tc[2, 2], Beta(tc[2, 6], tc[2, 7])) :
        tc[2, 5] == 1.0 ? truncated(Normal(tc[2, 6], tc[2, 7]), tc[2, 2], tc[2, 3]) :
        Uniform(tc[2, 2], tc[2, 3])

    # θ[3]: natural mortality M (log M in parameter layout, exposed as m_nat)
    m_nat ~ tc[3, 5] == 1.0 ? Normal(tc[3, 6], tc[3, 7]) :
            tc[3, 5] == 2.0 ? LogNormal(tc[3, 6], tc[3, 7]) :
            Uniform(tc[3, 2], tc[3, 3])

    # θ[4]: log(R_bar)
    log_rbar ~ tc[4, 5] == 1.0 ? Normal(tc[4, 6], tc[4, 7]) :
               Uniform(tc[4, 2], tc[4, 3])

    # θ[5]: log(R_init)
    log_rinit ~ tc[5, 5] == 1.0 ? Normal(tc[5, 6], tc[5, 7]) :
                Uniform(tc[5, 2], tc[5, 3])

    # θ[6]: rho (variance partition)
    rho ~ tc[6, 5] == 3.0 ? LocationScale(tc[6, 2], tc[6, 3] - tc[6, 2], Beta(tc[6, 6], tc[6, 7])) :
          tc[6, 5] == 1.0 ? truncated(Normal(tc[6, 6], tc[6, 7]), tc[6, 2], tc[6, 3]) :
          Uniform(tc[6, 2], tc[6, 3])

    # θ[7]: varphi (total precision)
    varphi ~ tc[7, 5] == 4.0 ? Gamma(tc[7, 6], tc[7, 7]) :
             tc[7, 5] == 1.0 ? Normal(tc[7, 6], tc[7, 7]) :
             Uniform(tc[7, 2], tc[7, 3])

    theta = [log_ro, h, m_nat, log_rbar, log_rinit, rho, varphi]

    # Selectivity parameters
    n_sel = length(L.sel_par)
    sel_pars = Vector{Matrix{Float64}}(undef, n_sel)
    for k in 1:n_sel
        b = L.sel_par[k]
        r, c = b.dims
        if c == 0
            sel_pars[k] = zeros(Float64, r, c)
        else
            p_mat = zeros(Float64, r, c)
            for j in 1:c
                for i in 1:r
                    p_mat[i, j] ~ Normal(0.0, 2.0)
                end
            end
            sel_pars[k] = p_mat
        end
    end

    # Fishing mortality parameters
    n_ft = length(L.log_ft_pars.range)
    log_ft = Vector{Float64}(undef, n_ft)
    for i in 1:n_ft
        log_ft[i] ~ Normal(log(model.control.misc.fbar), prior_ft_sd)
    end

    # Recruitment deviations
    n_init_rdev = length(L.init_log_rec_devs.range)
    init_rdevs = Vector{Float64}(undef, n_init_rdev)
    for i in 1:n_init_rdev
        init_rdevs[i] ~ Normal(0.0, prior_rec_sd)
    end

    n_rdev = length(L.log_rec_devs.range)
    rdevs = Vector{Float64}(undef, n_rdev)
    for i in 1:n_rdev
        rdevs[i] ~ Normal(0.0, prior_rec_sd)
    end

    # Natural mortality random walk nodes
    n_mnod = length(L.log_m_nodes.range)
    m_nodes = Vector{Float64}(undef, n_mnod)
    for i in 1:n_mnod
        m_nodes[i] ~ Normal(0.0, 0.5)
    end

    # Assemble full parameter vector x via assemble_x
    parts = (
        theta = theta,
        sel_par = sel_pars,
        log_ft_pars = log_ft,
        init_log_rec_devs = init_rdevs,
        log_rec_devs = rdevs,
        log_m_nodes = m_nodes
    )
    x = assemble_x(L, parts)

    # Evaluate integrated forward population dynamics
    ctx = PhaseContext(L.max_phase, true)
    p = unpack(L, x)
    lead = leading_parameters(model, p.theta)
    log_sel = calc_selectivities(model, p.sel_par)
    mort = calc_total_mortality(model, log_sel, p.log_ft_pars, p.log_m_nodes, lead.M, ctx)
    pop = calc_numbers_at_age(model, lead, mort, p.log_rec_devs, p.init_log_rec_devs)
    fish = calc_fishery_observations(model, log_sel, mort, pop.N)
    Ahat = calc_age_proportions(model, fish.Chat)
    surv = calc_survey_observations(model, log_sel, mort, pop.N)
    sr = calc_stock_recruitment(model, lead, mort, pop.N, pop.log_rt)

    ng = ngear(model)
    misc = model.control.misc

    # Catch likelihood (lognormal / normal on log catch)
    sig_c = misc.sig_c_last
    for k in 1:ng
        for t in 1:model.ny
            obs_c = model.data.obs_ct[k, t]
            if obs_c > 0.0
                obs_log_c = log(obs_c)
                pred_log_c = log(max(1e-4, fish.ct[k, t]))
                obs_log_c ~ Normal(pred_log_c, sig_c)
            end
        end
    end

    # Survey abundance index likelihood
    for (k, s) in enumerate(model.data.surveys)
        sig_k = (lead.rho / lead.varphi) ./ s.weight[model.survey_rows[k]]
        rows = model.survey_rows[k]
        for (idx, t) in enumerate(rows)
            obs_it = s.index[idx]
            if obs_it > 0.0
                obs_log_i = log(obs_it)
                pred_log_i = log(max(1e-4, surv.pit[k][idx]))
                obs_log_i ~ Normal(pred_log_i, sig_k[idx])
            end
        end
    end

    # Age-composition likelihood
    #
    # `Ahat` holds the predicted proportions for each age-composition dataset. The
    # likelihood is the same one the point-estimate objective minimises: multinomial
    # with pooled bins, or multivariate-logistic, selected by `misc.agecomp_like`.
    # The shared routines are called so the Bayesian and point estimates use one
    # formulation; the negative log-likelihood they return is added to the posterior
    # with a sign change because Turing accumulates log posterior.
    for (k, ac) in enumerate(model.data.agecomps)
        rows = model.agecomp_rows[k]
        isempty(rows) && continue
        O = ac.obs[rows, :]
        P = Ahat[k]
        nll = if misc.agecomp_like == 1
            dmvlogistic(O, P, misc.min_prop)[1]
        else
            dmultinom_pooled(O, P, misc.min_prop;
                             proc_err = model.control.penalties.agecomp_process_error)[1]
        end
        @addlogprob!(-nll)
    end

    # Stock-recruitment deviation likelihood
    tau = lead.tau
    sr_deltas = sr.delta
    for t in eachindex(sr_deltas)
        sr_delta_t = sr_deltas[t]
        sr_delta_t ~ Normal(0.0, tau)
    end

    return (theta = theta, log_ft = log_ft, rdevs = rdevs, ssb = sr.sbt, N = pop.N)
end
