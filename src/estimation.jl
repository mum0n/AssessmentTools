# ------------------------------------------------------------------------------------
# Phased model fitting and parameter estimation via L-BFGS and automatic differentiation.
# ------------------------------------------------------------------------------------

"""
    FitResult

Container holding results of the phased optimization:
- `x`: final estimated parameter vector
- `f`: final objective function value
- `converged`: boolean convergence flag
- `details`: breakdown of likelihoods, priors, and penalties
- `ref_points`: estimated MSY equilibrium reference points
- `optim_results`: vector of Optim.jl results across phases
"""
struct FitResult
    x::Vector{Float64}
    f::Float64
    converged::Bool
    details::ObjectiveComponents{Float64}
    ref_points::MSYEquilibrium
    optim_results::Vector{Any}
end

"""
    fit(model; max_phase=model.layout.max_phase, iterations=nothing, g_abstol=nothing, verbose=true) -> FitResult

Fit the statistical catch-at-age model using phased estimation:
In each phase `ph = 1:max_phase`, active parameters are updated while inactive
parameters remain fixed at their current estimates. Gradients are computed using
`ForwardDiff`.

Iteration limit and gradient tolerance default to the values declared in the
`[estimation]` section of the model's configuration file, so the solver settings
are user-level configuration rather than choices compiled into this function.
Passing `iterations` or `g_abstol` explicitly overrides the file.
"""
function fit(m::StockModel; max_phase::Integer = m.layout.max_phase,
              iterations::Union{Integer, Nothing} = nothing,
              g_abstol::Union{Real, Nothing} = nothing, verbose::Bool = true)
    ec = m.control.estimation
    iterations = isnothing(iterations) ? ec.max_iterations : iterations
    g_abstol = isnothing(g_abstol) ? ec.g_abstol : g_abstol
    x = copy(m.x0)
    L = m.layout
    optim_runs = Any[]

    for ph in 1:max_phase
        is_last = (ph == max_phase)
        ctx = PhaseContext(ph, is_last)
        act_idx = active_indices(L, ph)

        if isempty(act_idx)
            continue
        end

        if verbose
            println("--- Starting Phase $ph / $max_phase ($(length(act_idx)) active parameters) ---")
        end

        # Objective function with fixed inactive parameters
        function phase_obj(sub_x)
            full_x = copy(x)
            full_x[act_idx] .= sub_x
            return m(full_x, ctx)
        end

        function phase_grad!(G, sub_x)
            ForwardDiff.gradient!(G, phase_obj, sub_x)
        end

        sub_x0 = x[act_idx]
        sub_lb = L.lb[act_idx]
        sub_ub = L.ub[act_idx]

        # Use LBFGS from Optim.jl. Bounds come from the control table via
        # `ParameterLayout.lb` / `.ub`; the prior implementation left a phase's
        # active parameters unconstrained, so estimates such as steepness or
        # log R0 could land outside their bounds.
        opt = Optim.Options(
            iterations = iterations,
            g_abstol = g_abstol,
            show_trace = false
        )

        res = Optim.optimize(phase_obj, phase_grad!, sub_x0,
                             Optim.Fminbox(Optim.LBFGS(), sub_lb, sub_ub), opt)
        x[act_idx] .= Optim.minimizer(res)
        push!(optim_runs, res)

        if verbose
            println("Phase $ph finished. Objective: $(Optim.minimum(res)), Converged: $(Optim.converged(res))")
        end
    end

    final_ctx = PhaseContext(max_phase, true)
    final_f, details = objective_function(m, x, final_ctx)
    p = unpack(L, x)
    lead = leading_parameters(m, p.theta)
    log_sel = calc_selectivities(m, p.sel_par)
    mort = calc_total_mortality(m, log_sel, p.log_ft_pars, p.log_m_nodes, lead.M, final_ctx)
    ref = calc_reference_points(m, mort, log_sel)

    converged = isempty(optim_runs) ? true : Optim.converged(last(optim_runs))
    return FitResult(x, final_f, converged, details, ref, optim_runs)
end
