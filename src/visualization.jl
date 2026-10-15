# ------------------------------------------------------------------------------------
# Visualization tools for State Variable Trajectories & Retrospective Patterns
# Native Julia terminal and notebook plotting via UnicodePlots
# ------------------------------------------------------------------------------------

"""
    plot_trajectory(times, values; title = "State Variable Trajectory",
                    xlabel = "Time / Year", ylabel = "Value", color = :cyan)

Generate a high-contrast time-series plot of a population state variable.
Returns a `UnicodePlots.Plot` object that renders directly in terminals and Pluto notebooks.
"""
function plot_trajectory(times::AbstractVector,
                         values::AbstractVector{<:Real};
                         title::AbstractString = "State Variable Trajectory",
                         xlabel::AbstractString = "Time / Year",
                         ylabel::AbstractString = "Value",
                         color::Symbol = :cyan)
    length(times) == length(values) || error("times and values lengths must match.")
    p = lineplot(times, values; title = title, xlabel = xlabel, ylabel = ylabel,
                 color = color, canvas = DotCanvas, width = 60, height = 15)
    return p
end

"""
    plot_trajectory!(plot_obj, times, values; name = "", color = :red)

Overlay an additional time series onto an existing `UnicodePlots.Plot`.
"""
function plot_trajectory!(p::Plot,
                          times::AbstractVector,
                          values::AbstractVector{<:Real};
                          name::AbstractString = "",
                          color::Symbol = :red)
    lineplot!(p, times, values; name = name, color = color)
    return p
end

"""
    plot_state_variables(model_result) -> Plot

Dispatch automated multi-trajectory visualization for assessment results:
- `LogisticDiscreteResult`: Displays exploitable biomass and catch removals.
- `VpaResult`: Displays total and spawning biomass.
- `DdeModelConfig` / DDE simulation: Displays multi-stage numerical densities.
- `MseResult`: Displays true operating biomass, observed biomass, and catches.
"""
function plot_state_variables(res::LogisticDiscreteResult)
    p = lineplot(res.years, res.biomass; name = "Biomass", title = "Biomass Dynamics Trajectory",
                 xlabel = "Year", ylabel = "Biomass / Removals", color = :green,
                 width = 65, height = 15)
    lineplot!(p, res.years, res.removals; name = "Removals", color = :red)
    return p
end

function plot_state_variables(res::VpaResult; years = 1:size(res.N, 2))
    p = lineplot(collect(years), res.biomass_total; name = "Total Biomass",
                 title = "VPA Biomass Trajectory", xlabel = "Year",
                 ylabel = "Abundance / Biomass", color = :blue, width = 65, height = 15)
    lineplot!(p, collect(years), res.biomass_ssb; name = "SSB", color = :magenta)
    return p
end

function plot_state_variables(res::AdaptResult; years = 1:size(res.vpa.N, 2))
    p = lineplot(collect(years), res.vpa.biomass_total; name = "Total Biomass",
                 title = "ADAPT VPA Calibrated Trajectory", xlabel = "Year",
                 ylabel = "Abundance / Biomass", color = :blue, width = 65, height = 15)
    lineplot!(p, collect(years), res.vpa.biomass_ssb; name = "SSB", color = :magenta)
    return p
end

function plot_state_variables(res::DelayDifferenceResult)
    p = lineplot(res.years, res.biomass; name = "Biomass",
                 title = "Delay-Difference Trajectory", xlabel = "Year",
                 ylabel = "Biomass / Removals", color = :green, width = 65, height = 15)
    lineplot!(p, res.years, res.catches_biomass; name = "Removals", color = :red)
    lineplot!(p, res.years, res.recruits_biomass; name = "Recruitment (Wt)", color = :cyan)
    return p
end

function plot_state_variables(res::MseResult)
    p = lineplot(res.years, res.true_biomass; name = "True Biomass",
                 title = "MSE Feedback Simulation", xlabel = "Year",
                 ylabel = "Biomass / Catch", color = :cyan, width = 65, height = 15)
    lineplot!(p, res.years, res.observed_biomass; name = "Observed Biomass", color = :yellow)
    lineplot!(p, res.years, res.catches; name = "Catch Removals", color = :red)
    return p
end

"""
    plot_fishery_predictions(time, simulations; obs_time = nothing, obs_values = nothing,
                             title = "Fishery Predictions Trace", xlabel = "Year", ylabel = "Biomass / Index") -> Plot

Incorporate `model_fishery` ensemble trace visualization:
- Displays mean predicted trajectory overlaid with transparent/thin individual simulation realizations.
- Optionally displays observed survey abundance points.
"""
function plot_fishery_predictions(times::AbstractVector,
                                  sims::AbstractMatrix{<:Real};
                                  obs_time = nothing,
                                  obs_values = nothing,
                                  title::AbstractString = "Fishery Predictions Trace",
                                  xlabel::AbstractString = "Year",
                                  ylabel::AbstractString = "Biomass / Index")
    nt, nsim = size(sims)
    nt == length(times) || error("rows of sims must match times length.")

    mean_traj = vec(mean(sims, dims = 2))
    p = lineplot(times, mean_traj; name = "Mean Simulation", title = title,
                 xlabel = xlabel, ylabel = ylabel, color = :red, width = 65, height = 15)

    # Plot sample traces
    n_sample = min(nsim, 5)
    sample_indices = rand(1:nsim, n_sample)
    colors = [:yellow, :cyan, :magenta, :green, :blue]
    for (i, s_idx) in enumerate(sample_indices)
        col = colors[(i - 1) % length(colors) + 1]
        lineplot!(p, times, sims[:, s_idx]; name = "Sim $s_idx", color = col)
    end

    if !isnothing(obs_time) && !isnothing(obs_values)
        scatterplot!(p, obs_time, obs_values; name = "Observed Survey", color = :white)
    end

    return p
end

"""
    plot_fishery_footprint(times, biomass_fished, biomass_unfished; title = "Fishery Depletion Footprint") -> Plot

Incorporate `model_fishery` fishery footprint metric:
```math
\\text{Footprint}_t = \\frac{B^{\\text{unfished}}_t - B^{\\text{fished}}_t}{B^{\\text{unfished}}_t}
```
"""
function plot_fishery_footprint(times::AbstractVector,
                                b_fished::AbstractVector{<:Real},
                                b_unfished::AbstractVector{<:Real};
                                title::AbstractString = "Fishery Depletion Footprint")
    length(times) == length(b_fished) == length(b_unfished) ||
        error("times and biomass vectors must have identical lengths.")

    footprint = zeros(Float64, length(times))
    for t in eachindex(times)
        if b_unfished[t] > 0.0
            footprint[t] = max(0.0, (b_unfished[t] - b_fished[t]) / b_unfished[t])
        end
    end

    p = lineplot(times, footprint; name = "Footprint (Depletion Fraction)", title = title,
                 xlabel = "Year", ylabel = "Relative Footprint (0 - 1)", color = :magenta,
                 ylim = (0.0, 1.0), width = 65, height = 15)
    return p
end

"""
    plot_retrospective(retro_res::RetrospectiveResult; target = :biomass) -> Plot

Plot retrospective peel trajectories showing historical consistency or systematic bias.
Includes terminal year peel points and reports Mohn's rho.
"""
function plot_retrospective(retro::RetrospectiveResult; target::Symbol = :biomass)
    trajs = target == :biomass ? retro.biomass_trajectories : retro.fishing_mortalities
    rho_val = target == :biomass ? retro.rho_biomass : retro.rho_f
    ylabel_str = target == :biomass ? "Biomass" : "Fishing Mortality (F)"
    title_str = "Retrospective Peels (Mohn's ρ = $(round(rho_val, digits=3)))"

    # Base plot (peel 0)
    p = lineplot(retro.peel_years[1], trajs[1]; name = retro.peel_labels[1],
                 title = title_str, xlabel = "Year", ylabel = ylabel_str,
                 color = :white, width = 65, height = 15)

    colors = [:yellow, :cyan, :magenta, :red, :blue, :green]
    for i in 2:length(retro.peel_labels)
        col = colors[(i - 1) % length(colors) + 1]
        lineplot!(p, retro.peel_years[i], trajs[i]; name = retro.peel_labels[i], color = col)
    end

    return p
end
