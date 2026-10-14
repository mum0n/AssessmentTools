# ------------------------------------------------------------------------------------
# ASCII / Text report writer.
# ------------------------------------------------------------------------------------

"""
    write_report(io, model, x)
    write_report(filepath, model, x)

Generate a standard `.rep` text report containing all model estimates, derived biomasses,
mortalities, catch residuals, selectivity schedules, age-composition fits, and MSY
reference points.
"""
function write_report(io::IO, m::StockModel, x::AbstractVector{<:Real})
    p = unpack(m.layout, x)
    ctx = PhaseContext(m.layout.max_phase, true)
    lead = leading_parameters(m, p.theta)
    log_sel = calc_selectivities(m, p.sel_par)
    mort = calc_total_mortality(m, log_sel, p.log_ft_pars, p.log_m_nodes, lead.M, ctx)
    pop = calc_numbers_at_age(m, lead, mort, p.log_rec_devs, p.init_log_rec_devs)
    fish = calc_fishery_observations(m, log_sel, mort, pop.N)
    Ahat = calc_age_proportions(m, fish.Chat)
    surv = calc_survey_observations(m, log_sel, mort, pop.N)
    sr = calc_stock_recruitment(m, lead, mort, pop.N, pop.log_rt)
    f_total, details = objective_function(m, x, ctx)
    ref = calc_reference_points(m, mort, log_sel)

    ny = m.ny
    na = m.nages
    syr = m.syr
    nyr = m.nyr
    sf = m.control.misc.spawn_fraction

    println(io, "f\n", f_total)

    # Negative log-likelihood vector
    println(io, "nlvec")
    # Row 1: catch
    println(io, " ", join(details.nll_catch, " "))
    # Row 2: surveys
    println(io, " ", join(details.nll_survey, " "))
    # Row 3: age compositions
    println(io, " ", join(details.nll_agecomp, " "))
    # Row 4: recruitment
    println(io, " ", details.nll_rec)
    # Row 5: selectivity 2nd diff
    println(io, " ", join(details.nll_sel_diff2, " "))
    # Row 6: selectivity dome
    println(io, " ", join(details.nll_sel_dome, " "))

    println(io, "ro\n", lead.ro)
    println(io, "rbar\n", exp(lead.log_avgrec))
    println(io, "rinit\n", exp(lead.log_recinit))
    println(io, "sbo\n", sr.sbo)
    println(io, "kappa\n", lead.kappa)
    println(io, "steepness\n", lead.h)
    println(io, "m\n", lead.M)
    println(io, "tau\n", lead.tau)
    println(io, "sig\n", lead.sig)
    println(io, "age_tau2\n ", join(details.age_tau2, " "))

    println(io, "ngear\n", ngear(m))
    println(io, "yr\n ", join(syr:nyr, " "))
    println(io, "yrs\n ", join(syr:(nyr + 1), " "))
    println(io, "age\n ", join(m.ages, " "))
    println(io, "la\n ", join(m.la, " "))
    println(io, "wa\n ", join(m.wa, " "))

    println(io, "log_sel")
    for k in 1:ngear(m)
        for i in 1:ny
            println(io, "$k\t", join(log_sel[k][i, :], " "))
        end
    end

    println(io, "obs_ct")
    for k in 1:ngear(m)
        println(io, " ", join(m.data.obs_ct[k, :], " "))
    end

    println(io, "ct")
    for k in 1:ngear(m)
        println(io, " ", join(fish.ct[k, :], " "))
    end

    println(io, "eta")
    for k in 1:ngear(m)
        println(io, " ", join(fish.eta[k, :], " "))
    end

    println(io, "ft")
    for k in 1:ngear(m)
        println(io, " ", join(mort.ft[k, :], " "))
    end

    # bt: total biomass at start of year
    bt = [sum(pop.N[t, :] .* m.wt_obs[t, :]) for t in 1:(ny + 1)]
    println(io, "bt\n ", join(bt, " "))

    # sbt: spawning biomass
    println(io, "sbt\n ", join(sr.sbt, " "))

    println(io, "rectype\n", m.control.misc.rec_type)
    println(io, "rt\n ", join(sr.rt, " "))
    println(io, "ln_rt\n ", join(pop.log_rt, " "))
    println(io, "delta\n ", join(sr.delta, " "))

    println(io, "q\n ", join(surv.q, " "))

    println(io, "it")
    for (k, s) in enumerate(m.data.surveys)
        println(io, " ", join(s.index[m.survey_rows[k]], " "))
    end

    println(io, "pit")
    for p_it in surv.pit
        println(io, " ", join(p_it, " "))
    end

    println(io, "epsilon")
    for eps in surv.epsilon
        println(io, " ", join(eps, " "))
    end

    println(io, "F")
    for t in 1:ny
        println(io, " ", join(mort.F[t, :], " "))
    end

    println(io, "M_tot")
    for t in 1:ny
        println(io, " ", join(mort.M_tot[t, :], " "))
    end

    println(io, "N")
    for t in 1:(ny + 1)
        println(io, " ", join(pop.N[t, :], " "))
    end

    println(io, "bo\n", ref.bo)
    println(io, "fmsy\n ", join(ref.fmsy, " "))
    println(io, "msy\n ", join(ref.msy, " "))
    println(io, "bmsy\n", ref.bmsy)
    println(io, "Umsy\n", ref.umsy)
end

function write_report(path::AbstractString, m::StockModel, x::AbstractVector{<:Real})
    open(path, "w") do f
        write_report(f, m, x)
    end
end

"""
    extract_trajectories(model, x) -> NamedTuple

Extract key simulated or estimated population trajectories (numbers-at-age, biomass,
spawning biomass, fishing mortality, recruitment) evaluated at parameter vector `x`.
"""
function extract_trajectories(m::StockModel, x::AbstractVector{<:Real})
    p = unpack(m.layout, x)
    ctx = PhaseContext(m.layout.max_phase, true)
    lead = leading_parameters(m, p.theta)
    log_sel = calc_selectivities(m, p.sel_par)
    mort = calc_total_mortality(m, log_sel, p.log_ft_pars, p.log_m_nodes, lead.M, ctx)
    pop = calc_numbers_at_age(m, lead, mort, p.log_rec_devs, p.init_log_rec_devs)
    fish = calc_fishery_observations(m, log_sel, mort, pop.N)
    sr = calc_stock_recruitment(m, lead, mort, pop.N, pop.log_rt)
    ref = calc_reference_points(m, mort, log_sel)

    ny = m.ny
    years = collect(m.syr:m.nyr)
    bt = [sum(pop.N[t, :] .* m.wt_obs[t, :]) for t in 1:ny]
    rec = exp.(pop.log_rt[1:ny])

    return (
        year = years,
        biomass = bt,
        spawning_biomass = sr.sbt[1:ny],
        recruitment = rec,
        catch_total = [sum(fish.ct[:, t]) for t in 1:ny],
        f_mean = [mean(mort.ft[:, t]) for t in 1:ny],
        N = pop.N[1:ny, :],
        F = mort.F[1:ny, :],
        ref = ref
    )
end

"""
    write_parquet(filepath, model, x) -> String

Export annual assessment trajectories to Apache Parquet format for high-throughput
columnar analytics.
"""
function write_parquet(filepath::AbstractString, m::StockModel, x::AbstractVector{<:Real})
    traj = extract_trajectories(m, x)
    tbl = (
        year = traj.year,
        biomass = traj.biomass,
        spawning_biomass = traj.spawning_biomass,
        recruitment = traj.recruitment,
        catch_total = traj.catch_total,
        f_mean = traj.f_mean
    )
    Parquet2.writefile(filepath, tbl)
    return filepath
end

"""
    write_zarr(zarr_dir, model, x) -> String

Export multidimensional assessment states (numbers-at-age `N` [year × age], total
fishing mortality `F` [year × age], and annual scalar indicators) to a chunked Zarr
array store.
"""
function write_zarr(zarr_dir::AbstractString, m::StockModel, x::AbstractVector{<:Real})
    traj = extract_trajectories(m, x)
    if isdir(zarr_dir)
        rm(zarr_dir; recursive = true, force = true)
    end
    mkpath(zarr_dir)

    # Save 2D multi-dimensional arrays
    z_n = Zarr.zcreate(Float64, size(traj.N)...; path = joinpath(zarr_dir, "N"))
    z_n[:, :] = traj.N

    z_f = Zarr.zcreate(Float64, size(traj.F)...; path = joinpath(zarr_dir, "F"))
    z_f[:, :] = traj.F

    # Save 1D time series
    z_sb = Zarr.zcreate(Float64, length(traj.spawning_biomass);
                        path = joinpath(zarr_dir, "spawning_biomass"))
    z_sb[:] = traj.spawning_biomass

    z_rec = Zarr.zcreate(Float64, length(traj.recruitment);
                         path = joinpath(zarr_dir, "recruitment"))
    z_rec[:] = traj.recruitment

    z_yr = Zarr.zcreate(Int, length(traj.year); path = joinpath(zarr_dir, "year"))
    z_yr[:] = traj.year

    return zarr_dir
end
