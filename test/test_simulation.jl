using Test
using AssessmentTools
using LinearAlgebra
using Statistics
using TOML
using Random
using AssessmentTools.SimulationEngine
using AssessmentTools.StageMatrix
using AssessmentTools.ContinuousBiomassDynamics
using DynamicPPL, Turing

@testset "Continuous DDE, Continuous Biomass & SimulationEngine" begin
    # DelayDifferential (Continuous Multi-Stage DDE)
    cfg_dde = load_dde_config(normpath(joinpath(@__DIR__, "..", "data", "snowcrab", "dde.toml")))
    @test cfg_dde.num_stages == 6
    @test length(cfg_dde.lags) == 3

    sim_dde = simulate_dde_system(cfg_dde, (0.0, 5.0); dt = 0.25,
                              removals_times = [1.0, 2.0, 3.0, 4.0],
                              removals_amount = [2.0, 2.5, 3.0, 2.0])
    @test size(sim_dde.states, 2) == 6
    @test length(sim_dde.time) > 10
    @test all(sim_dde.states .> 0.0)

    ann_yrs, ann_rec = extract_annual_recruitment(sim_dde; stage_index = 2, target_month = 9.0)
    @test length(ann_yrs) >= 5
    @test all(ann_rec .> 0.0)

    dd_cfg = DelayDifferenceConfig(exp(-0.25), 0.165, 0.70, 0.55)
    dd_res = simulate_delay_difference(ann_yrs, fill(2.0, length(ann_yrs)), ann_rec, dd_cfg;
                                       b0 = 40.0)
    @test length(dd_res.biomass) == length(ann_yrs)
    @test all(dd_res.biomass .> 0.0)
    @test all(dd_res.numbers .> 0.0)

    b_model = bayesian_delay_difference_model(
        fill(2.0, length(ann_yrs)), ann_rec, dd_res.biomass .* 1.0, dd_cfg
    )
    sample_prior = rand(b_model)
    @test haskey(sample_prior, :b0)
    @test haskey(sample_prior, :s)
    @test haskey(sample_prior, :q)
    @test sample_prior[:b0] > 0.0

    s_times = [1.0, 2.0, 3.0, 4.0]
    s_data = [sim_dde.states[findmin(abs.(sim_dde.time .- t))[2], :]' for t in s_times]
    s_mat = vcat(s_data...)
    b_dde = bayesian_delay_differential_model(
        s_times, s_mat, cfg_dde, (0.0, 5.0); dt = 0.5
    )
    sample_dde = rand(b_dde)
    @test haskey(sample_dde, :k_scale)
    @test haskey(sample_dde, :rec_scale)
    @test sample_dde[:k_scale] > 0.0

    # SimulationEngine
    sc = load_stage_matrix_config(normpath(joinpath(@__DIR__, "..", "data", "stage_matrix",
                                                  "config.toml")))
    n = sc.n_stages
    _t(i_) = i_ < n && i_ <= length(sc.transition) ? sc.transition[i_] : 0.0
    pr = projection_parameters(
        n_groups = n,
        weights = sc.stage_weights,
        maturity = sc.maturity,
        fertility = sc.fertility,
        natural_mortality = -log(0.85),
        advance = [_t(i_) for i_ in 1:n],
        retention = [1.0 - _t(i_) for i_ in 1:n],
        initial_numbers = sc.initial_abundance,
        fishing_mortality = zeros(n, 1),
        harvest_rate = fill(0.05, n, 1),
        model_symbol = :stage_matrix)

    for i in 1:n
        @test pr.advance[i] + pr.retention[i] ≈ 1
    end

    st0 = SimulationEngine.ProjectionState(copy(pr.initial_numbers), 0,
                                           sum(pr.initial_numbers .* pr.weights),
                                           0.0, 0.0, 1)
    fwd = SimulationEngine.project_state(pr, st0, direction = :forward, n_years = 8)
    @test length(fwd) == 9
    @test all(s -> length(s.numbers) == n, fwd)
    @test all(isfinite, fwd[end].numbers)

    back = SimulationEngine.reconstruct_backward(pr, fwd[end].numbers, n_years = 8)
    @test isapprox(back[end].numbers, pr.initial_numbers, rtol = 1.0e-6)

    st3 = SimulationEngine.project_from_slice(pr, copy(pr.initial_numbers), 1,
                                              direction = :forward, n_years = 3)
    @test length(st3) == 4

    pop = spawn_population(pr, n_agents_per_group = 4)
    @test isapprox(group_abundance(pop), pr.initial_numbers, rtol = 1.0e-9)
    ag_states, ag_rem = run_agent_projection(pr, 4, n_agents_per_group = 8,
                                             rng = Random.MersenneTwister(7))
    @test length(ag_states) == 5
    @test all(s -> all(x -> x >= 0, s), ag_states)
    @test all(x -> x >= 0, ag_rem)

    net = build_reaction_network(pr)
    @test length(net.species) == n
    @test length(net.reactions) == 1 + (n - 1) + 3 * n
    @test net.reactions[1].name === :recruit

    ss = simulate_ssa(net, 2.0, n_bins = 8, rng = Random.MersenneTwister(11))
    @test size(ss.counts, 1) == n
    @test size(ss.counts, 2) == 9
    @test all(>=(0), ss.counts)
    @test ss.total_harvested >= 0.0

    ss_jump = simulate_ssa_jump(net, 2.0; year_index = 1, n_saves = 8,
                                rng = Random.MersenneTwister(11))
    @test size(ss_jump.counts, 1) == n
    @test size(ss_jump.counts, 2) >= 2
    @test all(>=(0), ss_jump.counts)
    @test all(isfinite, ss_jump.counts)

    pr2 = projection_parameters(
        n_groups = n, weights = sc.stage_weights, maturity = sc.maturity,
        fertility = sc.fertility, natural_mortality = -log(0.85),
        advance = [_t(i_) for i_ in 1:n], retention = [1.0 - _t(i_) for i_ in 1:n],
        initial_numbers = sc.initial_abundance,
        fishing_mortality = zeros(n, 3),
        harvest_rate = fill(0.05, n, 3),
        model_symbol = :stage_matrix)
    f2 = SimulationEngine.project_state(pr2, st0, direction = :forward, n_years = 7)
    @test length(f2) == 8
    @test all(isfinite, f2[end].numbers)

    rng_d = Random.MersenneTwister(4321)
    base_a = pr.advance
    base_r = pr.retention
    draws = map(1:40) do _
        a_raw = abs.(base_a .+ 0.03 .* randn(rng_d, n))
        r_raw = abs.(base_r .+ 0.03 .* randn(rng_d, n))
        tot = a_raw .+ r_raw
        adv = clamp.(a_raw ./ tot, 0.0, 1.0)
        projection_parameters(
            n_groups = n,
            weights = pr.weights .* (1.0 .+ 0.04 .* randn(rng_d, n)),
            maturity = pr.maturity,
            fertility = pr.fertility .* (1.0 .+ 0.15 .* randn(rng_d, n)),
            natural_mortality = -log(0.85) * exp(0.04 * randn(rng_d)),
            advance = adv,
            retention = 1.0 .- adv,
            initial_numbers = pr.initial_numbers,
            fishing_mortality = pr.fishing_mortality,
            harvest_rate = fill(0.05, n, 1),
            model_symbol = :stage_matrix)
    end
    ens = ParameterEnsemble(draws, "test_ensemble")

    mktempdir() do tmp
        z = joinpath(tmp, "ens.zarr")
        saved = SimulationEngine.save_ensemble(z, ens)
        back_ens = SimulationEngine.load_ensemble(saved)

        @test length(back_ens) == length(ens)
        @test back_ens.draws[1].natural_mortality ≈ draws[1].natural_mortality rtol = 1.0e-6
        @test back_ens.draws[1].advance ≈ draws[1].advance rtol = 1.0e-6
        @test back_ens.draws[1].weights ≈ draws[1].weights rtol = 1.0e-6

        head = SimulationEngine.load_ensemble(saved; n_draws = 5)
        @test length(head) == 5

        point_run = SimulationEngine.project_state(pr, st0, direction = :forward,
                                                   n_years = 6)[end].biomass
        resample = map(1:8) do k
            SimulationEngine.project_state(pr, st0, direction = :forward, n_years = 6,
                                          ensemble = back_ens,
                                          mode = :resample_each_step,
                                          rng = Random.MersenneTwister(100 + k))[end].biomass
        end
        single = map(1:8) do k
            SimulationEngine.project_state(pr, st0, direction = :forward, n_years = 6,
                                          ensemble = back_ens,
                                          mode = :single_draw,
                                          rng = Random.MersenneTwister(300 + k))[end].biomass
        end
        mean_run = SimulationEngine.project_state(pr, st0, direction = :forward,
                                                  n_years = 6, ensemble = back_ens,
                                                  mode = :posterior_mean)[end].biomass
        @test all(isfinite, resample)
        @test all(isfinite, single)
        @test mean_run > 0.0
        @test point_run ≈ SimulationEngine.project_state(pr, st0, direction = :forward,
                                                         n_years = 6)[end].biomass

        scfg = joinpath(tmp, "sim.toml")
        write(scfg, "[simulation]\nparameter_combination = 1\nn_draws = 40\n")
        sc_ctl = load_simulation_controls(scfg)
        @test sc_ctl.parameter_combination === :resample_each_step
        @test sc_ctl.n_draws == 40

        write(scfg, "[simulation]\nparameter_combination = 3\nn_draws = 7\n")
        @test load_simulation_controls(scfg).parameter_combination === :posterior_mean
    end

    # ContinuousBiomassDynamics
    cpath = normpath(joinpath(@__DIR__, "..", "data", "continuous_biomass", "config.toml"))
    cfg_c = load_continuous_biomass_config(cpath)
    @test cfg_c.n_hours == 12
    @test length(cfg_c.catches) == cfg_c.n_hours

    out_c = integrate_continuous_biomass(cfg_c)
    @test length(out_c.biomass) == cfg_c.n_hours + 1
    @test out_c.biomass[1] == cfg_c.b0
    @test all(x -> 0.0 <= x <= cfg_c.k, out_c.biomass)
    @test all(isfinite, out_c.biomass)

    c0 = ContinuousBiomassConfig(cfg_c.r, cfg_c.k, cfg_c.b0, 1, 1.0, [0.0], cfg_c.survey_q)
    r1 = integrate_continuous_biomass(c0)
    @test r1.biomass[2] ≈ (cfg_c.b0 + 1.0 * cfg_c.r * cfg_c.b0) /
                          (1.0 + 1.0 * cfg_c.r * cfg_c.b0 / cfg_c.k)

    rp = continuous_msy_reference_points(cfg_c)
    @test rp.bmsy ≈ cfg_c.k / 2
    @test rp.msy ≈ cfg_c.r * cfg_c.k / 4
    @test rp.umsy ≈ 0.5

    tmp_c = mktempdir()
    bad = joinpath(tmp_c, "bad.toml")
    write(bad, "[continuous]\nr = 0.35\nk = 1200.0\nn_hours = 3\ndt = 1.0\n")
    @test_throws Exception load_continuous_biomass_config(bad)

    bad_b0 = joinpath(tmp_c, "bad_b0.toml")
    write(bad_b0, "[continuous]\nr = 0.35\nk = 100.0\nb0 = 250.0\nn_hours = 2\ndt = 1.0\ncatches = [1.0, 1.0]\nsurvey_q = 0.01\n")
    @test_throws Exception load_continuous_biomass_config(bad_b0)

    bad_c = joinpath(tmp_c, "bad_c.toml")
    write(bad_c, "[continuous]\nr = 0.35\nk = 100.0\nb0 = 80.0\nn_hours = 2\ndt = 1.0\ncatches = [1.0, -2.0]\nsurvey_q = 0.01\n")
    @test_throws Exception load_continuous_biomass_config(bad_c)

    bad_r = joinpath(tmp_c, "bad_r.toml")
    write(bad_r, "[continuous]\nr = 12.0\nk = 100.0\nb0 = 80.0\nn_hours = 2\ndt = 1.0\ncatches = [1.0, 1.0]\nsurvey_q = 0.01\n")
    @test_throws Exception load_continuous_biomass_config(bad_r)

    bad_dt = joinpath(tmp_c, "bad_dt.toml")
    write(bad_dt, "[continuous]\nr = 0.35\nk = 100.0\nb0 = 80.0\nn_hours = 2\ndt = 20.0\ncatches = [1.0, 1.0]\nsurvey_q = 0.01\n")
    @test_throws Exception load_continuous_biomass_config(bad_dt)

    traj, obs = simulate_continuous_biomass(cfg_c; n_sims = 40, process_cv = 0.10,
                                            obs_cv = 0.20,
                                            rng = Random.MersenneTwister(99))
    @test size(traj) == (40, cfg_c.n_hours + 1)
    @test all(>=(0), traj)
    @test all(>(0), obs)

    mdl = bayesian_continuous_logistic_model(cfg_c.catches, cfg_c.catches, cfg_c)
    @test mdl isa DynamicPPL.Model
end
