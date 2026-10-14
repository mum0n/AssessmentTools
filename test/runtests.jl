using Test
using AssessmentTools
using LinearAlgebra
using TOML
using UnicodePlots

@testset "AssessmentTools Package Tests" begin

    @testset "Spline and Statistics primitives" begin
        x = [0.0, 0.25, 0.5, 0.75, 1.0]
        y = [0.0, 0.5, 1.0, 0.5, 0.0]
        s = AssessmentTools.StatisticalCatchAtAge.CubicSpline(x, y)
        @test isapprox(s(0.0), 0.0, atol=1e-5)
        @test isapprox(s(0.5), 1.0, atol=1e-5)
        @test isapprox(s(1.0), 0.0, atol=1e-5)

        # Logistic
        @test isapprox(AssessmentTools.plogis(5.0, 5.0, 1.0), 0.5)
        @test AssessmentTools.plogis(10.0, 5.0, 1.0) > 0.99
    end

    @testset "Silver Hake Assessment (TOML/CSV)" begin
        data_dir = normpath(joinpath(@__DIR__, "..", "data", "silverhake"))
        cfg_file = joinpath(data_dir, "config.toml")
        par_file = joinpath(data_dir, "parameters.toml")

        @test isfile(cfg_file)
        @test isfile(par_file)

        # Construct model directly from modular dataset directory
        model = StockModel(data_dir)
        x_sol = load_parameters(par_file)
        @test length(x_sol) == model.layout.n

        # Evaluate objective function at solution
        f_val, details = objective_function(model, x_sol, PhaseContext(model.layout.max_phase, true))

        println("Silver Hake Objective: $f_val")
        # Verify close agreement with expected benchmark objective
        @test isapprox(f_val, 280.385, rtol = 0.01)

        # Test reference points calculation
        p = AssessmentTools.StatisticalCatchAtAge.unpack(model.layout, x_sol)
        ctx = PhaseContext(model.layout.max_phase, true)
        lead = AssessmentTools.StatisticalCatchAtAge.leading_parameters(model, p.theta)
        log_sel = calc_selectivities(model, p.sel_par)
        mort = calc_total_mortality(model, log_sel, p.log_ft_pars, p.log_m_nodes, lead.M, ctx)
        ref = calc_reference_points(model, mort, log_sel)

        @test ref.bo > 0
        @test ref.bmsy > 0
        @test all(ref.fmsy .> 0)

        # Test decision table evaluation
        dt = decision_table(model, x_sol)
        @test length(dt) == length(model.projection.tac)

        # Verify load_dataset from CSV/TOML
        d_toml, c_toml, p_toml = load_dataset(data_dir)
        @test d_toml.obs_ct == model.data.obs_ct
        @test length(d_toml.surveys) == length(model.data.surveys)
        @test length(d_toml.agecomps) == length(model.data.agecomps)
    end

    @testset "Cusk Assessment (TOML/CSV)" begin
        data_dir = normpath(joinpath(@__DIR__, "..", "data", "cusk"))
        cfg_file = joinpath(data_dir, "config.toml")
        par_file = joinpath(data_dir, "parameters.toml")

        @test isfile(cfg_file)
        @test isfile(par_file)

        # Construct Cusk model directly from dataset directory
        model_cusk = StockModel(data_dir)
        x_sol_cusk = load_parameters(par_file)
        @test length(x_sol_cusk) == model_cusk.layout.n

        # Test reference points from benchmark solution
        p_c = AssessmentTools.StatisticalCatchAtAge.unpack(model_cusk.layout, x_sol_cusk)
        ctx_c = PhaseContext(model_cusk.layout.max_phase, true)
        lead_c = AssessmentTools.StatisticalCatchAtAge.leading_parameters(model_cusk, p_c.theta)
        log_sel_c = calc_selectivities(model_cusk, p_c.sel_par)
        mort_c = calc_total_mortality(model_cusk, log_sel_c, p_c.log_ft_pars,
                                      p_c.log_m_nodes, lead_c.M, ctx_c)
        ref_cusk = calc_reference_points(model_cusk, mort_c, log_sel_c, p_c.theta)

        ref_toml = TOML.parsefile(par_file)["reference_points"]
        @test isapprox(ref_cusk.bo, ref_toml["bo"], rtol = 1e-4)
        @test isapprox(ref_cusk.bmsy, ref_toml["bmsy"], rtol = 1e-4)
        @test isapprox(sum(ref_cusk.msy), ref_toml["msy"], rtol = 1e-4)
        @test isapprox(ref_cusk.fmsy[1], ref_toml["fmsy"], rtol = 1e-4)

        # Verify load_dataset from CSV/TOML
        d_toml, c_toml, _ = load_dataset(data_dir)
        @test d_toml.obs_ct == model_cusk.data.obs_ct
        @test length(d_toml.surveys) == length(model_cusk.data.surveys)
    end

    @testset "Empirical Assessment & Management Methods" begin
        # 1. Chapman-Robson & Catch Curves
        ages = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        # Exponential decline with Z = 0.5 (S ≈ 0.6065)
        true_s = exp(-0.5)
        catches = [1000.0 * (true_s^i) for i in 0:7]

        cr = chapman_robson(ages, catches)
        @test isapprox(cr.s, true_s, atol = 0.05)
        @test isapprox(cr.z, 0.5, atol = 0.05)
        @test cr.se_z > 0.0

        cc = catch_curve_linear(ages, catches)
        @test isapprox(cc.z, 0.5, atol = 0.01)
        @test isapprox(cc.r2, 1.0, atol = 1e-4)

        # Bayesian catch curve model check
        bcc = bayesian_catch_curve(ages, catches)
        s_bcc = rand(bcc)
        @test haskey(s_bcc, :z)
        @test haskey(s_bcc, :alpha)
        @test s_bcc[:z] > 0.0

        # 2. Leslie & DeLury Depletion
        # Continuous removal or discrete Leslie model where CPUE_t = q * (N0 - K_t)
        # Using midpoint cumulative catch K_t = sum_{i<t} C_i + 0.5 C_t:
        n0_true = 10000.0
        q_true = 0.001
        efforts = [100.0, 100.0, 100.0, 100.0, 100.0]
        # Exact Leslie linear relation: C_t = q * f_t * (N0 - K_t)
        # C_t = q * f_t * (N_t - 0.5 C_t) => C_t (1 + 0.5 q f_t) = q f_t N_t
        catches_dep = Float64[]
        curr_n = n0_true
        for f in efforts
            c = (q_true * f * curr_n) / (1.0 + 0.5 * q_true * f)
            push!(catches_dep, c)
            curr_n -= c
        end

        leslie = leslie_depletion(catches_dep, efforts)
        @test isapprox(leslie.n0, n0_true, rtol = 0.01)
        @test isapprox(leslie.q, q_true, rtol = 0.01)
        @test leslie.r2 > 0.999

        delury = delury_depletion(catches_dep, efforts)
        @test isapprox(delury.n0, n0_true, rtol = 0.06)
        @test isapprox(delury.q, q_true, rtol = 0.06)

        # Bayesian Depletion regressions
        b_leslie = bayesian_leslie_depletion(catches_dep, efforts)
        s_bleslie = rand(b_leslie)
        @test haskey(s_bleslie, :n0)
        @test haskey(s_bleslie, :q)
        @test s_bleslie[:n0] > 0.0

        b_delury = bayesian_delury_depletion(catches_dep, efforts)
        s_bdelury = rand(b_delury)
        @test haskey(s_bdelury, :n0)
        @test haskey(s_bdelury, :q)
        @test s_bdelury[:n0] > 0.0

        # 3. Age-Length Key
        # 3 lengths, 3 ages
        raw_alk = [
            50.0  10.0   0.0;
            10.0  60.0  10.0;
             0.0  10.0  50.0
        ]
        len_freq = [100.0, 200.0, 100.0]
        alk_res = apply_age_length_key(raw_alk, len_freq)
        @test length(alk_res.age_proportions) == 3
        @test isapprox(sum(alk_res.age_proportions), 1.0, atol = 1e-6)
        @test isapprox(sum(alk_res.age_numbers), sum(len_freq), atol = 1e-6)

        # 4. Harvest Control Rule (Hockey Stick)
        hcr = HockeyStickHCR(1000.0, 2500.0, 0.05, 0.30)
        @test eval_hcr(hcr, 500.0) == 0.05
        @test eval_hcr(hcr, 1000.0) == 0.05
        @test eval_hcr(hcr, 3000.0) == 0.30
        @test isapprox(eval_hcr(hcr, 1750.0), 0.175, atol = 1e-6)

        # TAC Stabilizer
        @test isapprox(apply_tac_stabilizer(80.0, 100.0; max_decrease = 0.15), 85.0)
        @test isapprox(apply_tac_stabilizer(130.0, 100.0; max_increase = 0.15), 115.0)
        @test isapprox(apply_tac_stabilizer(105.0, 100.0; max_increase = 0.15), 105.0)

        # 5. Management Strategy Evaluation (MSE) & Bayesian Feedback
        mse_years = collect(2020:2030)
        mse_hcr = HockeyStickHCR(20.0, 50.0, 0.05, 0.25)
        mse_res = simulate_mse(mse_years, mse_hcr; r = 0.40, k = 100.0, b0 = 80.0)
        @test length(mse_res.true_biomass) == length(mse_years)
        @test all(mse_res.true_biomass .> 0.0)
        @test all(mse_res.catches .>= 0.0)

        b_mse = bayesian_mse_feedback(mse_res.observed_biomass, mse_res.catches, mse_hcr)
        s_bmse = rand(b_mse)
        @test haskey(s_bmse, :r)
        @test haskey(s_bmse, :k)
        @test s_bmse[:k] > 0.0
    end

    @testset "BiomassDynamics (Discrete Logistic Surplus Production)" begin
        years = collect(2000:2010)
        removals = fill(10.0, length(years))
        r_true = 0.4
        k_true = 100.0

        # Step check
        b_next = logistic_step(50.0, r_true, k_true, 10.0)
        @test b_next > 0.0

        # Simulation check
        res = simulate_logistic_discrete(years, removals; r = r_true, k = k_true, b0 = 80.0)
        @test length(res.biomass) == length(years)
        @test all(res.biomass .> 0.0)
        @test res.biomass[1] == 80.0

        # MSY Reference points
        ref = logistic_msy_reference_points(r_true, k_true)
        @test ref.bmsy == 50.0
        @test ref.msy == 10.0
        @test ref.fmsy == 0.2

        # Bayesian logistic discrete model check
        b_log = bayesian_logistic_discrete(removals, res.survey_pred)
        s_log = rand(b_log)
        @test haskey(s_log, :k)
        @test haskey(s_log, :r)
        @test haskey(s_log, :q)
        @test s_log[:k] > 0.0
        @test s_log[:r] > 0.0
    end

    @testset "CohortAnalysis (VPA and Pope's Cohort Analysis)" begin
        # 4 ages x 5 years synthetic catch matrix
        catches = [
            500.0 450.0 400.0 380.0 350.0;
            300.0 280.0 260.0 240.0 220.0;
            150.0 140.0 130.0 120.0 110.0;
             50.0  45.0  40.0  35.0  30.0
        ]
        terminal_f = [0.2, 0.25, 0.3, 0.35]
        M = 0.2

        # 1. Pope's approximation
        res_pope = popes_cohort_analysis(catches, M; terminal_f = terminal_f)
        @test size(res_pope.N) == size(catches)
        @test size(res_pope.F) == size(catches)
        @test all(res_pope.N .> 0.0)
        @test all(res_pope.F .>= 0.0)

        # 2. Exact Baranov VPA
        res_vpa = standard_vpa(catches, M; terminal_f = terminal_f)
        @test size(res_vpa.N) == size(catches)
        @test size(res_vpa.F) == size(catches)
        @test all(res_vpa.N .> 0.0)
        @test isapprox(res_vpa.catch_pred[:, end], catches[:, end], rtol = 0.01)

        # 3. Bayesian cohort analysis
        surv_idx = catches .* 0.001
        b_vpa = bayesian_cohort_analysis(catches, surv_idx, M)
        s_vpa = rand(b_vpa)
        @test haskey(s_vpa, :ft)
        @test haskey(s_vpa, :q)
        @test s_vpa[:ft][1] > 0.0

        # 4. ADAPT Virtual Population Analysis (Least-Squares Calibration)
        adapt_res = adapt_vpa(catches, surv_idx, M)
        @test length(adapt_res.terminal_f) == size(catches, 1)
        @test all(adapt_res.terminal_f .> 0.0)
        @test all(adapt_res.q .> 0.0)
        @test adapt_res.ssq >= 0.0

        # 5. Bayesian ADAPT VPA
        b_adapt = bayesian_adapt_vpa(catches, surv_idx, M)
        s_badapt = rand(b_adapt)
        @test haskey(s_badapt, :ft)
        @test haskey(s_badapt, :q)
        @test s_badapt[:ft][1] > 0.0
    end

    @testset "DelayDifferential (Continuous Multi-Stage DDE)" begin
        cfg = default_snowcrab_dde_config()
        @test cfg.num_stages == 6
        @test length(cfg.lags) == 3

        # Simulation check over 5 years with pulse fishery removals
        sim = simulate_dde_system(cfg, (0.0, 5.0); dt = 0.25,
                                  removals_times = [1.0, 2.0, 3.0, 4.0],
                                  removals_amount = [2.0, 2.5, 3.0, 2.0])
        @test size(sim.states, 2) == 6
        @test length(sim.time) > 10
        @test all(sim.states .> 0.0)

        # Deriso-Schnute delay-difference integration test
        ann_yrs, ann_rec = extract_annual_recruitment(sim; stage_index = 2, target_month = 9.0)
        @test length(ann_yrs) >= 5
        @test all(ann_rec .> 0.0)

        dd_cfg = DelayDifferenceConfig(exp(-0.25), 0.165, 0.70, 0.55)
        dd_res = simulate_delay_difference(ann_yrs, fill(2.0, length(ann_yrs)), ann_rec, dd_cfg;
                                           b0 = 40.0)
        @test length(dd_res.biomass) == length(ann_yrs)
        @test all(dd_res.biomass .> 0.0)
        @test all(dd_res.numbers .> 0.0)

        # Bayesian delay difference model check (prior sampling)
        b_model = bayesian_delay_difference_model(
            fill(2.0, length(ann_yrs)), ann_rec, dd_res.biomass .* 1.0, dd_cfg
        )
        sample_prior = rand(b_model)
        @test haskey(sample_prior, :b0)
        @test haskey(sample_prior, :s)
        @test haskey(sample_prior, :q)
        @test sample_prior[:b0] > 0.0

        # Bayesian continuous DDE model check
        s_times = [1.0, 2.0, 3.0, 4.0]
        s_data = [sim.states[findmin(abs.(sim.time .- t))[2], :]' for t in s_times]
        s_mat = vcat(s_data...)
        b_dde = bayesian_delay_differential_model(
            s_times, s_mat, cfg, (0.0, 5.0); dt = 0.5
        )
        sample_dde = rand(b_dde)
        @test haskey(sample_dde, :k_scale)
        @test haskey(sample_dde, :rec_scale)
        @test sample_dde[:k_scale] > 0.0
    end

    @testset "Bayesian Statistical Catch-at-Age (Turing)" begin
        data_dir = normpath(joinpath(@__DIR__, "..", "data", "silverhake"))
        model_sh = StockModel(data_dir)
        b_scaa = bayesian_statistical_catch_at_age(model_sh)
        sample_scaa = rand(b_scaa)
        @test haskey(sample_scaa, :log_ro)
        @test haskey(sample_scaa, :h)
        @test haskey(sample_scaa, :m_nat)
        @test sample_scaa[:h] > 0.0
    end

    @testset "Storage Export (Parquet & Zarr)" begin
        mktempdir() do tmpdir
            data_dir = normpath(joinpath(@__DIR__, "..", "data", "silverhake"))
            model = StockModel(data_dir)
            par_file = joinpath(data_dir, "parameters.toml")
            x_sol = load_parameters(par_file)

            # Test Parquet export
            pq_file = joinpath(tmpdir, "trajectories.parquet")
            write_parquet(pq_file, model, x_sol)
            @test isfile(pq_file)
            @test filesize(pq_file) > 0

            # Test Zarr array export
            zarr_dir = joinpath(tmpdir, "trajectories.zarr")
            write_zarr(zarr_dir, model, x_sol)
            @test isdir(zarr_dir)
            @test isfile(joinpath(zarr_dir, "N", "zarr.json")) ||
                  isfile(joinpath(zarr_dir, "N", ".zarray"))
        end
    end

    @testset "Retrospective Pattern Analysis & Mohn's Rho" begin
        # 1. Biomass Dynamics retrospective peel
        years = collect(2000:2010)
        removals = fill(8.0, length(years))
        retro_bio = retrospective_analysis(years, removals; n_peels = 4, r = 0.4, k = 100.0)
        @test length(retro_bio.peel_labels) == 5
        @test length(retro_bio.terminal_biomass_peeled) == 4
        @test -1.0 <= retro_bio.rho_biomass <= 1.0

        # Plot retrospective
        p_retro = plot_retrospective(retro_bio)
        @test p_retro isa UnicodePlots.Plot

        # Plot state variables
        res_bio = simulate_logistic_discrete(years, removals; r = 0.4, k = 100.0)
        p_bio = plot_state_variables(res_bio)
        @test p_bio isa UnicodePlots.Plot

        # 2. ADAPT VPA retrospective peel
        catches_vpa = [
            500.0 450.0 400.0 380.0 350.0;
            300.0 280.0 260.0 240.0 220.0;
            150.0 140.0 130.0 120.0 110.0;
             50.0  45.0  40.0  35.0  30.0
        ]
        surv_vpa = catches_vpa .* 0.001
        retro_vpa = retrospective_analysis(catches_vpa, surv_vpa, 0.2; n_peels = 2)
        @test length(retro_vpa.peel_labels) == 3
        @test -1.0 <= retro_vpa.rho_biomass <= 1.0
        p_vpa_retro = plot_retrospective(retro_vpa)
        @test p_vpa_retro isa UnicodePlots.Plot
    end

    @testset "model_fishery visualization assimilations" begin
        times = collect(1990.0:2000.0)
        # Mock 10 realizations of a declining fishery
        sims = zeros(Float64, length(times), 10)
        for s in 1:10
            sims[:, s] = 1000.0 .* exp.(-0.15 .* (0:10)) .+ (randn(11) .* 15.0)
        end
        obs_t = [1992.0, 1995.0, 1998.0]
        obs_v = [740.0, 470.0, 300.0]

        p_pred = plot_fishery_predictions(
            times, sims;
            obs_time = obs_t,
            obs_values = obs_v,
            title = "Ensemble Prediction Test"
        )
        @test p_pred isa UnicodePlots.Plot

        b_fished = [1000.0, 900.0, 750.0, 600.0, 500.0]
        b_unfished = [1000.0, 1020.0, 980.0, 1010.0, 1005.0]
        p_foot = plot_fishery_footprint(
            collect(2000.0:2004.0), b_fished, b_unfished;
            title = "Footprint Test"
        )
        @test p_foot isa UnicodePlots.Plot
    end
end

