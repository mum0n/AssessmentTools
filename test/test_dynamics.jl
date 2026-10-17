using Test
using AssessmentTools
using LinearAlgebra
using Statistics
using TOML
using UnicodePlots
using Random

@testset "Empirical & Surplus Production Dynamics" begin
    # 1. Chapman-Robson & Catch Curves
    ages = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    true_s = exp(-0.5)
    catches = [1000.0 * (true_s^i) for i in 0:7]

    cr = chapman_robson(ages, catches)
    @test isapprox(cr.s, true_s, atol = 0.05)
    @test isapprox(cr.z, 0.5, atol = 0.05)
    @test cr.se_z > 0.0

    cc = catch_curve_linear(ages, catches)
    @test isapprox(cc.z, 0.5, atol = 0.01)
    @test isapprox(cc.r2, 1.0, atol = 1e-4)

    bcc = bayesian_catch_curve(ages, catches)
    s_bcc = rand(bcc)
    @test haskey(s_bcc, :z)
    @test haskey(s_bcc, :alpha)
    @test s_bcc[:z] > 0.0

    # 2. Leslie & DeLury Depletion
    n0_true = 10000.0
    q_true = 0.001
    efforts = [100.0, 100.0, 100.0, 100.0, 100.0]
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

    @test isapprox(apply_tac_stabilizer(80.0, 100.0; max_decrease = 0.15), 85.0)
    @test isapprox(apply_tac_stabilizer(130.0, 100.0; max_increase = 0.15), 115.0)
    @test isapprox(apply_tac_stabilizer(105.0, 100.0; max_increase = 0.15), 105.0)
    @test isapprox(apply_tac_stabilizer(60.0, 100.0; max_decrease = 0.15,
                                         biomass = 40.0, blim = 50.0), 60.0)
    @test isapprox(apply_tac_stabilizer(60.0, 100.0; max_decrease = 0.15,
                                         biomass = 60.0, blim = 50.0), 85.0)

    # 5. Management Strategy Evaluation (MSE) & Bayesian Feedback
    mse_years = collect(2020:2030)
    mse_hcr = HockeyStickHCR(20.0, 50.0, 0.05, 0.25)
    mse_res = simulate_mse(mse_years, mse_hcr; r = 0.40, k = 100.0, b0 = 80.0)
    @test length(mse_res.true_biomass) == length(mse_years)
    @test all(mse_res.true_biomass .> 0.0)
    @test all(mse_res.catches .>= 0.0)

    res_rng1 = simulate_mse(mse_years, mse_hcr; rng = Random.MersenneTwister(42))
    res_rng2 = simulate_mse(mse_years, mse_hcr; rng = Random.MersenneTwister(42))
    @test res_rng1.true_biomass == res_rng2.true_biomass
    @test res_rng1.observed_biomass == res_rng2.observed_biomass

    b_mse = bayesian_mse_feedback(mse_res.observed_biomass, mse_res.catches, mse_hcr)
    s_bmse = rand(b_mse)
    @test haskey(s_bmse, :r)
    @test haskey(s_bmse, :k)
    @test s_bmse[:k] > 0.0

    # 6. BiomassDynamics (Discrete Logistic Surplus Production)
    years = collect(2000:2010)
    removals = fill(10.0, length(years))
    r_true = 0.4
    k_true = 100.0

    b_next = logistic_step(50.0, r_true, k_true, 10.0)
    @test b_next > 0.0

    res = simulate_logistic_discrete(years, removals; r = r_true, k = k_true, b0 = 80.0)
    @test length(res.biomass) == length(years)
    @test all(res.biomass .> 0.0)
    @test res.biomass[1] == 80.0

    ref = logistic_msy_reference_points(r_true, k_true)
    @test ref.bmsy == 50.0
    @test ref.msy == 10.0
    @test ref.fmsy == 0.2

    b_log = bayesian_logistic_discrete(removals, res.survey_pred)
    s_log = rand(b_log)
    @test haskey(s_log, :k)
    @test haskey(s_log, :r)
    @test haskey(s_log, :q)
    @test s_log[:k] > 0.0
    @test s_log[:r] > 0.0

    # 7. Retrospective Pattern Analysis
    retro_bio = retrospective_analysis(years, removals; n_peels = 4, r = 0.4, k = 100.0)
    @test length(retro_bio.peel_labels) == 5
    @test length(retro_bio.terminal_biomass_peeled) == 4
    @test -1.0 <= retro_bio.rho_biomass <= 1.0

    p_retro = plot_retrospective(retro_bio)
    @test p_retro isa UnicodePlots.Plot

    p_bio = plot_state_variables(res)
    @test p_bio isa UnicodePlots.Plot

    # 8. Model Fishery Visualizations
    times_f = collect(1990.0:2000.0)
    sims = zeros(Float64, length(times_f), 10)
    for s_idx in 1:10
        sims[:, s_idx] = 1000.0 .* exp.(-0.15 .* (0:10)) .+ (randn(11) .* 15.0)
    end
    obs_t = [1992.0, 1995.0, 1998.0]
    obs_v = [740.0, 470.0, 300.0]

    p_pred = plot_fishery_predictions(
        times_f, sims;
        obs_time = obs_t,
        obs_values = obs_v,
        title = "Ensemble Prediction Test"
    )
    @test p_pred isa UnicodePlots.Plot

    p_pred_rng = plot_fishery_predictions(
        times_f, sims;
        rng = Random.MersenneTwister(123),
        title = "RNG Seeded Prediction"
    )
    @test p_pred_rng isa UnicodePlots.Plot

    b_fished = [1000.0, 900.0, 750.0, 600.0, 500.0]
    b_unfished = [1000.0, 1020.0, 980.0, 1010.0, 1005.0]
    p_foot = plot_fishery_footprint(
        collect(2000.0:2004.0), b_fished, b_unfished;
        title = "Footprint Test"
    )
    @test p_foot isa UnicodePlots.Plot
end
