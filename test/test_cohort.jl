using Test
using AssessmentTools
using LinearAlgebra
using Statistics
using TOML
using UnicodePlots
using Random

@testset "Cohort & Virtual Population Analysis" begin
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

    res_pope_gamma = popes_cohort_analysis(catches, M; terminal_f = terminal_f,
                                            gamma = 1.0, preceding_ages_count = 2)
    @test size(res_pope_gamma.N) == size(catches)
    @test all(res_pope_gamma.N .> 0.0)
    @test all(res_pope_gamma.F .>= 0.0)
    @test isapprox(res_pope_gamma.F[4, 1], mean(res_pope_gamma.F[2:3, 1]), rtol = 1e-6)

    # 2. Exact Baranov VPA
    res_vpa = standard_vpa(catches, M; terminal_f = terminal_f)
    @test size(res_vpa.N) == size(catches)
    @test size(res_vpa.F) == size(catches)
    @test all(res_vpa.N .> 0.0)
    @test isapprox(res_vpa.catch_pred[:, end], catches[:, end], rtol = 0.01)

    res_vpa_gamma = standard_vpa(catches, M; terminal_f = terminal_f,
                                 gamma = 0.90, preceding_ages_count = 2)
    @test size(res_vpa_gamma.N) == size(catches)
    @test all(res_vpa_gamma.N .> 0.0)
    @test isapprox(res_vpa_gamma.F[4, 1], 0.90 * mean(res_vpa_gamma.F[2:3, 1]), rtol = 1e-6)

    # 3. Bayesian cohort analysis
    surv_idx = catches .* 0.001
    b_vpa = bayesian_cohort_analysis(catches, surv_idx, M)
    s_vpa = rand(b_vpa)
    @test haskey(s_vpa, :ft)
    @test haskey(s_vpa, :q)
    @test s_vpa[:ft][1] > 0.0

    # 4. ADAPT Virtual Population Analysis
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

    # 6. Retrospective Analysis for VPA
    retro_vpa = retrospective_analysis(catches, surv_idx, 0.2; n_peels = 2)
    @test length(retro_vpa.peel_labels) == 3
    @test -1.0 <= retro_vpa.rho_biomass <= 1.0
    p_vpa_retro = plot_retrospective(retro_vpa)
    @test p_vpa_retro isa UnicodePlots.Plot
end
