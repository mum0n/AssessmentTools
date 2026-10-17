using Test
using AssessmentTools
using LinearAlgebra
using Statistics
using TOML
using UnicodePlots
using Random

@testset "Assessment Primitives & Single-Species Models" begin
    # Spline and Statistics primitives
    x = [0.0, 0.25, 0.5, 0.75, 1.0]
    y = [0.0, 0.5, 1.0, 0.5, 0.0]
    s = AssessmentTools.StatisticalCatchAtAge.CubicSpline(x, y)
    @test isapprox(s(0.0), 0.0, atol=1e-5)
    @test isapprox(s(0.5), 1.0, atol=1e-5)
    @test isapprox(s(1.0), 0.0, atol=1e-5)

    @test isapprox(AssessmentTools.plogis(5.0, 5.0, 1.0), 0.5)
    @test AssessmentTools.plogis(10.0, 5.0, 1.0) > 0.99

    # Silver Hake Assessment (TOML/CSV)
    data_dir_sh = normpath(joinpath(@__DIR__, "..", "data", "silverhake"))
    cfg_file_sh = joinpath(data_dir_sh, "config.toml")
    par_file_sh = joinpath(data_dir_sh, "parameters.toml")

    @test isfile(cfg_file_sh)
    @test isfile(par_file_sh)

    model_sh = StockModel(data_dir_sh)
    x_sol_sh = load_parameters(par_file_sh)
    @test length(x_sol_sh) == model_sh.layout.n

    f_val, _ = objective_function(model_sh, x_sol_sh, PhaseContext(model_sh.layout.max_phase, true))
    @test isapprox(f_val, 280.385, rtol = 0.01)

    p_sh = AssessmentTools.StatisticalCatchAtAge.unpack(model_sh.layout, x_sol_sh)
    ctx_sh = PhaseContext(model_sh.layout.max_phase, true)
    lead_sh = AssessmentTools.StatisticalCatchAtAge.leading_parameters(model_sh, p_sh.theta)
    log_sel_sh = calc_selectivities(model_sh, p_sh.sel_par)
    mort_sh = calc_total_mortality(model_sh, log_sel_sh, p_sh.log_ft_pars, p_sh.log_m_nodes, lead_sh.M, ctx_sh)
    ref_sh = calc_reference_points(model_sh, mort_sh, log_sel_sh)

    @test ref_sh.bo > 0
    @test ref_sh.bmsy > 0
    @test all(ref_sh.fmsy .> 0)

    dt_sh = decision_table(model_sh, x_sol_sh)
    @test length(dt_sh) == length(model_sh.projection.tac)

    d_toml_sh, c_toml_sh, p_toml_sh = load_dataset(data_dir_sh)
    @test d_toml_sh.obs_ct == model_sh.data.obs_ct
    @test length(d_toml_sh.surveys) == length(model_sh.data.surveys)
    @test length(d_toml_sh.agecomps) == length(model_sh.data.agecomps)

    # Cusk Assessment (TOML/CSV)
    data_dir_c = normpath(joinpath(@__DIR__, "..", "data", "cusk"))
    cfg_file_c = joinpath(data_dir_c, "config.toml")
    par_file_c = joinpath(data_dir_c, "parameters.toml")

    @test isfile(cfg_file_c)
    @test isfile(par_file_c)

    model_cusk = StockModel(data_dir_c)
    x_sol_cusk = load_parameters(par_file_c)
    @test length(x_sol_cusk) == model_cusk.layout.n

    p_c = AssessmentTools.StatisticalCatchAtAge.unpack(model_cusk.layout, x_sol_cusk)
    ctx_c = PhaseContext(model_cusk.layout.max_phase, true)
    lead_c = AssessmentTools.StatisticalCatchAtAge.leading_parameters(model_cusk, p_c.theta)
    log_sel_c = calc_selectivities(model_cusk, p_c.sel_par)
    mort_c = calc_total_mortality(model_cusk, log_sel_c, p_c.log_ft_pars,
                                  p_c.log_m_nodes, lead_c.M, ctx_c)
    ref_cusk = calc_reference_points(model_cusk, mort_c, log_sel_c, p_c.theta)

    ref_toml_c = TOML.parsefile(par_file_c)["reference_points"]
    @test isapprox(ref_cusk.bo, ref_toml_c["bo"], rtol = 1e-4)
    @test isapprox(ref_cusk.bmsy, ref_toml_c["bmsy"], rtol = 1e-4)
    @test isapprox(sum(ref_cusk.msy), ref_toml_c["msy"], rtol = 1e-4)
    @test isapprox(ref_cusk.fmsy[1], ref_toml_c["fmsy"], rtol = 1e-4)

    d_toml_c, c_toml_c, _ = load_dataset(data_dir_c)
    @test d_toml_c.obs_ct == model_cusk.data.obs_ct
    @test length(d_toml_c.surveys) == length(model_cusk.data.surveys)

    # Bayesian Statistical Catch-at-Age (Turing)
    b_scaa = bayesian_statistical_catch_at_age(model_sh)
    sample_scaa = rand(b_scaa)
    @test haskey(sample_scaa, :log_ro)
    @test haskey(sample_scaa, :h)
    @test haskey(sample_scaa, :m_nat)
    @test sample_scaa[:h] > 0.0

    # Storage Export (Parquet & Zarr)
    mktempdir() do tmpdir
        pq_file = joinpath(tmpdir, "trajectories.parquet")
        write_parquet(pq_file, model_sh, x_sol_sh)
        @test isfile(pq_file)
        @test filesize(pq_file) > 0

        zarr_dir = joinpath(tmpdir, "trajectories.zarr")
        write_zarr(zarr_dir, model_sh, x_sol_sh)
        @test isdir(zarr_dir)
        @test isfile(joinpath(zarr_dir, "N", "zarr.json")) ||
              isfile(joinpath(zarr_dir, "N", ".zarray"))
    end
end
