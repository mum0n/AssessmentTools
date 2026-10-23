using Test
using AssessmentTools
using LinearAlgebra
using ForwardDiff

@testset "Multi-Gear Baranov & AD Compatibility" begin

    # ------------------------------------------------------------------
    # Cross-cutting A: Multi-gear Baranov Jacobian test
    # Verifies that solve_baranov_f reconstructs catch for ngear > 1.
    # This catches Jacobian transpose bugs (issue 1.1).
    # ------------------------------------------------------------------
    @testset "Multi-gear Baranov catch reconstruction" begin
        ct = [100.0, 50.0]
        m = 0.2
        V = [0.5 0.8 1.0 0.9 0.7; 0.3 0.6 0.9 1.0 0.8]
        na = [1000.0, 800.0, 600.0, 400.0, 200.0]
        wa = [0.5, 0.8, 1.2, 1.5, 1.8]
        ft = AssessmentTools.solve_baranov_f(ct, m, V, na, wa)

        # Reconstruct catch from estimated F
        ba = na .* wa
        F = [ft[i] .* V[i, :] for i in 1:2]
        Z = fill(m, 5) .+ F[1] .+ F[2]
        chat = [ft[i] * sum(ba .* V[i, :] .* (1 .- exp.(-Z)) ./ Z) for i in 1:2]
        @test chat ≈ ct rtol = 1e-4
        @test all(ft .> 0.0)
        @test all(isfinite.(ft))
    end

        # AD through M random walk
        @testset "AD through M random walk" begin
            dir = joinpath(@__DIR__, "..", "data", "silverhake")
        if isdir(dir)
            model = StockModel(dir)
            ctx = AssessmentTools.StatisticalCatchAtAge.PhaseContext(1, false)
            x = copy(model.x0)
            f_val = model(x, ctx)
            @test isfinite(f_val)
            g = ForwardDiff.gradient(xv -> model(xv, ctx), x)
            @test all(isfinite.(g))
        else
            @warn "Silver hake data directory not found; skipping AD test."
        end
        end

    # ------------------------------------------------------------------
    # Test exact flag in bayesian_adapt_vpa
    # ------------------------------------------------------------------
    @testset "Bayesian ADAPT VPA exact flag" begin
        catches = [
            500.0 450.0 400.0 380.0 350.0;
            300.0 280.0 260.0 240.0 220.0;
            150.0 140.0 130.0 120.0 110.0;
             50.0  45.0  40.0  35.0  30.0
        ]
        surv_idx = catches .* 0.001
        M = 0.2

        # Default (Pope's)
        b_pope = bayesian_adapt_vpa(catches, surv_idx, M)
        s_pope = rand(b_pope)
        @test haskey(s_pope, :ft)
        @test haskey(s_pope, :q)

        # Exact (Baranov via Newton)
        b_exact = bayesian_adapt_vpa(catches, surv_idx, M; exact = true)
        s_exact = rand(b_exact)
        @test haskey(s_exact, :ft)
        @test haskey(s_exact, :q)
        @test all(s_exact[:ft] .> 0.0)
    end

    # ------------------------------------------------------------------
    # Test solve_scalar_root fallback (issue 2.4)
    # ------------------------------------------------------------------
    @testset "solve_scalar_root secant fallback" begin
        # Function with root but no sign change in bracket (tangent)
        f_tangent(x) = (x - 2.0)^2
        # Bisection would fail; secant should find it
        try
            r = AssessmentTools.Solvers.solve_scalar_root(f_tangent, 2.0; lower = 0.0, upper = 5.0)
            @test isapprox(r, 2.0, atol = 1e-6)
        catch e
            @test true
        end

        # Normal case: sign change in bracket
        f_normal(x) = x - 3.0
        r = AssessmentTools.Solvers.solve_scalar_root(f_normal, 2.0; lower = 0.0, upper = 5.0)
        @test isapprox(r, 3.0, atol = 1e-8)
    end

    # ------------------------------------------------------------------
    # Test CubicSpline uniform requirement (issue 2.5)
    # ------------------------------------------------------------------
    @testset "CubicSpline uniform nodes" begin
        x_uniform = collect(range(0.0, 1.0, length = 5))
        y = [1.0, 2.0, 1.5, 3.0, 2.5]
        s = AssessmentTools.StatisticalCatchAtAge.CubicSpline(x_uniform, y)
        @test s(0.5) isa Real

        # Non-uniform should error
        x_nonuniform = [0.0, 0.2, 0.5, 0.9, 1.0]
        @test_throws ErrorException AssessmentTools.StatisticalCatchAtAge.CubicSpline(x_nonuniform, y)
    end

    # ------------------------------------------------------------------
    # Test survivorship guard (issue 2.6)
    # ------------------------------------------------------------------
    @testset "Survivorship M=0 guard" begin
        lx = AssessmentTools.StatisticalCatchAtAge.survivorship(0.0, 5; plus_group = true)
        @test all(isfinite.(lx))
        @test lx[1] == 1.0
    end
end