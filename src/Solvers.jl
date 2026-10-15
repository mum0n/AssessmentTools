# ------------------------------------------------------------------------------------
# Solvers: one shared root-finding facility, backed by Roots.jl.
#
# This module exists to be the single place where scalar and vector root solving
# happens. Before it, `src/cohort_analysis.jl` and `src/baranov_msy.jl` each carried
# their own hand-written Newton-Raphson loop with its own convergence constants and
# its own bracketing behaviour; both are now expressed in terms of the two entry
# points below. Roots.jl supplies the methods; this module supplies only the wrappers
# that keep the assessment-specific problem formulation (and its named tolerance and
# iteration parameters) at the call site.
# ------------------------------------------------------------------------------------

module Solvers

using Roots
using LinearAlgebra

export solve_scalar_root, solve_newton_vector

"""
    solve_scalar_root(f, guess; lower = 0.0, upper = 100.0, tol = 1e-10) -> Float64

Find a single real root of `f(x) = 0` near `guess`, searching inside `[lower, upper]`.

Backed by `Roots.find_zero` on a bracketing method (`Bisection`), which converges
from any bracket that contains an odd number of roots. A bracket method is used
deliberately: the Baranov catch residual below very low `F` is poorly scaled, so the
plain secant method that `find_zero` would otherwise choose can step out of the
physically admissible range.

`f` must be scalar and return a `Real`, so it is compatible with `ForwardDiff`
differentiation.

# Arguments
- `f`: residual function, zero at the root.
- `guess`: starting value.
- `lower`, `upper`: search bracket; the root is returned inside it.
- `tol`: absolute root tolerance passed to `Roots`.
"""
function solve_scalar_root(f::Function, guess::Real;
                           lower::Real = 0.0, upper::Real = 100.0, tol::Real = 1.0e-10)
    upper > lower || error("require upper ($upper) > lower ($lower).")
    isfinite(guess) || error("guess must be finite, got $guess.")
    lo = min(lower, upper)
    hi = max(lower, upper)
    return find_zero(f, (lo, hi), Roots.Bisection(), xatol = tol, maxiter = 200)
end

"""
    solve_newton_vector(F, J, x0; tol = 1e-10, max_iter = 100, lower = nothing, upper = nothing) -> Vector{Float64}

Solve a system `F(x) = 0` by Newton's method with an explicitly supplied analytic
Jacobian `J`. `x0` is the starting vector. Each iteration solves the linear system
``J\\,dx = -F`` and updates ``x \\leftarrow x + dx``, so no inverse is formed.
Iteration stops when ``\\|F\\|_2 < \\text{tol}`` or after `max_iter` iterations, and
non-convergence is reported as an error with the final residual norm.

Optional `lower` / `upper` bounds are applied to `x` after each step, so the iterate
stays inside the admissible box. Passing `nothing` leaves a parameter unbounded.

This is the single Newton implementation in the package.
"""
function solve_newton_vector(F::Function, J::Function, x0::AbstractVector{<:Real};
                              tol::Real = 1.0e-10, max_iter::Integer = 100,
                              lower::Union{Nothing, AbstractVector} = nothing,
                              upper::Union{Nothing, AbstractVector} = nothing)
    n = length(x0)
    !isnothing(lower) && length(lower) == n || error("lower bound must have $n elements.")
    !isnothing(upper) && length(upper) == n || error("upper bound must have $n elements.")
    max_iter >= 1 || error("max_iter must be at least 1.")

    x = collect(Float64, x0)
    fx = F(x)
    iters = 0
    while norm(fx) >= tol
        iters += 1
        iters > max_iter &&
            error("Newton iteration did not converge in $max_iter iterations (‖F‖ = $(norm(fx))).")
        dx = J(x) \ fx
        x .-= dx
        if !isnothing(lower) || !isnothing(upper)
            @inbounds for i in 1:n
                if !isnothing(lower) && x[i] < lower[i]
                    x[i] = lower[i]
                end
                if !isnothing(upper) && x[i] > upper[i]
                    x[i] = upper[i]
                end
            end
        end
        fx = F(x)
    end
    return x
end

end # module Solvers
