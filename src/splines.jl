# ------------------------------------------------------------------------------------
# Spline and interpolation interface built upon Interpolations.jl and DataInterpolations.jl.
# Fully generic in ordinate types for ForwardDiff automatic differentiation.
# ------------------------------------------------------------------------------------

"""
    CubicSpline(x, y; yp1 = 0.0, ypn = 0.0)

Cubic spline interpolation of `y` over knots `x` powered by `Interpolations.jl`
with clamped flat boundary conditions (`Flat(OnGrid())`). Requires uniformly
spaced knots for ForwardDiff AD compatibility. Non-uniform grids are not supported.

"""
function CubicSpline end

function CubicSpline(x::AbstractVector{<:Real}, y::AbstractVector; yp1 = 0.0, ypn = 0.0)
    # Check if x is an equally-spaced range
    n = length(x)
    n >= 2 || error("A cubic spline needs at least 2 nodes, found $n.")
    step_val = (x[end] - x[1]) / (n - 1)
    is_uniform = all(i -> abs((x[i] - x[i - 1]) - step_val) < 1e-12, 2:n)
    is_uniform || error("CubicSpline requires uniformly spaced knots; non-uniform " *
                        "grids are not supported (Interpolations.jl provides " *
                        "guaranteed AD compatibility only for uniform grids).")
    r = range(first(x), last(x), length = n)
    itp = cubic_spline_interpolation(r, y, bc = Interpolations.Flat(Interpolations.OnGrid()))
    return itp
end

"""
    unit_nodes(n::Integer) -> Vector{Float64}

Equally spaced spline nodes spanning [0, 1].
"""
unit_nodes(n::Integer) = n >= 2 ? collect(range(0.0, 1.0; length = n)) :
                         error("At least 2 spline nodes are required, found $n.")

"""
    bicubic_spline(knots::AbstractMatrix, nrow::Integer, ncol::Integer) -> Matrix

Bicubic spline interpolation of an `m × n` knot matrix onto an `nrow × ncol` regular grid
spanning [0, 1] × [0, 1] using `Interpolations.jl`.
"""
function bicubic_spline(knots::AbstractMatrix, nrow::Integer, ncol::Integer)
    m, n = size(knots)
    r_row = range(0.0, 1.0, length = m)
    r_col = range(0.0, 1.0, length = n)
    itp = cubic_spline_interpolation((r_row, r_col), knots,
                                    bc = Interpolations.Line(Interpolations.OnGrid()))

    out_row = range(0.0, 1.0, length = nrow)
    out_col = range(0.0, 1.0, length = ncol)

    T = eltype(knots)
    S = Matrix{T}(undef, nrow, ncol)
    for j in 1:ncol
        cj = out_col[j]
        for i in 1:nrow
            S[i, j] = itp(out_row[i], cj)
        end
    end
    return S
end

"""
    linapprox(x::AbstractVector, y::AbstractVector, xout::AbstractVector) -> Vector

Piecewise-linear interpolation with flat boundary extrapolation using `Interpolations.jl`.
"""
function linapprox(x::AbstractVector, y::AbstractVector, xout::AbstractVector)
    issorted(x) || error("linapprox requires sorted x.")
    itp = linear_interpolation(x, y, extrapolation_bc = Flat())
    return [itp(v) for v in xout]
end
