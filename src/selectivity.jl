# ------------------------------------------------------------------------------------
# Selectivity formulations and transformations.
# ------------------------------------------------------------------------------------

const SEL_TINY = 1e-10

"""
    calc_selectivities(model, sel_par) -> Vector{Matrix}

Log selectivity `log v_{k,t,a}` for each gear `k` (matrix `ny × nages`). Forms by
`isel_type` (see [`selectivity_dims`](@ref)); `p₁ = exp(s₁)`, `p₂ = exp(s₂)`:

- 1: `log(plogis(a, p₁, p₂) + 10⁻¹⁰)`;  6: same without the constant (fixed);
- 2: `log v_a = s_a` for `a < A`, `log v_A = log v_{A-1}`;
- 3: clamped cubic spline through age nodes on [0, 1] (constant over years);
- 4: as 3 with a separate node vector each year;
- 5: bicubic spline over age × year nodes;
- 7: `log(plogis(x_{t,a}, p₁, p₂) + 10⁻¹⁰)`, `x_{t,a} = log(1000 w_{t,a}) / mean(log(1000 w))`;
- 8: type 1 plus `s₃ δ_{t,a}` (standardised weight-at-age deviations);
- 11: logistic in length `ℓ_{t,a} = (w_{t,a}/α)^{1/β}`;
- 12: clamped cubic spline in rescaled length `(ℓ - min ℓ)/(max ℓ - min ℓ)`;
- 0: uniform.

Each year is then scaled so that `mean_a exp(log v) = 1`:
`log v ← log v - log(mean(exp(log v)) + 10⁻¹⁰)`.
"""
function calc_selectivities(m::StockModel, sel_par::AbstractVector{<:AbstractMatrix})
    T = promote_type(Float64, mapreduce(eltype, promote_type, sel_par))
    ctl, data = m.control, m.data
    ages = Float64.(m.ages)
    agegrid = unit_nodes(m.nages)
    out = Vector{Matrix{T}}(undef, ngear(m))
    for k in 1:ngear(m)
        P = sel_par[k]
        t = ctl.isel_type[k]
        L = zeros(T, m.ny, m.nages)
        if t in (1, 6, 8)
            v = plogis.(ages, exp(P[1, 1]), exp(P[1, 2]))
            row = t == 6 ? log.(v) : log.(v .+ SEL_TINY)
            for i in 1:m.ny
                L[i, :] .= row
            end
            if t == 8
                all(isfinite, m.wt_dev) || error("Gear $k: selectivity type 8 needs " *
                    "empirical weight-at-age with non-zero variance at every age.")
                L .+= P[1, 3] .* m.wt_dev[1:m.ny, :]
            end
        elseif t == 2
            row = vcat(P[1, :], P[1, end])
            for i in 1:m.ny
                L[i, :] .= row
            end
        elseif t == 3
            row = CubicSpline(unit_nodes(size(P, 2)), P[1, :])(agegrid)
            for i in 1:m.ny
                L[i, :] .= row
            end
        elseif t == 4
            nodes = unit_nodes(size(P, 2))
            for i in 1:m.ny
                L[i, :] .= CubicSpline(nodes, P[i, :])(agegrid)
            end
        elseif t == 5
            L .= bicubic_spline(P, m.ny, m.nages)
        elseif t == 7
            p1, p2 = exp(P[1, 1]), exp(P[1, 2])
            mlw = sum(log.(m.wt_obs .* 1000)) / length(m.wt_obs)
            for i in 1:m.ny
                x = log.(m.wt_obs[i, :] .* 1000) ./ mlw
                L[i, :] .= log.(plogis.(x, p1, p2) .+ SEL_TINY)
            end
        elseif t == 11
            p1, p2 = exp(P[1, 1]), exp(P[1, 2])
            for i in 1:m.ny
                len = (m.wt_obs[i, :] ./ data.a) .^ (1 / data.b)
                L[i, :] .= log.(plogis.(len, p1, p2))
            end
        elseif t == 12
            nodes = unit_nodes(size(P, 2))
            s = CubicSpline(nodes, P[1, :])
            for i in 1:m.ny
                len = (m.wt_obs[i, :] ./ data.a) .^ (1 / data.b)
                L[i, :] .= s((len .- minimum(len)) ./ (maximum(len) - minimum(len)))
            end
        elseif t != 0
            error("Unknown selectivity type $t for gear $k.")
        end
        for i in 1:m.ny
            L[i, :] .-= log(sum(exp, L[i, :]) / m.nages + SEL_TINY)
        end
        out[k] = L
    end
    return out
end
