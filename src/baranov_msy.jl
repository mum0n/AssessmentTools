# ------------------------------------------------------------------------------------
# Baranov catch equation solver and equilibrium / MSY reference points calculations.
# ------------------------------------------------------------------------------------

"""
    solve_baranov_f(ct, m, V, na, wa; tol = 1e-10, max_iter = 100, f_lower = 1e-15, f_upper = 5.0) -> Vector{Float64}

Solves the multi-gear Baranov catch equation for instantaneous fishing mortalities `ft`
given:
- `ct`: observed catch by gear (in weight)
- `m`: instantaneous natural mortality rate
- `V`: gear-specific selectivity matrix (`ngear x nages`)
- `na`: numbers at age at the start of the year
- `wa`: mean weight-at-age
- `tol`: convergence tolerance on `norm(C_model - C_obs)`
- `max_iter`: iteration limit, passed to the shared Newton solver
- `f_lower`, `f_upper`: admissible box for each gear's fishing mortality

Newton's method is run by [`Solvers.solve_newton_vector`](@ref), the single Newton
implementation in the package, using the analytic Jacobian evaluated below. The residual
and its Jacobian are computed at the same `ft`, so there is no separate convergence loop
at this call site.
"""
function solve_baranov_f(ct::AbstractVector{<:Real}, m::Real, V::AbstractMatrix{<:Real},
                         na::AbstractVector{<:Real}, wa::AbstractVector{<:Real};
                         tol::Real = 1.0e-10, max_iter::Integer = 100,
                         f_lower::Real = 1.0e-15, f_upper::Real = 5.0)
    ng = length(ct)
    na_len = length(na)
    ba = na .* wa

    # Catch and Jacobian of the model catch with respect to the per-gear F vector.
    function catch_and_jacobian(ft::AbstractVector{<:Real})
        F_gear = [ft[i] .* V[i, :] for i in 1:ng]
        Z = fill(Float64(m), na_len)
        for i in 1:ng
            Z .+= F_gear[i]
        end
        sa = exp.(-Z)
        oa = 1.0 .- sa

        chat = zeros(Float64, ng)
        J = zeros(Float64, ng, ng)
        for i in 1:ng
            for j in 1:ng
                if i == j
                    k1 = ba .* (V[i, :] ./ Z)
                    k2 = k1 .* V[i, :]
                    k3 = k2 ./ Z
                    J[i, j] = -sum(k1 .* oa) - ft[i] * sum(k2 .* sa) + ft[i] * sum(k3 .* oa)
                    chat[i] = ft[i] * sum(k1 .* oa)
                else
                    t1 = (ft[i] .* ba .* V[i, :]) ./ Z
                    t2 = t1 .* V[j, :]
                    t3 = t2 ./ Z
                    J[j, i] = -sum(t2 .* sa) + sum(t3 .* oa)
                end
            end
        end
        return chat, J
    end

    residual(ft::AbstractVector) = ct .- first(catch_and_jacobian(ft))
    jacobian(ft::AbstractVector) = last(catch_and_jacobian(ft))

    # Initial guess: split the total catch over gears in proportion to their observed
    # catch, on vulnerable biomass.
    bt = sum(na .* exp(-0.5 * m) .* wa) + 1.0e-10
    ft0 = collect(Float64(ct[i]) / bt for i in 1:ng)
    ft0 = clamp.(ft0, f_lower, f_upper)

    return Solvers.solve_newton_vector(residual, jacobian, ft0;
                                       tol = tol, max_iter = max_iter,
                                       lower = fill(f_lower, ng),
                                       upper = fill(f_upper, ng))
end

"""
    MSYEquilibrium

Container for equilibrium and reference point quantities. All vectors are per gear.

# Fields
- `bo::Float64`: unfished (virgin) spawning biomass ``B_0``
- `bmsy::Float64`: spawning biomass expected to produce MSY
- `fmsy::Vector{Float64}`: fishing mortality resulting in MSY
- `fall::Vector{Float64}` fishing mortality rescaled to the historical catch allocation
- `msy::Vector{Float64}`: Maximum Sustainable Yield as a fleet-wise vector
- `umsy::Float64`: exploitation rate at MSY
- `spr_msy::Float64`: spawning potential ratio at MSY
- `re::Float64`: equilibrium recruitment at MSY
- `be::Float64`: biomass in the absence of fishing for the year of MSY

`rmsy` and `ye` were removed as duplicates of `re` and `msy` respectively.
"""
struct MSYEquilibrium
    bo::Float64
    bmsy::Float64
    fmsy::Vector{Float64}
    fall::Vector{Float64}
    msy::Vector{Float64}
    umsy::Float64
    spr_msy::Float64
    re::Float64
    be::Float64
end

"""
    calc_equilibrium_point(fe, ro, h, m, spawn_frac, wa, fa, V) -> NamedTuple

Compute equilibrium recruitment, yield, and spawning biomass for a given vector
of fishing mortalities `fe` across gears.
"""
function calc_equilibrium_point(fe::AbstractVector{<:Real}, ro::Real, h::Real, m::Real,
                                spawn_frac::Real, wa::AbstractVector{<:Real},
                                fa::AbstractVector{<:Real}, V::AbstractMatrix{<:Real})
    sage = 1
    nage = length(wa)
    ngear = size(V, 1)

    za = fill(Float64(m), nage)
    for k in 1:ngear
        za .+= fe[k] .* V[k, :]
    end
    pza = spawn_frac .* za
    sa = exp.(-za)
    psa = exp.(-pza)
    oa = 1.0 .- sa

    lz = zeros(Float64, nage)
    lw = zeros(Float64, nage)
    dlz = zeros(Float64, ngear, nage)
    d2lz = zeros(Float64, ngear, nage)
    dlw = zeros(Float64, ngear, nage)
    d2lw = zeros(Float64, ngear, nage)

    qa = zeros(Float64, ngear, nage)
    for k in 1:ngear
        qa[k, :] .= (V[k, :] .* wa .* oa) ./ za
        dlw[k, sage] = -psa[sage] * spawn_frac * V[k, sage]
        d2lw[k, sage] = psa[sage] * (spawn_frac * V[k, sage])^2
    end

    lz[sage] = 1.0
    lw[sage] = psa[sage]

    for j in (sage + 1):nage
        lz[j] = lz[j - 1] * sa[j - 1]
        lw[j] = lz[j] * psa[j]
        if j == nage
            lz[j] /= oa[j]
            lw[j] = lz[j - 1] * sa[j - 1] * psa[j] / oa[j]
        end

        for k in 1:ngear
            dlz[k, j] = sa[j - 1] * (dlz[k, j - 1] - lz[j - 1] * V[k, j - 1])
            d2lz[k, j] = sa[j - 1] * (d2lz[k, j - 1] + lz[j - 1] * V[k, j - 1]^2)

            dlw[k, j] = -lz[j] * spawn_frac * V[k, j] * psa[j]
            d2lw[k, j] = lz[j] * (spawn_frac * V[k, j])^2 * psa[j]

            if j == nage
                dlz[k, j] = dlz[k, j] / oa[j] - lz[j - 1] * sa[j - 1] * V[k, j] * sa[j] / (oa[j]^2)
                dlw[k, j] = -lz[j - 1] * sa[j - 1] * spawn_frac * V[k, j] / oa[j] -
                            lz[j - 1] * psa[j] * V[k, j] * sa[j] / (oa[j]^2)

                V1 = V[k, j - 1]
                V2 = V[k, j]
                oa2 = oa[j]^2
                d2lz[k, j] = d2lz[k, j] / oa[j] +
                             2 * lz[j - 1] * V1 * sa[j - 1] * V2 * sa[j] / oa2 +
                             2 * lz[j - 1] * sa[j - 1] * V2^2 * sa[j]^2 / (oa[j] * oa2) +
                             lz[j - 1] * sa[j - 1] * V2^2 * sa[j] / oa2

                d2lw[k, j] = lz[j - 1] * (spawn_frac * V2)^2 * psa[j] / oa[j] +
                             2 * lz[j - 1] * spawn_frac * V2^2 * psa[j] * sa[j] / oa2 +
                             2 * lz[j - 1] * psa[j] * V2^2 * sa[j]^2 / (oa[j] * oa2) +
                             lz[j - 1] * psa[j] * V2^2 * sa[j] / oa2
            end
        end
    end

    # Unfished eggs per recruit
    lx0 = zeros(Float64, nage)
    for i in 1:nage
        lx0[i] = exp(-m * (i - 1) - spawn_frac * m)
        if i == nage
            lx0[i] /= (1.0 - exp(-m))
        end
    end
    phie = sum(lx0 .* fa)
    bo = ro * phie

    kappa = 4.0 * h / (1.0 - h)
    km1 = kappa - 1.0

    phif = sum(lw .* fa)
    phif2 = phif^2

    dphif = zeros(Float64, ngear)
    d2phif = zeros(Float64, ngear)
    phiq = zeros(Float64, ngear)
    dphiq = zeros(Float64, ngear)
    d2phiq = zeros(Float64, ngear)
    dre = zeros(Float64, ngear)
    d2re = zeros(Float64, ngear)

    for k in 1:ngear
        dphif[k] = sum(dlz[k, :] .* fa)
        d2phif[k] = sum(d2lz[k, :] .* fa)

        phiq[k] = sum(lz .* qa[k, :])
        if ngear == 1
            t1 = (lz .* wa .* (V[k, :] .^ 2)) ./ za
        else
            t1 = (lz .* wa .* V[k, :]) ./ za
        end
        t0 = oa ./ za
        t3 = sa .- t0
        dphiq[k] = sum(qa[k, :] .* dlz[k, :]) + sum(t1 .* t3)

        t2 = 2.0 .* dlz[k, :]
        V2 = V[k, :] .^ 2
        t5 = (wa .* V2) ./ za
        t7 = V[k, :] ./ za
        t9 = t5 .* sa
        t11 = t5 .* t0
        t13 = lz .* t5
        t14 = V[k, :] .* sa
        t15 = t7 .* sa
        t17 = V[k, :] .* t0
        t18 = t17 ./ za
        d2phiq[k] = sum(d2lz[k, :] .* qa[k, :]) + sum(t2 .* t9) - sum(t2 .* t11) -
                    sum(t13 .* t14) - 2.0 * sum(t13 .* t15) + 2.0 * sum(t13 .* t18)

        dre[k] = ro * phie * dphif[k] / (phif2 * km1)
        d2re[k] = -2.0 * ro * phie * (dphif[k]^2) / (phif2 * phif * km1) +
                  ro * phie * d2phif[k] / (phif2 * km1)
    end

    re = ro * (kappa - phie / phif) / km1
    ye = re .* (fe .* phiq)
    dye = re .* phiq .+ fe .* phiq .* dre .+ fe .* re .* dphiq

    d2ye = zeros(Float64, ngear, ngear)
    for j in 1:ngear, k in 1:ngear
        d2ye[k, j] = fe[j] * phiq[j] * d2re[k] + 2.0 * fe[j] * dre[k] * dphiq[k] + fe[j] * re * d2phiq[k]
        if k == j
            d2ye[j, k] += 2.0 * dre[j] * phiq[j] + 2.0 * re * dphiq[j]
        end
    end

    return (
        bo = bo, phie = phie, re = re, ye = ye, be = re * phif, bi = re * sum(lz .* fa),
        spr = phif / phie, dye = dye, d2ye = d2ye, phiq = phiq
    )
end

"""
    calc_reference_points(model, mort, log_sel, theta = model.x0[model.layout.theta.range]) -> MSYEquilibrium

Compute MSY-based reference points (Bo, Bmsy, Fmsy, Umsy) using Newton-Raphson
optimization over steady-state yield equilibrium equations. Evaluates
equilibrium recruitment, yield, and spawning biomass at leading parameters `theta`.
"""
function calc_reference_points(m::StockModel, mort, log_sel, theta::AbstractVector = m.x0[m.layout.theta.range])
    nfleet = length(m.ifleet)
    if nfleet == 0
        error("calc_reference_points requires at least one fishing gear with positive allocation.")
    end

    V = zeros(Float64, nfleet, m.nages)
    ak = zeros(Float64, nfleet)
    for (k, j) in enumerate(m.ifleet)
        V[k, :] .= exp.(log_sel[j][end, :])
        ak[k] = m.data.allocation[j]
    end
    ak ./= sum(ak)

    ro = exp(theta[1])
    h = theta[2]
    m_val = mort.m_bar
    spawn_frac = m.control.misc.spawn_fraction
    wa = m.avg_wt
    fa = m.avg_fec

    # 1. Unconstrained Fmsy. Newton's method on the equilibrium yield gradient is run by
    # the shared solver; the residual stacks each gear's dYe with one scalar MSY
    # condition (total yield gradient zero) so the system is square and well determined.
    f_lower_msy = 1.0e-5
    f_upper_msy = 300.0
    tol_msy = 1.0e-9
    max_iter_msy = 300

    function msy_residual(f::AbstractVector{<:Real})
        eq = calc_equilibrium_point(f, ro, h, m_val, spawn_frac, wa, fa, V)
        return vcat(eq.dye, sum(eq.dye))
    end

    function msy_jacobian(f::AbstractVector{<:Real})
        eq = calc_equilibrium_point(f, ro, h, m_val, spawn_frac, wa, fa, V)
        # d(sum dYe)/dF_j = sum over k of the second derivative of yield
        extra = vec(sum(eq.d2ye, dims = 1))
        return vcat(eq.d2ye, extra')
    end

    fmsy0 = fill(0.6 * m_val, nfleet)
    fe = Solvers.solve_newton_vector(
        msy_residual, msy_jacobian, fmsy0;
        tol = tol_msy, max_iter = max_iter_msy,
        lower = fill(f_lower_msy, nfleet),
        upper = fill(f_upper_msy, nfleet))
    fmsy = copy(fe)
    eq_msy = calc_equilibrium_point(fmsy, ro, h, m_val, spawn_frac, wa, fa, V)

    # 2. Allocated Fall (respecting gear allocation ak). One scalar condition on F_bar:
    # d(sum Ye)/dF_bar == 0, with the gear allocation re-imposed at every trial.
    fbar = 0.6 * m_val
    fall = fbar .* ak
    for _ in 1:max_iter_msy
        eq1 = calc_equilibrium_point(fbar .* ak, ro, h, m_val, spawn_frac, wa, fa, V)
        tot_ye = sum(eq1.ye)
        lambda = ak ./ ((tot_ye > 0 ? eq1.ye ./ tot_ye : ak) .+ 1e-15)

        fk = fbar .* lambda
        eq2 = calc_equilibrium_point(fk, ro, h, m_val, spawn_frac, wa, fa, V)
        dYe = sum(eq2.dye)
        d2Ye = sum(diag(eq2.d2ye))

        if abs(dYe) < tol_msy
            fall = fk
            break
        end

        step = dYe / d2Ye
        fbar -= step
        if (f_lower_msy - fbar) * (fbar - f_upper_msy) < 0.0
            fbar += 0.98 * step
        end
        fall = fbar .* lambda
    end

    umsy = sum(eq_msy.ye) / (eq_msy.bi + 1e-15)

    return MSYEquilibrium(
        eq_msy.bo, eq_msy.be, fmsy, fall, eq_msy.ye,
        umsy, eq_msy.spr, eq_msy.re, eq_msy.be
    )
end
