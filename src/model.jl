# ------------------------------------------------------------------------------------
# Model construction: derived data quantities, parameter layout and estimation phases.
# ------------------------------------------------------------------------------------

"""
    PhaseContext(phase, last_phase)

Estimation phase state. A parameter with phase `φ` is active when `0 < φ ≤ phase`.
`last_phase` switches the catch and mean-F standard deviations and the recruitment
deviation penalties, as in the full model convergence phase.
"""
struct PhaseContext
    phase::Int
    last_phase::Bool
end

is_active(phase::Integer, ctx::PhaseContext) = 0 < phase <= ctx.phase

"""
    ParameterBlock

Contiguous block of the parameter vector. `dims = (rows, cols)`; matrices
scalar and vector parameters each get one element; matrices (`sel_par`) are stored
column-major. `phase` is per element.
"""
struct ParameterBlock
    name::String
    range::UnitRange{Int}
    dims::Tuple{Int, Int}
    phase::Vector{Int}
end

"""
    ParameterLayout

Ordering, bounds and phases of all estimated quantities:
`theta`, `sel_par[1:ngear]`, `log_ft_pars`, `init_log_rec_devs`, `log_rec_devs`,
`log_m_nodes`.
"""
struct ParameterLayout
    theta::ParameterBlock
    sel_par::Vector{ParameterBlock}
    log_ft_pars::ParameterBlock
    init_log_rec_devs::ParameterBlock
    log_rec_devs::ParameterBlock
    log_m_nodes::ParameterBlock
    n::Int
    lb::Vector{Float64}
    ub::Vector{Float64}
    phase::Vector{Int}
    names::Vector{String}
    max_phase::Int
end

"""
    StockModel(data, control, projection; retro_yrs = 0)

Assemble a statistical catch-at-age model from parsed inputs.

Derived quantities (`a` = age, `t` = year):

- length- and weight-at-age `l_a = L∞(1 - e^{-k(a - t₀)})`, `w_a = α l_a^β`;
- weight-at-age `w_{t,a}` (empirical rows replace `w_a`), and fecundity
  `f_{t,a} = w_{t,a} / (1 + exp((a_h - a)/g_h))`;
- mean weight and fecundity over `pf_cntrl[3]:pf_cntrl[4]`, used for reference points
  and projections; `w_{T+1,a}` is set to the mean weight;
- `f̄_a`, the mean fecundity over all data years, used in the stock–recruit slope;
- standardised weight deviations per age (selectivity type 8).

`retro_yrs > 0` removes the last `retro_yrs` years from the fitted time series
(retrospective analysis). Data-derived weight and fecundity averages use the full
data series.
"""
struct StockModel
    data::StockData
    control::StockControl
    projection::ProjectionControl
    pf_cntrl::Vector{Int}
    retro_yrs::Int
    syr::Int
    nyr::Int
    sage::Int
    nage::Int
    ny::Int
    nages::Int
    ages::Vector{Int}
    la::Vector{Float64}
    wa::Vector{Float64}
    wt_obs::Matrix{Float64}
    fec::Matrix{Float64}
    avg_wt::Vector{Float64}
    avg_fec::Vector{Float64}
    fa_bar::Vector{Float64}
    wt_dev::Matrix{Float64}
    ft_index::Matrix{Int}
    ifleet::Vector{Int}
    survey_rows::Vector{Vector{Int}}
    agecomp_rows::Vector{Vector{Int}}
    layout::ParameterLayout
    x0::Vector{Float64}
end

ngear(m::StockModel) = m.data.ngear
yrow(m::StockModel, year::Integer) = year - m.syr + 1

function StockModel(data::StockData, ctl::StockControl, pfc::ProjectionControl;
                    retro_yrs::Integer = 0)
    syr, sage, nage = data.syr, data.sage, data.nage
    retro_yrs >= 0 || error("retro_yrs must be non-negative.")
    nyr = data.nyr - retro_yrs
    nyr - sage > syr ||
        error("Too many retrospective years: the stock–recruitment series is empty.")
    pf = copy(pfc.pf_cntrl)
    (pf[1] < syr || pf[3] < syr || pf[5] < syr) &&
        error("A start year in the projection control file precedes $syr.")
    (pf[2] > data.nyr || pf[4] > data.nyr || pf[6] > data.nyr) &&
        error("An end year in the projection control file exceeds $(data.nyr).")
    pf[1] <= pf[2] && pf[3] <= pf[4] || error("Projection control year ranges reversed.")

    ages = collect(sage:nage)
    nages = length(ages)
    la = data.linf .* (1 .- exp.(-data.vonbk .* (ages .- data.to)))
    wa = data.a .* la .^ data.b
    mat = plogis.(ages, data.ah, data.gh)

    # weight and fecundity at age for years syr:(data.nyr + 1)
    nyw = data.nyr - syr + 2
    wt_obs = repeat(permutedims(wa), nyw, 1)
    for i in axes(data.wt_input, 1)
        yr = data.wt_input[i, 1]
        isinteger(yr) && syr <= yr <= data.nyr + 1 ||
            error("Empirical weight-at-age year $yr outside $syr:$(data.nyr + 1).")
        wt_obs[Int(yr) - syr + 1, :] .= data.wt_input[i, 2:end]
    end
    fec = wt_obs .* permutedims(mat)
    r34 = (pf[3] - syr + 1):(pf[4] - syr + 1)
    avg_wt = vec(sum(wt_obs[r34, :]; dims = 1)) ./ length(r34)
    avg_fec = vec(sum(fec[r34, :]; dims = 1)) ./ length(r34)
    wt_obs[end, :] .= avg_wt
    fa_bar = vec(sum(fec[1:(data.nyr - syr + 1), :]; dims = 1)) ./ (data.nyr - syr + 1)
    minimum(wt_obs) > 0 ||
        error("Observed mean weight-at-age of 0 in the data file; please fix.")
    wt_dev = similar(wt_obs)
    for j in 1:nages
        col = wt_obs[:, j]
        mu = sum(col) / length(col)
        sd = sqrt(sum(abs2, col .- mu) / length(col))   # population standard deviation
        wt_dev[:, j] .= (col .- mu) ./ sd
    end

    # Retrospective truncation of the projection control years
    if retro_yrs > 0
        pf[2] = min(pf[2], nyr)
        pf[4] = min(pf[4], nyr)
        pf[6] = min(pf[6], nyr)
        pf[1] <= pf[2] || error("Retrospective peel removes the mean-M year range.")
    end

    ny = nyr - syr + 1
    ng = data.ngear
    ft_index = zeros(Int, ng, ny)
    count = 0
    for k in 1:ng, t in 1:ny
        if data.obs_ct[k, t] > 0
            count += 1
            ft_index[k, t] = count
        end
    end
    ifleet = findall(>(0), data.allocation)

    survey_rows = Vector{Int}[]
    for (i, s) in enumerate(data.surveys)
        all(in(1:ng), s.gear) || error("Survey $i refers to an undefined gear.")
        all(>=(syr), s.year) || error("Survey $i has years before $syr.")
        rows = findall(<=(nyr), s.year)
        length(rows) >= 2 || error("Survey $i has fewer than 2 observations to fit.")
        push!(survey_rows, rows)
    end
    agecomp_rows = Vector{Int}[]
    for (k, a) in enumerate(data.agecomps)
        all(in(1:ng), a.gear) || error("Age-composition series $k refers to an undefined gear.")
        all(>=(syr), a.year) || error("Age-composition series $k has years before $syr.")
        push!(agecomp_rows, findall(<=(nyr), a.year))
    end

    layout, x0 = build_layout(data, ctl, ny, count)
    return StockModel(data, ctl, pfc, pf, retro_yrs, syr, nyr, sage, nage, ny,
                      nages, ages, la, wa, wt_obs, fec, avg_wt, avg_fec, fa_bar, wt_dev,
                      ft_index, ifleet, survey_rows, agecomp_rows, layout, x0)
end

"""
    StockModel(data::StockData, ctl::StockControl; kwargs...) -> StockModel

Construct a `StockModel` with default projection controls spanning the assessment years.
"""
function StockModel(data::StockData, ctl::StockControl; kwargs...)
    pfc = ProjectionControl([0.0], [data.syr, data.nyr, data.syr, data.nyr, data.syr, data.nyr])
    return StockModel(data, ctl, pfc; kwargs...)
end

"""
    StockModel(dir::AbstractString; kwargs...) -> StockModel

Construct a `StockModel` directly from a dataset directory containing `config.toml`
and related CSV data series.
"""
function StockModel(dir::AbstractString; kwargs...)
    data, ctl, pfc = load_dataset(dir)
    if isnothing(pfc)
        return StockModel(data, ctl; kwargs...)
    else
        return StockModel(data, ctl, pfc; kwargs...)
    end
end

"""
    selectivity_dims(isel_type, age_nodes, yr_nodes, nages, ny) -> (rows, cols)

Dimensions of the selectivity parameter matrix of one gear:

| type | form                                         | rows        | cols            |
|:----:|:---------------------------------------------|:------------|:----------------|
| 0    | uniform (no parameters)                      | 1           | 0               |
| 1, 6 | logistic in age (6 = fixed)                  | 1           | 2               |
| 2    | age-specific coefficients                    | 1           | 
ages - 1`     |
| 3    | cubic spline over age nodes                  | 1           | `age_nodes`     |
| 4    | cubic spline over age nodes, each year       | 
y`        | `age_nodes`     |
| 5    | bicubic spline over age × year nodes         | `age_nodes` | `yr_nodes`      |
| 7    | logistic in relative log weight              | 1           | 2               |
| 8    | logistic in age + weight deviations          | 1           | 3               |
| 11   | logistic in length                           | 1           | 2               |
| 12   | cubic spline in length                       | 1           | 
ages - 1`     |
"""
function selectivity_dims(t::Integer, age_nodes::Integer, yr_nodes::Integer,
                          nages::Integer, ny::Integer)
    t == 0 && return (1, 0)
    t in (1, 6, 7, 11) && return (1, 2)
    t in (2, 12) && return (1, nages - 1)
    t == 3 && return (1, age_nodes)
    t == 4 && return (ny, age_nodes)
    t == 5 && return (age_nodes, yr_nodes)
    t == 8 && return (1, 3)
    error("Unknown selectivity type $t.")
end

function build_layout(data::StockData, ctl::StockControl, ny::Integer, ft_count::Integer)
    nages = data.nage - data.sage + 1
    lb, ub, phase, names, x0 = Float64[], Float64[], Int[], String[], Float64[]
    pos = Ref(0)
    function add!(name, n, dims, lo, hi, ph, init)
        r = (pos[] + 1):(pos[] + n)
        pos[] += n
        append!(lb, lo); append!(ub, hi); append!(phase, ph); append!(x0, init)
        return ParameterBlock(name, r, dims, collect(Int, ph))
    end
    tc = ctl.theta_control
    theta = add!("theta", 7, (7, 1), tc[:, 2], tc[:, 3], Int.(tc[:, 4]), tc[:, 1])
    append!(names, ["theta[$i]" for i in 1:7])

    sel = ParameterBlock[]
    for k in 1:data.ngear
        t = ctl.isel_type[k]
        t in (3, 4, 5) && ctl.age_nodes[k] < 2 &&
            error("Gear $k: spline selectivity needs at least 2 age nodes.")
        t == 5 && ctl.yr_nodes[k] < 2 &&
            error("Gear $k: bicubic selectivity needs at least 2 year nodes.")
        rows, cols = selectivity_dims(t, ctl.age_nodes[k], ctl.yr_nodes[k], nages, ny)
        n = rows * cols
        ph = (t == 6 && ctl.sel_phz[k] > 0) ? -1 : ctl.sel_phz[k]
        init = zeros(n)
        if (t == 1 || t == 6 || t >= 7) && n >= 2
            ctl.ahat[k] > 0 && ctl.ghat[k] > 0 ||
                error("Gear $k: ahat and ghat must be positive for logistic forms.")
            init[1] = log(ctl.ahat[k])
            init[2] = log(ctl.ghat[k])
        end
        push!(sel, add!("sel_par[$k]", n, (rows, cols), fill(-15.0, n), fill(15.0, n),
                        fill(ph, n), init))
        # Column-major, matching the storage order used by `unpack`/`assemble_x`.
        append!(names, ["sel_par[$k][$i,$j]" for j in 1:cols for i in 1:rows])
    end

    ft = add!("log_ft_pars", ft_count, (ft_count, 1), fill(-30.0, ft_count),
              fill(3.0, ft_count), fill(1, ft_count), fill(log(0.1), ft_count))
    append!(names, ["log_ft_pars[$i]" for i in 1:ft_count])

    ni = nages - 1
    init_phz = ctl.misc.unfished_start ? -1 : 2
    idev = add!("init_log_rec_devs", ni, (ni, 1), fill(-15.0, ni), fill(15.0, ni),
                fill(init_phz, ni), zeros(ni))
    append!(names, ["init_log_rec_devs[$(data.sage + i)]" for i in 1:ni])
    rdev = add!("log_rec_devs", ny, (ny, 1), fill(-15.0, ny), fill(15.0, ny),
                fill(2, ny), zeros(ny))
    append!(names, ["log_rec_devs[$(data.syr + i - 1)]" for i in 1:ny])
    nm = ctl.misc.n_m_nodes
    mph = nm > 0 ? ctl.misc.m_dev_phase : -1
    mnod = add!("log_m_nodes", nm, (nm, 1), fill(-5.0, nm), fill(5.0, nm), fill(mph, nm),
                zeros(nm))
    append!(names, ["log_m_nodes[$i]" for i in 1:nm])

    pos_phases = filter(>(0), phase)
    max_phase = isempty(pos_phases) ? 1 : maximum(pos_phases)
    layout = ParameterLayout(theta, sel, ft, idev, rdev, mnod, pos[], lb, ub, phase,
                             names, max_phase)
    return layout, x0
end

"""
    unpack(layout, x) -> NamedTuple

Split the parameter vector into named blocks. `sel_par` is a vector of
`rows × cols` matrices (one per gear), and each is stored **column-major**, so
element `(i, j)` of a matrix block occupies `x[block_start + (j - 1) * rows + i]`.
"""
function unpack(L::ParameterLayout, x::AbstractVector)
    length(x) == L.n || error("Parameter vector has length $(length(x)); expected $(L.n)).")
    sel = map(L.sel_par) do b
        r, c = b.dims
        reshape(x[b.range], r, c)
    end
    return (theta = x[L.theta.range], sel_par = sel, log_ft_pars = x[L.log_ft_pars.range],
            init_log_rec_devs = x[L.init_log_rec_devs.range],
            log_rec_devs = x[L.log_rec_devs.range], log_m_nodes = x[L.log_m_nodes.range])
end

"""
    assemble_x(L::ParameterLayout, parts::NamedTuple) -> Vector{Float64}

Concatenate named parameter blocks back into a full vector. The inverse of
[`unpack`](@ref); the field names must match `unpack`'s output. `sel_par` matrices
are stored column-major, consistently with `unpack`.
"""
function assemble_x(L::ParameterLayout, parts::NamedTuple)
    out = Vector{Float64}(undef, L.n)
    out[L.theta.range] .= vec(collect(Float64, parts.theta))
    for (k, block) in enumerate(L.sel_par)
        mtx = collect(Float64, parts.sel_par[k])
        r, c = block.dims
        size(mtx) == (r, c) ||
            error("sel_par[$k] has size $(size(mtx)); expected ($r, $c).")
        out[block.range] .= vec(mtx)
    end
    out[L.log_ft_pars.range] .= vec(collect(Float64, parts.log_ft_pars))
    out[L.init_log_rec_devs.range] .= vec(collect(Float64, parts.init_log_rec_devs))
    out[L.log_rec_devs.range] .= vec(collect(Float64, parts.log_rec_devs))
    out[L.log_m_nodes.range] .= vec(collect(Float64, parts.log_m_nodes))
    all(isfinite, out) ||
        error("assemble_x: at least one value is non-finite.")
    return out
end

"Indices of parameters active in `phase`."
active_indices(L::ParameterLayout, phase::Integer) = findall(p -> 0 < p <= phase, L.phase)
