# ------------------------------------------------------------------------------------
# AssessmentTools I/O: Native TOML and CSV dataset loader and report writers.
# ------------------------------------------------------------------------------------

using ..Config

"""
    ProjectionControl(tac, pf_cntrl)

Controls for forward projections:
- `tac`: Vector of Total Allowable Catch values (tonnes) evaluated in decision tables.
- `pf_cntrl`: 6-element vector defining year ranges `[m_start, m_end, wt_start, wt_end, sel_start, sel_end]`.
"""
struct ProjectionControl
    tac::Vector{Float64}
    pf_cntrl::Vector{Int}
end

"""
    SurveySeries

One relative abundance index series:
- `year`: Vector of survey observation years.
- `index`: Relative abundance index values.
- `gear`: Associated gear index.
- `weight`: Relative precision weight for likelihood weighting.
- `timing`: Fraction of annual total mortality elapsed when the survey occurs.
- `survey_type`: 1 = vulnerable numbers, 2 = vulnerable biomass, 3 = spawning biomass.
"""
struct SurveySeries
    year::Vector{Int}
    index::Vector{Float64}
    gear::Vector{Int}
    weight::Vector{Float64}
    timing::Vector{Float64}
    survey_type::Int
end

"""
    AgeCompSeries

Age-composition observations for one series:
- `year`: Vector of observation years.
- `gear`: Associated gear index.
- `sage`: Youngest age class in series.
- `nage`: Oldest age class in series.
- `obs`: Matrix of observed counts or proportions (rows = years, columns = ages).
"""
struct AgeCompSeries
    year::Vector{Int}
    gear::Vector{Int}
    sage::Int
    nage::Int
    obs::Matrix{Float64}
end

"""
    StockData

Primary assessment dataset container:
- `syr`, `nyr`: Start and end years of the assessment horizon.
- `sage`, `nage`: Youngest and oldest age classes modeled.
- `ngear`: Number of fishing gears and survey indices.
- `allocation`: Relative catch allocation among gears (sums to 1.0).
- `catch_type`: Units of catch per gear (1 = biomass, 2 = numbers, 3 = harvest rate).
- `fixed_m`, `linf`, `vonbk`, `to`, `a`, `b`, `ah`, `gh`: Life-history parameters.
- `obs_ct`: Observed catch matrix of size `(ngear, nyears)`.
- `surveys`: Vector of [`SurveySeries`](@ref).
- `agecomps`: Vector of [`AgeCompSeries`](@ref).
- `wt_input`: Empirical weight-at-age matrix (optional).
"""
struct StockData
    syr::Int
    nyr::Int
    sage::Int
    nage::Int
    ngear::Int
    allocation::Vector{Float64}
    catch_type::Vector{Int}
    fixed_m::Float64
    linf::Float64
    vonbk::Float64
    to::Float64
    a::Float64
    b::Float64
    ah::Float64
    gh::Float64
    obs_ct::Matrix{Float64}
    surveys::Vector{SurveySeries}
    agecomps::Vector{AgeCompSeries}
    wt_input::Matrix{Float64}
end

"""
    MiscControls

Assessment model runtime options and hyper-parameters:
- `verbose`: Verbose printing toggle.
- `rec_type`: 1 = Beverton–Holt, 2 = Ricker.
- `sig_c_early`, `sig_c_last`: Standard deviations of log catch residuals.
- `unfished_start`: Initialise numbers at unfished equilibrium.
- `min_prop`: Minimum proportion-at-age before pooling composition bins.
- `fbar`: Mean fishing mortality target for regularisation.
- `fbar_sd_early`, `fbar_sd_last`: Standard deviation for mean-F penalty.
- `m_dev_phase`: Estimation phase of natural mortality spline nodes.
- `m_dev_sd`: Prior standard deviation for M spline variations.
- `n_m_nodes`: Number of natural mortality spline nodes.
- `spawn_fraction`: Fraction of annual total mortality occurring prior to spawning.
- `agecomp_like`: 1 = multivariate logistic, 2 = multinomial.
- `raw`: 14-element parameter vector.
"""
struct MiscControls
    verbose::Bool
    rec_type::Int
    sig_c_early::Float64
    sig_c_last::Float64
    unfished_start::Bool
    min_prop::Float64
    fbar::Float64
    fbar_sd_early::Float64
    fbar_sd_last::Float64
    m_dev_phase::Int
    m_dev_sd::Float64
    n_m_nodes::Int
    spawn_fraction::Float64
    agecomp_like::Int
    raw::Vector{Float64}
end

function MiscControls(c::AbstractVector{<:Real})
    length(c) == 14 || error("cntrl must have 14 elements, found $(length(c)).")
    for (i, v) in ((2, c[2]), (10, c[10]), (12, c[12]), (14, c[14]))
        isinteger(v) || error("cntrl($i) must be an integer, found $v.")
    end
    c[2] in (1, 2) || error("cntrl(2) recruitment model must be 1 or 2, found $(c[2]).")
    c[14] in (1, 2) || error("cntrl(14) age-composition likelihood must be 1 or 2.")
    c[12] >= 0 || error("cntrl(12) number of M nodes must be non-negative.")
    0 <= c[13] <= 1 || error("cntrl(13) spawn fraction must lie in [0, 1], found $(c[13]).")
    return MiscControls(c[1] != 0, Int(c[2]), c[3], c[4], c[5] != 0, c[6], c[7], c[8],
                        c[9], Int(c[10]), c[11], Int(c[12]), c[13], Int(c[14]),
                        collect(Float64, c))
end

"""
    EstimationControls

Solver settings for phased estimation, read from the `[estimation]` section of
`config.toml`. Exposed at the user level so run length and convergence tolerance
are not compiled into the optimiser call.

# Fields
- `max_iterations::Int`: maximum iterations per phase.
- `g_abstol::Float64`: absolute-gradient convergence tolerance.
"""
struct EstimationControls
    max_iterations::Int
    g_abstol::Float64
end

"""
    PenaltyControls

Penalty weights used in the objective function, read from the `[penalties]`
section of `config.toml`. Every value here was previously a literal in
`src/objective.jl`.

# Fields
- `rec_dev_sd::Float64`: prior standard deviation for recruitment deviations in the
  final phase.
- `rec_dev_penalty_weight::Float64`: weight of the early-phase quadratic penalty on
  recruitment deviations and initial recruitment deviations.
- `sel_zero_mean_weight::Float64`: weight of the zero-mean penalty on selectivity
  parameter rows and columns.
- `m_dev_node_weight::Float64`: weight of the quadratic penalty on natural-mortality
  spline nodes.
- `agecomp_process_error::Float64`: HCAM process-error adjustment used to convert
  raw age-composition counts into an effective sample size.
"""
struct PenaltyControls
    rec_dev_sd::Float64
    rec_dev_penalty_weight::Float64
    sel_zero_mean_weight::Float64
    m_dev_node_weight::Float64
    agecomp_process_error::Float64
end

"""
    DecisionControls

Decision-table thresholds used by [`project_population`](@ref), read from the
`[decision]` section of `config.toml`. Every threshold was previously a literal in
`src/simulation_projection.jl`.

# Fields
- `sb_bo_thresholds::Vector{Float64}`: spawning-biomass thresholds as fractions of
  `B₀`; a probability is reported for each.
- `sb_bmsy_thresholds::Vector{Float64}`: spawning-biomass thresholds as fractions of
  `B_MSY`.
- `u_umsy_thresholds::Vector{Float64}`: exploitation-rate thresholds as fractions of
  `U_MSY`.
- `age_floor::Int`: minimum age counted in the age-`3+` exploitation rate.
- `age3_exploitation_limit::Float64`: exploitation-rate limit on age-3+ fish.
- `epsilon::Float64`: small value guarding division by a zero denominator.
"""
struct DecisionControls
    sb_bo_thresholds::Vector{Float64}
    sb_bmsy_thresholds::Vector{Float64}
    u_umsy_thresholds::Vector{Float64}
    age_floor::Int
    age3_exploitation_limit::Float64
    epsilon::Float64
end

"""
    StockControl

Control parameters and priors:
- `theta_control`: `npar × 7` matrix with columns
  `[initial, lower, upper, phase, prior_type, p1, p2]` for leading parameters
  `θ = (log R₀, h, log M, log R̄, log R_init, ρ, ϑ)`.
- `isel_type`: Selectivity type identifier per gear.
- `ahat`, `ghat`: Selectivity inflection and slope per gear.
- `age_nodes`, `yr_nodes`: Spline node counts per gear.
- `sel_phz`: Estimation phase for selectivity parameters.
- `sel_2nd_diff_wt`, `sel_dome_wt`: Smoothing penalty weights for spline selectivity.
- `q_prior`, `q_mu`, `q_sd`: Survey catchability prior distributions.
- `misc`: [`MiscControls`](@ref) structure.
- `estimation`: [`EstimationControls`](@ref) solver settings.
- `penalties`: [`PenaltyControls`](@ref) penalty weights.
- `decision`: [`DecisionControls`](@ref) decision-table thresholds.
"""
struct StockControl
    theta_control::Matrix{Float64}
    isel_type::Vector{Int}
    ahat::Vector{Float64}
    ghat::Vector{Float64}
    age_nodes::Vector{Int}
    yr_nodes::Vector{Int}
    sel_phz::Vector{Int}
    sel_2nd_diff_wt::Vector{Float64}
    sel_dome_wt::Vector{Float64}
    q_prior::Vector{Int}
    q_mu::Vector{Float64}
    q_sd::Vector{Float64}
    misc::MiscControls
    estimation::EstimationControls
    penalties::PenaltyControls
    decision::DecisionControls
end

# ------------------------------------------------------------------------------------
# Native Modular Dataset Loader (TOML + CSV)
# ------------------------------------------------------------------------------------

"""
    load_dataset(dir::AbstractString) -> (StockData, StockControl, Union{ProjectionControl, Nothing})

Load a complete stock assessment dataset from a directory containing `config.toml`
and corresponding CSV files (`catches.csv`, `survey_series_*.csv`, `agecomp_series_*.csv`,
and optionally `weight_at_age.csv`).
"""
function load_dataset(dir::AbstractString)
    cfg_file = joinpath(dir, "config.toml")
    isfile(cfg_file) || error("Configuration file not found: $cfg_file")
    cfg = TOML.parsefile(cfg_file)

    dim = cfg["dimensions"]
    bio = cfg["biology"]
    ctrl_dict = cfg["controls"]
    sel_dict = cfg["selectivity"]
    qp_dict = cfg["survey_priors"]
    th_dict = cfg["theta_control"]

    syr = Int(dim["syr"])
    nyr = Int(dim["nyr"])
    sage = Int(dim["sage"])
    nage = Int(dim["nage"])
    ngear = Int(dim["ngear"])
    allocation = Float64.(dim["allocation"])
    catch_type = haskey(dim, "catch_type") ? Int.(dim["catch_type"]) : ones(Int, ngear)
    # Catches from catches.csv
    catch_file = joinpath(dir, "catches.csv")
    isfile(catch_file) || error("Catches file not found: $catch_file")
    catch_lines = filter(!isempty, strip.(readlines(catch_file)))
    nyears = nyr - syr + 1
    obs_ct = Matrix{Float64}(undef, ngear, nyears)
    for (t, line) in enumerate(catch_lines[2:end])
        tokens = split(line, ',')
        for g in 1:ngear
            obs_ct[g, t] = parse(Float64, tokens[g + 1])
        end
    end

    # Surveys from survey_series_*.csv
    surveys = SurveySeries[]
    s_idx = 1
    while true
        sf = joinpath(dir, "survey_series_$s_idx.csv")
        !isfile(sf) && break
        s_lines = filter(!isempty, strip.(readlines(sf)))
        s_yrs = Int[]
        s_idx_val = Float64[]
        s_gear = Int[]
        s_wt = Float64[]
        s_timing = Float64[]
        s_type = 2
        for line in s_lines[2:end]
            toks = split(line, ',')
            push!(s_yrs, parse(Int, toks[1]))
            push!(s_idx_val, parse(Float64, toks[2]))
            push!(s_gear, parse(Int, toks[3]))
            push!(s_wt, parse(Float64, toks[4]))
            push!(s_timing, parse(Float64, toks[5]))
            s_type = parse(Int, toks[6])
        end
        push!(surveys, SurveySeries(s_yrs, s_idx_val, s_gear, s_wt, s_timing, s_type))
        s_idx += 1
    end

    # Age compositions from agecomp_series_*.csv
    agecomps = AgeCompSeries[]
    ac_idx = 1
    while true
        acf = joinpath(dir, "agecomp_series_$ac_idx.csv")
        !isfile(acf) && break
        ac_lines = filter(!isempty, strip.(readlines(acf)))
        ac_yrs = Int[]
        ac_gear = Int[]
        ac_mat_rows = Vector{Float64}[]
        n_ages = nage - sage + 1
        for line in ac_lines[2:end]
            toks = split(line, ',')
            push!(ac_yrs, parse(Int, toks[1]))
            push!(ac_gear, parse(Int, toks[2]))
            row_vals = [parse(Float64, toks[2 + a]) for a in 1:n_ages]
            push!(ac_mat_rows, row_vals)
        end
        obs_mat = isempty(ac_mat_rows) ? Matrix{Float64}(undef, 0, n_ages) :
                  reduce(vcat, [transpose(r) for r in ac_mat_rows])
        push!(agecomps, AgeCompSeries(ac_yrs, ac_gear, sage, nage, obs_mat))
        ac_idx += 1
    end

    # Empirical weight-at-age (optional)
    wt_file = joinpath(dir, "weight_at_age.csv")
    wt_input = if isfile(wt_file)
        wt_lines = filter(!isempty, strip.(readlines(wt_file)))
        n_cols = nage - sage + 2
        wt_rows = Vector{Float64}[]
        for line in wt_lines[2:end]
            push!(wt_rows, parse.(Float64, split(line, ',')))
        end
        isempty(wt_rows) ? Matrix{Float64}(undef, 0, n_cols) :
        reduce(vcat, [transpose(r) for r in wt_rows])
    else
        Matrix{Float64}(undef, 0, nage - sage + 2)
    end

    data = StockData(
        syr, nyr, sage, nage, ngear, allocation, catch_type,
        Float64(bio["fixed_m"]), Float64(bio["linf"]), Float64(bio["vonbk"]),
        Float64(bio["to"]), Float64(bio["a"]), Float64(bio["b"]),
        Float64(bio["ah"]), Float64(bio["gh"]),
        obs_ct, surveys, agecomps, wt_input
    )

    # Miscellaneous controls
    misc = MiscControls(
        Bool(ctrl_dict["verbose"]), Int(ctrl_dict["rec_type"]),
        Float64(ctrl_dict["sig_c_early"]), Float64(ctrl_dict["sig_c_last"]),
        Bool(ctrl_dict["unfished_start"]), Float64(ctrl_dict["min_prop"]),
        Float64(ctrl_dict["fbar"]), Float64(ctrl_dict["fbar_sd_early"]),
        Float64(ctrl_dict["fbar_sd_last"]), Int(ctrl_dict["m_dev_phase"]),
        Float64(ctrl_dict["m_dev_sd"]), Int(ctrl_dict["n_m_nodes"]),
        Float64(ctrl_dict["spawn_fraction"]), Int(ctrl_dict["agecomp_like"]),
        [
            Float64(ctrl_dict["verbose"]), Float64(ctrl_dict["rec_type"]),
            Float64(ctrl_dict["sig_c_early"]), Float64(ctrl_dict["sig_c_last"]),
            Float64(ctrl_dict["unfished_start"]), Float64(ctrl_dict["min_prop"]),
            Float64(ctrl_dict["fbar"]), Float64(ctrl_dict["fbar_sd_early"]),
            Float64(ctrl_dict["fbar_sd_last"]), Float64(ctrl_dict["m_dev_phase"]),
            Float64(ctrl_dict["m_dev_sd"]), Float64(ctrl_dict["n_m_nodes"]),
            Float64(ctrl_dict["spawn_fraction"]), Float64(ctrl_dict["agecomp_like"])
        ]
    )

    # Leading parameter control table (7 leading parameters)
    theta_keys = ["log_ro", "steepness_h", "log_m", "log_rbar", "log_rinit", "rho", "vartheta"]
    n_theta = length(theta_keys)
    tc = Matrix{Float64}(undef, n_theta, 7)
    for (i, k) in enumerate(theta_keys)
        p = th_dict[k]
        tc[i, 1] = Float64(p["initial"])
        tc[i, 2] = Float64(p["lower"])
        tc[i, 3] = Float64(p["upper"])
        tc[i, 4] = Float64(p["phase"])
        tc[i, 5] = Float64(p["prior_type"])
        tc[i, 6] = Float64(p["p1"])
        tc[i, 7] = Float64(p["p2"])
    end

    ctl = StockControl(
        tc,
        Int.(sel_dict["isel_type"]),
        Float64.(sel_dict["ahat"]),
        Float64.(sel_dict["ghat"]),
        Int.(sel_dict["age_nodes"]),
        Int.(sel_dict["yr_nodes"]),
        Int.(sel_dict["sel_phz"]),
        Float64.(sel_dict["sel_2nd_diff_wt"]),
        Float64.(sel_dict["sel_dome_wt"]),
        Int.(qp_dict["q_prior"]),
        Float64.(qp_dict["q_mu"]),
        Float64.(qp_dict["q_sd"]),
        misc,
        _load_estimation(cfg, cfg_file),
        _load_penalties(cfg, cfg_file),
        _load_decision(cfg, cfg_file)
    )

    # Projection control
    pfc = if haskey(cfg, "projection")
        proj_dict = cfg["projection"]
        ProjectionControl(Float64.(proj_dict["tac"]), Int.(proj_dict["pf_cntrl"]))
    else
        nothing
    end

    return data, ctl, pfc
end

"""
    _load_estimation(cfg, path) -> EstimationControls

Read `[estimation]` from a configuration table. Every key is required, so the
optimiser settings are explicit in the file rather than defaulted in code.
"""
function _load_estimation(cfg::AbstractDict, path::AbstractString)
    d = Config.require_section(cfg, "estimation", path)
    where = "$path section [estimation]"
    n = Config.load_int(d, "max_iterations", where)
    n >= 1 || error("'max_iterations' in $where must be at least 1, got $n.")
    g = Config.positive(d, "g_abstol", where)
    return EstimationControls(n, g)
end

"""
    _load_penalties(cfg, path) -> PenaltyControls

Read `[penalties]` from a configuration table. Every key is required.
"""
function _load_penalties(cfg::AbstractDict, path::AbstractString)
    d = Config.require_section(cfg, "penalties", path)
    where = "$path section [penalties]"
    return PenaltyControls(
        Config.positive(d, "rec_dev_sd", where),
        Config.non_negative(d, "rec_dev_penalty_weight", where),
        Config.non_negative(d, "sel_zero_mean_weight", where),
        Config.non_negative(d, "m_dev_node_weight", where),
        Config.non_negative(d, "agecomp_process_error", where)
    )
end

"""
    _load_decision(cfg, path) -> DecisionControls

Read `[decision]` from a configuration table. Every key is required, so each
decision-table threshold is stated by the user rather than fixed in code.
"""
function _load_decision(cfg::AbstractDict, path::AbstractString)
    d = Config.require_section(cfg, "decision", path)
    where = "$path section [decision]"
    bo_t = Config.load_reals(d, "sb_bo_thresholds", where)
    bmsy_t = Config.load_reals(d, "sb_bmsy_thresholds", where)
    u_t = Config.load_reals(d, "u_umsy_thresholds", where)
    # `DecisionMetrics` has a fixed number of threshold fields, so the configured
    # vectors must match them. An explicit message beats an index error later.
    length(bo_t) == 2 ||
        error("'sb_bo_thresholds' in $where must have 2 entries (one per DecisionMetrics field), got $(length(bo_t)).")
    length(bmsy_t) == 2 ||
        error("'sb_bmsy_thresholds' in $where must have 2 entries, got $(length(bmsy_t)).")
    length(u_t) == 3 ||
        error("'u_umsy_thresholds' in $where must have 3 entries, got $(length(u_t)).")
    for (nm, v) in (("sb_bo_thresholds", bo_t), ("sb_bmsy_thresholds", bmsy_t),
                     ("u_umsy_thresholds", u_t))
        for x in v
            0 < x || error("Threshold $x in '$nm' ($where) must be positive.")
        end
    end
    age_floor = Config.load_int(d, "age_floor", where)
    age_floor >= 1 || error("'age_floor' in $where must be at least 1, got $age_floor.")
    u3 = Config.positive(d, "age3_exploitation_limit", where)
    eps = Config.positive(d, "epsilon", where)
    return DecisionControls(bo_t, bmsy_t, u_t, age_floor, u3, eps)
end

"""
    load_parameters(path::AbstractString) -> Vector{Float64}

Load estimated or benchmark parameter values from a native TOML file.
"""
function load_parameters(path::AbstractString)
    isfile(path) || error("Parameters file not found: $path")
    p_dict = TOML.parsefile(path)
    haskey(p_dict, "parameters") || error("Key 'parameters' not found in $path")
    return Float64.(p_dict["parameters"])
end
