# ------------------------------------------------------------------------------------
# Config: strict, typed accessors for user-level configuration files.
#
# Every model, solver, prior, penalty and decision-table parameter used by this
# package is read from a TOML configuration file through these accessors. Missing
# keys raise an error naming the key and the file, so a configuration file always
# states the complete set of parameters it controls: there is no silent fallback to
# a value compiled into the source.
#
# Constants that are part of the mathematics rather than tunable parameters
# (a factor of 2 in a variance formula, a mid-year scaling of 0.5, 12 months per
# year) are named constants in the corresponding module and documented there.
# ------------------------------------------------------------------------------------

module Config

using TOML

export require_key, require_section, parse_file

"""
    parse_file(path::AbstractString) -> Dict{String, Any}

Read a TOML configuration file. Errors if the file does not exist or does not
parse.
"""
function parse_file(path::AbstractString)
    isfile(path) || error("Configuration file not found: $path")
    try
        return TOML.parsefile(path)
    catch err
        error("Configuration file $path could not be parsed as TOML: $(sprint(showerror, err))")
    end
end

"""
    require_section(table, section, path) -> Dict{String, Any}

Return `table[section]`, raised as a `Dict`. Used for TOML tables.
"""
function require_section(table::AbstractDict, section::AbstractString, path::AbstractString)
    haskey(table, section) ||
        error("Configuration file $path is missing required section [$section].")
    v = table[section]
    v isa AbstractDict ||
        error("Configuration file $path: section [$section] must be a TOML table.")
    return v
end

"""
    require_key(table, key, where) -> Any

Return `table[key]`, raising an error that names the key and the owning
configuration location if it is absent.
"""
function require_key(table::AbstractDict, key::AbstractString, where::AbstractString)
    haskey(table, key) ||
        error("Missing required configuration key '$key' in $where. Add it to the configuration file.")
    return table[key]
end

_as_float(v, key, where) = v isa Real ? Float64(v) :
    error("Configuration key '$key' in $where must be a real number, got $(typeof(v)).")
_as_int(v, key, where) = v isa Integer ? Int(v) :
    error("Configuration key '$key' in $where must be an integer, got $(typeof(v)).")
_as_bool(v, key, where) = v isa Bool ? v :
    error("Configuration key '$key' in $where must be true or false, got $(typeof(v)).")

_reals(v, key, where) = (v isa AbstractVector && all(x -> x isa Real, v)) ?
    Float64.(v) :
    error("Configuration key '$key' in $where must be a vector of real numbers.")

"""
    load_float(table, key, where) -> Float64
    load_int(table, key, where) -> Int
    load_bool(table, key, where) -> Bool
    load_reals(table, key, where) -> Vector{Float64}

Typed readers. Each validates the type and raises an error naming the key and the
owning configuration location when the key is missing or of the wrong type.
"""
load_float(table::AbstractDict, key::AbstractString, where::AbstractString) =
    _as_float(require_key(table, key, where), key, where)

load_int(table::AbstractDict, key::AbstractString, where::AbstractString) =
    _as_int(require_key(table, key, where), key, where)

load_bool(table::AbstractDict, key::AbstractString, where::AbstractString) =
    _as_bool(require_key(table, key, where), key, where)

load_reals(table::AbstractDict, key::AbstractString, where::AbstractString) =
    _reals(require_key(table, key, where), key, where)

"""
    load_ints(table, key, where) -> Vector{Int}

Read a vector of integers, validating each element.
"""
function load_ints(table::AbstractDict, key::AbstractString, where::AbstractString)
    v = require_key(table, key, where)
    v isa AbstractVector ||
        error("Configuration key '$key' in $where must be a vector of integers.")
    return [load_int(Dict("v" => x), "v", "$where (element of '$key')") for x in v]
end

"""
    positive(table, key, where) -> Float64

Read a strictly positive real.
"""
function positive(table::AbstractDict, key::AbstractString, where::AbstractString)
    v = load_float(table, key, where)
    v > 0 || error("Configuration key '$key' in $where must be strictly positive, got $v.")
    return v
end

"""
    non_negative(table, key, where) -> Float64

Read a non-negative real.
"""
function non_negative(table::AbstractDict, key::AbstractString, where::AbstractString)
    v = load_float(table, key, where)
    v >= 0 || error("Configuration key '$key' in $where must be non-negative, got $v.")
    return v
end

"""
    unit_range_ok(table, key, where) -> Float64

Read a real in `[0, 1]`.
"""
function unit_range_ok(table::AbstractDict, key::AbstractString, where::AbstractString)
    v = load_float(table, key, where)
    (0 <= v <= 1) ||
        error("Configuration key '$key' in $where must lie in [0, 1], got $v.")
    return v
end

end # module Config
