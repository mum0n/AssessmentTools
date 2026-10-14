# Age-Length Key (ALK) expansion and conditional age assignment.

"""
Result of age-length key expansion.

# Fields
- `age_proportions::Vector{Float64}`: Estimated overall age proportions ``P(a)``.
- `age_numbers::Vector{Float64}`: Estimated total number or expanded catch at age ``C_a``.
- `transition_matrix::Matrix{Float64}`: Conditional probability matrix ``P(a \\mid l)``
  with rows corresponding to length bins and columns corresponding to ages.
"""
struct AlkResult
    age_proportions::Vector{Float64}
    age_numbers::Vector{Float64}
    transition_matrix::Matrix{Float64}
end

"""
    compute_alk_matrix(aged_counts::AbstractMatrix{<:Real}) -> Matrix{Float64}

Convert a matrix of aged sample counts into a conditional probability matrix ``P(a \\mid l)``,
where rows correspond to length intervals and columns correspond to ages.

# Mathematical Formulation
```math
P(a \\mid l) = \\frac{A_{l,a}}{\\sum_{a'} A_{l,a'}}
```
If a length row contains zero sampled fish, the conditional row probabilities are assigned
uniformly or left zero.

# Inputs
- `aged_counts`: Matrix of dimensions `(n_lengths, n_ages)` where element `(l, a)` is the
  number of individuals in length class `l` confirmed to be age `a`.

# Outputs
A row-stochastic `Matrix{Float64}` of dimensions `(n_lengths, n_ages)`.
"""
function compute_alk_matrix(aged_counts::AbstractMatrix{<:Real})
    n_len, n_age = size(aged_counts)
    alk = zeros(Float64, n_len, n_age)

    for l in 1:n_len
        row_tot = sum(aged_counts[l, :])
        if row_tot > 0
            alk[l, :] .= aged_counts[l, :] ./ row_tot
        end
    end
    return alk
end

"""
    apply_age_length_key(alk_matrix::AbstractMatrix{<:Real},
                         length_freq::AbstractVector{<:Real}) -> AlkResult

Apply an Age-Length Key (conditional transition matrix) to expand an observed
length frequency distribution into estimated age compositions.

# Mathematical Formulation
Let ``L_l`` denote the observed or expanded number of individuals in length class ``l``,
and ``P(a \\mid l)`` the conditional probability of age ``a`` given length ``l``:

```math
C_a = \\sum_{l=1}^{N_L} L_l \\cdot P(a \\mid l)
```
The overall age proportions are:
```math
P(a) = \\frac{C_a}{\\sum_{a'} C_{a'}}
```

# Inputs
- `alk_matrix`: Matrix of size `(n_lengths, n_ages)` representing ``P(a \\mid l)`` or raw counts.
- `length_freq`: Vector of length `n_lengths` containing total length-frequency counts.

# Outputs
An [`AlkResult`](@ref) containing expanded numbers at age and proportion at age.

# References
- Fridriksson, A. 1934. On the calculation of age-distribution within a stock of cod by
  means of relatively few age-determinations as a key to measurements on a large scale.
  Rapp. P.-V. Reun. Cons. Int. Explor. Mer 86:1-14.
- Kimura, D.K. 1977. Statistical study of the AMH (Age-Length-Key) method.
  Fish. Bull. 75:219-224.
"""
function apply_age_length_key(
    alk_matrix::AbstractMatrix{<:Real},
    length_freq::AbstractVector{<:Real}
)
    n_len, n_age = size(alk_matrix)
    length(length_freq) == n_len ||
        throw(DimensionMismatch("length_freq length must match alk_matrix row count"))

    # If rows do not sum to 1.0 (raw counts supplied), normalize
    alk = zeros(Float64, n_len, n_age)
    for l in 1:n_len
        row_sum = sum(alk_matrix[l, :])
        if row_sum > 0
            alk[l, :] .= alk_matrix[l, :] ./ row_sum
        end
    end

    l_vec = Float64.(length_freq)
    # C_a = sum_l (L_l * P(a | l)) = L' * P
    c_a = vec(l_vec' * alk)

    tot = sum(c_a)
    p_a = tot > 0.0 ? c_a ./ tot : zeros(Float64, n_age)

    return AlkResult(p_a, c_a, alk)
end
