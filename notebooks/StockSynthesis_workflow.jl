### A Pluto.jl notebook ###
# v1.0.4

using Markdown
using InteractiveUtils

macro bind(def, element)
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 3f2c8a41-6b57-4d1e-9a83-2c5b7e1d0f44
using Pkg

# ╔═╡ 7a3f0c15-9b42-4e8d-a6f1-2b8c5d0e7a93
Pkg.activate(joinpath(@__DIR__, ".."))

# ╔═╡ 8e2b6d47-3f19-4c7b-8d5a-1f4c9b2e0a36
using AssessmentTools

# ╔═╡ 9d4e1b72-8c05-4a63-b7e1-6f0d3a8c5e21
md"""
# StockSynthesis.jl — Size-Structured Stock Assessment

This notebook demonstrates the size-structured stock assessment framework
implemented in `AssessmentTools.StockSynthesis`. It covers:

1. Size class definition and growth parameters
2. Size transition matrix computation
3. Size selectivity
4. Size composition data handling (individual sizes → binned)
5. Multi-series likelihood (Turing-native)
6. MSY reference points for size-structured models
7. Bayesian size-structured model
8. Sex-specific selectivity and maturity
9. Age-structured stock synthesis

**References:**
- Methot & Wetzel (2013). Stock Synthesis. *Fisheries Research*, 142, 86-99.
- von Bertalanffy (1938). A quantitative theory of organic growth. *Human Biology*, 10, 181-213.
- Haddon (2011). *Modelling and Quantitative Methods in Fisheries* (2nd ed.).
"""

# ╔═╡ 2b7f5d93-1e48-4c0a-8f6d-9a1c4e7b0d35
md"""
## 1. Size Classes and Growth Parameters
"""

# ╔═╡ 5c1d8e64-2a79-4b5f-9e07-3d6a1c8b4f29
begin
    ss = AssessmentTools.StockSynthesis
    size_classes = ss.make_size_classes(10.0, 120.0, 15)
    growth = ss.GrowthParameters(130.0, 0.15; t0 = -0.5, cv = 0.1)
    (size_classes = size_classes, growth = growth)
end

# ╔═╡ 7f3c9e52-4b18-4d6a-8e05-2c9b1d6a3f47
md"""
Size classes: $(length(size_classes)) bins from $(size_classes[1].lower) to $(size_classes[end].upper) cm.
Growth: L_inf = $(growth.linf), k = $(growth.k), t0 = $(growth.t0), CV = $(growth.cv).
"""

# ╔═╡ e1000001-0001-0001-0001-000000000200
md"""
## 2. Size Transition Matrix (Growth)
"""

# ╔═╡ e1000001-0001-0001-0001-000000000201
G = ss.size_transition_matrix(growth, size_classes, 1.0)

# ╔═╡ e1000001-0001-0001-0001-000000000202
md"""
Transition matrix G (size classes × size classes). Column j = current size, row i = next size.
Column sums should be ~1.0 (conservative growth).
"""

# ╔═╡ e1000001-0001-0001-0001-000000000203
begin
    col_sums = [sum(G[:, j]) for j in 1:length(size_classes)]
    html"""
    <table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
      <tr><th style="border:1px solid #ccc; padding:6px;">Metric</th><th style="border:1px solid #ccc; padding:6px;">Value</th></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Matrix size</td><td style="border:1px solid #ccc; padding:6px;">$(size(G))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Column sum range</td><td style="border:1px solid #ccc; padding:6px;">[$(round(minimum(col_sums); digits=6)), $(round(maximum(col_sums); digits=6))]</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Max transition prob</td><td style="border:1px solid #ccc; padding:6px;">$(round(maximum(G); digits=4))</td></tr>
    </table>
    """
end

# ╔═╡ e1000001-0001-0001-0001-000000000204
md"""
## 3. Size Selectivity
"""

# ╔═╡ e1000001-0001-0001-0001-000000000205
begin
    midpoints = [sc.midpoint for sc in size_classes]
    sel_logistic = ss.size_selectivity(midpoints, [40.0, 0.1]; type = :logistic)
    sel_knife = ss.size_selectivity(midpoints, [30.0]; type = :knife_edge)
    sel_double = ss.size_selectivity(midpoints, [60.0, 10.0, 20.0, 0.8]; type = :double_normal)
    (logistic = sel_logistic, knife_edge = sel_knife, double_normal = sel_double)
end

# ╔═╡ e1000001-0001-0001-0001-000000000206
md"""
Three selectivity types: logistic (S-shaped), knife-edge (step function), and
double-normal (dome-shaped for optimal size targeting).
"""

# ╔═╡ e1000001-0001-0001-0001-000000000207
md"""
## 4. Size Composition Data (Individual Sizes)
"""

# ╔═╡ e1000001-0001-0001-0001-000000000208
begin
    using Random, Distributions
    rng = Random.MersenneTwister(42)
    lengths = vcat(
        rand(rng, Normal(35.0, 8.0), 200),
        rand(rng, Normal(65.0, 12.0), 150),
        rand(rng, Normal(90.0, 10.0), 50)
    )
    sc_data = ss.SizeCompositionData(lengths; fleet = 1, year = 1, n_bins = 12)
    (n_fish = sc_data.n_fish, n_bins = length(sc_data.counts), counts = sc_data.counts)
end

# ╔═╡ e1000001-0001-0001-0001-000000000209
md"""
Individual lengths are binned into size classes. Effective sample size is adjusted
for process error using the HCAM method.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000210
begin
    props = ss.size_proportions(sc_data)
    n_eff = ss.effective_sample_size(sc_data.counts)
    (proportions = round.(props; digits=3), n_effective = round(n_eff; digits=1))
end

# ╔═╡ e1000001-0001-0001-0001-000000000211
md"""
## 5. Multi-Series Likelihood (Turing-Native)
"""

# ╔═╡ e1000001-0001-0001-0001-000000000212
begin
    obs_series = [
        [10.0, 12.0, 15.0, 11.0],
        [50.0, 55.0, 60.0, 52.0],
        Float64.(sc_data.counts)
    ]
    pred_series = [
        [9.5, 11.5, 14.0, 10.5],
        [48.0, 53.0, 58.0, 50.0],
        fill(1.0/length(sc_data.counts), length(sc_data.counts))
    ]
    types = [:lognormal, :lognormal, :multinomial]

    ms_model = ss.bayesian_multi_series(obs_series, pred_series, types)
    (model_type = typeof(ms_model), n_series = length(obs_series))
end

# ╔═╡ e1000001-0001-0001-0001-000000000213
md"""
The Turing model can be used with any inference engine:

```julia
# MAP/ML
result = fit_map(ms_model, x0)

# MCMC
chain = sample(ms_model, NUTS(), 1000)

# VI
vi_res = fit_vi(ms_model)
```
"""

# ╔═╡ e1000001-0001-0001-0001-000000000214
md"""
## 6. MSY Reference Points (Size-Structured)
"""

# ╔═╡ e1000001-0001-0001-0001-000000000215
begin
    model = ss.SizeStructuredModel(size_classes, growth, 10, 2, 1;
                                    m = 0.2, n_sel_params = 2)
    rp = ss.size_msy_reference_points(model)
    html"""
    <table style="border-collapse: collapse; width: 100%; font-family: sans-serif;">
      <tr><th style="border:1px solid #ccc; padding:6px;">Reference Point</th><th style="border:1px solid #ccc; padding:6px;">Value</th></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Bmsy</td><td style="border:1px solid #ccc; padding:6px;">$(round(rp.bmsy; digits=2))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">MSY</td><td style="border:1px solid #ccc; padding:6px;">$(round(rp.msy; digits=2))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Fmsy</td><td style="border:1px solid #ccc; padding:6px;">$(round(rp.fmsy; digits=4))</td></tr>
      <tr><td style="border:1px solid #ccc; padding:6px;">Umsy</td><td style="border:1px solid #ccc; padding:6px;">$(round(rp.umsy; digits=4))</td></tr>
    </table>
    """
end

# ╔═╡ e1000001-0001-0001-0001-000000000216
md"""
## 7. Bayesian Size-Structured Model
"""

# ╔═╡ e1000001-0001-0001-0001-000000000217
md"""
The Bayesian model uses Turing.jl with:
- State-space formulation on biomass (Schaefer-like)
- Multiple likelihoods: catch (lognormal), survey (lognormal), size composition (multinomial)
- Priors on r, K, q, process error, observation error

```julia
model = bayesian_size_structured(obs_catch, obs_indices, size_comps)
chain = sample(model, NUTS(), 1000)
```
"""

# ╔═╡ e1000001-0001-0001-0001-000000000218
md"""
## 8. Sex-Specific Selectivity and Maturity

Survey selectivity and maturity oogives can be specified per sex:
- `survey_sel_params`: (n_surveys × n_sexes × n_sel_params)
- `mat_at_size`: (n_sexes × n_size)
- `w_at_size`: (n_sexes × n_size)
- `fec_at_size`: (n_sexes × n_size)
- `m_size`: (n_sexes × n_size)

Default is single sex (n_sexes=1) for backward compatibility.
"""

# ╔═╡ e1000001-0001-0001-0001-000000000219
begin
    model_2sex = ss.SizeStructuredModel(size_classes, growth, 10, 2, 2;
                                         n_sexes = 2, m = 0.2,
                                         mat_at_size = [sc.midpoint > 25.0 ? 1.0 : 0.0 for sc in size_classes],
                                         survey_sel_params = cat([40.0 0.1; 35.0 0.15],
                                                                  [45.0 0.08; 40.0 0.12]; dims = 3))
    (n_sexes = model_2sex.n_sexes,
     mat_size = size(model_2sex.mat_at_size),
     survey_sel_size = size(model_2sex.survey_sel_params))
end

# ╔═╡ e1000001-0001-0001-0001-000000000220
md"""
## 9. Age-Structured Stock Synthesis

An age-based variation using numbers-at-age dynamics with the Baranov catch equation.
Supports the same unified inference interface (MAP, MCMC, VI).
"""

# ╔═╡ e1000001-0001-0001-0001-000000000221
begin
    ac = ss.AgeCompositionData([10, 20, 15, 5, 2]; ages = [1, 2, 3, 4, 5])
    age_model = ss.bayesian_age_synthesis(
        [100.0, 110.0, 105.0, 95.0],
        [50.0, 55.0, 52.0, 48.0],
        [ac])
    (model_type = typeof(age_model), n_params = ss.InferenceBridge.parameter_dimension(age_model))
end

# ╔═╡ Cell order:
# ╠═3f2c8a41-6b57-4d1e-9a83-2c5b7e1d0f44
# ╠═7a3f0c15-9b42-4e8d-a6f1-2b8c5d0e7a93
# ╠═8e2b6d47-3f19-4c7b-8d5a-1f4c9b2e0a36
# ╠═9d4e1b72-8c05-4a63-b7e1-6f0d3a8c5e21
# ╠═2b7f5d93-1e48-4c0a-8f6d-9a1c4e7b0d35
# ╠═5c1d8e64-2a79-4b5f-9e07-3d6a1c8b4f29
# ╠═7f3c9e52-4b18-4d6a-8e05-2c9b1d6a3f47
# ╠═e1000001-0001-0001-0001-000000000200
# ╠═e1000001-0001-0001-0001-000000000201
# ╠═e1000001-0001-0001-0001-000000000202
# ╠═e1000001-0001-0001-0001-000000000203
# ╠═e1000001-0001-0001-0001-000000000204
# ╠═e1000001-0001-0001-0001-000000000205
# ╠═e1000001-0001-0001-0001-000000000206
# ╠═e1000001-0001-0001-0001-000000000207
# ╠═e1000001-0001-0001-0001-000000000208
# ╠═e1000001-0001-0001-0001-000000000209
# ╠═e1000001-0001-0001-0001-000000000210
# ╠═e1000001-0001-0001-0001-000000000211
# ╠═e1000001-0001-0001-0001-000000000212
# ╠═e1000001-0001-0001-0001-000000000213
# ╠═e1000001-0001-0001-0001-000000000214
# ╠═e1000001-0001-0001-0001-000000000215
# ╠═e1000001-0001-0001-0001-000000000216
# ╠═e1000001-0001-0001-0001-000000000217
# ╠═e1000001-0001-0001-0001-000000000218
# ╠═e1000001-0001-0001-0001-000000000219
# ╠═e1000001-0001-0001-0001-000000000220
# ╠═e1000001-0001-0001-0001-000000000221