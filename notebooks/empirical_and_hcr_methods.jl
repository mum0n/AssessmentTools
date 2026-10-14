### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 00000003-0000-0000-0000-000000000001
md"""
# AssessmentTools — Empirical Assessment & Feedback Methods

This Pluto notebook illustrates the complementary empirical and simulation methods in
`AssessmentTools.jl`:

1. **Catch Curves & Total Mortality ($Z$)**:
   - Chapman and Robson (1960) minimum-variance unbiased estimator (MVUE) for annual survival $S$ and total mortality $Z = -\log(S)$ with closed-form standard errors.
   - Linear catch curve regressions (both unweighted and Maceina & Bettoli 1998 weighted least squares).
2. **Depletion Methods ($N_0$ and $q$)**:
   - Leslie & Davis (1939) linear depletion estimator of initial abundance $N_0$ and catchability $q$ with delta-method variance.
   - DeLury (1947) logarithmic depletion model.
3. **Age-Length Key (ALK) Conversion**:
   - Constructing conditional transition matrices $P(a \mid l)$ from aged subsamples.
   - Expanding commercial or survey length-frequency distributions into age compositions.
4. **Precautionary Harvest Control Rules (HCR) & Stabilizers**:
   - Piecewise-linear hockey-stick HCR with limit and precautionary reference points ($B_{\mathrm{lim}}, B_{\mathrm{pa}}, F_{\mathrm{low}}, F_{\mathrm{target}}$).
   - Interannual TAC change constraints ($\pm 15\%$ dampeners) to smooth quota volatility.
5. **Integration with Integrated Catch-at-Age Assessment**:
   - Applying empirical estimators directly to the internal Silver Hake and Cusk assessment data.
"""

# ╔═╡ 00000003-0000-0000-0000-000000000002
begin
    import Pkg
    Pkg.activate(normpath(joinpath(@__DIR__, "..")))
    using AssessmentTools
    using AssessmentTools.Empirical
    using AssessmentTools.Management
    using LinearAlgebra
    using Statistics
    using Printf
end

# ╔═╡ 00000003-0000-0000-0000-000000000003
md"""
---
## 1. Catch Curves & Chapman-Robson Mortality Estimation

In empirical fisheries analysis, catch curve methods provide direct estimates of total instantaneous mortality $Z$ and annual survival $S$ from age-composition data on the descending limb (ages $\ge a_{\mathrm{rec}}$).

### Mathematical Formulation:
- **Chapman & Robson (1960)**:
  Let $a_0$ be the peak recruitment age, and $i = a - a_0 \ge 0$. With $n = \sum C_i$ and coded mean age $\bar{T} = \frac{1}{n} \sum i C_i$:
  ```math
  S = \frac{\bar{T}}{1 + \bar{T} - 1/n}, \quad \widehat{\mathrm{Var}}(S) = S \left( S - \frac{\bar{T}-1}{n + \bar{T} - 1} \right)
  ```
  ```math
  Z = -\log(S), \quad \mathrm{SE}(Z) \approx \frac{\mathrm{SE}(S)}{S}
  ```
- **Weighted Linear Catch Curve (Maceina & Bettoli 1998)**:
  ```math
  \log(C_a) = \alpha - Z a + \epsilon_a, \quad w_a = C_a
  ```
"""

# ╔═╡ 00000003-0000-0000-0000-000000000004
begin
    # Load Silver Hake fishery catch-at-age composition for an empirical mortality check
    hake_data_dir = normpath(joinpath(@__DIR__, "..", "data", "silverhake"))
    d_hake, c_hake, _ = load_dataset(hake_data_dir)

    # Extract the age composition matrix for the commercial fishery (series 1)
    ac = d_hake.agecomps[1]
    ages_vec = Float64.(ac.sage:ac.nage)
    
    # Calculate average catch-at-age across all sampled years
    mean_catch_at_age = vec(mean(ac.obs, dims = 1)) .* 1000.0

    # 1. Chapman-Robson estimator
    cr_est = chapman_robson(ages_vec, mean_catch_at_age)

    # 2. Linear catch curve (unweighted)
    cc_unweighted = catch_curve_linear(ages_vec, mean_catch_at_age; weighted = false)

    # 3. Linear catch curve (weighted by sample catch count)
    cc_weighted = catch_curve_linear(ages_vec, mean_catch_at_age; weighted = true)

    (
        ages = "$(Int(ages_vec[1])) to $(Int(ages_vec[end]))",
        recruitment_age = argmax(mean_catch_at_age) + Int(ages_vec[1]) - 1,
        chapman_robson_S = round(cr_est.s, digits = 4),
        chapman_robson_Z = round(cr_est.z, digits = 4),
        chapman_robson_SE_Z = round(cr_est.se_z, digits = 4),
        catch_curve_Z_unweighted = round(cc_unweighted.z, digits = 4),
        catch_curve_R2_unweighted = round(cc_unweighted.r2, digits = 4),
        catch_curve_Z_weighted = round(cc_weighted.z, digits = 4),
        catch_curve_R2_weighted = round(cc_weighted.r2, digits = 4)
    )
end

# ╔═╡ 00000003-0000-0000-0000-000000000005
md"""
---
## 2. Depletion Methods (Leslie & DeLury Estimators)

When a stock experiences intensive removals over a short period without substantial recruitment or natural mortality, depletion methods estimate initial abundance $N_0$ and catchability $q$ directly from catch and effort time series.

### Mathematical Formulation:
- **Leslie & Davis (1939)**:
  Regresses $\mathrm{CPUE}_t = \frac{C_t}{f_t}$ on midpoint cumulative catch $K_t = \sum_{i<t} C_i + \frac{1}{2} C_t$:
  ```math
  \mathrm{CPUE}_t = q N_0 - q K_t \implies q = -\beta, \quad N_0 = \frac{\alpha}{q}
  ```
- **DeLury (1947)**:
  Regresses $\log(\mathrm{CPUE}_t)$ on cumulative effort $E_t = \sum_{i<t} f_i + \frac{1}{2} f_t$:
  ```math
  \log(\mathrm{CPUE}_t) = \log(q N_0) - q E_t \implies q = -\beta, \quad N_0 = \frac{\exp(\alpha)}{q}
  ```
"""

# ╔═╡ 00000003-0000-0000-0000-000000000006
begin
    # Synthetic multi-pass depletion experiment with known parameters:
    # N0 = 50,000 fish, q = 0.0004 per boat-day, effort = 80 boat-days per pass
    true_n0 = 50000.0
    true_q = 0.0004
    passes = 6
    sim_effort = fill(80.0, passes)
    
    sim_catches = let rem_pop = true_n0
        sc = zeros(Float64, passes)
        for p in 1:passes
            f = sim_effort[p]
            c = (true_q * f * rem_pop) / (1.0 + 0.5 * true_q * f)
            sc[p] = c
            rem_pop -= c
        end
        sc
    end

    leslie_fit = leslie_depletion(sim_catches, sim_effort)
    delury_fit = delury_depletion(sim_catches, sim_effort)

    (
        true_N0 = true_n0,
        true_q = true_q,
        leslie_N0 = round(leslie_fit.n0, digits = 1),
        leslie_SE_N0 = round(leslie_fit.se_n0, digits = 1),
        leslie_q = round(leslie_fit.q, digits = 6),
        leslie_R2 = round(leslie_fit.r2, digits = 4),
        delury_N0 = round(delury_fit.n0, digits = 1),
        delury_SE_N0 = round(delury_fit.se_n0, digits = 1),
        delury_q = round(delury_fit.q, digits = 6),
        delury_R2 = round(delury_fit.r2, digits = 4)
    )
end

# ╔═╡ 00000003-0000-0000-0000-000000000007
md"""
---
## 3. Age-Length Key (ALK) Expansion

Surveys and commercial sampling often measure lengths for thousands of fish, but age determinations (from otoliths) are expensive and performed on smaller stratified subsamples. The Age-Length Key converts lengths to age compositions.

### Mathematical Formulation:
```math
P(a \mid l) = \frac{A_{l,a}}{\sum_{a'} A_{l,a'}}, \quad C_a = \sum_{l=1}^{N_L} L_l \cdot P(a \mid l)
```
"""

# ╔═╡ 00000003-0000-0000-0000-000000000008
begin
    # Example length bins (20-25cm, 25-30cm, 30-35cm, 35-40cm, 40-45cm)
    # Ages: 1, 2, 3, 4, 5
    aged_otolith_samples = [
        45.0  12.0   1.0   0.0   0.0;
        10.0  55.0  20.0   2.0   0.0;
         1.0  15.0  60.0  18.0   2.0;
         0.0   2.0  14.0  52.0  15.0;
         0.0   0.0   2.0  10.0  40.0
    ]

    # Total observed length frequency from market or survey sampling
    bulk_length_frequencies = [1200.0, 3500.0, 4800.0, 2900.0, 850.0]

    # Apply age-length key
    alk_expanded = apply_age_length_key(aged_otolith_samples, bulk_length_frequencies)

    (
        total_lengths_sampled = sum(bulk_length_frequencies),
        expanded_catch_at_age = round.(alk_expanded.age_numbers, digits = 1),
        proportions_at_age = round.(alk_expanded.age_proportions, digits = 4),
        ages = [1, 2, 3, 4, 5]
    )
end

# ╔═╡ 00000003-0000-0000-0000-000000000009
md"""
---
## 4. Precautionary Harvest Control Rules (HCR) & TAC Dampening

Standard management procedures evaluate target fishing mortalities using a hockey-stick
decision rule parameterized by limit ($B_{\mathrm{lim}}$) and precautionary ($B_{\mathrm{pa}}$) biomass reference points.

### Mathematical Formulation:
```math
F(SSB) = \begin{cases}
F_{\mathrm{low}} & SSB \le B_{\mathrm{lim}} \\
F_{\mathrm{low}} + \dfrac{SSB - B_{\mathrm{lim}}}{B_{\mathrm{pa}} - B_{\mathrm{lim}}} (F_{\mathrm{target}} - F_{\mathrm{low}}) & B_{\mathrm{lim}} < SSB < B_{\mathrm{pa}} \\
F_{\mathrm{target}} & SSB \ge B_{\mathrm{pa}}
\end{cases}
```
Annual quota stability constraint:
```math
\mathrm{TAC}_t = \max\Big(\mathrm{TAC}_{t-1}(1 - \delta_{\mathrm{dec}}), \; \min\big(\mathrm{TAC}_{t-1}(1 + \delta_{\mathrm{inc}}), \; \mathrm{TAC}_{\mathrm{target}}\big)\Big)
```
"""

# ╔═╡ 00000003-0000-0000-0000-000000000010
begin
    # Reference points (e.g. from Cusk benchmark)
    b_lim = 12000.0   # Limit biomass (t)
    b_pa  = 25000.0   # Precautionary biomass / Bmsy proxy (t)
    f_low = 0.02      # Bycatch or residual F
    f_tar = 0.20      # Fmsy target

    hcr_rule = HockeyStickHCR(b_lim, b_pa, f_low, f_tar)

    # Evaluate HCR across a gradient of stock states
    test_biomasses = [5000.0, 10000.0, 12000.0, 15000.0, 18500.0, 22000.0, 25000.0, 32000.0]
    prescribed_f = [eval_hcr(hcr_rule, b) for b in test_biomasses]

    # Demonstrate interannual TAC stabilization (±15% rule)
    prev_year_tac = 1500.0
    raw_tac_options = [900.0, 1200.0, 1550.0, 1800.0, 2200.0]
    stabilized_tacs = [apply_tac_stabilizer(tac, prev_year_tac; max_decrease = 0.15, max_increase = 0.15)
                       for tac in raw_tac_options]

    (
        biomasses = test_biomasses,
        target_F = round.(prescribed_f, digits = 4),
        tac_prev = prev_year_tac,
        unconstrained_tac = raw_tac_options,
        stabilized_tac = round.(stabilized_tacs, digits = 1)
    )
end

# ╔═╡ 00000003-0000-0000-0000-000000000011
md"""
---
## 5. Closed-Loop Management Strategy Evaluation (MSE) Simulation

Simulate closed-loop feedback management over a 20-year horizon combining stochastic biological production, observation error, HCR decision-making, and TAC constraints:
"""

# ╔═╡ 00000003-0000-0000-0000-000000000012
begin
    using UnicodePlots
    mse_years = 2025:2045
    mse_hcr = HockeyStickHCR(25.0, 50.0, 0.02, 0.25)
    
    mse_sim = simulate_mse(
        mse_years, mse_hcr;
        r = 0.35, k = 100.0, b0 = 70.0,
        obs_cv = 0.15, proc_cv = 0.05, tac_constraint = 0.15
    )
    
    p_mse = plot_state_variables(mse_sim)
    (
        mse_years = "$(minimum(mse_sim.years)) - $(maximum(mse_sim.years))",
        terminal_true_biomass = round(mse_sim.true_biomass[end]; digits = 2),
        mean_catch = round(mean(mse_sim.catches); digits = 2),
        plot_type = typeof(p_mse)
    )
end

# ╔═╡ 00000003-0000-0000-0000-000000000013
md"""
---
## 6. Summary & References
These modular capabilities enrich `AssessmentTools.jl` with standard empirical and management procedures:
- **Empirical Diagnostics**: Catch curves and depletion methods provide quick checks and priors independent of complex likelihood spaces.
- **Data Preprocessing**: Age-Length Keys automate conversion of abundant survey length frequencies into age compositions.
- **Closed-Loop Management**: Precautionary HCRs and TAC stabilizers enable Management Strategy Evaluation (MSE) directly within native Julia.

### Key References
- Chapman, D.G., & Robson, D.S. (1960). The analysis of a catch curve. *Biometrics*, 16(3): 354–368.
- Leslie, P.H., & Davis, D.H.S. (1939). An attempt to determine the absolute number of rats on a given area. *J. Anim. Ecol.*, 8(1): 94–113.
- DeLury, D.B. (1947). On the estimation of biological populations. *Biometrics*, 3(4): 145–167.
- Maceina, M.J., & Bettoli, P.W. (1998). Variation in large-scale fishing mortality estimates. *N. Am. J. Fish. Manage.*, 18(2): 375–382.
- Punt, A.E., Butterworth, D.S., de Moor, C.L., De Oliveira, J.A., & Szuwalski, C. (2016). Management strategy evaluation: best practices. *Fish and Fisheries*, 17(2): 303–334.
"""

# ╔═╡ Cell order:
# ╠═00000003-0000-0000-0000-000000000001
# ╠═00000003-0000-0000-0000-000000000002
# ╠═00000003-0000-0000-0000-000000000003
# ╠═00000003-0000-0000-0000-000000000004
# ╠═00000003-0000-0000-0000-000000000005
# ╠═00000003-0000-0000-0000-000000000006
# ╠═00000003-0000-0000-0000-000000000007
# ╠═00000003-0000-0000-0000-000000000008
# ╠═00000003-0000-0000-0000-000000000009
# ╠═00000003-0000-0000-0000-000000000010
# ╠═00000003-0000-0000-0000-000000000011
# ╠═00000003-0000-0000-0000-000000000012
# ╠═00000003-0000-0000-0000-000000000013
