### A Pluto.jl notebook ###
# v0.20.0

using Markdown
using InteractiveUtils

# ╔═╡ 40000001-0000-0000-0000-000000000001
md"""
# Multi-Stage Continuous Delay Differential Models (DDE)

This notebook introduces the continuous size- and stage-structured delay differential equation (DDE) population dynamics framework implemented in **`AssessmentTools.DelayDifferential`** (Choi 2024).

### Key Biological Assumptions & Analytical Formulation
1. **Multi-Stage Ontogenetic Structure**:
   Tracks 6 developmental and demographic stages:
   - $M_0$: Commercial legal males ($\ge 95\text{ mm}$ carapace width, CW)
   - $M_1$: Male recruits approaching commercial size (instar 12, sub-legal recruit)
   - $M_2$: Pre-recruits (instar 11)
   - $M_3$: Immature males (instar 10)
   - $M_4$: Immature males (instar 9, entry stage from larval settlement)
   - $f_{\text{mat}}$: Mature reproductive females
2. **Reproductive Lags ($\tau$)**:
   Biological recruitment into instar 9 ($M_4$) is delayed by $\tau \approx 7\text{--}9$ years following egg extrusion and fertilization by mature females $f_{\text{mat}}(t - \tau)$:
   $$R(t) = \beta f_{\text{mat}}(t - \tau) \exp\left(-\frac{f_{\text{mat}}(t - \tau)}{K_f}\right)$$
3. **Moult Transition Rates & Developmental Stanzas**:
   Crustaceans grow discretely through ecdysis. Continuous inter-instar progression is mediated by instar-specific transition rates $v_i$:
   $$\frac{d M_i}{dt} = v_{i+1} M_{i+1}(t) - (v_i + d_i) M_i(t) - d_{2,i} \frac{M_i(t)^2}{H_i(t)}$$
4. **Partitioned Mortality & Dynamic Habitat Carrying Capacity**:
   Total mortality is decomposed into baseline linear natural mortality $d_i$ and non-linear density-dependent space-competition mortality scaled by bottom temperature and oceanographic habitat capacity $H_i(t)$.
5. **Continuous Pulse Fishery Removals**:
   Captures pulse fishing seasons occurring within narrow intra-annual windows (e.g. spring/autumn openings) without artificial annual step discretization:
   $$M_0(t^+) = \max\left(0, M_0(t^-) - C(t)\right)$$

### Key References
- Choi, J.S. (2024). Continuous size-structured delay differential models for snow crab (*Chionoecetes opilio*) fisheries assessment. *DFO Can. Sci. Advis. Sec. Res. Doc.* 2024/nnn.
- Gurney, W.S.C., & Nisbet, R.M. (1998). *Ecological Dynamics*. Oxford University Press.
- Macdonald, N. (1978). *Time Lags in Biological Models*. Lecture Notes in Biomathematics, Springer-Verlag, Berlin.
"""

# ╔═╡ 40000001-0000-0000-0000-000000000002
begin
    import Pkg
    Pkg.activate(normpath(joinpath(@__DIR__, "..")))
    using AssessmentTools
    using AssessmentTools.DelayDifferential
    using Parquet2
    using DataFrames
    using UnicodePlots
end

# ╔═╡ 40000001-0000-0000-0000-000000000003
md"""
## 1. Load Ingested Size-Structured Snow Crab Data (Parquet)
"""

# ╔═╡ 40000001-0000-0000-0000-000000000004
begin
    num_pq_file = normpath(joinpath(@__DIR__, "..", "data", "snowcrab", "biodyn_number_size_struct.parquet"))
    table_num = Parquet2.readfile(num_pq_file)
    df_stages = DataFrame(table_num)
    
    (
        years = "$(minimum(df_stages.yrs)) - $(maximum(df_stages.yrs))",
        n_rows = nrow(df_stages),
        n_columns = ncol(df_stages)
    )
end

# ╔═╡ 40000001-0000-0000-0000-000000000005
md"""
## 2. Configure 6-Stage Snow Crab DDE System
"""

# ╔═╡ 40000001-0000-0000-0000-000000000006
begin
    cfg = default_snowcrab_dde_config()
    (
        stages = cfg.num_stages,
        delays_years = cfg.lags,
        asymptotic_carrying_capacities = cfg.k_carrying_capacity,
        baseline_natural_mortality = cfg.natural_mortality
    )
end

# ╔═╡ 40000001-0000-0000-0000-000000000007
md"""
## 3. Simulate Continuous Intra-Annual Dynamics (Monthly Resolution)
"""

# ╔═╡ 40000001-0000-0000-0000-000000000008
begin
    sim_tspan = (0.0, 10.0) # 10 years
    dt_monthly = 1.0 / 12.0 # Monthly time step
    
    # Pulse fishing occurring annually in the autumn (month 9)
    pulse_times = [1.75, 2.75, 3.75, 4.75, 5.75, 6.75, 7.75, 8.75, 9.75]
    pulse_catches = fill(3.5, length(pulse_times)) # kt removals
    
    dde_sim = simulate_dde_system(
        cfg, sim_tspan;
        dt = dt_monthly,
        removals_times = pulse_times,
        removals_amount = pulse_catches
    )
    
    (
        time_steps = length(dde_sim.time),
        stage_names = dde_sim.stage_names,
        initial_legal_males = round(dde_sim.states[1, 1]; digits = 2),
        terminal_legal_males = round(dde_sim.states[end, 1]; digits = 2),
        terminal_mature_females = round(dde_sim.states[end, 6]; digits = 2)
    )
end

# ╔═╡ 40000001-0000-0000-0000-000000000009
md"""
## 4. Multi-Stage Visualizations & Trajectory Traces

Render continuous time series for commercial legal males ($M_0$) alongside mature females ($f_{\text{mat}}$):
"""

# ╔═╡ 40000001-0000-0000-0000-000000000010
begin
    p_m0 = plot_trajectory(
        dde_sim.time, dde_sim.states[:, 1];
        title = "Commercial Males (M0) Continuous DDE Trajectory",
        xlabel = "Year", ylabel = "Biomass (kt)", color = :blue
    )
    plot_trajectory!(p_m0, dde_sim.time, dde_sim.states[:, 6]; name = "Mature Females", color = :red)
    p_m0
end

# ╔═╡ Cell order:
# ╠═40000001-0000-0000-0000-000000000001
# ╠═40000001-0000-0000-0000-000000000002
# ╠═40000001-0000-0000-0000-000000000003
# ╠═40000001-0000-0000-0000-000000000004
# ╠═40000001-0000-0000-0000-000000000005
# ╠═40000001-0000-0000-0000-000000000006
# ╠═40000001-0000-0000-0000-000000000007
# ╠═40000001-0000-0000-0000-000000000008
# ╠═40000001-0000-0000-0000-000000000009
# ╠═40000001-0000-0000-0000-000000000010
