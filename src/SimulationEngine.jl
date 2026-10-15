"""
    AssessmentTools.SimulationEngine

Forward and backward projection from an arbitrary time slice, driven by the estimated
parameters of any model in the package, in two interchangeable representations:

- Agent-based: each group (age class, life stage, cohort or gear) is an agent whose
  per-step transition is the producing model's own forward recursion.
- Reaction network: each group is a species, and recruitment, advancement, retention,
  natural mortality and fishing removals are reactions, simulated with a time-binned
  Gillespie SSA.

Parameters may come from a point estimate or from posterior draws, so the same engine
runs a best-estimate projection and a full uncertainty propagation with no changes at
the call site.
"""
module SimulationEngine

include("simulation_engine.jl")

end # module SimulationEngine
