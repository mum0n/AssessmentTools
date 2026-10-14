# Module wrapper for fisheries management procedures and harvest control rules

"""
    AssessmentTools.Management

Harvest control rules (HCR) and management procedure components:
- Precautionary piecewise-linear hockey-stick HCR with limit and trigger thresholds.
- Interannual TAC change constraints and stabilizers.
"""
module Management

using LinearAlgebra
using Statistics
using Distributions
using Turing

export HockeyStickHCR, eval_hcr, apply_tac_stabilizer
export MseResult, simulate_mse, bayesian_mse_feedback

include("harvest_control.jl")

end # module Management
