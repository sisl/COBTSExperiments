module CMCTS

using CPOMDPs

using Infiltrator
using POMDPs
import POMDPs: action, update, updater, initialize_belief
using POMDPTools
using MCTS
import MCTS: tooltip_tag, node_tag, estimate_value, convert_estimator

using Random
using Printf
using ProgressMeter
using POMDPLinter: @show_requirements, requirements_info, @POMDP_require, @req, @subreq
import POMDPLinter

using D3Trees
using Colors

#export
#    StateNode

#export
#    AbstractStateNode,
#    StateActionStateNode,
#    DPWStateActionNode,
#    DPWStateNode,

abstract type AbstractCMCTSPlanner{P<:Union{CMDP,CPOMDP}} <: Policy end
abstract type AbstractCMCTSSolver <: Solver end

#include("requirements_info.jl") # TODO

export 
    estimate_value
include("rollout.jl")

# export 
#    CMCTSSolver,
#    CMCTSPlanner
# include("vanilla_types.jl")
# include("vanilla_solver.jl")


export
    CDPWSolver,
    CDPWPlanner,
    BeliefCMCTSSolver,
    AbstractCMCTSPlanner,
    AbstractCMCTSSolver,
    solve,
    action,
    action_info,
    clear_tree!

include("cdpw_types.jl")
include("cdpw_solver.jl")

export
    BeliefCMCTSSolver,
    CMCTSBudgetUpdateWrapper,
    updater,
    update,
    initialize_belief
include("belief_cmcts.jl")

export
    tooltip_tag,
    node_tag
include("visualization.jl")

end # module
