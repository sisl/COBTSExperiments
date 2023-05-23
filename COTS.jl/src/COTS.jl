module COTS

using CPOMDPs
import CPOMDPs: rng, low_level, select_option, update!

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
using Parameters
using D3Trees
using Colors

#include("requirements_info.jl") # TODO

export 
    estimate_value
include("rollout.jl")

export
    COTSSolver,
    COTSPlanner,
    solve,
    action,
    action_info,
    clear_tree!
include("cots_types.jl")
include("cots_solver.jl")

export
    COBTSSolver,
    COBTSBudgetUpdateWrapper,
    updater,
    update,
    initialize_belief
include("cobts.jl")

export
    tooltip_tag,
    node_tag
include("visualization.jl")

end # module
