module COTS

using Infiltrator

using CPOMDPs
using POMDPs
import POMDPs: action, update, updater, initialize_belief
using POMDPTools
using MCTS
import MCTS: tooltip_tag, node_tag, estimate_value, convert_estimator
using POMDPLinter: @show_requirements, requirements_info, @POMDP_require, @req, @subreq
import POMDPLinter

using Random
using Printf
using ProgressMeter
using Parameters
using D3Trees
using Colors

#include("requirements_info.jl") # TODO

export 
    next_option
include("option_gen.jl")

export
    COTSSolver,
    COTSPlanner,
    solve,
    action,
    action_info,
    clear_tree!
include("cots_types.jl")
include("cots_solver.jl")
include("rollout.jl")

export
    COBTSSolver,
    COBTSBudgetUpdateWrapper,
    updater,
    update,
    initialize_belief,
    OptionsUpdateWrapper
include("cobts.jl")

export
    tooltip_tag,
    node_tag
include("visualization.jl")

end # module
