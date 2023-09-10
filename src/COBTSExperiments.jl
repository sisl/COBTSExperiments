module COBTSExperiments

using Infiltrator
using Debugger 
using ProgressMeter
using Parameters
using POMDPGifs
using ParticleFilters
using LinearAlgebra
using Plots
using Statistics
import Statistics: mean, std
using Random
using Distributed
using FileIO
using Distributions
using Printf
using Cairo
using POMDPs
using POMDPTools


using CPOMDPs
import CPOMDPs: costs, costs_limit, n_costs, terminate, reset!, update_option!
import MCTS: node_tag

# models 
using POMDPModels
using RoombaPOMDPs
export 
    LightDarkNew, 
    LightDarkCPOMDP,
    RoombaCPOMDP,
    RoombaCPOMDPInitBounds,
    CNav, 
    zeroV_trueC
include("cpomdps/cnav.jl")
include("cpomdps/clightdark.jl")
include("cpomdps/croomba.jl")

using SpillpointPOMDP
export SpillpointInjectionCPOMDP
include("cpomdps/cspillpoint.jl")

# Options Policies
export
    # Nav + LightDark
    Navigate,
    NavigateSlow,
    GoToGoal,
    LocalizeFast,
    LocalizeSlow,
    LocalizeSafe,
    MultiActionWrapper,
    # Roomba
    GoToGoal2D,
    Localize2D,
    GreedyGoToGoal,
    SafeGoToGoal,
    TurnThenGo,
    BigSpin,
    Spin,
    #Spillpoint
    SingleActionWrapper,
    InferGeology,
    SafeFill
include("policies/ld_nav_policies.jl")
include("policies/roomba_policies.jl")
include("policies/spillpoint_policies.jl")

# helpers
export
    plot_lightdark_beliefs,
    LightExperimentResults,
    ExperimentResults,
    print_and_save,
    load_and_print,
    mean,
    std,
    zero_V,
    QMDP_V,
    heuristicV,
    SearchProgress,
    max_clip
include("utils.jl") 

# experiment scripts
export
    run_cmdp_simulation,
    run_cpomdp_simulation
include("experiments.jl")

end # module COBTSExperiments
