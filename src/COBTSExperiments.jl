module COBTSExperiments

using Infiltrator
using Debugger 
using ProgressMeter
using Parameters
using POMDPGifs
using ParticleFilters
using LinearAlgebra
using Plots
import Statistics
using Random
using Distributed
using FileIO

using POMDPs
using POMDPTools

using CPOMDPs
import CPOMDPs: costs, costs_limit, n_costs, terminate
import MCTS: node_tag

# constrained solvers
# using CMCTS
# using COTS

# models 
using POMDPModels
export 
    LightDarkNew, 
    CLightDarkNew,
    CNav, 
    zeroV_trueC
include("cpomdps/cnav.jl")
include("cpomdps/clightdark.jl")

#using RockSample
#export RockSampleCPOMDP
#include("cpomdps/crocksample.jl")

#using SpillpointPOMDP
#export SpillpointInjectionCPOMDP
#include("cpomdps/cspillpoint.jl")

#using RoombaPOMDPs ????

# Policies
export
    Navigate,
    NavigateSlow,
    GoToGoal
include("policies.jl")

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
