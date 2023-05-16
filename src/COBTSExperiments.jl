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

# non-constrained baseline
using BasicPOMCP
using MCTS # belief-mcts for belief dpw
using POMCPOW

using CPOMDPs
import CPOMDPs: costs, costs_limit, n_costs

# constrained solvers
# using CMCTS

# models 
using POMDPModels
export LightDarkNew, CLightDarkNew, zeroV_trueC
include("cpomdps/clightdark.jl")

#using RockSample
#export RockSampleCPOMDP
#include("cpomdps/crocksample.jl")

#using SpillpointPOMDP
#export SpillpointInjectionCPOMDP
#include("cpomdps/cspillpoint.jl")

#using RoombaPOMDPs ????

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
    SearchProgress,
    max_clip
include("utils.jl") 

# experiment scripts
export
    run_pomdp_simulation,
    run_cpomdp_simulation,
    run_lambda_experiments
include("experiments.jl")


end # module COBTSExperiments
