using COBTSExperiments
using RoombaPOMDPs
using POMDPs
using CPOMDPs
using ParticleFilters
using Cairo
using Gtk
using Random
using Printf
using POMDPSimulators
using CPOMCPOW
using CMCTS
using COTS
using Infiltrator
using D3Trees
using Plots
using Statistics

search_times = []
n_seeds = 5
for seed = 1:n_seeds
Random.seed!(seed)

cpomdp = LightDarkCPOMDP(cost_budget=0.1)

options = [GoToGoal(cpomdp), 
    LocalizeFast(cpomdp,10.,0.2), LocalizeSlow(cpomdp,10.,0.2), LocalizeSafe(cpomdp, 10., 12., 1., 0.2),
    LocalizeFast(cpomdp,10.,0.5), LocalizeSlow(cpomdp,10.,0.5), LocalizeSafe(cpomdp, 10., 12., 1., 0.5), 
    LocalizeFast(cpomdp,10.,1.), LocalizeSlow(cpomdp,10.,1.), LocalizeSafe(cpomdp, 10., 12., 1., 1.)]

as = 0.5 # 1.
search_pf_size = Int(10)
cpomdp_pf_size = Int(1e4)

max_steps = 100
global run_policy = 3 # [1 = CPOMCPOW, 2 = CPFT-DPW, 3=COBETS]

# same kwargs for pft and cobts algorithms
kwargs = Dict(:n_iterations=>Int(1e4), 
        :k_state => 1., # 0.1, # ld experiments use 5
        :alpha_state => 1/5, #0.5, # ld experiments use 1/15
        :enable_action_pw=>false,
        :depth => 10,
        :exploration_constant=>200., # ld experiments use 90
        :nu=>0., 
        :estimate_value => zero_V,
        :tree_in_info => false,
        :search_progress_info => false)

cpomcpow_kwargs = Dict(:tree_queries=>kwargs[:n_iterations]*search_pf_size, # fair to total number of particles
                        :k_observation => kwargs[:k_state], 
                        :alpha_observation => kwargs[:alpha_state], 
                        :enable_action_pw => kwargs[:enable_action_pw],
                        :check_repeat_obs => false,
                        :max_depth => kwargs[:depth],
                        :criterion => MaxCUCB(kwargs[:exploration_constant], kwargs[:nu]), 
                        :alpha_schedule => CPOMCPOW.ConstantAlphaSchedule(as),
                        :estimate_value=>kwargs[:estimate_value])

if run_policy == 1
    solver = CPOMCPOWSolver(;cpomcpow_kwargs...)
    planner = solve(solver, cpomdp)
    belief_updater = CPOMCPOWBudgetUpdateWrapper(BootstrapFilter(cpomdp, cpomdp_pf_size), planner)

elseif run_policy == 2
    search_updater = BootstrapFilter(cpomdp, search_pf_size)
    solver = BeliefCMCTSSolver(
        CDPWSolver(;kwargs...,
            alpha_schedule = CMCTS.ConstantAlphaSchedule(as),
            return_safe_action=false,
        ), search_updater;
        exact_rewards=false)
    planner = solve(solver, cpomdp)
    belief_updater = CMCTSBudgetUpdateWrapper(BootstrapFilter(cpomdp, cpomdp_pf_size), planner)

elseif run_policy == 3
    n_options = 4
    search_updater = BootstrapFilter(cpomdp, search_pf_size)
    solver = COBTSSolver(
        COTSSolver(;kwargs...,
            options = options[1:n_options],
            alpha_schedule = COTS.ConstantAlphaSchedule(as),
            return_safe_action=true,
        ), search_updater;
        exact_rewards=true)
    planner = solve(solver, cpomdp)
    belief_updater = BootstrapFilter(cpomdp, cpomdp_pf_size)
else
    error("Not Implemented")
end

search_time = Inf
for (t, step) in enumerate(stepthrough(cpomdp, planner, belief_updater, max_steps=1))  # set to one as we only want to find the search time here
    if planner isa OptionsPolicy 
        if step.action_info.select !== nothing
            # tree = step.action_info.select[:tree]
            search_time = step.action_info.select[:search_time]
        else
            skip=true
        end
    else
        # tree = step.action_info[:tree]
        search_time = step.action_info[:search_time]
    end
end
println("... search_time: ", search_time)
global search_times = [search_times; search_time]
end
println("Average search_time: ", mean(search_times))
using JLD2, FileIO
@save "results/time_lightdark_policy$(run_policy)_$(n_seeds)seeds.jld2" search_times
