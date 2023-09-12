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

sensor = FrontBumper(π/2-0.01) # Bumper() or Lidar()
vs = [0, 3]
oms = [-π/2, 0, π/2] # with a dt of 0.5 seconds, this is 45 degrees per step
RoombaActSpace = [RoombaAct(v, om) for v in vs for om in oms]
v_noise_coefficient = 0.15 #1.0
om_noise_coefficient = 0.02 #0.5
v_max = maximum(vs) + v_noise_coefficient/2 # allow PF to hit maximum target noise
om_max = maximum(oms) + om_noise_coefficient/2
pomdp = RoombaPOMDP(sensor=sensor,
                    mdp=RoombaMDP(v_max=v_max, om_max=om_max,config=4, 
                    aspace=RoombaActSpace, 
                    stairs_penalty=-10.0, goal_reward=10., 
                    contact_pen=0.,time_pen=-0.05,
                    discount=0.999))
cpomdp = RoombaCPOMDP(pomdp, cost_budget=0.1,
    # init_bounds=RoombaCPOMDPInitBounds(-24.5,-15.5,-19.5,4.5,0.,3π/2), # general
    init_bounds=RoombaCPOMDPInitBounds(-24.,-16.,-14.,4.,π/2-0.1,π/2+0.1), # target
    # init_bounds=RoombaCPOMDPInitBounds(-15.5,-15.5,-16.,-16.,0.,0.), # specific
    )

options = [
    GreedyGoToGoal(cpomdp;max_steps=80, max_std=[2.,2.]), #GreedyGoToGoal(cpomdp;max_steps=20),
    SafeGoToGoal(cpomdp;max_steps=80, max_std=[2.,2.]), #SafeGoToGoal(cpomdp;max_steps=20),
    TurnThenGo(cpomdp;turn_steps=0,max_steps=40), 
    TurnThenGo(cpomdp;turn_steps=2,max_steps=40), TurnThenGo(cpomdp;turn_steps=-2,max_steps=40), 
    #TurnThenGo(cpomdp;turn_steps=3,max_steps=40),TurnThenGo(cpomdp;turn_steps=-3,max_steps=40), 
    TurnThenGo(cpomdp;turn_steps=4,max_steps=40),
    ]
    
num_particles = 10000


belief_updater = RoombaParticleFilter(cpomdp.pomdp, 
    num_particles, v_noise_coefficient, om_noise_coefficient)
max_steps = 100
global run_policy = 3 # [1 = CPOMCPOW, 2 = CPFT-DPW, 3=COBETS]

### CPOMCPOW policy p
if run_policy == 1
    # CPOMCPOW kwargs 
    cpomcpow_kwargs = Dict(
        :tree_queries=>Int(1e5),
        # observation widening: Bumper = None, Lidar = (1., 1/5)
        :k_observation => 1., 
        :alpha_observation => 1/5., 
        :check_repeat_obs => true,

        # action widening: If |A| is small, None
        :enable_action_pw => true,
        :k_action => 1.,
        :alpha_action => 1/5., 
        :check_repeat_act => true,

        :max_depth => max_steps,
        # Q + k sqrt(N(sa)/log(N))
        :criterion => MaxCUCB(2., 0.), 
        :alpha_schedule => CPOMCPOW.ConstantAlphaSchedule(1.), # lambda = max(0, lambda + alphaschedule(iteration)*( Qsim - budget ))
        :estimate_value=>zero_V,
    )
    solver = CPOMCPOWSolver(;cpomcpow_kwargs...)
    p = solve(solver, cpomdp)
    belief_updater = CPOMCPOWBudgetUpdateWrapper(belief_updater, p)
elseif run_policy == 2
    # CPFT-DPW kwargs
    cpft_kwargs = Dict(
        :n_iterations=>Int(1e5),

        # belief-state widening: Bumper = None, Lidar = (1., 1/5)
        :enable_state_pw => true,
        :k_state => 1.,
        :alpha_state => 1/5, 

        # action-state widening: If |A| is small, None
        :enable_action_pw=>true,
        :k_action => 1.,
        :alpha_action => 1/5.,
        
        :depth => max_steps,
        :estimate_value => zero_V, # heuristicV
        :exploration_constant => 2.,
        :nu => 0.,
        :alpha_schedule => CMCTS.ConstantAlphaSchedule(1.),
    )
    search_updater = RoombaParticleFilter(cpomdp.pomdp, 30, 
        v_noise_coefficient, om_noise_coefficient)
#    search_updater = BootstrapFilter(cpomdp, 30)
    solver = BeliefCMCTSSolver(
        CDPWSolver(;cpft_kwargs...), search_updater;
        exact_rewards=false)
    planner = solve(solver, cpomdp)
    p = solve(solver, cpomdp)
    belief_updater = CMCTSBudgetUpdateWrapper(belief_updater, p)

elseif run_policy == 3
    # CoBETS kwargs
    cobts_kwargs = Dict(
        :options => options,
        :n_iterations=>Int(1e5),

        # belief-state widening: Bumper = None, Lidar = (1., 1/5)
        :enable_state_pw => true,
        :k_state => 1.,
        :alpha_state => 1/5, 

        # option widening: Default false
        :enable_action_pw=>false,
        :return_safe_action=>true,
        :depth => max_steps,
        :estimate_value => zero_V, # heuristicV
        :exploration_constant => 4.,
        :nu => 0.,
        :alpha_schedule => COTS.ConstantAlphaSchedule(1.),
    )
    search_updater = RoombaSearchParticleFilter(cpomdp.pomdp, 30, 
        v_noise_coefficient, om_noise_coefficient)
    #search_updater = BootstrapFilter(cpomdp, 30)
    solver = COBTSSolver(
        COTSSolver(;cobts_kwargs...), search_updater;
        exact_rewards=true)
    p = solve(solver, cpomdp)
    belief_updater = OptionsUpdateWrapper(belief_updater, p) # external action wrapper
else
    error("Not Implemented")
end

search_time = Inf
for (t, step) in enumerate(stepthrough(cpomdp, p, belief_updater, max_steps=1))  # set to one as we only want to find the search time here
    if p isa OptionsPolicy 
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
@save "results/time_roomba_bumper_policy$(run_policy)_$(n_seeds)seeds.jld2" search_times
