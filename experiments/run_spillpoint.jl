using COBTSExperiments
using COTS 
using CMCTS
using CPOMCPOW
using CPOMDPs
using Random
using ProgressMeter
using ParticleFilters
using SpillpointPOMDP
using POMDPs
using D3Trees

# experiments to run
experiments = Dict(
    "cpft-infogain"=>false,
    "cpft-noheur"=>false,
    "cobts"=>true,
    "cpomcpow"=>false,
)
nsims = 10

# same kwargs for pft and cobts algorithms
kwargs = Dict(:n_iterations=>Int(1e2), 
        :k_state => 5., # 0.1, # ld experiments use 5
        :alpha_state => 0.25, #0.5, # ld experiments use 1/15
        :enable_action_pw=>false,
        :depth => 25,
        :exploration_constant=>30., # ld experiments use 90
        :nu=>0., 
        :estimate_value => zero_V,
        :tree_in_info => false,
        :search_progress_info => false)

as = 10000.
max_steps = 25
search_pf_size = Int(1e2)
cpomdp_pf_size = Int(1e2)


# CPOMCPOW kwargs 
cpomcpow_kwargs = Dict(
    :tree_queries=>kwargs[:n_iterations]*search_pf_size, # fair to total number of particles
    :k_observation => kwargs[:k_state], 
    :alpha_observation => kwargs[:alpha_state], 
    :enable_action_pw => kwargs[:enable_action_pw],
    :check_repeat_obs => false,
    :max_depth => kwargs[:depth],
    :criterion => MaxCUCB(kwargs[:exploration_constant], kwargs[:nu]), 
    :alpha_schedule => CPOMCPOW.ConstantAlphaSchedule(as),
    :estimate_value=>kwargs[:estimate_value],
)

# options
cpomdp = SpillpointInjectionCPOMDP(constraint_budget=1e-6)
m = cpomdp.pomdp
options() = [
    SingleActionWrapper((:stop, 0.0)),
    [SingleActionWrapper((:inject, val)) for val in m.injection_rates]...,
    [SingleActionWrapper((:observe, config)) for config in m.obs_configurations]...,
    InferGeology(cpomdp, [(:observe,  m.obs_configurations[1])]),
    InferGeology(cpomdp, [(:observe,  m.obs_configurations[1]), (:observe,  m.obs_configurations[1])]),
    InferGeology(cpomdp, [(:observe,  m.obs_configurations[2])]),
    InferGeology(cpomdp, [(:observe,  m.obs_configurations[2]), (:observe,  m.obs_configurations[2])]),
    InferGeology(cpomdp, [(:observe,  m.obs_configurations[1]), (:observe,  m.obs_configurations[2])]),
    SafeFill(cpomdp)    
]

default_up(rng) = SpillpointPOMDP.SIRParticleFilter(
	model=m, 
	N=200, 
	state2param=SpillpointPOMDP.state2params, 
	param2state=SpillpointPOMDP.params2state,
	N_samples_before_resample=100,
    clampfn=SpillpointPOMDP.clamp_distribution,
	prior=SpillpointPOMDP.param_distribution(initialstate(m)),
	elite_frac=0.3,
	bandwidth_scale=.5,
	max_cpu_time=60,
    rng=rng
)

cobts_updater(rng) = SpillpointPOMDP.SIRParticleFilter(
	model=m, 
	N=search_pf_size, 
	state2param=SpillpointPOMDP.state2params, 
	param2state=SpillpointPOMDP.params2state,
	N_samples_before_resample=50,
    clampfn=SpillpointPOMDP.clamp_distribution,
	prior=SpillpointPOMDP.param_distribution(initialstate(m)),
	elite_frac=0.3,
	bandwidth_scale=.5,
	max_cpu_time=1,
    rng=rng
)



function COTS.options(sol::COTSSolver, mdp::GenerativeBeliefCMDP{P, U, B, A}, b) where {P <: SpillpointInjectionCPOMDP, U, B, A}
    m = mdp.cpomdp.pomdp
    state = rand(b)
    if isnothing(state.x_inj)
		return [SingleActionWrapper((:drill, val)) for val in m.drill_locations]
	else
		return sol.options
	end
end

rng_stseed = 0

### Experiments
results = Dict(k=>LightExperimentResults(nsims) for (k,v) in experiments if v)

# CPFT-Infogain
exp = "cpft-infogain"
if experiments[exp]
    Threads.@threads for i = 1:nsims
        println("Experiment $exp, sim $i")
        infogain_kwargs = copy(kwargs)
        infogain_kwargs[:estimate_value] = QMDP_V
        rng = MersenneTwister(rng_stseed+i)
        search_updater = cobts_updater(rng) #BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = BeliefCMCTSSolver(
            CDPWSolver(;infogain_kwargs..., 
                rng = rng,
                alpha_schedule = CMCTS.ConstantAlphaSchedule(as),
                return_safe_action=true,
            ), search_updater;
            exact_rewards=false)
        planner = solve(solver, cpomdp)
        updater = CMCTSBudgetUpdateWrapper(default_up(rng), planner)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater, max_steps; rng=rng, track_history=false)
    end
    println("Results for $(exp)")
    print_and_save(results[exp], "results/spillpoint_$(exp)_$(nsims)sims.jld2") 
end

# CPFT-No Heuristic
exp = "cpft-noheur"
if experiments[exp]
    Threads.@threads for i = 1:nsims
        println("Experiment $exp, sim $i")
        rng = MersenneTwister(rng_stseed+i)
        search_updater = cobts_updater(rng) # BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = BeliefCMCTSSolver(
            CDPWSolver(;kwargs...,
                rng = rng,
                alpha_schedule = CMCTS.ConstantAlphaSchedule(as),
                return_safe_action=true,
            ), search_updater;
            exact_rewards=false)
        planner = solve(solver, cpomdp)
        updater = CMCTSBudgetUpdateWrapper(default_up(rng), planner)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater, max_steps; rng=rng, track_history=false)
    end
    println("Results for $(exp)")
    print_and_save(results[exp], "results/spillpoint_$(exp)_$(nsims)sims.jld2") 
end



# COBTS
exp = "cobts"
if experiments[exp]
    Threads.@threads for i = 1:nsims
        println("Experiment $exp, sim $i")
        rng = MersenneTwister(rng_stseed+i)
        search_updater = cobts_updater(rng) # BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = COBTSSolver(
            COTSSolver(;kwargs..., 
                rng = rng,
                options = options(),
                alpha_schedule = COTS.ConstantAlphaSchedule(as),
                return_safe_action=true,
                estimate_value = QMDP_V,
            ), search_updater;
            exact_rewards=false) # TODO: Maybe switch back
        planner = solve(solver, cpomdp)
        updater = default_up(rng)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater, max_steps; rng=rng, track_history=false)
    end
    println("Results for $(exp)")
    print_and_save(results[exp], "results/spillpoint_$(exp)_$(nsims)sims.jld2") 
end

# CPOMCPOW
exp = "cpomcpow"
if experiments[exp]
    Threads.@threads for i = 1:nsims
        println("Experiment $exp, sim $i")
        rng = MersenneTwister(rng_stseed+i)
        solver = CPOMCPOWSolver(;cpomcpow_kwargs..., rng = rng)
        planner = solve(solver, cpomdp)
        updater = default_up(rng)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater, max_steps; rng=rng, track_history=false)
    end
    println("Results for $(exp)")
    print_and_save(results[exp], "results/spillpoint_$(exp)_$(nsims)sims.jld2") 
end

# Print and save
for (key,result) in results
    println("Results for $(key)")
    print_and_save(result, "results/all_spillpoint_$(key)_$(nsims)sims.jld2") 
end