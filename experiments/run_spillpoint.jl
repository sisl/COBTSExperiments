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

# experiments to run
experiments = Dict(
    "cpft-infogain"=>true,
    "cpft-noheur"=>true,
    "cobts"=>true,
    "cpomcpow"=>true,
)
nsims = 10

# same kwargs for pft and cobts algorithms
kwargs = Dict(:n_iterations=>Int(1e2), 
        :k_state => 10., # 0.1, # ld experiments use 5
        :alpha_state => 0.3, #0.5, # ld experiments use 1/15
        :enable_action_pw=>false,
        :depth => 25,
        :exploration_constant=>30., # ld experiments use 90
        :nu=>0., 
        :estimate_value => zero_V,
        :tree_in_info => false,
        :search_progress_info => false)

as = 10000.
max_steps = 25
search_pf_size = Int(10)
cpomdp_pf_size = Int(1e4)


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
cpomdp = SpillpointInjectionCPOMDP(constraint_budget=0.001)
m = cpomdp.pomdp
options = [
    SingleActionWrapper((:stop, 0.0)),
    [SingleActionWrapper((:observe, config)) for config in m.obs_configurations]...,
    InferGeology(cpomdp, 0.01, [(:observe,  m.obs_configurations[1])]),
    InferGeology(cpomdp, 0.05, [(:observe,  m.obs_configurations[1])]),
    InferGeology(cpomdp, 0.1, [(:observe,  m.obs_configurations[1])]),
    InferGeology(cpomdp, 0.01, [(:observe,  m.obs_configurations[1]), (:observe,  m.obs_configurations[1])]),
    InferGeology(cpomdp, 0.05, [(:observe,  m.obs_configurations[1]), (:observe,  m.obs_configurations[1])]),
    InferGeology(cpomdp, 0.1, [(:observe,  m.obs_configurations[1]), (:observe,  m.obs_configurations[1])]),
    InferGeology(cpomdp, 0.01, [(:observe,  m.obs_configurations[2])]),
    InferGeology(cpomdp, 0.05, [(:observe,  m.obs_configurations[2])]),
    InferGeology(cpomdp, 0.1, [(:observe,  m.obs_configurations[2])]),
    SafeFill(cpomdp)    
]

default_up() = SpillpointPOMDP.SIRParticleFilter(
	model=m, 
	N=200, 
	state2param=SpillpointPOMDP.state2params, 
	param2state=SpillpointPOMDP.params2state,
	N_samples_before_resample=100,
    clampfn=SpillpointPOMDP.clamp_distribution,
	prior=SpillpointPOMDP.param_distribution(initialstate(m)),
	elite_frac=0.3,
	bandwidth_scale=.5,
	max_cpu_time=60
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
    @showprogress for i = 1:nsims
        infogain_kwargs = copy(kwargs)
        infogain_kwargs[:estimate_value] = QMDP_V
        rng = MersenneTwister(rng_stseed+i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = BeliefCMCTSSolver(
            CDPWSolver(;infogain_kwargs..., 
                rng = rng,
                alpha_schedule = CMCTS.ConstantAlphaSchedule(as),
                return_safe_action=true,
            ), search_updater;
            exact_rewards=true)
        planner = solve(solver, cpomdp)
        updater = CMCTSBudgetUpdateWrapper(default_up(), planner)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater, max_steps; rng=rng, track_history=false)
        println("Results for $(exp)")
        print_and_save(results[exp], "results/spillpoint_$(exp)_$(nsims)sims.jld2") 
    end
end

# CPFT-No Heuristic
exp = "cpft-noheur"
if experiments[exp]
    @showprogress for i = 1:nsims
        rng = MersenneTwister(rng_stseed+i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = BeliefCMCTSSolver(
            CDPWSolver(;kwargs...,
                rng = rng,
                alpha_schedule = CMCTS.ConstantAlphaSchedule(as),
                return_safe_action=true,
            ), search_updater;
            exact_rewards=true)
        planner = solve(solver, cpomdp)
        updater = CMCTSBudgetUpdateWrapper(default_up(), planner)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater, max_steps; rng=rng, track_history=false)
        println("Results for $(exp)")
        print_and_save(results[exp], "results/spillpoint_$(exp)_$(nsims)sims.jld2") 
    end
end



# COBTS
exp = "cobts"
if experiments[exp]
    @showprogress for i = 1:nsims
        rng = MersenneTwister(rng_stseed+i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = COBTSSolver(
            COTSSolver(;kwargs..., 
                rng = rng,
                options = options,
                alpha_schedule = COTS.ConstantAlphaSchedule(as),
                return_safe_action=true,
            ), search_updater;
            exact_rewards=false) # TODO: Maybe switch back
        planner = solve(solver, cpomdp)
        updater = default_up()
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater, max_steps; rng=rng, track_history=false)
        println("Results for $(exp)")
        print_and_save(results[exp], "results/spillpoint_$(exp)_$(nsims)sims.jld2") 
    end
end

# CPOMCPOW
exp = "cpomcpow"
if experiments[exp]
    @showprogress for i = 1:nsims
        rng = MersenneTwister(rng_stseed+i)
        solver = CPOMCPOWSolver(;cpomcpow_kwargs..., rng = rng)
        planner = solve(solver, cpomdp)
        updater = default_up()
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater, max_steps; rng=rng, track_history=false)
        println("Results for $(exp)")
        print_and_save(results[exp], "results/spillpoint_$(exp)_$(nsims)sims.jld2") 
    end
end

# Print and save
for (key,result) in results
    println("Results for $(key)")
    print_and_save(result, "results/all_spillpoint_$(key)_$(nsims)sims.jld2") 
end