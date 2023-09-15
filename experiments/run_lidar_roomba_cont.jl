using COBTSExperiments
using RoombaPOMDPs
using POMDPs
using CPOMDPs
using ParticleFilters
using Random
using Printf
using POMDPSimulators
using CPOMCPOW
using CMCTS
using COTS
using ProgressMeter

# experiments to run
experiments = Dict(
    "cpft-heur"=>false,
    "cpft-noheur"=>false,
    "cobts"=>true,
    "cpomcpow"=>false,
)
nsims = 50

problem = "continuous"
sensor = Lidar()
v_noise_coefficient = 0.2 #0.5
om_noise_coefficient = 0.15 #0.5
v_max = 5. #5.
om_max = π/2 #π/2
pomdp = RoombaPOMDP(sensor=sensor,
                    mdp=RoombaMDP(v_max=v_max, om_max=om_max,config=4, 
                    stairs_penalty=-10.0, goal_reward=10.,
                    contact_pen=0.,time_pen=-0.05,
                    discount=0.999))
cpomdp = RoombaCPOMDP(pomdp, cost_budget=0.1, init_bounds=RoombaCPOMDPInitBounds(-24.5,-15.5,-19.5,4.5,π/2,3π/2))

options = [
    GreedyGoToGoal(cpomdp;max_steps=13, max_std=[10.,10.]),
    SafeGoToGoal(cpomdp;max_steps=13, barrier_penalty=Inf,  max_std=[10.,10.]),
    Spin(cpomdp;max_steps=5),Spin(cpomdp;max_steps=8), Spin(cpomdp;max_steps=15),
    BigSpin(cpomdp;max_steps=10), BigSpin(cpomdp;turn_every=2,max_steps=10)
    ]
    
num_particles = 10000
max_steps = 100

# Define next action function
rng = Random.GLOBAL_RNG # change this if needed
function generate_random_action(p::RoombaCPOMDP, s, o, rng)
    v_max = p.pomdp.mdp.v_max
    om_max = p.pomdp.mdp.om_max
    # Define uniform distribution over actions
    v = rand(rng, 0:v_max)
    om = rand(rng, -om_max:om_max)
    return RoombaAct(v, om)
end
generate_random_action(p::GenerativeBeliefCMDP, s, o, rng) = generate_random_action(p.cpomdp, s, o, rng)

# CPOMCPOW kwargs 
cpomcpow_kwargs = Dict(
    :tree_queries=>Int(1e4),
    # observation widening: Bumper = None, Lidar = (1., 1/5)
    :k_observation => 1., 
    :alpha_observation => 1/5., 
    :check_repeat_obs => true,

    # action widening: If |A| is small, None
    :enable_action_pw => true,
    :k_action => 1.,
    :alpha_action => 1/5., 
    :check_repeat_act => true,
    :next_action => (p,s,o) -> generate_random_action(p,s,o,rng),

    :max_depth => max_steps,
    # Q + k sqrt(N(sa)/log(N))
    :criterion => MaxCUCB(2., 0.), 
    :alpha_schedule => CPOMCPOW.ConstantAlphaSchedule(1.), # lambda = max(0, lambda + alphaschedule(iteration)*( Qsim - budget ))
    :estimate_value=>zero_V,
)


# CPFT-DPW kwargs
cpft_kwargs = Dict(
    :n_iterations=>Int(1e3),

    # belief-state widening: Bumper = None, Lidar = (1., 1/5)
    :enable_state_pw => true,
    :k_state => 1.,
    :alpha_state => 1/5, 

    # action-state widening: If |A| is small, None
    :enable_action_pw=>true,
    :k_action => 1.,
    :alpha_action => 1/5.,
    :next_action => (p,s,o) -> generate_random_action(p,s,o,rng),
    
    :depth => max_steps,
    :estimate_value => zero_V, # heuristicV
    :exploration_constant => 2.,
    :nu => 0.,
    :alpha_schedule => CMCTS.ConstantAlphaSchedule(1.),
)

# CoBETS kwargs
cobts_kwargs = Dict(
    :options => options,
    :n_iterations=>Int(4e3),

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

rng_stseed = 0

### Experiments
results = Dict(k=>LightExperimentResults(nsims) for (k,v) in experiments if v)

# COBTS
exp = "cobts"
if experiments[exp]
    println("Results for $(exp)")
    @showprogress for i = 1:nsims
        Random.seed!(i)
        belief_updater = RoombaParticleFilter(cpomdp.pomdp, 
            num_particles, v_noise_coefficient, om_noise_coefficient)
        search_updater = RoombaSearchParticleFilter(cpomdp.pomdp, 30, 
            v_noise_coefficient, om_noise_coefficient)
        solver = COBTSSolver(
            COTSSolver(;cobts_kwargs...), search_updater;
            exact_rewards=true)
        p = solve(solver, cpomdp)
        belief_updater = OptionsUpdateWrapper(belief_updater, p) # external action wrapper
    
        results[exp][i] = run_cpomdp_simulation(cpomdp, p, belief_updater, max_steps; track_history=false)
    end
    print_and_save(results[exp], "results/roomba_lidar_cont_$(exp)_$(nsims)sims.jld2") 
end

# CPOMCPOW
exp = "cpomcpow"
if experiments[exp]
    println("Results for $(exp)")
    @showprogress for i = 1:nsims
        Random.seed!(i)
        solver = CPOMCPOWSolver(;cpomcpow_kwargs...)
        p = solve(solver, cpomdp)
        belief_updater = RoombaParticleFilter(cpomdp.pomdp, 
            num_particles, v_noise_coefficient, om_noise_coefficient)
        belief_updater = CPOMCPOWBudgetUpdateWrapper(belief_updater, p)
        results[exp][i] = run_cpomdp_simulation(cpomdp, p, belief_updater, max_steps; track_history=false)
    end
    print_and_save(results[exp], "results/roomba_lidar_cont_$(exp)_$(nsims)sims.jld2") 
end

# CPFT-Infogain
exp = "cpft-heur"
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
        updater = CMCTSBudgetUpdateWrapper(belief_updater, planner)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater, max_steps; track_history=false)
        println("Results for $(exp)")
        print_and_save(results[exp], "results/roomba_lidar_cont_$(exp)_$(nsims)sims.jld2") 
    end
end

# CPFT-No Heuristic
exp = "cpft-noheur"
if experiments[exp]
    println("Results for $(exp)")
    @showprogress for i = 1:nsims
        Random.seed!(i)
        belief_updater = RoombaParticleFilter(cpomdp.pomdp, 
            num_particles, v_noise_coefficient, om_noise_coefficient)
        search_updater = RoombaSearchParticleFilter(cpomdp.pomdp, 30, 
            v_noise_coefficient, om_noise_coefficient)        
        solver = BeliefCMCTSSolver(
            CDPWSolver(;cpft_kwargs...), search_updater;
            exact_rewards=false)
        p = solve(solver, cpomdp)
        belief_updater = CMCTSBudgetUpdateWrapper(belief_updater, p)
        results[exp][i] = run_cpomdp_simulation(cpomdp, p, belief_updater, max_steps; track_history=false)
    end
    print_and_save(results[exp], "results/roomba_lidar_cont_$(exp)_$(nsims)sims.jld2") 
end

# Print and save
for (key,result) in results
    println("Results for $(key)")
    print_and_save(result, "results/all_roomba_lidar_cont_$(key)_$(nsims)sims.jld2") 
end
