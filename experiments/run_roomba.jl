using COBTSExperiments
using RoombaPOMDPs
using POMDPs
using ParticleFilters
using Random
using Printf
using POMDPSimulators
using POMDPModels
using POMDPTools
using CMCTS
using COTS
using ProgressMeter

kwargs = Dict(:n_iterations=>Int(1e4), 
        :k_state => 1., # 0.1,
        :alpha_state => 1/5, #0.5,
        :enable_action_pw=>false,
        :depth => 10,
        :exploration_constant=>200.,
        :nu=>0., 
        :estimate_value=>zeroV_trueC,
        :tree_in_info => true,
        :search_progress_info=>true)

sensor = Bumper() # Bumper() or Lidar()
config = 1 # 1,2, or 3 for different room configurations
vs = [0, 1, 5]
oms = [-π/2, 0, π/2] # with a dt of 0.5 seconds, this is 45 degrees per step
RoombaActSpace = [RoombaAct(v, om) for v in vs for om in oms]
pomdp = RoombaPOMDP(sensor=sensor,
    mdp=RoombaMDP(v_max=maximum(vs), om_max=maximum(oms),config=config, aspace=RoombaActSpace, stairs_penalty=0.0))
cpomdp = RoombaCPOMDP(pomdp)

as = 0.5
search_pf_size = Int(20)
cpomdp_pf_size = Int(1e4)

# v_noise_coeff = 2.0
# om_noise_coeff = 0.5

options1 = [GoToGoal2D(cpomdp), Localize2D(cpomdp, 0.2), Localize2D(cpomdp, 0.5)]
options2 = [GoToGoal2D(cpomdp), Localize2D(cpomdp, 0.1), Localize2D(cpomdp, 0.8)]

runs = [true, false, false]
nsims = 50
if runs[1]
    e1 = LightExperimentResults(nsims)
    @showprogress for i = 1:nsims
        rng = MersenneTwister(i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = BeliefCMCTSSolver(
            CDPWSolver(;kwargs..., 
                rng = rng,
                alpha_schedule = CMCTS.ConstantAlphaSchedule(as)
            ), search_updater)
        planner = solve(solver, cpomdp)
        # updater = CMCTSBudgetUpdateWrapper(RoombaParticleFilter(pomdp, cpomdp_pf_size, v_noise_coeff, om_noise_coeff, rng), planner)
        updater = CMCTSBudgetUpdateWrapper(BootstrapFilter(cpomdp, cpomdp_pf_size, rng), planner)
        e1[i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
    end
    print_and_save(e1,"results/roomba_cmcts_$(nsims)sims.jld2")
end

if runs[2]
    e2 = LightExperimentResults(nsims)
    @showprogress for i = 1:nsims
        rng = MersenneTwister(i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = COBTSSolver(
            COTSSolver(;kwargs..., 
                rng = rng,
                options = options1,
                alpha_schedule = COTS.ConstantAlphaSchedule(as)
            ), search_updater)
        planner = solve(solver, cpomdp)
        updater = RoombaParticleFilter(pomdp, cpomdp_pf_size, v_noise_coeff, om_noise_coeff, rng)
        e2[i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
    end
    print_and_save(e2,"results/roomba_cobts1_$(nsims)sims.jld2")
end

if runs[3]
    e3 = LightExperimentResults(nsims)
    @showprogress for i = 1:nsims
        rng = MersenneTwister(i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = COBTSSolver(
            COTSSolver(;kwargs..., 
                rng = rng,
                options = options2,
                alpha_schedule = COTS.ConstantAlphaSchedule(as)
            ), search_updater)
        planner = solve(solver, cpomdp)
        updater = RoombaParticleFilter(pomdp, cpomdp_pf_size, v_noise_coeff, om_noise_coeff, rng)
        e3[i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
    end
    print_and_save(e3,"results/roomba_cobts2_$(nsims)sims.jld2")
end
