using COBTSExperiments
using COTS 
using CMCTS
using Random
using D3Trees
using ParticleFilters
using Infiltrator
using RoombaPOMDPs

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

options1 = [GoToGoal2D(cpomdp), Localize2D(cpomdp, [0.2, 0.2, 0.3]), Localize2D(cpomdp, [0.5, 0.5, 0.6])]
options2 = [GoToGoal2D(cpomdp), Localize2D(cpomdp, [0.1, 0.1, 0.2]), Localize2D(cpomdp, [0.8, 0.8, 0.9])]

runs = [false, false, true]

if runs[1]
    for i = 1:2 # time second run through
        rng = MersenneTwister(i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = BeliefCMCTSSolver(
            CDPWSolver(;kwargs..., 
                rng = rng,
                alpha_schedule = CMCTS.ConstantAlphaSchedule(as)
            ), search_updater)
        planner = solve(solver, cpomdp)
        updater = CMCTSBudgetUpdateWrapper(BootstrapFilter(cpomdp, cpomdp_pf_size, rng), planner)
        @time hist1, R1, C1 = run_cpomdp_simulation(cpomdp, planner, updater, 1; rng=rng)
        i==2 && inchrome(D3Tree(hist1[1][:tree];lambda=hist1[1][:lambda][end]))
    end
end

if runs[2]
    for i = 1:2
        rng = MersenneTwister(i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = COBTSSolver(
            COTSSolver(;kwargs..., 
                rng = rng,
                options = options1,
                alpha_schedule = COTS.ConstantAlphaSchedule(as)
            ), search_updater)
        planner = solve(solver, cpomdp)
        updater = BootstrapFilter(cpomdp, cpomdp_pf_size, rng)
        @time hist2, R2, C2 = run_cpomdp_simulation(cpomdp, planner, updater, 1; rng=rng)
        i==2 && inchrome(D3Tree(hist2[1][:tree];lambda=hist2[1][:lambda][end]))
    end
end

if runs[3]
    for i = 1:2
        rng = MersenneTwister(i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = COBTSSolver(
            COTSSolver(;kwargs..., 
                rng = rng,
                options = options2,
                alpha_schedule = COTS.ConstantAlphaSchedule(as)
            ), search_updater)
        planner = solve(solver, cpomdp)
        updater = BootstrapFilter(cpomdp, cpomdp_pf_size, rng)
        @time hist3, R3, C3 = run_cpomdp_simulation(cpomdp, planner, updater, 1; rng=rng)
        i==2 && inchrome(D3Tree(hist3[1][:tree];lambda=hist3[1][:lambda][end]))
    end
end

# plot([l[1] for l=hist3[1][:lambda]])
