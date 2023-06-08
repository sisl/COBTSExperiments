using COBTSExperiments
using COTS 
using CMCTS
using Random
using ProgressMeter
using ParticleFilters

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

cpomdp = LightDarkCPOMDP(cost_budget=0.1)

as = 0.5
search_pf_size = Int(20)
cpomdp_pf_size = Int(1e4)
options1 = [GoToGoal(cpomdp), LocalizeFast(cpomdp,10.,1.), LocalizeFast(cpomdp,10.,0.5), LocalizeFast(cpomdp,10.,0.2), 
    LocalizeSlow(cpomdp,10.,1.), LocalizeSlow(cpomdp,10.,0.5), LocalizeSlow(cpomdp,10.,0.2)]
options2 = [GoToGoal(cpomdp), LocalizeFast(cpomdp,10.,0.2), LocalizeSlow(cpomdp,10.,0.2), LocalizeSafe(cpomdp,10., 12., 1., 0.2)]

# runs = [PFT-7, COBTS-7, COBTS-4]
runs = [true, true, true]
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
        updater = CMCTSBudgetUpdateWrapper(BootstrapFilter(cpomdp, cpomdp_pf_size, rng), planner)
        e1[i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
    end
    print_and_save(e1,"results/ld_cmcts_$(nsims)sims_noheuristic.jld2")
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
        updater = BootstrapFilter(cpomdp, cpomdp_pf_size, rng)
        e2[i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
    end
    print_and_save(e2,"results/ld_cobts7_$(nsims)sims_noheuristic.jld2")
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
        updater = BootstrapFilter(cpomdp, cpomdp_pf_size, rng)
        e3[i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
    end
    print_and_save(e3,"results/ld_cobts4_$(nsims)sims_noheuristic.jld2")
end