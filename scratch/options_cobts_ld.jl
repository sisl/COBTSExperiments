using COBTSExperiments
using COTS 
using CMCTS
using Random
using D3Trees
using ParticleFilters
using Infiltrator

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
target = 27
search_pf_size = Int(20)
cpomdp_pf_size = Int(1e4)
options1 = [GoToGoal(cpomdp), LocalizeFast(cpomdp,10.,1.), LocalizeFast(cpomdp,10.,0.5), LocalizeFast(cpomdp,10.,0.2), 
    LocalizeSlow(cpomdp,10.,1.), LocalizeSlow(cpomdp,10.,0.5), LocalizeSlow(cpomdp,10.,0.2)]
options2 = [GoToGoal(cpomdp), LocalizeFast(cpomdp,10.,1.), LocalizeFast(cpomdp,10.,0.5), LocalizeFast(cpomdp,10.,0.2), 
    LocalizeSlow(cpomdp,10.,1.), LocalizeSlow(cpomdp,10.,0.5), LocalizeSlow(cpomdp,10.,0.2),
    LocalizeSafe(cpomdp,10., 12., 1., 0.5), LocalizeSafe(cpomdp,10.,12., 0.5, 0.5), LocalizeSafe(cpomdp,10.,12., 1., 0.2)]
options3 = [GoToGoal(cpomdp), LocalizeFast(cpomdp,10.,0.5), LocalizeSlow(cpomdp,10.,0.5), LocalizeSafe(cpomdp,10., 12., 1., 0.5)]

# runs = [PFT-7, ...COBTS with differing options, ]
runs = [false, false, false, true, true]

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


if runs[4]
    for i = 1:2
        rng = MersenneTwister(i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = COBTSSolver(
            COTSSolver(;kwargs..., 
                rng = rng,
                options = options3,
                alpha_schedule = COTS.ConstantAlphaSchedule(as)
            ), search_updater)
        planner = solve(solver, cpomdp)
        updater = BootstrapFilter(cpomdp, cpomdp_pf_size, rng)
        @time hist4, R4, C4 = run_cpomdp_simulation(cpomdp, planner, updater, 1; rng=rng)
        i==2 && inchrome(D3Tree(hist4[1][:tree];lambda=hist4[1][:lambda][end]))
    end
end

if runs[5]
    for i = 1:2
        rng = MersenneTwister(i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = COBTSSolver(
            COTSSolver(;kwargs..., 
                rng = rng,
                options = options3,
                alpha_schedule = COTS.ConstantAlphaSchedule(as)
            ), 
            search_updater; 
            exact_rewards=true)
        planner = solve(solver, cpomdp)
        updater = BootstrapFilter(cpomdp, cpomdp_pf_size, rng)
        @time hist4, R4, C4 = run_cpomdp_simulation(cpomdp, planner, updater, 1; rng=rng)
        i==2 && inchrome(D3Tree(hist4[1][:tree];lambda=hist4[1][:lambda][end]))
    end
end

# plot([l[1] for l=hist3[1][:lambda]])