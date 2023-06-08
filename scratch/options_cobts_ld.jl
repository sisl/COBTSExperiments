using COBTSExperiments
using COTS 
using CMCTS
using Random
using D3Trees


kwargs = Dict(:n_iterations=>Int(1e4), 
        :k_state => 5., # 0.1,
        :alpha_state => 1/15, #0.5,
        :enable_action_pw=>false,
        :depth => 10,
        :exploration_constant=>90.,
        :nu=>0., 
        :estimate_value=>heuristicV,
        :tree_in_info => true,
        :search_progress_info=>true)

cpomdp = LightDarkCPOMDP(cost_budget=0.1)

as = 0.5
target = 27
search_pf_size = Int(100)
cpomdp_pf_size = Int(1e4)
options1 = [GoToGoal(cmdp), NavigateSlow(cmdp,cmdp.max_y,1.), Navigate(cmdp,cmdp.max_y+5,5.), [Navigate(cmdp, i, 2.) for i in [ -20., 10, 20, 30]]...]
options2 = [GoToGoal(cmdp), NavigateSlow(cmdp,cmdp.max_y,1.), Navigate(cmdp,cmdp.max_y+5,5.)]

# runs = [PFT-7, COBTS-7, COBTS-3]
runs = [true, true, true]

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
        @time hist2, R2, C2 = run_cpomdp_simulation(cmdp, planner, updater, 1; rng=rng)
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
        @time hist3, R3, C3 = run_cpomdp_simulation(cmdp, planner, updater, 1; rng=rng)
        i==2 && inchrome(D3Tree(hist3[1][:tree];lambda=hist3[1][:lambda][end]))
    end
end