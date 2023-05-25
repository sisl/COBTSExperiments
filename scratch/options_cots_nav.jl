using COBTSExperiments
using COTS 
using CMCTS
using Random

kwargs = Dict(:n_iterations=>Int(1e3), 
        :k_state => 5., # 0.1,
        :alpha_state => 1/15, #0.5,
        :enable_action_pw=>false,
        :depth => 10,
        :exploration_constant=>90.,
        :nu=>0., 
        :estimate_value=>heuristicV,
        :tree_in_info => true,
        :search_progress_info=>true)
as = 0.5
cmdp = CNav()
options1 = [GoToGoal(cmdp), [Navigate(cmdp,i) for i in [-20., -10, -5, 5, 10, 20]]...]
options2 = [GoToGoal(cmdp), Navigate(cmdp,10.), Navigate(cmdp,20.)]

# runs = [PFT-7, COTS-7, COTS-3]
runs = [true, true, true]

if runs[1]
    for i = 1:2 # time second run through
        rng = MersenneTwister(i)
        solver = CDPWSolver(;kwargs..., 
            rng = rng,
            alpha_schedule = CMCTS.ConstantAlphaSchedule(as)
            )
        planner = solve(solver, cmdp)
        @time hist1, R1, C1 = run_cmdp_simulation(cmdp, planner, 1;rng=rng)
    end
end

if runs[2]
    for i = 1:2
        rng = MersenneTwister(i)
        solver = COTSSolver(;kwargs..., 
            rng = rng,
            options = options1,
            alpha_schedule = COTS.ConstantAlphaSchedule(as)
            )
        planner = solve(solver, cmdp)
        @time hist2, R2, C2 = run_cmdp_simulation(cmdp, planner, 1; rng=rng)
    end
end

if runs[3]
    for i = 1:2
        rng = MersenneTwister(i)
        solver = COTSSolver(;kwargs..., 
            rng = rng,
            options = options2,
            alpha_schedule = COTS.ConstantAlphaSchedule(as)
            )
        planner = solve(solver, cmdp)
        @time hist3, R3, C3 = run_cmdp_simulation(cmdp, planner, 1; rng=rng)
    end
end