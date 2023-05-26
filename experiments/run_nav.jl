using COBTSExperiments
using COTS 
using CMCTS
using Random
using ProgressMeter

kwargs = Dict(:n_iterations=>Int(1e4), 
        :k_state => 5., # 0.1,
        :alpha_state => 1/15, #0.5,
        :enable_action_pw=>false,
        :depth => 10,
        :exploration_constant=>90.,
        :nu=>0., 
        :estimate_value=>heuristicV,
        :tree_in_info => false,#true,
        :search_progress_info=>false) #true)
as = 0.5
target = 27
cmdp = CNav(;nav_y=target,max_y=target+1)
options1 = [GoToGoal(cmdp), NavigateSlow(cmdp,cmdp.max_y,1.), Navigate(cmdp,cmdp.max_y+5,5.), [Navigate(cmdp, i, 2.) for i in [ -20., 10, 20, 30]]...]
options2 = [GoToGoal(cmdp), NavigateSlow(cmdp,cmdp.max_y,1.), Navigate(cmdp,cmdp.max_y+5,5.)]

# runs = [PFT-7, COTS-7, COTS-3]
runs = [true, true, true]
nsims = 50
if runs[1]
    e1 = LightExperimentResults(nsims)
    @showprogress for i = 1:nsims
        rng = MersenneTwister(i)
        solver = CDPWSolver(;kwargs..., 
            rng = rng,
            alpha_schedule = CMCTS.ConstantAlphaSchedule(as)
            )
        planner = solve(solver, cmdp)
        e1[i] = run_cmdp_simulation(cmdp, planner; rng=rng, track_history=false)
    end
    print_and_save(e1,"results/cnav_cmcts_$(nsims)sims.jld2")
end

if runs[2]
    e2 = LightExperimentResults(nsims)
    @showprogress for i = 1:nsims
        rng = MersenneTwister(i)
        solver = COTSSolver(;kwargs..., 
            rng = rng,
            options = options1,
            alpha_schedule = COTS.ConstantAlphaSchedule(as)
            )
        planner = solve(solver, cmdp)
        e2[i] = run_cmdp_simulation(cmdp, planner; rng=rng, track_history=false)
    end
    print_and_save(e2,"results/cnav_cots7_$(nsims)sims.jld2")
end

if runs[3]
    e3 = LightExperimentResults(nsims)
    @showprogress for i = 1:nsims
        rng = MersenneTwister(i)
        solver = COTSSolver(;kwargs..., 
            rng = rng,
            options = options2,
            alpha_schedule = COTS.ConstantAlphaSchedule(as)
            )
        planner = solve(solver, cmdp)
        e3[i] = run_cmdp_simulation(cmdp, planner; rng=rng, track_history=false)
    end
    print_and_save(e3,"results/cnav_cots3_$(nsims)sims.jld2")
end