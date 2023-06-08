using COBTSExperiments
using COTS 
using CMCTS
using Random
using ProgressMeter
using Plots

searches = Int[1e1, 1e2, 1e3, 1e4, 1e5]
#greedy_action = false
kwargs = Dict(#:n_iterations=>Int(1e4), 
        :k_state => 5., # 0.1,
        :alpha_state => 1/15, #0.5,
        :enable_action_pw=>false,
        :depth => 10,
        :exploration_constant=>90.,
        :nu=>0., 
#        :estimate_value=>heuristicV,
        :tree_in_info => false,#true,
        :search_progress_info=>false) #true)
as = 1.
target = 27
cmdp = CNav(;nav_y=target,max_y=target+1)
options1 = [GoToGoal(cmdp), NavigateSlow(cmdp,cmdp.max_y,1.), Navigate(cmdp,cmdp.max_y+5,5.), [Navigate(cmdp, i, 2.) for i in [ -20., 10, 20, 30]]...]
options2 = [GoToGoal(cmdp), NavigateSlow(cmdp,cmdp.max_y,1.), Navigate(cmdp,cmdp.max_y+5,5.)]

labels = ["PFT-7", "COTS-7", "COTS-3"]
runs = [true, true, true]
nsims = 20
er = [[LightExperimentResults(nsims) for _ in 1:length(searches)] for _ in  1:length(runs)]

@showprogress for (iS,s) in enumerate(searches)
    if runs[1]
        @showprogress for i = 1:nsims
            rng = MersenneTwister(i)
            solver = CDPWSolver(;kwargs..., 
                n_iterations = s,
                rng = rng,
                alpha_schedule = CMCTS.ConstantAlphaSchedule(as)
                )
            planner = solve(solver, cmdp)
            er[1][iS][i] = run_cmdp_simulation(cmdp, planner; rng=rng, track_history=false)
        end
    end

    if runs[2]
        @showprogress for i = 1:nsims
            rng = MersenneTwister(i)
            solver = COTSSolver(;kwargs...,  
                n_iterations = s,
                rng = rng,
                options = options1,
                alpha_schedule = COTS.ConstantAlphaSchedule(as)
                )
            planner = solve(solver, cmdp)
            er[2][iS][i] = run_cmdp_simulation(cmdp, planner; rng=rng, track_history=false)
        end
    end

    if runs[3]
        @showprogress for i = 1:nsims
            rng = MersenneTwister(i)
            solver = COTSSolver(;kwargs...,  
                n_iterations = s,
                rng = rng,
                options = options2,
                alpha_schedule = COTS.ConstantAlphaSchedule(as)
                )
            planner = solve(solver, cmdp)
            er[3][iS][i] = run_cmdp_simulation(cmdp, planner; rng=rng, track_history=false)
        end
    end
end

## plot results
ms = [[mean(er[iR][iS]) for iS = 1:length(searches)] for iR =1:length(runs)]

plt = plot(xaxis=:log)
for ir = 1:3
    plot!(searches, first.(ms[ir]), label = labels[ir]*" value")
end
savefig(plt,"results/cnav_values.png")

plt = plot(xaxis=:log)
for ir = 1:3
    plot!(searches, first.(last.(ms[ir])), label = labels[ir]*" cost")
end
savefig(plt,"results/cnav_costs.png")
#print_and_save(e3,"results/cnav_cots3_$(nsims)sims.jld2")
