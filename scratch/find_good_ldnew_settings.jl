using CPOMCPPlusExperiments
using D3Trees
using Plots 

### Find Good Settings
kwargs = Dict(:tree_queries=>2e5, 
        :k_observation => 5., # 0.1,
        :alpha_observation => 1/15, #0.5,
        :enable_action_pw=>false,
        :max_depth => 10,
        :alpha_schedule => CPOMCPPlusExperiments.CPOMCPOW.ConstantAlphaSchedule(0.5))
c = 90.0 # 250
nu = 0.0
λ_test = [1.]
runs = [false, true, false]

if runs[1]
    cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew(cost_budget=20.);λ=λ_test)
    solver = CPOMCPPlusExperiments.CPOMCPOWSolver(;kwargs..., 
        criterion=CPOMCPPlusExperiments.CPOMCPOW.MaxCUCB(c, nu), 
        estimate_value=zeroV_trueC,
        tree_in_info=true,
        search_progress_info=true)
    updater(planner) = CPOMCPPlusExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
        CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.rng), 
        planner)
    hist3, R3, C3, RC3 = run_cpomdp_simulation(cpomdp, solver, updater)

    R3
    C3[1]
    RC3
    plot_lightdark_beliefs(hist3,"figs/belief_ldn_unconstrained.png")

    inchrome(D3Tree(hist3[1][:tree]))


    sp3 = SearchProgress(hist3[1])
end
#plot(sp3.v_best)
#plot(sp3.cv_best)
#plot(sp3.v_taken)
#plot(sp3.cv_taken)
#plot(sp3.lambda)

## constrained

if runs[2]
    cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew(cost_budget=0.1);λ=λ_test)
    solver = CPOMCPPlusExperiments.CPOMCPOWSolver(;kwargs..., 
        criterion=CPOMCPPlusExperiments.CPOMCPOW.MaxCUCB(c, nu), 
        estimate_value=zeroV_trueC,
        tree_in_info=true,
        search_progress_info=true)
    updater(planner) = CPOMCPPlusExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
        CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.rng), 
        planner)
    hist4, R4, C4, RC4 = run_cpomdp_simulation(cpomdp, solver, updater)

    R4
    C4[1]
    RC4
    plot_lightdark_beliefs(hist4,"figs/belief_ldn_constrained.png")

    inchrome(D3Tree(hist4[1][:tree]))


    sp4 = SearchProgress(hist4[1])

    #plot(sp4.v_best)
    #plot(sp4.cv_best)
    #plot(sp4.v_taken)
    #plot(sp4.cv_taken)
    #plot(sp4.lambda)

    layer1 = hist4[1][:tree].tried[1]
    hist4[1][:tree].v[layer1]
    hist4[1][:tree].cv[layer1]

    # OBSERVATION: cost for 5 is still really high (0.43), despite the fact that there exists a policy that doesnt violate constraint 
    # reason - because taken action costs get propagated up tree, not min costs (even though there exists a safe policy)
    # solution - propagate lambda-weighted min costs
end 

if runs[3]
    cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew(cost_budget=0.1);λ=λ_test)
    solver = CPOMCPPlusExperiments.CPOMCPOWSolver(;kwargs..., 
        criterion=CPOMCPPlusExperiments.CPOMCPOW.MaxCUCB(c, nu), 
        estimate_value=zeroV_trueC,
        tree_in_info=true,
        search_progress_info=true,
        return_best_cost=true)
    updater(planner) = CPOMCPPlusExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
        CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.rng), 
        planner)
    hist5, R5, C5, RC5 = run_cpomdp_simulation(cpomdp, solver, updater)

    R5
    C5[1]
    RC5
    plot_lightdark_beliefs(hist5,"figs/belief_ldn_constrained_2.png")

    inchrome(D3Tree(hist5[1][:tree]))


    sp5 = SearchProgress(hist5[1])

    #plot(sp4.v_best)
    #plot(sp4.cv_best)
    #plot(sp4.v_taken)
    #plot(sp4.cv_taken)
    #plot(sp4.lambda)

    layer1 = hist5[1][:tree].tried[1]
    hist5[1][:tree].v[layer1]
    hist5[1][:tree].cv[layer1]

    # OBSERVATION: cost for 5 is still really high (0.43), despite the fact that there exists a policy that doesnt violate constraint 
    # reason - because taken action costs get propagated up tree, not min costs (even though there exists a safe policy)
    # solution - propagate lambda-weighted min costs

end