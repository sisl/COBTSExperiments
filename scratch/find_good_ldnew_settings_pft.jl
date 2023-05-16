using CPOMCPPlusExperiments
using D3Trees
using Plots 

### Find Good Settings
kwargs = Dict(:n_iterations=>Int(5e4), 
        :k_state => 5., 
        :alpha_state => 1/15, 
        :enable_action_pw=>false,
        :depth => 10,
        :alpha_schedule => CPOMCPPlusExperiments.CMCTS.ConstantAlphaSchedule(0.5),
        :exploration_constant => 100., #90.
        ) 

nu = 0.0
λ_test = [1.]
runs = [true, true, true]
npart = Int(10)
if runs[1]
    cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew(cost_budget=20.);λ=λ_test)

    up = CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, npart)
    
    solver = CPOMCPPlusExperiments.CMCTS.BeliefCMCTSSolver(
        CPOMCPPlusExperiments.CMCTS.CDPWSolver(;kwargs..., 
            nu=nu, 
            estimate_value=CPOMCPPlusExperiments.heuristicV,
            tree_in_info=true,
            search_progress_info=true), 
        up)
    
    updater(planner) = CPOMCPPlusExperiments.CMCTS.CMCTSBudgetUpdateWrapper(
        CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.solver.rng), 
        planner)
    
    hist3, R3, C3, RC3 = run_cpomdp_simulation(cpomdp, solver, updater)

    R3
    C3[1]
    RC3
    plot_lightdark_beliefs(hist3,"figs/belief_ldn_unconstrained_pft.png")

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
    
    up = CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, npart)
    
    solver = CPOMCPPlusExperiments.CMCTS.BeliefCMCTSSolver(
        CPOMCPPlusExperiments.CMCTS.CDPWSolver(;kwargs..., 
            nu=nu, 
            estimate_value=CPOMCPPlusExperiments.heuristicV,
            tree_in_info=true,
            search_progress_info=true), 
        up)
    
    updater(planner) = CPOMCPPlusExperiments.CMCTS.CMCTSBudgetUpdateWrapper(
        CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.solver.rng), 
        planner)

    hist4, R4, C4, RC4 = run_cpomdp_simulation(cpomdp, solver, updater)

    R4
    C4[1]
    RC4
    plot_lightdark_beliefs(hist4,"figs/belief_ldn_constrained_pft.png")

    inchrome(D3Tree(hist4[1][:tree]))


    sp4 = SearchProgress(hist4[1])

    #plot(sp4.v_best)
    #plot(sp4.cv_best)
    #plot(sp4.v_taken)
    #plot(sp4.cv_taken)
    #plot(sp4.lambda)

    layer1 = hist4[1][:tree].children[1]
    hist4[1][:tree].q[layer1]
    hist4[1][:tree].qc[layer1]

    # OBSERVATION: cost for 5 is still really high (0.43), despite the fact that there exists a policy that doesnt violate constraint 
    # reason - because taken action costs get propagated up tree, not min costs (even though there exists a safe policy)
    # solution - propagate lambda-weighted min costs
end 

if runs[3]
    cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew(cost_budget=0.1);λ=λ_test)
    
    up = CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, npart)
    
    solver = CPOMCPPlusExperiments.CMCTS.BeliefCMCTSSolver(
        CPOMCPPlusExperiments.CMCTS.CDPWSolver(;kwargs..., 
            nu=nu, 
            estimate_value=CPOMCPPlusExperiments.heuristicV,
            tree_in_info=true,
            search_progress_info=true,
            return_best_cost=true), 
        up)
    
    updater(planner) = CPOMCPPlusExperiments.CMCTS.CMCTSBudgetUpdateWrapper(
        CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.solver.rng), 
        planner)

    hist5, R5, C5, RC5 = run_cpomdp_simulation(cpomdp, solver, updater)

    R5
    C5[1]
    RC5
    plot_lightdark_beliefs(hist5,"figs/belief_ldn_constrained_2_pft.png")

    inchrome(D3Tree(hist5[1][:tree]))


    sp5 = SearchProgress(hist5[1])

    #plot(sp4.v_best)
    #plot(sp4.cv_best)
    #plot(sp4.v_taken)
    #plot(sp4.cv_taken)
    #plot(sp4.lambda)

    layer1 = hist5[1][:tree].children[1]
    hist5[1][:tree].q[layer1]
    hist5[1][:tree].qc[layer1]

    # OBSERVATION: cost for 5 is still really high (0.43), despite the fact that there exists a policy that doesnt violate constraint 
    # reason - because taken action costs get propagated up tree, not min costs (even though there exists a safe policy)
    # solution - propagate lambda-weighted min costs

end