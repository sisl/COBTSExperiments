using CPOMCPPlusExperiments
using D3Trees
using Plots 
using Random
using LinearAlgebra
import Statistics
using ProgressMeter
using Distributed

### Solver Settings
kwargs = Dict(:tree_queries=>1e5, 
        :k_observation => 5., # 0.1,
        :alpha_observation => 1/15, #0.5,
        :enable_action_pw=>false,
        :max_depth => 10,
        :alpha_schedule => CPOMCPPlusExperiments.CPOMCPOW.ConstantAlphaSchedule(0.5),
        :criterion=>CPOMCPPlusExperiments.CPOMCPOW.MaxCUCB(90.,0.), 
        :estimate_value=>zeroV_trueC,
        :tree_in_info => true,
        :search_progress_info=>true)
λ_test = [1.]
runs = [true, true, true]

nsims = 100

# unconstrained
if runs[1]
    Ns = []
    Q_cs = []
    Q_lams = []
    hist = nothing
    @showprogress 1 @distributed for i in 1:nsims
    
        cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew(cost_budget=20.);λ=λ_test)
        solver = CPOMCPPlusExperiments.CPOMCPOWSolver(;kwargs..., 
            rng = MersenneTwister(i)
            )
        updater(planner) = CPOMCPPlusExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
            CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.rng), 
            planner)
        hist, _, _, _ = run_cpomdp_simulation(cpomdp, solver, updater, 1)
        lam = hist[1][:lambda][end]
        n = hist[1][:tree].n[1:7]
        q = hist[1][:tree].v[1:7]
        qc = hist[1][:tree].cv[1:7]
        ql = q .- [lam⋅i for i in qc]
        dql = ql .- maximum(ql)
        np = n ./ sum(n)
        push!(Ns, np)
        push!(Q_cs, vcat(qc...))
        push!(Q_lams, dql)
    end
    println("Unconstrained")
    println("Ns: $(Statistics.mean(Ns))")
    println("DQls: $(Statistics.mean(Q_lams))")
    println("QCs: $(Statistics.mean(Q_cs))")
    println("Ns_std: $(Statistics.std(Ns)/sqrt(nsims))")
    println("DQls_std: $(Statistics.std(Q_lams)/sqrt(nsims))")
    println("QCs_std: $(Statistics.std(Q_cs)/sqrt(nsims))")
end

if runs[2]
    Ns = []
    Q_cs = []
    Q_lams = []
    hist = nothing
    @showprogress 1 @distributed for i in 1:nsims
    
        cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew(cost_budget=0.1);λ=λ_test)
        solver = CPOMCPPlusExperiments.CPOMCPOWSolver(;kwargs..., 
            rng = MersenneTwister(i),
            max_clip = max_clip(cpomdp.cpomdp)
            )
        updater(planner) = CPOMCPPlusExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
            CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.rng), 
            planner)
        hist, _, _, _ = run_cpomdp_simulation(cpomdp, solver, updater, 1)
        lam = hist[1][:lambda][end]
        n = hist[1][:tree].n[1:7]
        q = hist[1][:tree].v[1:7]
        qc = hist[1][:tree].cv[1:7]
        ql = q .- [lam⋅i for i in qc]
        dql = ql .- maximum(ql)
        np = n ./ sum(n)
        push!(Ns, np)
        push!(Q_cs, vcat(qc...))
        push!(Q_lams, dql)
    end
    println("Constrained, normal propagation")
    println("Ns: $(Statistics.mean(Ns))")
    println("DQls: $(Statistics.mean(Q_lams))")
    println("QCs: $(Statistics.mean(Q_cs))")
    println("Ns_std: $(Statistics.std(Ns)/sqrt(nsims))")
    println("DQls_std: $(Statistics.std(Q_lams)/sqrt(nsims))")
    println("QCs_std: $(Statistics.std(Q_cs)/sqrt(nsims))")
end 

if runs[3]
    Ns = []
    Q_cs = []
    Q_lams = []
    hist = nothing
    @showprogress 1 @distributed for i in 1:nsims
    
        cpomdp = SoftConstraintPOMDPWrapper(CLightDarkNew(cost_budget=0.1);λ=λ_test)
        solver = CPOMCPPlusExperiments.CPOMCPOWSolver(;kwargs...,
            return_best_cost=true, 
            rng = MersenneTwister(i),
            max_clip = max_clip(cpomdp.cpomdp)
            )
        updater(planner) = CPOMCPPlusExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
            CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, Int(1e4), solver.rng), 
            planner)
        hist, _, _, _ = run_cpomdp_simulation(cpomdp, solver, updater, 1)
        lam = hist[1][:lambda][end]
        n = hist[1][:tree].n[1:7]
        q = hist[1][:tree].v[1:7]
        qc = hist[1][:tree].cv[1:7]
        ql = q .- [lam⋅i for i in qc]
        dql = ql .- maximum(ql)
        np = n ./ sum(n)
        push!(Ns, np)
        push!(Q_cs, vcat(qc...))
        push!(Q_lams, dql)
    end
    println("Constrained, minimum cost propagation")
    println("Ns: $(Statistics.mean(Ns))")
    println("DQls: $(Statistics.mean(Q_lams))")
    println("QCs: $(Statistics.mean(Q_cs))")
    println("Ns_std: $(Statistics.std(Ns)/sqrt(nsims))")
    println("DQls_std: $(Statistics.std(Q_lams)/sqrt(nsims))")
    println("QCs_std: $(Statistics.std(Q_cs)/sqrt(nsims))")
end