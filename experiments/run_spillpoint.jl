using CPOMCPPlusExperiments
using Infiltrator
using ProgressMeter
using Distributed
using Random
using POMDPs, CPOMDPs

nsims = 10
run = [true, true, true] #(pomcpow, pomcp, pft-dpw)

cpomdp = SoftConstraintPOMDPWrapper(SpillpointInjectionCPOMDP(constraint_budget=0.);λ=[1000.])

# global parameters
max_steps=25
tree_queries = Int(1000) 
pft_tree_queries = Int(100)
k_observation = 10.
alpha_observation = 0.3
max_depth = 10
c = 30.0
nu = 0.0
asched = 10000.
update_filter_size = Int(1e4)
pf_filter_size = 10
init_lam = [1000.]
mc = max_clip(cpomdp.cpomdp)

default_updater = CPOMCPPlusExperiments.SpillpointPOMDP.SIRParticleFilter(
        model=cpomdp.cpomdp,  
        N=200, 
        state2param=CPOMCPPlusExperiments.SpillpointPOMDP.state2params, 
        param2state=CPOMCPPlusExperiments.SpillpointPOMDP.params2state,
        N_samples_before_resample=100,
        clampfn=CPOMCPPlusExperiments.SpillpointPOMDP.clamp_distribution,
        fraction_prior = .5,
        prior=CPOMCPPlusExperiments.SpillpointPOMDP.param_distribution(
            CPOMCPPlusExperiments.initialstate(cpomdp.cpomdp)),
        elite_frac=0.3,
        bandwidth_scale=.5,
        max_cpu_time=20 #60 FIXME 
    )


if run[1] # POMCPOW

    kwargs = Dict(
        :tree_queries=>tree_queries, 
        :k_observation => k_observation, 
        :alpha_observation => alpha_observation, 
        :check_repeat_obs => false,
        :check_repeat_act => false,
        :max_depth => max_depth,
        :criterion=>CPOMCPPlusExperiments.CPOMCPOW.MaxCUCB(c, nu), 
        :alpha_schedule => CPOMCPPlusExperiments.CPOMCPOW.ConstantAlphaSchedule(asched),
        :init_λ=>init_lam,
        :estimate_value=>QMDP_V,
        :max_clip => mc
    )
    exp1 = LightExperimentResults(nsims)
    @showprogress 1 @distributed for i = 1:nsims
        Random.seed!(i)
        solver = CPOMCPPlusExperiments.CPOMCPOWSolver(;kwargs..., rng = MersenneTwister(i))
        updater(planner) = CPOMCPPlusExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(default_updater, planner)
        exp1[i] = run_cpomdp_simulation(cpomdp, solver, updater, max_steps;track_history=false)
    end
    print_and_save(exp1,"results/spillpoint_pomcpow_$(nsims)sims.jld2")
end

if run[2] # POMCP
    kwargs = Dict(
        :tree_queries=>pft_tree_queries, # POMCP trash anyway, keep small
        :k_observation => k_observation, 
        :alpha_observation => alpha_observation, 
        :check_repeat_obs => false,
        :check_repeat_act => false,
        :max_depth => max_depth,
        :c=>c,
        :nu=>nu, 
        :alpha_schedule => CPOMCPPlusExperiments.CPOMCP.ConstantAlphaSchedule(asched),
        :init_λ=>init_lam,
        :estimate_value=>QMDP_V,
        :max_clip => mc
    )
    exp2 = LightExperimentResults(nsims)
    @showprogress 1 @distributed for i = 1:nsims
        Random.seed!(i)
        solver = CPOMCPPlusExperiments.CPOMCPDPWSolver(;kwargs..., rng = MersenneTwister(i))
        updater(planner) = CPOMCPPlusExperiments.CPOMCP.CPOMCPBudgetUpdateWrapper(default_updater, planner)
        exp2[i] = run_cpomdp_simulation(cpomdp, solver, updater, max_steps;track_history=false)
    end
    print_and_save(exp2,"results/spillpoint_pomcpdpw_$(nsims)sims.jld2")
end

if run[3] # PFT
    kwargs = Dict(
        :n_iterations=>pft_tree_queries, 
        :k_state => k_observation, 
        :alpha_state => alpha_observation, 
        :check_repeat_state => false,
        :check_repeat_action => false,
        :depth => max_depth,
        :exploration_constant => c,
        :nu => nu, 
        :alpha_schedule => CPOMCPPlusExperiments.CMCTS.ConstantAlphaSchedule(asched),
        :init_λ=>init_lam,
        :estimate_value=>QMDP_V,
        :max_clip => mc
    )
    exp3 = LightExperimentResults(nsims)
    @showprogress 1 @distributed for i = 1:nsims
        Random.seed!(i)
        rng = MersenneTwister(i)
        up = CPOMCPPlusExperiments.ParticleFilters.BootstrapFilter(cpomdp, pf_filter_size, rng)
        solver = CPOMCPPlusExperiments.CMCTS.BeliefCMCTSSolver(
            CPOMCPPlusExperiments.CMCTS.CDPWSolver(;kwargs..., rng=rng),
            up)
        updater(planner) = CPOMCPPlusExperiments.CMCTS.CMCTSBudgetUpdateWrapper(default_updater, planner)
        exp3[i] = run_cpomdp_simulation(cpomdp, solver, updater, max_steps;track_history=false)
    end
    print_and_save(exp3,"results/spillpoint_pft_$(nsims)sims.jld2")
end