using CPOMCPPlusExperiments
using D3Trees
using Plots 
using Random
using SpillpointPOMDP
using POMDPTools

### Find Good Settings
c = 20.0 
nu = 0.0
solver_kwargs = Dict(:tree_queries=>1000, #5000,
        :init_λ => [1000.0],
        :k_observation => 10.,
        :alpha_observation => 0.3,
        :max_depth => 30,
        :alpha_schedule => CPOMCPPlusExperiments.CPOMCPOW.ConstantAlphaSchedule(10000.), # how much to update lambda by        
        :tree_in_info => true,
        :search_progress_info => true,
        :estimate_value => QMDP_V,
        :criterion=>CPOMCPPlusExperiments.CPOMCPOW.MaxCUCB(c, nu),
)

# default pomdp for reference
# SpillpointInjectionPOMDP(;exited_reward_amount=-1000, exited_reward_binary=-1000, obs_rewards=[-0.3, -0.7] , height_noise_std=0.01, sat_noise_std=0.01)


max_steps = 25 #100
λ_test = [0.] #  gets used when running unconstrained problem with scalarized reward

cpomdp = SoftConstraintPOMDPWrapper(SpillpointInjectionCPOMDP(constraint_budget=0.);
    λ=λ_test)

solver = CPOMCPPlusExperiments.CPOMCPOWSolver(;solver_kwargs..., 
    )

updater(planner) = CPOMCPPlusExperiments.CPOMCPOW.CPOMCPOWBudgetUpdateWrapper(
    CPOMCPPlusExperiments.SpillpointPOMDP.SIRParticleFilter(
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
        max_cpu_time=20 #60
    ), 
    planner)

hist, R, C, RC = run_cpomdp_simulation(cpomdp, solver, updater, max_steps);


R
C[1]
RC

inchrome(D3Tree(hist[end][:tree]))

R

hist[1]

ss = [h.s for h in hist]
as = [h.a for h in hist]
bs = [h.b for h in hist]

anim = @animate for i=1:length(ss)
    render(cpomdp.pomdp, ss[i], as[i], belief=bs[i])
    title!("index: $i")
end

gif(anim, "test.gif", fps=1)

sp = SearchProgress(hist[end-1])

# can plot values of best actions, taken actions, and lambdas across the search
plot(sp.v_best)
plot(sp.cv_best)
plot(sp.v_taken)
plot(sp.cv_taken)
plot(sp.lambda)


## constrained