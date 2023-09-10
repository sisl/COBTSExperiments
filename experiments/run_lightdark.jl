using COBTSExperiments
using COTS 
using CMCTS
using CPOMCPOW
using Random
using ProgressMeter
using ParticleFilters

# experiments to run
experiments = Dict(
    "cpft-infogain"=>true,
    "cpft-noheur"=>true,
    "cobts4"=>true,
    "cobts7"=>true,
    "cpomcpow"=>true,
)
nsims = 100

# same kwargs for pft and cobts algorithms
kwargs = Dict(:n_iterations=>Int(1e4), 
        :k_state => 1., # 0.1, # ld experiments use 5
        :alpha_state => 1/5, #0.5, # ld experiments use 1/15
        :enable_action_pw=>false,
        :depth => 10,
        :exploration_constant=>200., # ld experiments use 90
        :nu=>0., 
        :estimate_value => zero_V,
        :tree_in_info => false,
        :search_progress_info => false)

as = 0.5 # 1.
search_pf_size = Int(10)
cpomdp_pf_size = Int(1e4)

# CPOMCPOW kwargs 
cpomcpow_kwargs = Dict(
    :tree_queries=>kwargs[:n_iterations]*search_pf_size, # fair to total number of particles
    :k_observation => kwargs[:k_state], 
    :alpha_observation => kwargs[:alpha_state], 
    :enable_action_pw => kwargs[:enable_action_pw],
    :check_repeat_obs => false,
    :max_depth => kwargs[:depth],
    :criterion => MaxCUCB(kwargs[:exploration_constant], kwargs[:nu]), 
    :alpha_schedule => CPOMCPOW.ConstantAlphaSchedule(as),
    :estimate_value=>kwargs[:estimate_value],
)

# options
cpomdp = LightDarkCPOMDP(cost_budget=0.1)
options = [GoToGoal(cpomdp), 
LocalizeFast(cpomdp,10.,0.2), LocalizeSlow(cpomdp,10.,0.2), LocalizeSafe(cpomdp, 10., 12., 1., 0.2),
LocalizeFast(cpomdp,10.,0.5), LocalizeSlow(cpomdp,10.,0.5), LocalizeSafe(cpomdp, 10., 12., 1., 0.5), 
LocalizeFast(cpomdp,10.,1.), LocalizeSlow(cpomdp,10.,1.), LocalizeSafe(cpomdp, 10., 12., 1., 1.)]

rng_stseed = 0

### Experiments
results = Dict(k=>LightExperimentResults(nsims) for (k,v) in experiments if v)

# CPFT-Infogain
exp = "cpft-infogain"
if experiments[exp]
    println(exp)
    Threads.@threads for i = 1:nsims
        infogain_kwargs = copy(kwargs)
        infogain_kwargs[:estimate_value] = heuristicV
        rng = MersenneTwister(rng_stseed+i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = BeliefCMCTSSolver(
            CDPWSolver(;infogain_kwargs..., 
                rng = rng,
                alpha_schedule = CMCTS.ConstantAlphaSchedule(as),
                return_safe_action=false,
            ), search_updater;
            exact_rewards=false)
        planner = solve(solver, cpomdp)
        updater = CMCTSBudgetUpdateWrapper(BootstrapFilter(cpomdp, cpomdp_pf_size, rng), planner)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
    end
    print_and_save(results[exp], "results/lightdark_$(exp)_$(nsims)sims.jld2") 
end

# CPFT-No Heuristic
exp = "cpft-noheur"
if experiments[exp]
    println(exp)
    Threads.@threads for i = 1:nsims
        rng = MersenneTwister(rng_stseed+i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = BeliefCMCTSSolver(
            CDPWSolver(;kwargs...,
                rng = rng,
                alpha_schedule = CMCTS.ConstantAlphaSchedule(as),
                return_safe_action=false,
            ), search_updater;
            exact_rewards=false)
        planner = solve(solver, cpomdp)
        updater = CMCTSBudgetUpdateWrapper(BootstrapFilter(cpomdp, cpomdp_pf_size, rng), planner)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
    end
    print_and_save(results[exp], "results/lightdark_$(exp)_$(nsims)sims.jld2") 
end


# COBTS-4
exp = "cobts4"
if experiments[exp]
    println(exp)
    Threads.@threads for i = 1:nsims
        rng = MersenneTwister(rng_stseed+i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = COBTSSolver(
            COTSSolver(;kwargs..., 
                rng = rng,
                options = options[1:4],
                alpha_schedule = COTS.ConstantAlphaSchedule(as),
                return_safe_action = true,
            ), search_updater;
            exact_rewards=true)
        planner = solve(solver, cpomdp)
        updater = BootstrapFilter(cpomdp, cpomdp_pf_size, rng)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
    end
    print_and_save(results[exp], "results/lightdark_$(exp)_$(nsims)sims.jld2") 
end

# COBTS-7
exp = "cobts7"
if experiments[exp]
    println(exp)
    Threads.@threads for i = 1:nsims
        rng = MersenneTwister(rng_stseed+i)
        search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
        solver = COBTSSolver(
            COTSSolver(;kwargs..., 
                rng = rng,
                options = options[1:7],
                alpha_schedule = COTS.ConstantAlphaSchedule(as),
                return_safe_action=true,
            ), search_updater;
            exact_rewards=true)
        planner = solve(solver, cpomdp)
        updater = BootstrapFilter(cpomdp, cpomdp_pf_size, rng)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
    end
    print_and_save(results[exp], "results/lightdark_$(exp)_$(nsims)sims.jld2") 
end

# CPOMCPOW
exp = "cpomcpow"
if experiments[exp]
    println(exp)
    Threads.@threads for i = 1:nsims
        rng = MersenneTwister(rng_stseed+i)
        solver = CPOMCPOWSolver(;cpomcpow_kwargs..., rng = rng)
        planner = solve(solver, cpomdp)
        updater = CPOMCPOWBudgetUpdateWrapper(
            BootstrapFilter(cpomdp, cpomdp_pf_size, rng),
            planner)
        results[exp][i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
    end
    print_and_save(results[exp], "results/lightdark_$(exp)_$(nsims)sims.jld2") 
end
