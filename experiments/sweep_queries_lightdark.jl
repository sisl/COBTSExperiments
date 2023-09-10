using COBTSExperiments
using COTS 
using CMCTS
using CPOMCPOW
using Random
using ProgressMeter
using ParticleFilters

# experiments to run (num_options, num_queries)

num_options = [2, 3, 4, 5] 
num_queries = [10,20, 50, 100,200, 500, 1000,2000,5000,10000,20000, 50000,100000]
nsims = 20

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
        :search_progress_info => false,
        :return_safe_action => true)

as = 0.5
search_pf_size = Int(10)
cpomdp_pf_size = Int(1e4)

cpomdp = LightDarkCPOMDP(cost_budget=0.1)
options = [GoToGoal(cpomdp), 
    LocalizeSafe(cpomdp, 10., 12., 1., 0.2),
    LocalizeSafe(cpomdp, 10., 12., 1., 0.5),
    LocalizeSafe(cpomdp, 10., 12., 0.5, 0.2),
    LocalizeSafe(cpomdp, 10., 12., 0.5, 0.5)]

# options

### Experiments
for nq in num_queries
    kwargs[:n_iterations] = nq
    @showprogress for k = num_options
        results = LightExperimentResults(nsims)
        @showprogress for i = 1:nsims
            rng = MersenneTwister(i)
            search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
            solver = COBTSSolver(
                COTSSolver(;kwargs..., 
                    rng = rng,
                    options = options[1:k],
                    alpha_schedule = COTS.ConstantAlphaSchedule(as)
                ), search_updater;
                exact_rewards=true)
            planner = solve(solver, cpomdp)
            updater = BootstrapFilter(cpomdp, cpomdp_pf_size, rng)
            results[i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
        end
        print_and_save(results, "results/sweep_queries/lightdark_cobts_$(nq)queries_$(k)options_$(nsims)sims.jld2") 
    end
end
