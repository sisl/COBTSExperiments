using COBTSExperiments
using COTS 
using CMCTS
using CPOMCPOW
using Random
using ProgressMeter
using ParticleFilters

# experiments to run
experiments=Dict(
    "repeat_options"=>true,
    "goal_options"=>true,
    "target_uncertainty"=>true,
    "random_3actions"=>true,
    "random_6actions"=>true,
)
nsims = 50
max_options = 40

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
#base_options = [GoToGoal(cpomdp), LocalizeFast(cpomdp,10.,0.5), 
#    LocalizeSlow(cpomdp,10.,0.5),  LocalizeSafe(cpomdp, 10., 12., 1., 0.5)]

base_options = [GoToGoal(cpomdp), 
    LocalizeFast(cpomdp,10.,0.2), LocalizeSlow(cpomdp,10.,0.2), LocalizeSafe(cpomdp, 10., 12., 1., 0.2),
    LocalizeFast(cpomdp,10.,0.5), LocalizeSlow(cpomdp,10.,0.5), LocalizeSafe(cpomdp, 10., 12., 1., 0.5)]

# options

rng_stseed = 0
experiment_runs = Dict(
    "repeat_options"=>[4;collect(length(base_options):4:max_options)],
    "goal_options"=>[4;collect(length(base_options):1:max_options)],
    "target_uncertainty"=>[4;collect(length(base_options):3:max_options)],
    "random_3actions"=>[4;collect(length(base_options):4:max_options)],
    "random_6actions"=>[4;collect(length(base_options):4:max_options)],
)
actgenrng = MersenneTwister(1000)
options = Dict(
    "repeat_options"=>repeat(base_options, ceil(Int, max_options/length(base_options))),
    "goal_options"=>[base_options; repeat([GoToGoal(cpomdp)], max_options-length(base_options))],
    "target_uncertainty"=>[base_options; vcat([[LocalizeFast(cpomdp,10.,u), 
        LocalizeSlow(cpomdp,10.,u),  LocalizeSafe(cpomdp,10.,12.,1.,u)] for u in 0.2:.1:1.5]...)],
    "random_3actions"=>[base_options; [MultiActionWrapper(cpomdp,3;rng=actgenrng) for i=1:(max_options-length(base_options))]],
    "random_6actions"=>[base_options; [MultiActionWrapper(cpomdp,6;rng=actgenrng) for i=1:(max_options-length(base_options))]]

)

### Experiments
results = Dict{Tuple{String,Int},LightExperimentResults}()
for (expname, todo) in experiments
    !todo && continue
    @showprogress for k = experiment_runs[expname]
        results[(expname,k)] = LightExperimentResults(nsims)
        @showprogress for i = 1:nsims
            rng = MersenneTwister(rng_stseed+i)
            search_updater = BootstrapFilter(cpomdp, search_pf_size, rng)
            solver = COBTSSolver(
                COTSSolver(;kwargs..., 
                    rng = rng,
                    options = options[expname][1:k],
                    alpha_schedule = COTS.ConstantAlphaSchedule(as)
                ), search_updater;
                exact_rewards=true)
            planner = solve(solver, cpomdp)
            updater = BootstrapFilter(cpomdp, cpomdp_pf_size, rng)
            results[(expname,k)][i] = run_cpomdp_simulation(cpomdp, planner, updater; rng=rng, track_history=false)
        end
        println("Results for COBTS with $(expname) options and k=$(k)")
        print_and_save(results[(expname,k)], "results/sweep_options/lightdark_COBTS_$(expname)_$(k)options_$(nsims)sims.jld2") 
    end
end

# Print and save
# for (label,result) in results
#     expname, k = label
#     println("Results for COBTS with $(expname) options and k=$(k)")
#     print_and_save(result, "results/lightdark_COBTS_$(expname)_$(k)options_$(nsims)sims.jld2") 
# end