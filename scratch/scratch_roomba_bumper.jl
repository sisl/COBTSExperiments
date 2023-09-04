using COBTSExperiments
using RoombaPOMDPs
using POMDPs
using ParticleFilters
using Cairo
using Gtk
using Random
using Printf
using POMDPSimulators
using CPOMCPOW
using CMCTS
using COTS
using Infiltrator
using D3Trees
using Plots
using Statistics

problem = "discrete"
sensor = Bumper() # Bumper() or Lidar()
config = 1 # 1,2, or 3 for different room configurations
vs = [0, 1, 3]
oms = [-π/2, 0, π/2] # with a dt of 0.5 seconds, this is 45 degrees per step
RoombaActSpace = [RoombaAct(v, om) for v in vs for om in oms]
pomdp = RoombaPOMDP(sensor=sensor,
                    mdp=RoombaMDP(v_max=maximum(vs), om_max=maximum(oms),config=config, aspace=RoombaActSpace))
cpomdp = RoombaCPOMDP(pomdp, cost_budget=1000.)

num_particles = 10000
v_noise_coefficient = 1.0
om_noise_coefficient = 0.5

belief_updater = RoombaParticleFilter(cpomdp.pomdp, num_particles, v_noise_coefficient, om_noise_coefficient);
# belief_updater = BootstrapFilter(cpomdp, num_particles)
max_steps = 50
check_ts = [1] # time steps to check tree and lambda history
run_policy = 3 # [1 = CPOMCPOW, 2 = CPFT-DPW, 3=COBETS]

### CPOMCPOW policy p
if run_policy == 1
    # CPOMCPOW kwargs 
    cpomcpow_kwargs = Dict(
        :tree_queries=>Int(1e4),
        # observation widening: Bumper = None, Lidar = (1., 1/5)
        :k_observation => 1., 
        :alpha_observation => 1/5., 
        :check_repeat_obs => true,

        # action widening: If |A| is small, None
        :enable_action_pw => true,
        :k_action => 1.,
        :alpha_action => 1/5., 
        :check_repeat_act => true,

        :max_depth => 80,
        # Q + k sqrt(N(sa)/log(N))
        :criterion => MaxCUCB(1., 0.), 
        :alpha_schedule => CPOMCPOW.ConstantAlphaSchedule(1.), # lambda = max(0, lambda + alphaschedule(iteration)*( Qsim - budget ))
        :estimate_value=>zero_V,
    )
    solver = CPOMCPOWSolver(;cpomcpow_kwargs...)
    p = solve(solver, cpomdp)
    belief_updater = CPOMCPOWBudgetUpdateWrapper(belief_updater, p)
elseif run_policy == 2
    # CPFT-DPW kwargs
    cpft_kwargs = Dict(
        :n_iterations=>Int(1e3),

        # belief-state widening: Bumper = None, Lidar = (1., 1/5)
        :enable_state_pw => true,
        :k_state => 1.,
        :alpha_state => 1/5, 

        # action-state widening: If |A| is small, None
        :enable_action_pw=>true,
        :k_action => 1.,
        :alpha_action => 1/5.,
        
        :depth => 50,
        :estimate_value => zero_V, # heuristicV
        :exploration_constant => 1.,
        :nu => 0.,
        :alpha_schedule => CMCTS.ConstantAlphaSchedule(1.),
    )
    search_updater = BootstrapFilter(cpomdp, 30)
    solver = BeliefCMCTSSolver(
        CDPWSolver(;cpft_kwargs...), search_updater;
        exact_rewards=false)
    planner = solve(solver, cpomdp)
    p = solve(solver, cpomdp)
    belief_updater = CMCTSBudgetUpdateWrapper(belief_updater, p)

elseif run_policy == 3
    # CoBETS kwargs
    cobts_kwargs = Dict(
        :options => [GoToGoal2D(cpomdp), Localize2D(cpomdp, [0.1, 0.1, 0.2]), Localize2D(cpomdp, [0.8, 0.8, 0.9])],
        :n_iterations=>Int(1e3),

        # belief-state widening: Bumper = None, Lidar = (1., 1/5)
        :enable_state_pw => true,
        :k_state => 1.,
        :alpha_state => 1/5, 

        # option widening: Default false
        :enable_action_pw=>false,
        
        :depth => 50,
        :estimate_value => zero_V, # heuristicV
        :exploration_constant => 1.,
        :nu => 0.,
        :alpha_schedule => COTS.ConstantAlphaSchedule(1.),
    )
    search_updater = BootstrapFilter(cpomdp, 30)
    solver = COBTSSolver(
        COTSSolver(;cobts_kwargs...), search_updater;
        exact_rewards=false)
    p = solve(solver, cpomdp)
else
    error("Not Implemented")
end


# first seed the environment
Random.seed!(2)

# run the simulation
c = @GtkCanvas()
win = GtkWindow(c, "Roomba Environment", 600, 600)
if 1 in check_ts
    p.solver.tree_in_info = true
    p.solver.search_progress_info = true
end
for (t, step) in enumerate(stepthrough(cpomdp, p, belief_updater, max_steps=max_steps))
    
    if p.solver.tree_in_info && p.solver.search_progress_info
        # extract tree
        if run_policy == 3
            tree = step.action_info.select[:tree]
            lambdas = step.action_info.select[:lambda]
        else
            tree = step.action_info[:tree]
            lambdas = step.action_info[:lambda]
        end

        # plot tree
        inchrome(D3Tree(tree; lambda=lambdas[end]))
        
        # plot lambdas
        plt = plot(1:length(lambdas), transpose(hcat(lambdas...)))
        savefig(plt, "scratch/roomba_$(problem)_lambda_step$(t).png")
        p.solver.tree_in_info = false
        p.solver.search_progress_info = false
    end
    if t+1 in check_ts # turn on for next step
        p.solver.tree_in_info = true
        p.solver.search_progress_info = true
    end
    @guarded draw(c) do widget
        
        # the following lines render the room, the particles, and the roomba
        ctx = getgc(c)
        set_source_rgb(ctx,1,1,1)
        paint(ctx)
        render(ctx, cpomdp, step)
        
        # render some information that can help with debugging
        # here, we render the time-step, the state, and the observation
        move_to(ctx,300,400)
        show_text(ctx, @sprintf("t=%d, state=[%.1f,%.1f,%.1f,%i], o=%.1f, r=%.1f, c=%.1f",
        t,step.s..., step.o, step.r, step.c[1]))
        end
    show(c)
    sleep(0.1) # to slow down the simulation
end
