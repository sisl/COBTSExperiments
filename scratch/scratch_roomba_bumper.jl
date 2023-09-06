using COBTSExperiments
using RoombaPOMDPs
using POMDPs
using CPOMDPs
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

Random.seed!(2) # debug seed 2, 3

problem = "discrete"
sensor = FrontBumper(π/2-0.01) # Bumper() or Lidar()
vs = [0, 1, 3]
oms = [-π/2, 0, π/2] # with a dt of 0.5 seconds, this is 45 degrees per step
RoombaActSpace = [RoombaAct(v, om) for v in vs for om in oms]
v_noise_coefficient = 0.2 #1.0
om_noise_coefficient = 0.01 #0.5
v_max = maximum(vs) + v_noise_coefficient/2 # allow PF to hit maximum target noise
om_max = maximum(oms) + om_noise_coefficient/2
pomdp = RoombaPOMDP(sensor=sensor,
                    mdp=RoombaMDP(v_max=v_max, om_max=om_max,config=4, 
                    aspace=RoombaActSpace, 
                    stairs_penalty=-100.0, goal_reward=10., 
                    contact_pen=0.,time_pen=0.05))
cpomdp = RoombaCPOMDP(pomdp, cost_budget=1000.,
    #init_bounds=RoombaCPOMDPInitBounds(-24.5,-15.5,-19.5,4.5,0.,3π/2), # general
    #init_bounds=RoombaCPOMDPInitBounds(-24.,-16.,-19.,4.,π/2,3π/2), # target
    init_bounds=RoombaCPOMDPInitBounds(-15.5,-15.5,-16.,-16.,0.,0.), # specific
    )

options = [GreedyGoToGoal(cpomdp;max_steps=80, max_std=[30.,30.]), #GreedyGoToGoal(cpomdp;max_steps=20),
    #SafeGoToGoal(cpomdp;max_steps=40), SafeGoToGoal(cpomdp;max_steps=20),
    TurnThenGo(cpomdp;turn_steps=0,max_steps=40), TurnThenGo(cpomdp;turn_steps=2,max_steps=40),
    TurnThenGo(cpomdp;turn_steps=-2,max_steps=40), TurnThenGo(cpomdp;turn_steps=4,max_steps=40),
    ]
    
num_particles = 10000


belief_updater = RoombaParticleFilter(cpomdp.pomdp, 
    num_particles, v_noise_coefficient, om_noise_coefficient);
# belief_updater = BootstrapFilter(cpomdp, num_particles)
max_steps = 100
check_ts = collect(1:max_steps) #[1, 6] # [1,30] # time steps to check tree and lambda history
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

        :max_depth => max_steps,
        # Q + k sqrt(N(sa)/log(N))
        :criterion => MaxCUCB(2., 0.), 
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
        
        :depth => max_steps,
        :estimate_value => zero_V, # heuristicV
        :exploration_constant => 2.,
        :nu => 0.,
        :alpha_schedule => CMCTS.ConstantAlphaSchedule(1.),
    )
    search_updater = RoombaParticleFilter(cpomdp.pomdp, 30, v_noise_coefficient, om_noise_coefficient, 
        resamper=ParticleDeathResampler())
#    search_updater = BootstrapFilter(cpomdp, 30)
    solver = BeliefCMCTSSolver(
        CDPWSolver(;cpft_kwargs...), search_updater;
        exact_rewards=false)
    planner = solve(solver, cpomdp)
    p = solve(solver, cpomdp)
    belief_updater = CMCTSBudgetUpdateWrapper(belief_updater, p)

elseif run_policy == 3
    # CoBETS kwargs
    cobts_kwargs = Dict(
        :options => options,
        :n_iterations=>Int(1e3),

        # belief-state widening: Bumper = None, Lidar = (1., 1/5)
        :enable_state_pw => true,
        :k_state => 1.,
        :alpha_state => 1/5, 

        # option widening: Default false
        :enable_action_pw=>false,
        :return_safe_action=>false,
        :depth => max_steps,
        :estimate_value => zero_V, # heuristicV
        :exploration_constant => 2.,
        :nu => 0.,
        :alpha_schedule => COTS.ConstantAlphaSchedule(1.),
    )
    search_updater = RoombaParticleFilter(cpomdp.pomdp, 30, v_noise_coefficient, om_noise_coefficient)
    #search_updater = BootstrapFilter(cpomdp, 30)
    solver = COBTSSolver(
        COTSSolver(;cobts_kwargs...), search_updater;
        exact_rewards=true)
    p = solve(solver, cpomdp)
    belief_updater = OptionsUpdateWrapper(belief_updater, p) # external action wrapper
else
    error("Not Implemented")
end

# run the simulation
c = @GtkCanvas()
win = GtkWindow(c, "Roomba Environment", 600, 600)
if 1 in check_ts
    p.solver.tree_in_info = true
    p.solver.search_progress_info = true
end
hl_action = nothing
for (t, step) in enumerate(stepthrough(cpomdp, p, belief_updater, max_steps=max_steps))
    @infiltrate false
    (p isa OptionsPolicy) && (global hl_action = low_level(p))
    plotted = false
    if p.solver.tree_in_info && p.solver.search_progress_info
        skip = false
        # extract tree
        if p isa OptionsPolicy 
            if step.action_info.select !== nothing
                tree = step.action_info.select[:tree]
                lambdas = step.action_info.select[:lambda]
            else
                skip=true
            end
        else
            tree = step.action_info[:tree]
            lambdas = step.action_info[:lambda]
        end

        if !skip
            # plot tree
            inchrome(D3Tree(tree; lambda=lambdas[end]))
            plotted = true
            # plot lambdas
            plt = plot(1:length(lambdas), transpose(hcat(lambdas...)))
            saveloc="scratch/figs/roomba_$(problem)_lambda_step$(t).png"
            dir = dirname(saveloc)
            !isdir(dir) && mkpath(dir)
            savefig(plt, saveloc)
        end
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
        show_text(ctx, @sprintf("t=%d, state=[%.1f,%.1f,%.1f,%i], a=[%.1f,%.1f], o=%.1f, r=%.1f, c=%.1f",
        t,step.s..., step.a..., step.o, step.r, step.c[1]))
        if hl_action !== nothing
            move_to(ctx,300,420)
            show_text(ctx, @sprintf("Option: %s", COBTSExperiments.node_tag(hl_action)))
        end
    end
    show(c)
    sleep(0.1) # to slow down the simulation
    @infiltrate (t in check_ts) && plotted
end
