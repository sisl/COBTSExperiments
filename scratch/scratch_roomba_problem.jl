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
using Infiltrator
using D3Trees
# using Statistics


sensor = Lidar() # Bumper() or Lidar()
config = 1 # 1,2, or 3 for different room configurations
vs = [0, 1, 3]
oms = [-π/2, 0, π/2] # with a dt of 0.5 seconds, this is 45 degrees per step
RoombaActSpace = [RoombaAct(v, om) for v in vs for om in oms]
pomdp = RoombaPOMDP(sensor=sensor,
                    mdp=RoombaMDP(v_max=maximum(vs), om_max=maximum(oms),config=config, aspace=RoombaActSpace))
cpomdp = RoombaCPOMDP(pomdp, cost_budget=100000.0)

num_particles = 10000
v_noise_coefficient = 1.0
om_noise_coefficient = 0.5

belief_updater = RoombaParticleFilter(cpomdp.pomdp, num_particles, v_noise_coefficient, om_noise_coefficient);
# belief_updater = BootstrapFilter(cpomdp, num_particles)


### CPOMCPOW policy p
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
    :tree_queries=>Int(1e4),
    # observation widening: Bumper = None, Lidar = (1., 1/5)
    :k_observation => 1., 
    :alpha_observation => 1/5, 
    :check_repeat_obs => true,

    # action widening: If |A| is small, None
    :enable_action_pw => true,
    :k_action => 1.,
    :alpha_action => 1/5., 
    :check_repeat_act => true,

    :max_depth => 50,
    # Q + k sqrt(N(sa)/log(N))
    :criterion => MaxCUCB(1., 0.), 
    :alpha_schedule => CPOMCPOW.ConstantAlphaSchedule(1.), # lambda = max(0, lambda + alphaschedule(iteration)*( Qsim - budget ))
    :estimate_value=>zero_V,
)

solver = CPOMCPOWSolver(;cpomcpow_kwargs...)
p = solve(solver, cpomdp)


# first seed the environment
Random.seed!(2)

# run the simulation
c = @GtkCanvas()
win = GtkWindow(c, "Roomba Environment", 600, 600)
p.solver.tree_in_info = true
#p.solver.search_progress_info = true # put things that were changing into action_info, like lambda
for (t, step) in enumerate(stepthrough(cpomdp, p, belief_updater, max_steps=100))
    if t==2
        # extract tree
        tree = step.action_info[:tree]
        # plot tree
        inchrome(D3Tree(tree;)) #lambda=hist2[1][:lambda][end]))

    end

    @guarded draw(c) do widget
        
        # the following lines render the room, the particles, and the roomba
        ctx = getgc(c)
        set_source_rgb(ctx,1,1,1)
        paint(ctx)
        render(ctx, cpomdp.pomdp, step)
        
        # render some information that can help with debugging
        # here, we render the time-step, the state, and the observation
        move_to(ctx,300,400)
        show_text(ctx, @sprintf("t=%d, state=%s, o=%.3f",t,string(step.s),step.o))
    end
    show(c)
    sleep(0.1) # to slow down the simulation
end
