### Constrained Roomba Problem
"""
Some default parameters:
goal_reward = 10
stairs-penalty: -10
time_pen = -0.1
contact_pen = -1
"""

# Unconstrained version is defined as
"""
struct RoombaPOMDP{SS, AS, S, T, O} <: POMDP{S, RoombaAct, O}
    sensor::T
    mdp::RoombaMDP{SS, AS, S}
end
"""

struct RoombaCPOMDPInitBounds
    xmin::Float64
    xmax::Float64
    ymin::Float64
    ymax::Float64
    thmin::Float64
    thmax::Float64
end

# Constrained version
struct RoombaCPOMDP{P<:RoombaPOMDP,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P
    avoid_region::Vector{Float64}  # [xmin, xmax, ymin, ymax]
    cost_budget::Float64
    init_bounds::RoombaCPOMDPInitBounds
end

function RoombaCPOMDP(pomdp::P; avoid_region::Vector{Float64}=[-5., 5, -5, 0], cost_budget::Float64=0.5,
    init_bounds::RoombaCPOMDPInitBounds=RoombaCPOMDPInitBounds(-24.5,-15.5,-19.5,4.5,-π,π)) where {P<:RoombaPOMDP}
    return RoombaCPOMDP{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp, avoid_region, cost_budget, init_bounds)
end

POMDPs.initialstate(p::RoombaCPOMDP) = p.init_bounds
function Base.rand(rng::AbstractRNG, d::RoombaCPOMDPInitBounds)
    x = rand(rng)*(d.xmax-d.xmin)+d.xmin
    y = rand(rng)*(d.ymax-d.ymin)+d.ymin
    th = rand(rng)*(d.thmax-d.thmin)+d.thmin
    th = RoombaPOMDPs.wrap_to_pi(th)
    return RoombaState(x, y, th, 0.0)
end
in_avoid_region(cpomdp::RoombaCPOMDP, s::RoombaState) = cpomdp.avoid_region[1] <= s.x <= cpomdp.avoid_region[2] && cpomdp.avoid_region[3] <= s.y <= cpomdp.avoid_region[4]

# New cost function is defined for a certain region that the robot is not allowed to enter
costs(cpomdp::RoombaCPOMDP, s::RoombaState, a::RoombaAct) = Float64[in_avoid_region(cpomdp, s)]

# NOTE: now we have to set RoombaMDP.stairs_penalty to zero as it's part of the costs now

# transform coordinates using window from RoombaPOMDPs
transform_coords(x,y) = (x + 30.0)/50.0*600, -(y - 20.0)/50.0*600

function RoombaPOMDPs.render(ctx::CairoContext, m::RoombaCPOMDP, step)
    # find constraint region coordinates
    xmin, xmax, ymin, ymax = m.avoid_region
    xmin, ymin = transform_coords(xmin,ymin)
    xmax, ymax = transform_coords(xmax,ymax)
    
    # fill light constraint region
    set_source_rgba(ctx,0.82,0.15,0.19, 0.3)
    rectangle(ctx,xmin,ymax,xmax-xmin,ymin-ymax)  # region
    fill(ctx)
    
    # render POMDP
    RoombaPOMDPs.render(ctx,m.pomdp, step)

    # draw region outline
    set_source_rgb(ctx,1.0,0.,0.)
    set_line_width(ctx, 2)
    move_to(ctx, xmin, ymin)
    line_to(ctx, xmax, ymin)
    Cairo.stroke(ctx)
    move_to(ctx, xmax, ymin)
    line_to(ctx, xmax, ymax)
    Cairo.stroke(ctx)
    move_to(ctx, xmax, ymax)
    line_to(ctx, xmin, ymax)
    Cairo.stroke(ctx)
    move_to(ctx, xmin, ymax)
    line_to(ctx, xmin, ymin)
    Cairo.stroke(ctx)

    #reset rgb
    set_source_rgb(ctx,0.,0.,0.)
end


costs_limit(p::RoombaCPOMDP) = [p.cost_budget]
n_costs(::RoombaCPOMDP) = 1
# min_reward(p::RoombaCPOMDP) = p.pomdp.mdp.time_pen + p.pomdp.mdp.contact_pen
# max_reward(p::RoombaCPOMDP) = p.pomdp.mdp.time_pen + p.pomdp.mdp.goal_reward

zeroV_trueC(p::RoombaCPOMDP, s::RoombaState, args...) = (0, [0])

function zeroV_trueC(p::CPOMDPs.GenerativeBeliefCMDP{P}, s::ParticleFilters.ParticleCollection{S}, args...) where {P<:RoombaCPOMDP, S<:RoombaState}
    # C = [0.]
    # ws = weights(s)
    # for (part, w) in zip(particles(s),ws)
    #     C .+= trueC(p.cpomdp, part) * w
    # end
    # C ./= sum(ws)
    # return (0, C)
    return (0, [0.])
end

### statistics and belief-state node labels
function stats(b::Union{ParticleCollection{S}, WeightedParticleBelief{S}}) where {S<:RoombaState}
    ws = weights(b)
    ws /= sum(ws)
    locs = [[s.x, s.y, s.theta] for s in particles(b)]
    m_x = dot(ws, [loc[1] for loc in locs])
    m_y = dot(ws, [loc[2] for loc in locs])
    m_theta = get_mean_th(ws, [loc[3] for loc in locs])
    # m_theta = dot(ws, [loc[3] for loc in locs])
    m = [m_x, m_y, m_theta]
    diffs = [loc .- m for loc in locs]
    var_x = dot(ws, [diff[1]^2 for diff in diffs])
    var_y = dot(ws, [diff[2]^2 for diff in diffs])
    var_theta = dot(ws, [RoombaPOMDPs.wrap_to_pi.(diff[3])^2 for diff in diffs]) # var_theta isbs
    var = [var_x, var_y, var_theta]
    return m, sqrt.(var)
end

function node_tag(b::Union{ParticleCollection{S},WeightedParticleBelief{S}}) where {S<:RoombaState}
    y, std = stats(b)
    return @sprintf "RoombaParticles(%.3f±%.3f,%.3f±%.3f,%.3f±%.3f)" y[1] std[1] y[2] std[2] y[3] std[3]
end

# get mean th by using polar coordinates
function get_mean_th(ws, ths)
    xs = cos.(ths)
    ys = sin.(ths)
    mx = dot(ws,xs)
    my = dot(ws,ys)
    th = atan(my,mx)
    return RoombaPOMDPs.wrap_to_pi(th)
end