### Constrained Roomba Problem

# Unconstrained version is defined as
"""
struct RoombaPOMDP{SS, AS, S, T, O} <: POMDP{S, RoombaAct, O}
    sensor::T
    mdp::RoombaMDP{SS, AS, S}
end
"""

# Constrained version
struct RoombaCPOMDP{P<:RoombaPOMDP,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P
    cost_budget::Float64
end

function RoombaCPOMDP(pomdp::P; cost_budget::Float64=0.5) where {P<:RoombaPOMDP}
    return RoombaCPOMDP{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp,cost_budget)
end


# If collided with wall, state status becomes negative, positive for goal, 0 otherwise
costs(pomdp::RoombaCPOMDP, s::RoombaState, a::RoombaAct) = Float64[s.status < 0]

# NOTE: now we have to set RoombaMDP.stairs_penalty to zero as it's part of the costs now

costs_limit(p::RoombaCPOMDP) = [p.cost_budget]
n_costs(::RoombaCPOMDP) = 1
min_reward(p::RoombaCPOMDP) = p.pomdp.mdp.time_pen + p.pomdp.mdp.contact_pen
max_reward(p::RoombaCPOMDP) = p.pomdp.mdp.time_pen + p.pomdp.mdp.goal_reward

zeroV_trueC(p::RoombaCPOMDP, s::RoombaState, args...) = (0, [0])

function zeroV_trueC(p::CPOMDPs.GenerativeBeliefCMDP{P}, s::ParticleFilters.ParticleCollection{S}, args...) where {P<:RoombaCPOMDP, S<:RoombaState}
    # C = [0.]
    # ws = weights(s)
    # for (part, w) in zip(particles(s),ws)
    #     C .+= trueC(p.cpomdp, part) * w
    # end
    # C ./= sum(ws)
    # return (0, C)
    return (0, [0])
end

### statistics and belief-state node labels
function stats(b::Union{ParticleCollection{S}, WeightedParticleBelief{S}}) where {S<:RoombaState}
    ws = weights(b)
    ws /= sum(ws)
    locs = [[s.x, s.y, s.theta] for s in particles(b)]
    m_x = dot(ws, [loc[1] for loc in locs])
    m_y = dot(ws, [loc[2] for loc in locs])
    m_theta = dot(ws, [loc[3] for loc in locs])
    m = [m_x, m_y, m_theta]
    diffs = [loc .- m for loc in locs]
    var_x = dot(ws, [diff[1]^2 for diff in diffs])
    var_y = dot(ws, [diff[2]^2 for diff in diffs])
    var_theta = dot(ws, [diff[3]^2 for diff in diffs])
    var = [var_x, var_y, var_theta]
    return m, sqrt.(var)
end


function node_tag(b::Union{ParticleCollection{S},WeightedParticleBelief{S}}) where {S<:RoombaState}
    y, std = stats(b)
    return @sprintf "RoombaParticles(%.3f±%.3f,%.3f±%.3f,%.3f±%.3f)" y[1] std[1] y[2] std[2] y[3] std[3]
end