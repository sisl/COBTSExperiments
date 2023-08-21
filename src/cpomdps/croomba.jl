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
