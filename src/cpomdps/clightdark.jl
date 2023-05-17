### Light Dark New
## Same as lightdark, but with a different action space, and constraints on going above 12

import Base: ==, +, *, -
using Distributions

mutable struct LightDarkNew{F<:Function} <: POMDPs.POMDP{LightDark1DState,Int,Float64}
    discount_factor::Float64
    correct_r::Float64
    incorrect_r::Float64
    step_size::Float64
    movement_cost::Float64
    sigma::F
end

default_sigma(x::Float64) = abs(x - 10)/sqrt(2) + 1e-2

LightDarkNew() = LightDarkNew(0.95, 100.0, -100.0, 1.0, 1.0, default_sigma)
POMDPs.discount(p::LightDarkNew) = p.discount_factor
POMDPs.isterminal(::LightDarkNew, act::Int64) = act == 0
POMDPs.isterminal(::LightDarkNew, s::LightDark1DState) = s.status < 0
POMDPs.actions(::LightDarkNew) = [-10, -5, -1, 0, 1, 5, 10]
POMDPs.initialstate(::LightDarkNew) = POMDPModels.LDNormalStateDist(2, 2)
POMDPs.initialobs(m::LightDarkNew, s) = POMDPs.observation(m, s)
POMDPs.observation(p::LightDarkNew, sp::LightDark1DState) = Normal(sp.y, p.sigma(sp.y))

function POMDPs.transition(p::LightDarkNew, s::LightDark1DState, a::Int)
    if a == 0
        return Deterministic(LightDark1DState(-1, s.y+a*p.step_size))
    else
        return Deterministic(LightDark1DState(s.status, s.y+a*p.step_size))
    end
end

function POMDPs.reward(p::LightDarkNew, s::LightDark1DState, a::Int)
    if s.status < 0
        return 0.0
    elseif a == 0
        if abs(s.y) < 1
            return p.correct_r
        else
            return p.incorrect_r
        end
    else
        return -p.movement_cost
    end
end

POMDPs.convert_s(::Type{A}, s::LightDark1DState, p::LightDarkNew) where A<:AbstractArray = eltype(A)[s.status, s.y]
POMDPs.convert_s(::Type{LightDark1DState}, s::A, p::LightDarkNew) where A<:AbstractArray = LightDark1DState(Int64(s[1]), s[2])



## CPOMDP

struct CLightDarkNew{P<:LightDarkNew,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P 
    cost_budget::Float64
    max_y::Float64
end

function CLightDarkNew(;pomdp::P=LightDarkNew(),
    cost_budget::Float64=0.5,
    max_y::Float64=12.,
    ) where {P<:LightDarkNew}
    return CLightDarkNew{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp,cost_budget,max_y)
end

costs(p::CLightDarkNew, s::LightDark1DState, a::Int) = Float64[s.y >= p.max_y]
costs_limit(p::CLightDarkNew) = [p.cost_budget]
n_costs(::CLightDarkNew) = 1
max_reward(p::CLightDarkNew) = p.pomdp.correct_r
min_reward(p::CLightDarkNew) = -p.pomdp.movement_cost

function QMDP_V(p::LightDarkNew, s::LightDark1DState, args...) 
    y = abs(s.y)
    steps = floor(Int, y/10)
    steps += floor(Int, y-10*steps)
    γ = discount(p)
    return -sum([(γ^i)*p.movement_cost  for i in 0:steps-1]) + (γ^steps)*p.correct_r 
end

function QMDP_V(p::CLightDarkNew, s::LightDark1DState, args...)
    V = QMDP_V(p.pomdp,s,args...)
    γ = discount(p)
    steps = floor(Int, (s.y+10-p.max_y)/10)
    C = sum([γ^i for i in 0:steps-1])
    return (V, [C])
end

function trueC(p::CLightDarkNew, s::LightDark1DState)
    γ = discount(p)
    steps = floor(Int, (s.y+10-p.max_y)/10)
    C = sum([γ^i for i in 0:steps-1])
    return [C]
end

zeroV_trueC(p::CLightDarkNew, s::LightDark1DState, args...) = (0, trueC(p,s))

function zeroV_trueC(p::CPOMDPs.GenerativeBeliefCMDP{P}, s::ParticleFilters.ParticleCollection{S}, args...) where {P<:CLightDarkNew, S<:LightDark1DState}
    C = [0.]
    ws = weights(s)
    for (part, w) in zip(particles(s),ws)
        C .+= trueC(p.cpomdp, part) * w
    end
    C ./= sum(ws)
    return (0, C) # replaces old weight_sum(particle collections) that was 1 

end

function heuristicV(p::POMDP, s::ParticleFilters.ParticleCollection{S}) where {S<:LightDark1DState}
    ys = [p.y for p in particles(s)]
    m = Statistics.mean(ys)
    sig = Statistics.std(ys)
    V = 0
    γ = discount(p)
    steps = 1
    if sig > 1 # go to 10 first for two time steps
        steps += ceil(abs(10-m)/5) + 2
    end
    V += -sum([(γ^i)*p.movement_cost  for i in 0:steps-1]) + (γ^steps)*p.correct_r
    return V

end

heuristicV(p::CPOMDPs.GenerativeBeliefCMDP{P}, s::ParticleFilters.ParticleCollection{S}, 
    args...) where {P<:CLightDarkNew, S<:LightDark1DState} = return (heuristicV(p.cpomdp.pomdp, s), zeroV_trueC(p,s,args...)[2])

heuristicV(p::POMDPTools.GenerativeBeliefMDP{P}, s::ParticleFilters.ParticleCollection{S}, 
    args...) where {P<:LightDarkNew, S<:LightDark1DState} = return heuristicV(p.pomdp, s)

abstract type LowLevelPolicy <: POMDPs.Policy end
terminate(p::LowLevelPolicy, s, a) = terminate(p, s)
action(p::LowLevelPolicy,s) = first(POMDPTools.action_info(p,s))

### CMDP Low Level Policies

# navigate
function navigate(problem::UnderlyingMDP{P}, s::LightDark1DState, goal::Float64) where P<:CLightDarkNew
    action = 0
    best_dist = Inf
    for a in actions(problem)
        (a ≈ 0) && continue
        dist = abs(s.y+a*g2g.problem.step_size-goal)
        if dist < best_dist
            best_dist = dist
            action = a
        end
    end
    return action
end

# Low-Level Policies
struct Navigate <: LowLevelPolicy
    problem::UnderlyingCMDP{<:CLightDarkNew}
    info::NamedTuple
end
Navigate(p::UnderlyingCMDP{P}, goal::Float64) = Navigate(p,(;goal=goal)) where P<:CLightDarkNew

terminate(nav::Navigate, s::LightDark1DState) = Deterministic((abs(s.y-nav.info[:goal]) <= 1))

POMDPTools.action_info(nav::Navigate, s::LightDark1DState) = navigate(g2g.problem, s, nav.info[:goal])

# navigate to the goal and terminate
struct GoToGoal <: LowLevelPolicy
    problem::UnderlyingCMDP{<:CLightDarkNew}
    info::NamedTuple
end

terminate(::GoToGoal, ::LightDark1DState) = Deterministic(false)

function POMDPTools.action_info(g2g::GoToGoal, s::LightDark1DState)
    action = (abs(s.y) < 1) ? action = 0 : action = navigate(g2g.problem, s, 0.)
    return action, g2g.info
end
            
# High Level Policy

mutable struct RandomHighLevelPolicy <: POMDPs.Policy
    options::Vector{LowLevelPolicy}
    running::LowLevelPolicy
    rng::Random.RNG
    step_counter::Int
end

function POMDPTools.action_info(p::RandomHighLevelPolicy, s)
    dist = terminate(p, s)
    info = (;new_option=false)
    if rand(dist, p.rng)
        # terminate and start new
        p.running = rand(p.options)
        p.step_counter = 0
        info[:new_option]= true
    end
    a, ai = POMDPTools.action_info(p.running,s)
    p.step_counter += 1
    return a, merge(ai, info)
end
        
    


