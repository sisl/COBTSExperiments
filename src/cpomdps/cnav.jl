### Constrained Nav Problem. Similar to 
## Same as lightdark, but with a different action space, and constraints on going above 12

import Base: ==, +, *, -, rand

"""
    NavState
## Fields
- `y`: position
- `target_hit`: 0 = no, 1 = yes
- `status`: 0 = normal, negative = terminal
"""
struct NavState
    status::Int64
    target_hit::Int64
    y::Float64
end

*(n::Number, s::NavState) = NavState(s.status, s.target_hit, n*s.y)

struct NavStateDist
    mean::Float64
    std::Float64
    target_hit::Int64
end
sampletype(::Type{NavStateDist}) = NavState
rand(rng::AbstractRNG, d::NavStateDist) = NavState(0, d.target_hit, d.mean + randn(rng)*d.std)



@with_kw mutable struct CNav <: CMDP{NavState,Int}
    discount_factor::Float64 = 0.95
    correct_r::Float64 = 100.
    incorrect_r::Float64 = -100.
    step_size::Float64 = 1. 
    movement_cost::Float64 = 1.
    cost_budget::Float64 = 0.5
    max_y::Float64 = 15.
    nav_y::Float64 = 9.
end

POMDPs.discount(p::CNav) = p.discount_factor
POMDPs.isterminal(::CNav, act::Int64) = act == 0
POMDPs.isterminal(::CNav, s::NavState) = s.status < 0
POMDPs.actions(::CNav) = [-10, -5, -1, 0, 1, 5, 10]
POMDPs.initialstate(::CNav) = NavStateDist(2, 2, 0)

function POMDPs.transition(p::CNav, s::NavState, a::Int)
    next_pos = s.y+a*p.step_size
    target_hit = Int64((s.target_hit>0) || (next_pos > p.nav_y))
    status = a==0 ? -1 : s.status
    return Deterministic(NavState(status, target_hit, next_pos))
end

function POMDPs.reward(p::CNav, s::NavState, a::Int)
    if s.status < 0
        return 0.0
    elseif a == 0
        if abs(s.y) < 1 && s.target_hit == 1
            return p.correct_r
        else
            return p.incorrect_r
        end
    else
        return -p.movement_cost
    end
end

POMDPs.convert_s(::Type{A}, s::NavState, p::CNav) where A<:AbstractArray = eltype(A)[s.status, s.target_hit, s.y]
POMDPs.convert_s(::Type{NavState}, s::A, p::CNav) where A<:AbstractArray = NavState(Int64(s[1]),Int64(s[2]), s[3])


costs(p::CNav, s::NavState, a::Int) = Float64[s.y >= p.max_y]
costs_limit(p::CNav) = [p.cost_budget]
n_costs(::CNav) = 1
max_reward(p::CNav) = p.pomdp.correct_r
min_reward(p::CNav) = -p.pomdp.movement_cost

function QMDP_V(p::CNav, s::NavState, args...)
    y = abs(s.y)
    steps = floor(Int, y/10)
    steps += floor(Int, y-10*steps)
    γ = discount(p)
    V = -sum([(γ^i)*p.movement_cost  for i in 0:steps-1]) + (γ^steps)*p.correct_r
    steps2 = floor(Int, (s.y+10-p.max_y)/10)
    C = sum([γ^i for i in 0:steps2-1])
    return (V, [C])
end

function trueC(p::CNav, s::NavState)
    γ = discount(p)
    steps = floor(Int, (s.y+10-p.max_y)/10)
    C = sum([γ^i for i in 0:steps-1])
    return [C]
end

zeroV_trueC(p::CNav, s::NavState, args...) = (0, trueC(p,s))

function zeroV_trueC(p::CPOMDPs.GenerativeBeliefCMDP{P}, s::ParticleFilters.ParticleCollection{S}, args...) where {P<:CNav, S<:NavState}
    C = [0.]
    ws = weights(s)
    for (part, w) in zip(particles(s),ws)
        C .+= trueC(p.cpomdp, part) * w
    end
    C ./= sum(ws)
    return (0, C) # replaces old weight_sum(particle collections) that was 1 

end

function heuristicV(p::CNav, s::NavState, args...)
    m = s.y
    γ = discount(p)
    steps=0
    if s.target_hit==0 # go above p.nav_y
        steps = ceil(abs(p.nav_y-m)/10) 
        m += 10*steps 
    end
    steps += n_steps(m, 0.)
    V = -sum([(γ^i)*p.movement_cost  for i in 0:steps-1]) + (γ^steps)*p.correct_r
    return V, trueC(p, s)

end

# number of steps to get from x to \pm 1 of goal in increments of 1, 5, 10
function n_steps(x::Float64, goal::Float64)
    steps = 0
    x = abs(x-goal)
    steps += Int(x ÷ 10)
    x = (x % 10)
    steps += Int(x ÷ 5)
    x = (x % 5)
    steps += Int(x ÷ 1)
    return steps
end


