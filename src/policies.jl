### Navigation CMDP Low Level Policies

# navigate
function navigate(problem::Union{LightDarkNew,CNav}, s::Union{LightDark1DState,NavState}, goal::Float64) 
    action = 0
    best_dist = Inf
    for a in actions(problem)
        (a ≈ 0) && continue
        dist = abs(s.y+a*problem.step_size-goal)
        if dist < best_dist
            best_dist = dist
            action = a
        end
    end
    return action
end

# navigate to a target
struct Navigate{P} <: LowLevelPolicy
    problem::P
    goal::Float64
end
terminate(nav::Navigate, s) = Deterministic((abs(s.y-nav.goal) <= 1))

# navigate to the goal and terminate
struct GoToGoal{P} <: LowLevelPolicy
    problem::P
end
terminate(p::GoToGoal, s) = Deterministic(POMDPs.isterminal(p,s))

POMDPTools.action_info(nav::Navigate{P}, s::LightDark1DState) where {P<:UnderlyingCMDP{<:CLightDarkNew}} = navigate(nav.problem.cpomdp.pomdp, s, nav.goal), (;goal=nav.goal, distance=abs(s.y-nav.goal))

function POMDPTools.action_info(g2g::GoToGoal{P}, s::LightDark1DState) where {P<:UnderlyingCMDP{<:CLightDarkNew}}
    action = (abs(s.y) < 1) ? action = 0 : action = navigate(g2g.problem.cpomdp.pomdp, s, 0.)
    return action, (;goal=0., distance=abs(s.y))
end
            

POMDPTools.action_info(nav::Navigate{P}, s::NavState) where {P<:CNav} = navigate(nav.problem, s, nav.goal), (;goal=nav.goal, distance=abs(s.y-nav.goal))
function POMDPTools.action_info(g2g::GoToGoal{P}, s::NavState) where {P<:CNav}
    action = (abs(s.y) < 1) ? action = 0 : action = navigate(g2g.problem.cpomdp, s, 0.)
    return action, (;goal=0., distance=abs(s.y))
end