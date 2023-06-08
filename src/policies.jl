### Navigation CMDP Low Level Policies

get_step_size(p::Union{LightDarkNew,CNav}) = p.step_size
get_step_size(p::LightDarkCPOMDP) = p.pomdp.step_size

# navigate
function navigate(problem::Union{LightDarkCPOMDP,LightDarkNew,CNav}, y::Float64, goal::Float64) 
    action = 0
    best_dist = Inf
    step_size = get_step_size(problem)
    for a in actions(problem)
        (a ≈ 0) && continue
        dist = abs(y+a*step_size-goal)
        if dist < best_dist
            best_dist = dist
            action = a
        end
    end
    return action
end

function navigate_slow(problem::Union{LightDarkCPOMDP,LightDarkNew,CNav}, y::Float64, goal::Float64) 
    above = y > goal
    action = 0
    best_dist = Inf
    step_size = get_step_size(problem)
    for a in actions(problem)
        (a ≈ 0) && continue
        offset = y+a*step_size-goal
        new_above = offset > 0
        ((above && !new_above) || (!above && new_above)) && continue
        dist = abs(offset)
        if dist < best_dist
            best_dist = dist
            action = a
        end
    end
    return action
end

# navigate to a target, get within abs
struct Navigate{P} <: LowLevelPolicy
    problem::P
    goal::Float64
    abs::Float64
end
terminate(nav::Navigate, s) = Deterministic((abs(s.y-nav.goal) <= nav.abs))
node_tag(p::Navigate) = "Navigate($(p.goal))"

# navigate to a target, get within abs, no passing target
struct NavigateSlow{P} <: LowLevelPolicy
    problem::P
    goal::Float64
    abs::Float64
end
terminate(nav::NavigateSlow, s) = Deterministic((abs(s.y-nav.goal) <= nav.abs))
node_tag(p::NavigateSlow) = "NavigateSlow($(p.goal))"

# navigate to the goal and terminate
struct GoToGoal{P} <: LowLevelPolicy
    problem::P
end
terminate(p::GoToGoal, s) = Deterministic(POMDPs.isterminal(p.problem,s))
node_tag(::GoToGoal) = "GoToGoal"

# LightDarkCMDP
POMDPTools.action_info(nav::Navigate{P}, s::LightDark1DState) where {P<:UnderlyingCMDP{<:LightDarkCPOMDP}} = navigate(nav.problem.cpomdp.pomdp, s.y, nav.goal), (;goal=nav.goal, distance=abs(s.y-nav.goal))
function POMDPTools.action_info(g2g::GoToGoal{P}, s::LightDark1DState) where {P<:UnderlyingCMDP{<:LightDarkCPOMDP}}
    action = (abs(s.y) < 1) ? action = 0 : action = navigate(g2g.problem.cpomdp.pomdp, s.y, 0.)
    return action, (;goal=0., distance=abs(s.y))
end      

# CNav
POMDPTools.action_info(nav::Navigate{P}, s::NavState) where {P<:CNav} = navigate(nav.problem, s.y, nav.goal), (;goal=nav.goal, distance=abs(s.y-nav.goal))
POMDPTools.action_info(nav::NavigateSlow{P}, s::NavState) where {P<:CNav} = navigate_slow(nav.problem, s.y, nav.goal), (;goal=nav.goal, distance=abs(s.y-nav.goal))
function POMDPTools.action_info(g2g::GoToGoal{P}, s::NavState) where {P<:CNav}
    action = (abs(s.y) < 1) ? action = 0 : action = navigate(g2g.problem, s.y, 0.)
    return action, (;goal=0., distance=abs(s.y))
end

# LightDarkCPOMDP

function statistics(b::Union{ParticleCollection{S},WeightedParticleBelief{S}}) where {S<:LightDark1DState}
    ws = weights(b)
    ws /= sum(ws)
    ys = [s.y for s in particles(b)]
    m = dot(ws, ys)
    diffs = ys .- m
    var = dot(ws, diffs.^2)  
    return m, sqrt(var)
end

function node_tag(b::Union{ParticleCollection{S},WeightedParticleBelief{S}}) where {S<:LightDark1DState}
    y, std = statistics(b)
    return @sprintf "LightDarkParticles(%.3f±%.3f)" y std
end

"""
GoToGoal policy navigates the particle mean to the goal and terminates
"""
function POMDPTools.action_info(g2g::GoToGoal{P}, 
    b::Union{ParticleCollection{S},WeightedParticleBelief{S}}) where {P<:LightDarkCPOMDP, S<:LightDark1DState}
    y, std = statistics(b)
    action = (abs(y) < 1) ? action = 0 : action = navigate(g2g.problem, y, 0.)
    return action, (;goal=0., distance=y, std=std)
end

"""
LocalizeFast policy navigates the particle mean greedily towards `goal` until the particle std is <= `max_std`
"""
struct LocalizeFast{P} <: LowLevelPolicy
    problem::P
    goal::Float64
    max_std::Float64
end
function POMDPTools.action_info(p::LocalizeFast, b)
    y, std = statistics(b)
    return navigate(p.problem, y, p.goal), (;goal=p.goal, distance=abs(y-p.goal), std=std)
end
terminate(p::LocalizeFast, b) = Deterministic((last(statistics(b)) <= p.max_std))
node_tag(p::LocalizeFast) = "LocalizeFast($(p.max_std))"


"""
LocalizeSlow policy navigates the particle mean slowly towards `goal` until the particle std is <= `max_std`
"""
struct LocalizeSlow{P} <: LowLevelPolicy
    problem::P
    goal::Float64
    max_std::Float64
end
function POMDPTools.action_info(p::LocalizeSlow, b)
    y, std = statistics(b)
    return navigate_slow(p.problem, y, p.goal), (;goal=p.goal, distance=abs(y-p.goal), std=std)
end
terminate(p::LocalizeSlow, b) = Deterministic((last(statistics(b)) <= p.max_std))
node_tag(p::LocalizeSlow) = "LocalizeSlow($(p.max_std))"

"""
LocalizeSafe policy navigates the particle mean towards `goal` from below while keeping at least `α` stds away from `max_y`, until the particle std is <= `max_std`
"""
struct LocalizeSafe{P} <: LowLevelPolicy
    problem::P
    goal::Float64
    max_y::Float64
    α::Float64
    max_std::Float64
end
function POMDPTools.action_info(p::LocalizeSafe, b)
    y, std = statistics(b)
    as = actions(p.problem)
    action = minimum(as)
    best_dist = Inf
    for a in as
        (a ≈ 0.) && continue
        if (y + a) < (p.max_y - p.α*std)
            dist = abs(p.goal - y - a)
            if dist < best_dist
                best_dist = dist
                action = a
            end
        end
    end
    return action, (;goal=p.goal, distance=abs(y-p.goal), std=std)
end
terminate(p::LocalizeSafe, b) = Deterministic((last(statistics(b)) <= p.max_std))
node_tag(p::LocalizeSafe) = "LocalizeSafe($(p.α),$(p.max_std))"

#CPOMDPs.gbmdp_handle_terminal(pomdp::LightDarkCPOMDP, updater::Updater, b, s, a, rng) = b # dont update terminal beliefs (should just return 0 reward)

