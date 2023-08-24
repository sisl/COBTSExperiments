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


### RoombaCPOMDP

# Get statistics
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

# Function to calculate distance from a point (px, py) to a line segment (x1, y1) - (x2, y2)
function point_to_line_segment(px, py, x1, y1, x2, y2)
    dx, dy = x2 - x1, y2 - y1
    if dx != 0 || dy != 0
        t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)
        if t > 1
            x1, y1 = x2, y2
        elseif t > 0
            x1 += dx * t
            y1 += dy * t
        end
    end
    dx, dy = px - x1, py - y1
    return sqrt(dx * dx + dy * dy), atan(dy, dx), [x1, y1] # distance, direction, nearest point
end

# Now we can calculate the nearest wall to the robot
function nearest_wall(room, rx, ry)
    min_distance = Inf
    min_direction = 0
    min_point = (0, 0)
    for rectangle in room.rectangles
        for i in 1:4
            # Get the coordinates of the two points representing the current wall
            x1, y1 = rectangle.corners[i, :]
            x2, y2 = rectangle.corners[i % 4 + 1, :]
            # Calculate the distance and direction from the robot to this wall, and the coordinates of the nearest point on the wall
            distance, direction, point = point_to_line_segment(rx, ry, x1, y1, x2, y2)
            # Update the minimum distance, direction and point if this wall is closer
            if distance < min_distance
                min_distance = distance
                min_direction = direction
                min_point = point
            end
        end
    end
    return min_direction, min_distance, min_point
end

function adjust_angle(angle, clockwise=true)
    if clockwise
        return angle - pi/2
    else
        return angle + pi/2
    end
end


# navigate2D function navigates the robot towards `goal`
function navigate2D(problem::RoombaCPOMDP, b, goal::Vector{Float64})
    s = stats(b)[1]
    s = RoombaState(s..., 0)
    best_action = RoombaAct(0, 0)
    best_dist = Inf
    for a in actions(problem)
        new_s = rand(transition(problem, s, a))  # not quite random, in fact deterministic
        dist = norm([new_s.x, new_s.y] - goal)
        if dist < best_dist
            best_dist = dist
            best_action = a
        end
    end
    return best_action
end

# follow_wall function navigates the robot along the wall in the direction of the goal
function follow_wall(problem::RoombaCPOMDP, b)
    s = stats(b)[1]

    # Get the goal and room layout of the problem
    room = problem.pomdp.mdp.room
    goal_vec = get_goal_xy(problem.pomdp)

    # Compute the direction to the goal
    goal_dir = atan(goal_vec[2] - s.y, goal_vec[1] - s.x)

    # Get the direction and distance to the nearest wall
    wall_dir, _, _ = nearest_wall(room, s.x, s.y)

    # Compute the two possible directions the robot can move along the wall
    clockwise_dir = adjust_angle(wall_dir, true)
    counterclockwise_dir = adjust_angle(wall_dir, false)

    # Choose the direction that is closer to the goal direction
    if abs(clockwise_dir - goal_dir) < abs(counterclockwise_dir - goal_dir)
        wall_follow_dir = clockwise_dir
    else
        wall_follow_dir = counterclockwise_dir
    end

    dt = problem.pomdp.mdp.dt
    om_max = problem.pomdp.mdp.om_max
    v_max = problem.pomdp.mdp.v_max

    # Determine the necessary turn and its direction
    turn = s.theta - wall_follow_dir
    turns_required = ceil(abs(turn) / (om_max*dt))

    possible_oms = [a[2] for a in actions(problem.pomdp) if a[1] == 0]
    om = possible_oms[argmin(abs.(turn - dt .* possible_oms))]

    # Generate the action to turn in the chosen direction with zero velocity if multiple turns are required
    if turns_required <= 1
        action = RoombaAct(v_max, om)
    else
        action = RoombaState(0, om)
    end
    return action
end


"""
GoToGoal2D policy directs particle mean to goal and terminates
"""
struct GoToGoal2D{P} <: LowLevelPolicy
    problem::P
end
function POMDPTools.action_info(p::GoToGoal2D, b)
    s = stats(b)[1]
    goal_vec = get_goal_xy(p.problem.pomdp)
    action = navigate2D(p.problem, b, goal_vec)
    return action, (;goal=goal_vec, distance=norm(s - goal_vec))
end
terminate(p::GoToGoal2D, b) = Deterministic((stats(b)[1] ≈ get_goal_xy(p.problem.pomdp)))
node_tag(p::GoToGoal2D) = "GoToGoal2D"

"""
Localize2D policy localizes itself by going to the nearest wall (bumper sensor) or rotate (lidar sensor) until the particle std is <= `max_std`
"""
struct Localize2D{P} <: LowLevelPolicy
    problem::P
    max_std::Vector{Float64}
end
function POMDPTools.action_info(p::Localize2D, b)
    s, std = stats(b)
    # If we have a bumper sensor, then we navigate towards the nearest wall, and then we follow the wall until we bump into a new one
    if p.problem.pomdp.sensor == Bumper()
        # Get the direction and distance to the nearest wall
        _, wall_dist, wall_point = nearest_wall(p.problem.pomdp.mdp.room, s[1], s[2])
        # If we are close enough to the wall, then we follow it
        if wall_dist < 0.1
            action = follow_wall(p.problem, b)
        # Otherwise, we navigate towards the wall
        else
            action = navigate2D(p.problem, b, wall_point)
        end
    # If we instead have a lidar sensor, we simply rotate until we have a low enough std (no bumps)
    elseif p.problem.pomdp.sensor == Lidar()
        om_max = p.problem.pomdp.mdp.om_max
        action = RoombaState(0, om_max)
    end
    goal = get_goal_xy(p.problem.pomdp)
    dist = norm([s[1], s[2]] - goal)
    return action, (;goal=goal, distance=dist, std=std)
end
terminate(p::Localize2D, b) = Deterministic((stats(b)[2] <= p.max_std))
node_tag(p::Localize2D) = "Localize2D($(p.max_std))"
