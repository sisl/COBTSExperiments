
### RoombaCPOMDP Policies

## Turn then go
mutable struct TurnThenGo{P<:RoombaCPOMDP} <: LowLevelPolicy
    problem::P
    turn_steps::Int # number of turn steps to take (+ for ccw, - for cw)
    max_std::Vector{Float64} # [std_x,std_y] below which to terminate
    max_steps::Int # number of option steps above which to terminate
    v::Float64
    om::Float64
    steps::Int # number of steps taken in option
    bump::Bool
end
TurnThenGo(cpomdp::P; turn_steps = 0, max_std = [0.3, 0.3], max_steps = 10, v=3., om=π/2) where {P<:RoombaCPOMDP} = TurnThenGo{P}(cpomdp, turn_steps, max_std, max_steps, v, om, 0, false)
function POMDPTools.action_info(p::TurnThenGo, b)    
    if p.steps < abs(p.turn_steps) 
        action = RoombaAct(0,sign(p.turn_steps)*p.om)
    else
        action = RoombaAct(p.v,0)
    end
    p.steps += 1
    return action, (;)
end
function terminate(p::TurnThenGo, b) 
    std_met = all(stats(b)[2][1:2] .<= p.max_std)
    max_steps = (p.steps >= p.max_steps)
    bump_met = p.bump && (p.steps >= abs(p.turn_steps)+3) # had enough time to turn away from wall
    return Deterministic(std_met || max_steps || bump_met)
end 
function reset!(p::TurnThenGo) 
    p.steps = 0
    p.bump = false
end
function update_option!(p::TurnThenGo, b, a, o)
    o && (p.bump = true)
end
node_tag(p::TurnThenGo) = "TurnThenGo($(p.turn_steps), $(p.max_steps))"

# Greedy Go To Goal Option
mutable struct GreedyGoToGoal{P<:RoombaCPOMDP} <: LowLevelPolicy
    problem::P
    max_std::Vector # [std_x,std_y] above which to terminate
    max_steps::Int # number of option steps above which to terminate
    samples::Int # number of samples of belief to take for distance calculations
    steps::Int # number of steps taken in option
    steps_at_wall::Int
end
GreedyGoToGoal(cpomdp::P; max_std = [20.0, 20.0], max_steps = 40, samples=1) where {
    P<:RoombaCPOMDP} = GreedyGoToGoal{P}(cpomdp, max_std, max_steps, samples, 0, 0)
node_tag(p::GreedyGoToGoal) = "GreedyGoToGoal($(p.max_steps))"

# Safe Go To Goal Option
mutable struct SafeGoToGoal{P<:RoombaCPOMDP} <: LowLevelPolicy
    problem::P
    barrier_penalty::Float64 # factor by which to penalize avoid region distances
    max_std::Vector # [std_x,std_y] above which to terminate
    max_steps::Int # number of option steps above which to terminate
    samples::Int # number of samples of belief to take for distance calculations  
    steps::Int # number of steps taken in option
    steps_at_wall::Int
end
SafeGoToGoal(cpomdp::P; max_std = [10.0, 10.0], max_steps = 40, barrier_penalty=10., samples=1) where {
    P<:RoombaCPOMDP} = SafeGoToGoal{P}(cpomdp, barrier_penalty, max_std, max_steps, samples, 0, 0)
node_tag(p::SafeGoToGoal) = "SafeGoToGoal($(p.max_steps))"

# go_to_goal termination, reseting, and updating
const CRoombaGoToGoal = Union{GreedyGoToGoal,SafeGoToGoal}
function POMDPTools.action_info(p::CRoombaGoToGoal, b)
    if p.samples == 1
        s = stats(b)[1]
        ss = [RoombaState(s..., 0)] # take mean state
    else
        ss = [rand(b) for i in 1:p.samples] # take random states
    end
    action, distance = navigate(p, ss)
    p.steps += 1
    return action, (;distance=distance)
end
function terminate(p::CRoombaGoToGoal, b)
    above_max_std = any(stats(b)[2][1:2] .>= p.max_std)
    above_max_steps = p.steps >= p.max_steps
    stuck_at_wall = p.steps > 5 && p.steps_at_wall > 3
    @infiltrate false # stuck_at_wall
    return Deterministic( above_max_std || above_max_steps || stuck_at_wall)
end
function reset!(p::CRoombaGoToGoal)
    p.steps = 0
    p.steps_at_wall = 0
end
function update_option!(p::CRoombaGoToGoal, b, a, o)
    if (o isa Bool) 
        o ? p.steps_at_wall += 1 : p.steps_at_wall = 0
    end
end

# navigation helpers
distance(s::Vector{Float64}, goal) = norm(s - goal)
distance(s::RoombaState, goal) = distance(s[1:2], goal)

function distance_function(p::RoombaCPOMDP, s)
    """Compute the distance to the goal in the base POMDP"""
    @assert p.pomdp.mdp.config==4 "distance for wrong config"
    x, y, th = s[1:3]
    # come up with a distance map to goal
    # method 0 uses waypoints:
    waypoints = [[-15., 0], [12,0]]
    dist = 0.
    goal_vec = get_goal_xy(p.pomdp) + [1,-2.]
    if x >= waypoints[2][1] || distance(s,waypoints[2]) <= 4
        # close to goal
        dist += distance(s, goal_vec)
        #dist += abs(wrap_to_pi(th+π/2))/π # get it to turn down
    else
        # past waypoint 2
        dist += distance(waypoints[2], goal_vec)
        if x >= waypoints[1][1] || distance(s,waypoints[1]) <= 3
            # between waypoints 1 and 2
            dist += distance(s, waypoints[2])
            #dist += abs(wrap_to_pi(th))/π # get it to turn right
        else 
            # before waypoint 1
            dist += distance(waypoints[1], waypoints[2])
            dist += distance(s, waypoints[1])
            #dist += abs(wrap_to_pi(th-π/4))/π # get it to turn upish
        end        
    end
    return dist
end
distance_function(p::GreedyGoToGoal, s::RoombaState) = distance_function(p.problem, s)
distance_function(p::SafeGoToGoal, s::RoombaState) = distance_function(p.problem, s) + p.barrier_penalty * in_avoid_region(p.problem, s)

function navigate(p::CRoombaGoToGoal, ss::Vector{RoombaState})
    """
    Compute the action that minimizes the distance from the mean next state to the goal
    ss is a vector of roomba states to average over
    """
    best_action = RoombaAct(0, 0)
    best_dist = Inf
    for a in actions(p.problem)
        sps = [rand(transition(p.problem, s, a)) for s in ss]  # not quite random, in fact deterministic
        distances = [distance_function(p,s) for s in sps]
        dist = Statistics.mean(distances)
        if dist < best_dist
            best_dist = dist
            best_action = a
        end
    end
    return best_action, best_dist
end

# Spin Option
mutable struct Spin{P<:RoombaCPOMDP} <: LowLevelPolicy
    problem::P
    max_std::Vector # [std_x,std_y] below which to terminate
    max_steps::Int # number of option steps above which to terminate
    om::Float64
    steps::Int # number of steps taken in option
end
Spin(cpomdp::P; max_std = [0.3, 0.3], max_steps = 10, om=π/2) where {P<:RoombaCPOMDP} = Spin{P}(cpomdp, max_std, max_steps, om, 0)
function POMDPTools.action_info(p::Spin, b)
    action = RoombaAct(0,p.om)
    p.steps += 1
    return action, (;)
end
terminate(p::Spin, b) = Deterministic(all(stats(b)[2][1:2] .<= p.max_std) || p.steps >= p.max_steps)
reset!(p::Spin) = (p.steps = 0)
node_tag(p::Spin) = "Spin($(p.max_steps),$(p.max_std))"

mutable struct BigSpin{P<:RoombaCPOMDP} <: LowLevelPolicy
    problem::P
    turn_every::Int # steps to turn on
    max_std::Vector # [std_x,std_y] below which to terminate
    max_steps::Int # number of option steps above which to terminate
    v::Float64
    om::Float64
    steps::Int # number of steps taken in option
end
BigSpin(cpomdp::P; turn_every=3, max_std = [0.3, 0.3], max_steps = 10, v=3., om=π/2) where {P<:RoombaCPOMDP} = BigSpin{P}(cpomdp, turn_every, max_std, max_steps, v, om, 0)
function POMDPTools.action_info(p::BigSpin, b)
    speed = p.v
    turn = p.steps % p.turn_every == 0 ? p.om : 0.
    action = RoombaAct(speed,turn)
    p.steps += 1
    return action, (;)
end
terminate(p::BigSpin, b) = Deterministic(all(stats(b)[2][1:2] .<= p.max_std) || p.steps >= p.max_steps)
reset!(p::BigSpin) = (p.steps = 0)
node_tag(p::BigSpin) = "BigSpin($(p.turn_every),$(p.max_steps),$(p.max_std))"


### Utilities

isterminal_fraction(c::RoombaCPOMDP, b) = mean([isterminal(c,s) for s in particles(b)])


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

function navigate2Dsafe(problem::RoombaCPOMDP, ss::Vector{RoombaState}, goal::Vector{Float64})
    s = ss[1]
    best_action = RoombaAct(0, 0)
    best_dist = Inf
    for a in actions(problem)
        new_s = rand(transition(problem, s, a))
        safe_state = !in_avoid_region(problem,s)
        dist = norm([new_s.x, new_s.y] - goal)
        if dist < best_dist && safe_state
            best_dist = dist
            best_action = a
        end
    end
    return best_action
end

function navigate2Drobust(problem::RoombaCPOMDP, b, goal::Vector{Float64}; n_samples=5, safe_prob=0.7)
    ss = [rand(b) for i=1:n_samples]
    best_action = RoombaAct(0, 0)
    best_dist = Inf
    for a in actions(problem)
        sps = [rand(transition(problem, s, a)) for s in ss]
        safe_states = [!in_avoid_region(problem,s) for s in sps]
        dists = [norm([sp.x, sp.y] - goal) for sp in sps]
        if mean(dists) < best_dist && mean(safe_states) > safe_prob
            best_dist = mean(dists)
            best_action = a
        end
    end
    return best_action
end

# navigate2D function greedily navigates the robot towards `goal`
function navigate2D(problem::RoombaCPOMDP, ss::Vector{RoombaState}, goal::Vector{Float64})
    s = ss[1]
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
    goal_dir = atan(goal_vec[2] - s[2], goal_vec[1] - s[1])

    # Get the direction and distance to the nearest wall
    wall_dir, _, _ = nearest_wall(room, s[1], s[2])

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
    turn = s[3] - wall_follow_dir
    turns_required = ceil(abs(turn) / (om_max*dt))

    possible_oms = [a[2] for a in actions(problem.pomdp) if a[1] == 0]
    om = possible_oms[argmin(abs.(turn .- dt .* possible_oms))]

    # Generate the action to turn in the chosen direction with zero velocity if multiple turns are required
    if turns_required <= 1
        action = RoombaAct(v_max, om)
    else
        action = RoombaAct(0, om)
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
    dist = norm([s[1], s[2]] - goal_vec)
    return action, (;goal=goal_vec, distance=dist)
end
# terminate(p::GoToGoal2D, b) = Deterministic(([stats(b)[1][1], stats(b)[1][1]] ≈ get_goal_xy(p.problem.pomdp)))
terminate(p::GoToGoal2D, b) = Deterministic(norm([stats(b)[1][1], stats(b)[1][2]] - get_goal_xy(p.problem.pomdp)) <= 15)
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
        action = RoombaAct(0, om_max)
    end
    goal_vec = get_goal_xy(p.problem.pomdp)
    dist = norm([s[1], s[2]] - goal_vec)
    return action, (;goal=goal_vec, distance=dist, std=std)
end
terminate(p::Localize2D, b) = Deterministic(all(stats(b)[2] .<= p.max_std))
node_tag(p::Localize2D) = "Localize2D($(p.max_std))"
