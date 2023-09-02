
### RoombaCPOMDP Policies

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
