
# no computation is done in solve - the solver is just given the mdp model that it will work with
POMDPs.solve(solver::CMCTSSolver, mdp::Union{POMDP,MDP}) = CMCTSPlanner(solver, mdp)

@POMDP_require POMDPs.action(policy::AbstractCMCTSPlanner, state) begin
    @subreq simulate(policy, state, policy.solver.depth)
end

function POMDPTools.action_info(p::AbstractCMCTSPlanner, s)
    tree = plan!(p, s)
    best = best_sanode_Q(StateNode(tree, s))
    return action(best), (tree=tree,)
end

POMDPs.action(p::AbstractCMCTSPlanner, s) = first(action_info(p, s))

"""
Query the tree for a value estimate at state s. If the planner does not already have a tree, run the planner first.
"""
function POMDPs.value(planner::CMCTSPlanner, s)
    if planner.tree == nothing
        plan!(planner, s)
    end
    return value(planner.tree, s)
end

function POMDPs.value(tr::CMCTSTree, s)
    id = get(tr.state_map, s, 0)
    if id == 0
        error("State $s not present in CMCTS tree.")
    end
    return maximum(q(san) for san in children(StateNode(tr, id)))
end

function POMDPs.value(planner::CMCTSPlanner{<:Union{POMDP,MDP}, S, A}, s::S, a::A) where {S,A}
    if planner.tree == nothing
        plan!(planner, s)
    end
    return value(planner.tree, s, a)
end

function POMDPs.value(tr::CMCTSTree{S,A}, s::S, a::A) where {S,A}
    for san in children(StateNode(tr, s)) # slow search through children
        if action(san) == a
            return q(san)
        end
    end
end


"""
Build tree and store it in the planner.
"""
function plan!(planner::AbstractCMCTSPlanner, s)
    tree = build_tree(planner, s)
    planner.tree = tree
    return tree
end

function build_tree(planner::AbstractCMCTSPlanner, s)
    n_iterations = planner.solver.n_iterations
    depth = planner.solver.depth

    if planner.solver.reuse_tree
        tree = planner.tree
    else
        tree = CMCTSTree{statetype(planner.mdp), actiontype(planner.mdp)}(n_iterations)
    end

    sid = get(tree.state_map, s, 0)
    if sid == 0
        root = insert_node!(tree, planner, s)
    else
        root = StateNode(tree, sid)
    end

    timer = planner.solver.timer
    start_s = timer()
    # build the tree
    for n = 1:n_iterations
        simulate(planner, root, depth)
        if timer() - start_s >= planner.solver.max_time
            break
        end
    end
    return tree
end


function simulate(planner::AbstractCMCTSPlanner, node::StateNode, depth::Int64)
    mdp = planner.mdp
    rng = planner.rng
    s = state(node)
    tree = node.tree

    # once depth is zero return
    if isterminal(planner.mdp, s)
	return 0.0
    elseif depth == 0 
        return estimate_value(planner.solved_estimate, planner.mdp, s, depth)
    end

    # pick action using UCT
    sanode = best_sanode_UCB(node, planner.solver.exploration_constant)
    said = sanode.id

    # transition to a new state
    sp, r = @gen(:sp, :r)(mdp, s, action(sanode), rng)

    spid = get(tree.state_map, sp, 0)
    if spid == 0
        spn = insert_node!(tree, planner, sp)
        spid = spn.id
        q = r + discount(mdp) * estimate_value(planner.solved_estimate, planner.mdp, sp, depth-1)
    else
        q = r + discount(mdp) * simulate(planner, StateNode(tree, spid) , depth-1)
    end
    if planner.solver.enable_tree_vis
        record_visit!(tree, said, spid)
    end

    tree.total_n[node.id] += 1
    tree.n[said] += 1
    tree.q[said] += (q - tree.q[said]) / tree.n[said] # moving average of Q value
    return q
end

@POMDP_require simulate(planner::AbstractCMCTSPlanner, s, depth::Int64) begin
    mdp = planner.mdp
    P = typeof(mdp)
    S = statetype(P)
    A = actiontype(P)
    @req discount(::P)
    @req isterminal(::P, ::S)
    @subreq insert_node!(planner, s)
    @subreq estimate_value(planner.solved_estimate, mdp, s, depth)
    @req gen(::P, ::S, ::A, ::typeof(planner.rng)) # XXX this is not exactly right - it could be satisfied with transition
    @req isequal(::S, ::S) # for hasnode
    @req hash(::S) # for hasnode
end

function insert_node!(tree::CMCTSTree, planner::CMCTSPlanner, s)
    push!(tree.s_labels, s)
    tree.state_map[s] = length(tree.s_labels)
    push!(tree.child_ids, [])
    total_n = 0
    for a in actions(planner.mdp, s)
        n = init_N(planner.solver.init_N, planner.mdp, s, a)
        total_n += n
        push!(tree.n, n)
        push!(tree.q, init_Q(planner.solver.init_Q, planner.mdp, s, a))
        push!(tree.a_labels, a)
        push!(last(tree.child_ids), length(tree.n))
    end
    push!(tree.total_n, total_n)
    return StateNode(tree, length(tree.total_n))
end

@POMDP_require insert_node!(tree::CMCTSTree, planner::AbstractCMCTSPlanner, s) begin
    # from the StateNode constructor
    P = typeof(planner.mdp)
    A = actiontype(P)
    S = typeof(s)
    IQ = typeof(planner.solver.init_Q)
    if !(IQ <: Number) && !(IQ <: Function)
        @req init_Q(::IQ, ::P, ::S, ::A)
    end
    IN = typeof(planner.solver.init_N)
    if !(IN <: Number) && !(IN <: Function)
        @req init_N(::IN, ::P, ::S, ::A)
    end
    @req actions(::P, ::S)
    as = actions(planner.mdp, s)
    @req isequal(::S, ::S) # for tree[s]
    @req hash(::S) # for tree[s]
end

function record_visit!(tree::CMCTSTree, said::Int, spid::Int)
    vs = tree._vis_stats
    if !haskey(vs, said=>spid)
        vs[said=>spid] = 0
    end
    vs[said=>spid] += 1
end

"""
Return the best action based on the Q score
"""
function best_sanode_Q(snode::StateNode)
    best_Q = -Inf
    best = first(children(snode))
    for sanode in children(snode)
        if q(sanode) > best_Q
            best_Q = q(sanode)
            best = sanode
        end
    end
    return best
end

"""
Return the best action node based on the UCB score with exploration constant c
"""
function best_sanode_UCB(snode::StateNode, c::Float64)
    best_UCB = -Inf
    best = first(children(snode))
    sn = total_n(snode)
    for sanode in children(snode)
	
	# if sn==0, log(sn) = -Inf. We want to avoid this.
        # in most cases, if n(sanode)==0, UCB will be Inf, which is desired,
	# but if sn==1 as well, then we have 0/0, which is NaN
        if c == 0 || sn == 0 || (sn == 1 && n(sanode) == 0)
            UCB = q(sanode)
        else
            UCB = q(sanode) + c*sqrt(log(sn)/n(sanode))
        end
		
        if isnan(UCB)
            @show sn
            @show n(sanode)
            @show q(sanode)
        end
		
        @assert !isnan(UCB)
        @assert !isequal(UCB, -Inf)
		
        if UCB > best_UCB
            best_UCB = UCB
            best = sanode
        end
    end
    return best
end
