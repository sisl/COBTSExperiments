"""
CMCTS solver type

Fields:

    n_iterations::Int64
        Number of iterations during each action() call.
        default: 100

    max_time::Float64
        Maximum amount of CPU time spent iterating through simulations.
        default: Inf

    depth::Int64:
        Maximum rollout horizon and tree depth.
        default: 10

    exploration_constant::Float64:
        Specifies how much the solver should explore.
        In the UCB equation, Q + c*sqrt(log(t/N)), c is the exploration constant.
        default: 1.0

    rng::AbstractRNG:
        Random number generator

    estimate_value::Any (rollout policy)
        Function, object, or number used to estimate the value at the leaf nodes.
        If this is a function `f`, `f(mdp, s, remaining_depth)` will be called to estimate the value (remaining_depth can be ignored).
        If this is an object `o`, `estimate_value(o, mdp, s, remaining_depth)` will be called.
        If this is a number, the value will be set to that number
        default: RolloutEstimator(RandomSolver(rng); max_depth=50, eps=nothing)

    init_Q::Any
        Function, object, or number used to set the initial Q(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_Q(o, mdp, s, a)` will be called.
        If this is a number, Q will be set to that number
        default: 0.0

    init_N::Any
        Function, object, or number used to set the initial N(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_N(o, mdp, s, a)` will be called.
        If this is a number, N will be set to that number
        default: 0

    reuse_tree::Bool:
        If this is true, the tree information is re-used for calculating the next plan.
        Of course, clear_tree! can always be called to override this.
        default: false

    enable_tree_vis::Bool:
        If this is true, extra information needed for tree visualization will
        be recorded. If it is false, the tree cannot be visualized.
        default: false

    timer::Function:
        Timekeeping method. Search iterations ended when `timer() - start_time ≥ max_time`.
"""
mutable struct CMCTSSolver <: AbstractCMCTSSolver
    n_iterations::Int64
    max_time::Float64
    depth::Int64
    exploration_constant::Float64
    rng::AbstractRNG
    estimate_value::Any
    init_Q::Any
    init_N::Any
    reuse_tree::Bool
    enable_tree_vis::Bool
    timer::Function
end

"""
    CMCTSSolver()

Use keyword arguments to specify values for the fields.
"""
function CMCTSSolver(;n_iterations::Int64=100,
                     max_time::Float64=Inf,
                     depth::Int64=10,
                     exploration_constant::Float64=1.0,
                     rng=Random.GLOBAL_RNG,
                     estimate_value=RolloutEstimator(RandomSolver(rng)),
                     init_Q=0.0,
                     init_N=0,
                     reuse_tree::Bool=false,
                     enable_tree_vis::Bool=false,
                     timer=() -> 1e-9 * time_ns())
    return CMCTSSolver(n_iterations, max_time, depth, exploration_constant, rng, estimate_value, init_Q, init_N,
                      reuse_tree, enable_tree_vis, timer)
end

mutable struct CMCTSTree{S,A}
    state_map::Dict{S,Int}

    # these vectors have one entry for each state node
    child_ids::Vector{Vector{Int}}
    total_n::Vector{Int}
    s_labels::Vector{S}

    # these vectors have one entry for each action node
    n::Vector{Int}
    q::Vector{Float64}
    a_labels::Vector{A}

    _vis_stats::Union{Nothing, Dict{Pair{Int,Int}, Int}} # maps (said=>sid)=>number of transitions. THIS MAY CHANGE IN THE FUTURE

    function CMCTSTree{S,A}(sz::Int=1000) where {S,A}
        sz = min(sz, 100_000)

        return new(Dict{S, Int}(),

                   sizehint!(Vector{Int}[], sz),
                   sizehint!(Int[], sz),
                   sizehint!(S[], sz),

                   sizehint!(Int[], sz),
                   sizehint!(Float64[], sz),
                   sizehint!(A[], sz),
                   Dict{Pair{Int,Int},Int}()
                  )
    end
end

Base.isempty(t::CMCTSTree) = isempty(t.state_map)
state_nodes(t::CMCTSTree) = (StateNode(t, id) for id in 1:length(t.total_n))

struct StateNode{S,A}
    tree::CMCTSTree{S,A}
    id::Int
end
StateNode(tree::CMCTSTree{S}, s::S) where S = StateNode(tree, tree.state_map[s])

"""
    get_state_node(tree::CMCTSTree, s)

Return the StateNode in the tree corresponding to s.
"""
get_state_node(tree::CMCTSTree, s) = StateNode(tree, s)


# accessors for state nodes
@inline state(n::StateNode) = n.tree.s_labels[n.id]
@inline total_n(n::StateNode) = n.tree.total_n[n.id]
@inline child_ids(n::StateNode) = n.tree.child_ids[n.id]
@inline children(n::StateNode) = (ActionNode(n.tree, id) for id in child_ids(n))

struct ActionNode{S,A}
    tree::CMCTSTree{S,A}
    id::Int
end

# accessors for action nodes
@inline POMDPs.action(n::ActionNode) = n.tree.a_labels[n.id]
@inline n(n::ActionNode) = n.tree.n[n.id]
@inline q(n::ActionNode) = n.tree.q[n.id]


mutable struct CMCTSPlanner{P<:Union{MDP,POMDP}, S, A, SE, RNG} <: AbstractCMCTSPlanner{P}
	solver::CMCTSSolver # containts the solver parameters
	mdp::P # model
    tree::Union{Nothing,CMCTSTree{S,A}} # the search tree
    solved_estimate::SE
    rng::RNG
end

function CMCTSPlanner(solver::CMCTSSolver, mdp::Union{POMDP,MDP})
    # tree = Dict{statetype(mdp), StateNode{actiontype(mdp)}}()
    tree = CMCTSTree{statetype(mdp), actiontype(mdp)}(solver.n_iterations)
    se = convert_estimator(solver.estimate_value, solver, mdp)
    return CMCTSPlanner(solver, mdp, tree, se, solver.rng)
end


"""
Delete existing decision tree.
"""
function clear_tree!(p::CMCTSPlanner{S,A}) where {S,A} p.tree = nothing end

"""
    get_state_node(tree::CMCTSTree, s, planner::CMCTSPlanner)

Return the StateNode in the tree corresponding to s. If there is no such node, add it using the planner.
"""
function get_state_node(tree::CMCTSTree, s, planner::CMCTSPlanner)
    if haskey(tree.state_map, s)
        return StateNode(tree, s)
    else
        return insert_node!(tree, planner, s)
    end
end

