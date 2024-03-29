abstract type AlphaSchedule end
struct ConstantAlphaSchedule <: AlphaSchedule 
    scale::Float32
end
ConstantAlphaSchedule() = ConstantAlphaSchedule(1.e-3)
alpha(sched::ConstantAlphaSchedule, ::Int) = sched.scale

struct InverseAlphaSchedule <: AlphaSchedule 
    scale::Float32
end
InverseAlphaSchedule() = InverseAlphaSchedule(1.)
alpha(sched::InverseAlphaSchedule, query::Int) = sched.scale/query

"""
Constrained Options Tree Search solver

Fields:

    depth::Int64
        Maximum rollout horizon and tree depth.
        default: 10

    exploration_constant::Float64
        Specified how much the solver should explore.
        In the UCB equation, Q + c*sqrt(log(t/N)), c is the exploration constant.
        default: 1.0

    n_iterations::Int64
        Number of iterations during each action() call.
        default: 100

    max_time::Float64
        Maximum amount of CPU time spent iterating through simulations.
        default: Inf

    k_action::Float64
    alpha_action::Float64
    k_state::Float64
    alpha_state::Float64
        These constants control the double progressive widening. A new state
        or action will be added if the number of children is less than or equal to kN^alpha.
        defaults: k:10, alpha:0.5

    keep_tree::Bool
        If true, store the tree in the planner for reuse at the next timestep (and every time it is used in the future). There is a computational cost for maintaining the state dictionary necessary for this.
        default: false

    enable_action_pw::Bool
        If true, enable progressive widening on the action space; if false just use the whole action space.
        default: true

    enable_state_pw::Bool
        If true, enable progressive widening on the state space; if false just use the single next state (for deterministic problems).
        default: true

    check_repeat_state::Bool
    check_repeat_action::Bool
        When constructing the tree, check whether a state or action has been seen before (there is a computational cost to maintaining the dictionaries necessary for this)
        default: true

    tree_in_info::Bool
        If true, return the tree in the info dict when action_info is called. False by default because it can use a lot of memory if histories are being saved.
        default: false

    rng::AbstractRNG
        Random number generator

    estimate_value::Any (rollout policy)
        Function, object, or number used to estimate the value at the leaf nodes.
        If this is a function `f`, `f(mdp, s, depth)` will be called to estimate the value (depth can be ignored).
        If this is an object `o`, `estimate_value(o, mdp, s, depth)` will be called.
        If this is a number, the value will be set to that number.
        default: RolloutEstimator(RandomSolver(rng))

    init_Q::Any
        Function, object, or number used to set the initial Q(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_Q(o, mdp, s, a)` will be called.
        If this is a number, Q will always be set to that number.
        default: 0.0

    init_N::Any
        Function, object, or number used to set the initial N(s,a) value at a new node.
        If this is a function `f`, `f(mdp, s, a)` will be called to set the value.
        If this is an object `o`, `init_N(o, mdp, s, a)` will be called.
        If this is a number, N will always be set to that number.
        default: 0

    next_action::Any
        Function or object used to choose the next action to be considered for progressive widening.
        The next action is determined based on the MDP, the state, `s`, and the current `DPWStateNode`, `snode`.
        If this is a function `f`, `f(mdp, s, snode)` will be called to set the value.
        If this is an object `o`, `next_action(o, mdp, s, snode)` will be called.
        default: RandomActionGenerator(rng)

    default_action::Any
        Function, action, or Policy used to determine the action if POMCP fails with exception `ex`.
        If this is a Function `f`, `f(pomdp, belief, ex)` will be called.
        If this is a Policy `p`, `action(p, belief)` will be called.
        If it is an object `a`, `default_action(a, pomdp, belief, ex)` will be called, and if this method is not implemented, `a` will be returned directly.
        default: `ExceptionRethrow()`

    reset_callback::Function
        Function used to reset/reinitialize the MDP to a given state `s`.
        Useful when the simulator state is not truly separate from the MDP state.
        `f(mdp, s)` will be called.
        default: `(mdp, s)->false` (optimized out)

    show_progress::Bool
        Show progress bar during simulation.
        default: false

    timer::Function:
        Timekeeping method. Search iterations ended when `timer() - start_time ≥ max_time`.
"""
@with_kw mutable struct COTSSolver <: Solver
    options::Union{Nothing,Vector{<:LowLevelPolicy}} = nothing
    enable_constraint_pw::Bool = false
    depth::Int = 10
    exploration_constant::Float64 = 1.0
    nu::Float64 = 0.0
    n_iterations::Int = 100
    max_time::Float64 = Inf
    k_action::Float64 = 10.0
    alpha_action::Float64 = 0.5
    k_state::Float64 = 10.0
    alpha_state::Float64 = 0.5
    keep_tree::Bool = false
    enable_action_pw::Bool = true
    enable_state_pw::Bool = true
    check_repeat_state::Bool = true
    check_repeat_action::Bool = true
    return_safe_action::Bool = false
    tree_in_info::Bool = false
    search_progress_info::Bool = false
    rng::AbstractRNG = Random.GLOBAL_RNG
    alpha_schedule::AlphaSchedule = InverseAlphaSchedule()
    estimate_value::Any = RolloutEstimator(RandomSolver(rng))
    init_Q::Any = 0.
    init_N::Any = 0
    init_Qc::Any = 0.
    init_λ::Union{Nothing,Vector{Float64}} = nothing
    max_clip::Union{Float64,Vector{Float64}} = Inf
    next_action::Any = RandomOptionGenerator(rng, options, enable_constraint_pw)
    default_option::Any = ExceptionRethrow()
    reset_callback::Function = (mdp, s) -> false
    show_progress::Bool = false
    timer::Function = () -> 1e-9 * time_ns()
end

#=
mutable struct StateActionStateNode
    N::Int
    R::Float64
    StateActionStateNode() = new(0,0)
end

mutable struct DPWStateActionNode{S}
    V::Dict{S,StateActionStateNode}
    N::Int
    Q::Float64
    DPWStateActionNode(N,Q) = new(Dict{S,StateActionStateNode}(), N, Q)
end

mutable struct DPWStateNode{S,A} <: AbstractStateNode
    A::Dict{A,DPWStateActionNode{S}}
    N::Int
    DPWStateNode{S,A}() where {S,A} = new(Dict{A,DPWStateActionNode{S}}(),0)
end
=#
mutable struct COTSTree{S}
    # for each state node
    total_n::Vector{Int}
    children::Vector{Vector{Int}}
    s_labels::Vector{S}
    s_lookup::Dict{S, Int}

    # for each state-action node
    n::Vector{Int}
    q::Vector{Float64}
    qc::Vector{Vector{Float64}}
    transitions::Vector{Vector{Tuple{Int,Float64,Vector{Float64},Int}}}
    a_labels::Vector{LowLevelPolicy}
    a_lookup::Dict{Tuple{Int,LowLevelPolicy}, Int}

    # for tracking transitions
    n_a_children::Vector{Int}
    unique_transitions::Set{Tuple{Int,Int}}

    function COTSTree{S}(sz::Int=1000) where {S} 
        sz = min(sz, 100_000)
        return new(sizehint!(Int[], sz),
                   sizehint!(Vector{Int}[], sz),
                   sizehint!(S[], sz),
                   Dict{S, Int}(),

                   sizehint!(Int[], sz),
                   sizehint!(Float64[], sz),
                   sizehint!(Vector{Vector{Float64}}[], sz), #qc
                   sizehint!(Vector{Tuple{Int,Float64,Vector{Float64}}}[], sz),
                   sizehint!(LowLevelPolicy[], sz),
                   Dict{Tuple{Int,LowLevelPolicy}, Int}(),

                   sizehint!(Int[], sz),
                   Set{Tuple{Int,Int}}(),
                  )
    end
end


function insert_state_node!(tree::COTSTree{S}, s::S, maintain_s_lookup=true) where {S}
    push!(tree.total_n, 0)
    push!(tree.children, Int[])
    push!(tree.s_labels, s)
    snode = length(tree.total_n)
    if maintain_s_lookup
        tree.s_lookup[s] = snode
    end
    return snode
end


function insert_action_node!(tree::COTSTree{S}, snode::Int, a::LowLevelPolicy, n0::Int, q0::Float64, qc0::Vector{Float64}, 
        maintain_a_lookup=true) where {S}
    push!(tree.n, n0)
    push!(tree.q, q0)
    push!(tree.qc, qc0)
    push!(tree.a_labels, a)
    push!(tree.transitions, Vector{Tuple{Int,Float64,Vector{Float64}}}[])
    sanode = length(tree.n)
    push!(tree.children[snode], sanode)
    push!(tree.n_a_children, 0)

    if maintain_a_lookup
        tree.a_lookup[(snode, a)] = sanode
    end
    return sanode
end

Base.isempty(tree::COTSTree) = isempty(tree.n) && isempty(tree.q)

struct COTSStateNode{S} <: AbstractStateNode
    tree::COTSTree{S}
    index::Int
end

children(n::COTSStateNode) = n.tree.children[n.index]
n_children(n::COTSStateNode) = length(children(n))
isroot(n::COTSStateNode) = n.index == 1


mutable struct COTSPlanner{P<:Union{CMDP,CPOMDP}, S, A, SE, NA, RCB, RNG} <: OptionsPolicy
    solver::COTSSolver
    mdp::P
    tree::Union{Nothing, COTSTree{S}}
    solved_estimate::SE
    next_action::NA
    reset_callback::RCB
    rng::RNG
    budget::Vector{Float64} # remaining budget for constraint search
    
    # options parameters
    _running::Union{Nothing,LowLevelPolicy} # option that is currently running
    _step_counter::Int # steps that option has been running

    # cpomdp parameters
    _lambda::Union{Nothing,Vector{Float64}} # weights for dual ascent

end

function COTSPlanner(solver::COTSSolver, mdp::P) where P<:Union{CPOMDP,CMDP}
    se = convert_estimator(solver.estimate_value, solver, mdp)
    @assert solver.options !== nothing "No options specified in COTSSolver."
    return COTSPlanner{P,
                      statetype(P),
                      actiontype(P),
                      typeof(se),
                      typeof(solver.next_action),
                      typeof(solver.reset_callback),
                      typeof(solver.rng)}(solver,
                                          mdp,
                                          nothing,
                                          se,
                                          solver.next_action,
                                          solver.reset_callback,
                                          solver.rng,
                                          costs_limit(mdp),
                                          # options parameters
                                          nothing,
                                          0, 

                                          #cpomdp parameters
                                          costs_limit(mdp),
                     )
end
CPOMDPs.rng(p::COTSPlanner) = p.rng
CPOMDPs.low_level(p::COTSPlanner) = p._running
Random.seed!(p::COTSPlanner, seed) = Random.seed!(p.rng, seed)
POMDPs.solve(solver::COTSSolver, mdp::Union{CPOMDP,CMDP}) = COTSPlanner(solver, mdp)