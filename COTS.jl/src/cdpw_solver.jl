dot(a::Vector,b::Vector) = sum(a .* b)

"""
Delete existing decision tree.
"""
function clear_tree!(p::CDPWPlanner)
    p.tree = nothing
end

"""
Construct an MCTSCDPW tree and choose the best action.
"""
POMDPs.action(p::CDPWPlanner, s) = first(action_info(p, s))

"""
Construct an MCTSCDPW tree and choose the best action. Also output some information.
"""
function POMDPTools.action_info(p::CDPWPlanner, s; tree_in_info=false)
    local a::actiontype(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """)
        end
        # initialize tree and state node
        S = statetype(p.mdp)
        if !(s isa S)
            s = convert(S,s)
        end
        A = actiontype(p.mdp)
        if p.solver.keep_tree && p.tree != nothing
            tree = p.tree
            if haskey(tree.s_lookup, s)
                snode = tree.s_lookup[s]
            else
                snode = insert_state_node!(tree, s, true)
            end
        else
            tree = CDPWTree{S,A}(p.solver.n_iterations)
            p.tree = tree
            snode = insert_state_node!(tree, s, p.solver.check_repeat_state)
        end

        # perform search for stochastic policy
        policy = search(p, snode, info)
        info[:policy] = policy  
        
        # perform callback and add to tree to info
        p.reset_callback(p.mdp, s) # Optional: leave the MDP in the current state.
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        # take random action from resulting best policy and adjust one-step cost memory
        a = tree.a_labels[rand(p.rng,policy)]
        tlc = map(i->get(tree.top_level_costs, i, zeros(Float64, size(p._lambda))), policy.vals)
        p._cost_mem = dot(tlc, policy.probs)
    catch ex
        a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, info
end

function search(p::CDPWPlanner, snode::Int, info::Dict)
    timer = p.solver.timer
    p.solver.show_progress ? progress = Progress(p.solver.n_iterations) : nothing
    nquery = 0
    start_s = timer()
    # p._lambda = rand(p.rng, n_costs(p.mdp)) .* p.max_clip # random initialization
    if p.solver.init_λ === nothing
        p._lambda = zeros(Float64, n_costs(p.mdp)) # start unconstrained
    else
        p._lambda = p.solver.init_λ
    end
    if p.solver.search_progress_info
        info[:lambda] = sizehint!(Vector{Float64}[p._lambda], p.solver.n_iterations)
        info[:v_best] = sizehint!(Float64[], p.solver.n_iterations)
        info[:cv_best] = sizehint!(Vector{Float64}[], p.solver.n_iterations)
        info[:v_taken] = sizehint!(Float64[], p.solver.n_iterations)
        info[:cv_taken] = sizehint!(Vector{Float64}[], p.solver.n_iterations)
    end
    
    for i = 1:p.solver.n_iterations
        nquery += 1
        simulate(p, snode, p.solver.depth) # (not 100% sure we need to make a copy of the state here)
        p.solver.show_progress ? next!(progress) : nothing
        if timer() - start_s >= p.solver.max_time
            p.solver.show_progress ? finish!(progress) : nothing
            break
        end

        # dual ascent w/ clipping
        sa = rand(p.rng, action_policy_UCB(p.tree, snode, p._lambda, 0.0, 0.0))
        p._lambda += alpha(p.solver.alpha_schedule,i) .* (p.tree.qc[sa]-p.budget)
        p._lambda = min.(max.(p._lambda, 0.), p.solver.max_clip)

        # tracking
        if p.solver.search_progress_info
            push!(info[:lambda], p._lambda)
            push!(info[:v_taken], p.tree.q[sa])
            push!(info[:cv_taken], p.tree.qc[sa])

            # get absolute best node (no lambda weights)
            max_q = -Inf
            ha_best = nothing
            for nd in p.tree.children[1]
                if p.tree.q[nd] > max_q
                    max_q = p.tree.q[nd]
                    ha_best = nd
                end
            end
            push!(info[:v_best],p.tree.q[ha_best] )
            push!(info[:cv_best],p.tree.qc[ha_best] )
        end
    end
    info[:tree_queries] = nquery
    info[:search_time] = timer() - start_s
    info[:search_time_us] = info[:search_time]*1e6

    return action_policy_UCB(p.tree, snode, p._lambda, 0., p.solver.nu)
end

"""
Return the reward for one iteration of MCTSCDPW.
"""
function simulate(dpw::CDPWPlanner, snode::Int, d::Int)
    S = statetype(dpw.mdp)
    A = actiontype(dpw.mdp)
    sol = dpw.solver
    tree = dpw.tree
    s = tree.s_labels[snode]
    dpw.reset_callback(dpw.mdp, s) # Optional: used to reset/reinitialize MDP to a given state.
    if isterminal(dpw.mdp, s)
        return 0.0, zeros(Float64, n_costs(dpw.mdp))
    elseif d == 0
        return estimate_value(dpw.solved_estimate, dpw.mdp, s, d)
    end

    # action progressive widening
    if dpw.solver.enable_action_pw
        if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
            a = next_action(dpw.next_action, dpw.mdp, s, CDPWStateNode(tree, snode)) # action generation step
            if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                n0 = init_N(sol.init_N, dpw.mdp, s, a)
                q0 = init_Q(sol.init_Q, dpw.mdp, s, a)
                qc0 = init_Qc(sol.init_Qc, dpw.mdp, s, a)
                insert_action_node!(tree, snode, a, n0, q0, qc0,
                                    sol.check_repeat_action
                                   )
                tree.total_n[snode] += n0
            end
        end
    elseif isempty(tree.children[snode])
        for a in actions(dpw.mdp, s)
            n0 = init_N(sol.init_N, dpw.mdp, s, a)
            q0 = init_Q(sol.init_Q, dpw.mdp, s, a)
            qc0 = init_Qc(sol.init_Qc, dpw.mdp, s, a)
            insert_action_node!(tree, snode, a, n0, q0, qc0,
                                false)
            tree.total_n[snode] += n0
        end
    end
    acts = action_policy_UCB(tree, snode, dpw._lambda, sol.exploration_constant, sol.nu)
    sanode = rand(dpw.rng, acts)
    a = tree.a_labels[sanode]

    # state progressive widening
    new_node = false
    if (dpw.solver.enable_state_pw && tree.n_a_children[sanode] <= sol.k_state*tree.n[sanode]^sol.alpha_state) || tree.n_a_children[sanode] == 0
        sp, r, c = @gen(:sp, :r, :c)(dpw.mdp, s, a, dpw.rng)

        if sol.check_repeat_state && haskey(tree.s_lookup, sp)
            spnode = tree.s_lookup[sp]
        else
            spnode = insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
            new_node = true
        end
        
        push!(tree.transitions[sanode], (spnode, r, c))

        if !sol.check_repeat_state
            tree.n_a_children[sanode] += 1
        elseif !((sanode,spnode) in tree.unique_transitions)
            push!(tree.unique_transitions, (sanode,spnode))
            tree.n_a_children[sanode] += 1
        end
    else
        spnode, r, c = rand(dpw.rng, tree.transitions[sanode])
    end

    if new_node
        v, cv = estimate_value(dpw.solved_estimate, dpw.mdp, sp, d-1)
    else
        v, cv = simulate(dpw, spnode, d-1)
    end
    q = r + discount(dpw.mdp) * v
    qc = c + discount(dpw.mdp) * cv 

    tree.n[sanode] += 1
    tree.total_n[snode] += 1
    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]
    tree.qc[sanode] += (qc - tree.qc[sanode])/tree.n[sanode]

    # top level cost estimator
    if d == sol.depth
        if !(sanode in keys(tree.top_level_costs))
            tree.top_level_costs[sanode] = c
        else
            tree.top_level_costs[sanode] += (c-tree.top_level_costs[sanode])/tree.n[sanode]
        end
    end

    if sol.return_best_cost
        LC = dot(dpw._lambda .+ 1e-3, qc)
        for ch in tree.children[snode]
            LC_temp = dot(dpw._lambda .+ 1e-3, tree.qc[ch])
            if LC_temp < LC
                LC = LC_temp
                qc = tree.qc[ch]
            end
        end
    end

    return q, qc
end


# return sparse categorical policy over best action node indices
function action_policy_UCB(t::CDPWTree, s::Int, lambda::Vector{Float64}, c::Float64, nu::Float64)
    # Q_lambda = Q_value - lambda'Q_c + c sqrt(log(N)/N(h,a))
    ltn = log(t.total_n[s])
    best_nodes = Int[]
    criterion_values = sizehint!(Float64[],length(t.children[s]))
    best_criterion_val = -Inf
    for node in t.children[s]
        n = t.n[node]
        if n == 0 && ltn <= 0.0
            criterion_value = t.q[node] - dot(lambda,t.qc[node])
        elseif n == 0 && t.q[node] == -Inf
            criterion_value = Inf
        else
            criterion_value = t.q[node] - dot(lambda,t.qc[node])
            if c > 0
                criterion_value += c*sqrt(ltn/n)
            end
        end
        push!(criterion_values,criterion_value)
        if criterion_value > best_criterion_val
            best_criterion_val = criterion_value
            empty!(best_nodes)
            push!(best_nodes, node)
        elseif criterion_value == best_criterion_val
            push!(best_nodes, node)
        end
    end
    
    # get next best nodes
    if nu > 0.
        val_diff = best_criterion_val .- criterion_values
        next_best_nodes = t.children[s][0 .< val_diff .< nu]
        append!(best_nodes, next_best_nodes)
    end
    
    # weigh actions
    if length(best_nodes) == 1
        weights = [1.0]
    else
        weights = solve_lp(t, best_nodes)
    end
    return SparseCat(best_nodes, weights)
end

function solve_lp(t::CDPWTree, best_nodes::Vector{Int})
    # error("Multiple CDPW best actions not implemented")
    # random for now
    return ones(Float64, length(best_nodes)) / length(best_nodes)
end
