function generate_gif(p::POMDP, s, fname::String)
    try
        sim = GifSimulator(filename=fname, max_steps=30)
        simulate(sim, p, s)
    catch err
        println("Simulation $(fname) failed")
    end
end

function step_through(p::POMDP, planner::Policy, max_steps=100)
    for (s, a, o, r) in stepthrough(p, planner, "s,a,o,r", max_steps=max_steps)
        print("State: $s, ")
        print("Action: $a, ")
        print("Observation: $o, ")
        println("Reward: $r.")
    end
end

function step_through(p::CPOMDP, planner::Policy, max_steps=100)
    #@infiltrate
    for (s, a, o, r, c) in stepthrough(p, planner, "s,a,o,r,c", max_steps=max_steps)
        print("State: $s, ")
        print("Action: $a, ")
        print("Observation: $o, ")
        println("Reward: $r, ")
        println("Cost: $c.")
    end
end


function get_tree(planner)
    if hasproperty(planner, :tree)
        return planner.tree
    elseif hasproperty(planner, :_tree)
        return planner._tree
    else
        @error "Can't find tree for planner of type $(typeof(planner))"
    end
end

function run_cpomdp_simulation(p::CPOMDP, planner::Policy, 
    updater::Union{Nothing,Updater,Function}=nothing, max_steps=100;
    track_history::Bool=true, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    
    if updater===nothing
        updater = POMDPs.updater(planner)
    elseif updater isa Function
        updater = updater(planner)
    end
    R = 0
    C = zeros(n_costs(p.cpomdp))
    γ = 1
    hist = NamedTuple[]
    
    for (s, a, o, r, c, sp, b, ai) in stepthrough(p.cpomdp, planner, updater, "s,a,o,r,c,sp,b,action_info", max_steps=max_steps; rng=rng)
        
        # track fictitions augmented reward
        
        R += r*γ
        C .+= c.*γ

        γ *= discount(p)
        if track_history
            si = :select in keys(ai) ? ai[:select] : ai
            push!(hist, merge((;s, a, o, r, c, sp, b, ai), search_info(si))) 
        end
    end
    hist, R, C
end

function run_cmdp_simulation(p::CMDP, planner::Policy, max_steps=100; track_history::Bool=true, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    R = 0
    C = zeros(n_costs(p))
    γ = 1
    hist = NamedTuple[]
    
    for (s, a, r, c, sp, ai) in stepthrough(p, planner, "s,a,r,c,sp,action_info", max_steps=max_steps; rng=rng)
        R += r*γ
        C .+= c.*γ

        γ *= discount(p)
        if track_history
            si = :select in keys(ai) ? ai[:select] : ai
            push!(hist, merge((;s, a, r, c, sp, ai), search_info(si))) 
        end
    end
    hist, R, C
end

function search_info(si) 
    add_keys = (:tree, :lambda, :v_best, :cv_best, :v_taken, :cv_taken)
    si_keys = keys(si)
    NamedTuple{add_keys}([k in si_keys ? si[k] : nothing for k in add_keys])
end