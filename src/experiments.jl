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

function run_cpomdp_simulation(p::CPOMDP, solver::Solver, 
    bu::Union{Nothing,Updater,Function}=nothing, max_steps=100;track_history::Bool=true)
    planner = solve(solver, p.cpomdp)
    if bu===nothing
        bu = POMDPs.updater(planner)
    elseif bu isa Function
        bu = bu(planner)
    end
    R = 0
    C = zeros(n_costs(p.cpomdp))
    RC = 0
    γ = 1
    hist = NamedTuple[]
    
    for (s, a, o, r, c, sp, b, ai) in stepthrough(p.cpomdp, planner, bu, "s,a,o,r,c,sp,b,action_info", max_steps=max_steps)
        
        # track fictitions augmented reward
        rc = r - p.λ⋅c
        
        R += r*γ
        C .+= c.*γ
        RC += rc*γ 

        γ *= discount(p)
        if track_history
            push!(hist, (;s, a, o, r, c, rc, sp, b, 
                tree = :tree in keys(ai) ? ai[:tree] : nothing,
                lambda = :lambda in keys(ai) ? ai[:lambda] : nothing,
                v_best = :v_best in keys(ai) ? ai[:v_best] : nothing,
                cv_best = :cv_best in keys(ai) ? ai[:cv_best] : nothing,
                v_taken = :v_taken in keys(ai) ? ai[:v_taken] : nothing,
                cv_taken = :cv_taken in keys(ai) ? ai[:cv_taken] : nothing,
                ))
        end
    end
    hist, R, C, RC
end

function run_cmdp_simulation(p::CMDP, planner::Policy, max_steps=100;track_history::Bool=true, rng::Random.AbstractRNG=Random.GLOBAL_RNG)
    R = 0
    C = zeros(n_costs(p))
    RC = 0
    γ = 1
    hist = NamedTuple[]
    
    for (s, a, r, c, sp, ai) in stepthrough(p, planner, "s,a,r,c,sp,action_info", max_steps=max_steps; rng=rng)
        R += r*γ
        C .+= c.*γ

        γ *= discount(p)
        if track_history
            push!(hist, (;s, a, r, c, sp, ai))
        end
    end
    hist, R, C
end
