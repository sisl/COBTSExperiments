function convert_estimator(ev::RolloutEstimator, solver::COTSSolver, mdp::Union{CPOMDP,CMDP})
    return MCTS.SolvedRolloutEstimator(MCTS.convert_to_policy(ev.solver, mdp), solver.rng, ev.max_depth, ev.eps)
end

function estimate_value(estimator::MCTS.SolvedRolloutEstimator, mdp::CMDP, state, remaining_depth)
    if estimator.max_depth == -1
        max_steps = remaining_depth
    else
        max_steps = estimator.max_depth
    end
    sim = ConstrainedRolloutSimulator(rng=estimator.rng, max_steps=max_steps, eps=estimator.eps)
    return POMDPs.simulate(sim, mdp, estimator.policy, state)
end

"""
    init_Qc(initializer, mdp, s, a)
Return a value to initialize Qc(s,a) to based on domain knowledge.
"""
function init_Qc end
init_Qc(f::Function, mdp::Union{CMDP,CPOMDP}, s, a) = f(mdp, s, a)
init_Qc(n::Vector{Number}, mdp::Union{CMDP,CPOMDP}, s, a) = convert.(Float64, n)
init_Qc(n::Number, mdp::Union{CMDP,CPOMDP}, s, a) = convert(Float64, n) * ones(Float64, n_costs(mdp)) # set to uniform n