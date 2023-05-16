mutable struct RolloutEstimator
    solver::Union{Solver,Policy,Function} # rollout policy or solver
    max_depth::Union{Int, Nothing}
    eps::Union{Float64, Nothing}

    function RolloutEstimator(solver::Union{Solver,Policy,Function};
                              max_depth::Union{Int, Nothing}=50,
                              eps::Union{Float64, Nothing}=nothing)
        new(solver, max_depth, eps)
    end
end

mutable struct SolvedRolloutEstimator{P<:Policy, RNG<:AbstractRNG}
    policy::P
    rng::RNG
    max_depth::Union{Int, Nothing}
    eps::Union{Float64, Nothing}
end

function convert_estimator(ev::RolloutEstimator, solver::AbstractCMCTSSolver, mdp::Union{CPOMDP,CMDP})
    return SolvedRolloutEstimator(MCTS.convert_to_policy(ev.solver, mdp), solver.rng, ev.max_depth, ev.eps)
end

@POMDP_require estimate_value(estimator::SolvedRolloutEstimator, mdp::CMDP, state, remaining_depth) begin
    sim = ConstrainedRolloutSimulator(rng=estimator.rng, max_steps=estimator.max_depth, eps=estimator.eps)
    @subreq POMDPs.simulate(sim, mdp, estimator.policy, s)
end

function estimate_value(estimator::SolvedRolloutEstimator, mdp::CMDP, state, remaining_depth)
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