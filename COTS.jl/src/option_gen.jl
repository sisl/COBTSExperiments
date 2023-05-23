"""
Generate a new option when the set of options is widened.
"""
function next_option end

mutable struct RandomOptionGenerator{RNG<:AbstractRNG,OP<:LowLevelPolicy}
    rng::RNG
    options::Vector{OP}
    budget_setting::Bool
end
RandomOptionGenerator(options, budget_setting) = RandomOptionGenerator(Random.GLOBAL_RNG, options, budget_setting)

function next_option(gen::RandomOptionGenerator, mdp::Union{POMDP,MDP}, s, snode::AbstractStateNode, rem_budget::Vector{Float64})
    op = deepcopy(rand(gen.rng, gen.options))
    gen.budget_setting && set_budget!(op, rand(gen.rng,length(rem_budget)).*rem_budget)
    return op
end
next_option(f::Function, mdp::Union{POMDP,MDP}, s, snode::AbstractStateNode, rem_budget::Vector{Float64}) = f(mdp, s, snode, rem_budget)