"""
    BeliefCMCTSSolver(CMCTS_solver, updater)

The belief CMCTS solver solves POMDPs by modeling them as an MDP on the belief space. The `updater` is used to update the belief as part of the belief MDP generative model.

Example:

    using ParticleFilters
    using POMDPModels
    using CMCTS

    pomdp = BabyPOMDP()
    updater = SIRParticleFilter(pomdp, 1000)

    solver = BeliefCMCTSSolver(CDPWSolver(), updater)
    planner = solve(solver, pomdp)
    
    external_updater = CMCTSBudgetUpdateWrapper(updater, planner)
    simulate(HistoryRecorder(max_steps=10), pomdp, planner, external_updater)

    # alternatively, can use the default CMCTSBudgetUpdateWrapper(updater, planner) by leaving out 
    # external_updater in simulate()
"""
mutable struct BeliefCMCTSSolver <: AbstractCMCTSSolver
    solver::AbstractCMCTSSolver
    search_updater::Updater
end

# will return an AbstractCMCTSPlanner with a generative belief cmdp
# that gbcmdp handles b', r, c = update(updater, b,a,o) internally for search
function POMDPs.solve(sol::BeliefCMCTSSolver, p::POMDP)
    bmdp = GenerativeBeliefCMDP(p, sol.search_updater)
    return solve(sol.solver, bmdp)
end

struct CMCTSBudgetUpdateWrapper <: Updater
    updater::Updater
    planner::AbstractCMCTSPlanner
end

function update(up::CMCTSBudgetUpdateWrapper, b, a, o)
    if up.planner.tree != nothing
        up.planner.budget = (up.planner.budget - up.planner._cost_mem)/discount(up.planner.mdp)
        up.planner.budget = max.(0, up.planner.budget) .+ eps(Float64)
    end
    return update(up.updater, b, a, o)
end

initialize_belief(bu::CMCTSBudgetUpdateWrapper, dist) = initialize_belief(bu.updater, dist)

function updater(p::AbstractCMCTSPlanner)
    P = typeof(p.mdp)
    @assert P <: GenerativeBeliefCMDP "updater called on a AbstractCMCTSPlanner without an underlying BeliefMDP"
    return CMCTSBudgetUpdateWrapper(p.mdp.updater, p)
    # XXX It would be better to automatically use an SIRParticleFilter if possible
    # if !@implemented ParticleFilters.obs_weight(::P, ::S, ::A, ::S, ::O)
    #     return UnweightedParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
    # end
    # return SIRParticleFilter(p.problem, p.solver.tree_queries, rng=p.rng)
end