"""
    COBTSSolver(COBT_solver, updater)

The COBTS solver solves POMDPs by modeling them as an MDP on the belief space. The `updater` is used to update the belief as part of the belief MDP generative model.

Example:

    using ParticleFilters
    using POMDPModels
    using CMCTS

    pomdp = BabyPOMDP()
    search_updater = SIRParticleFilter(pomdp, 300)

    solver = COBTSSolver(COTSSolver(), search_updater)
    planner = solve(solver, pomdp)
    
    external_updater = SIRParticleFilter(pomdp, 1000)
    simulate(HistoryRecorder(max_steps=10), pomdp, planner, external_updater)

    # alternatively, can use the default updater by leaving out external_updater in simulate()
"""
mutable struct COBTSSolver <: Solver
    solver::COTSSolver
    search_updater::Updater
end

# will return a COTSPlanner with a generative belief cmdp
# that gbcmdp handles b', r, c = update(updater, b,a,o) internally for search
function POMDPs.solve(sol::COBTSSolver, p::CPOMDP)
    bmdp = GenerativeBeliefCMDP(p, sol.search_updater)
    return solve(sol.solver, bmdp)
end

function updater(p::COTSPlanner)
    P = typeof(p.mdp)
    @assert P <: GenerativeBeliefCMDP "updater called on a COTSPlanner without an underlying BeliefMDP"
    return p.mdp.updater
end

#=
"""
    COTSBudgetUpdateWrapper(updater::Updater, planner::COTSPlanner)

Defines a wrapper updater that updates the cost budget in the planner in between execution steps by using results from 
the previous search.
"""
struct COTSBudgetUpdateWrapper <: Updater
    updater::Updater
    planner::COTSPlanner
end

# update the cost budgets based on estimated cost budgets in the search phase (cannot use true costs)
function update(up::COTSBudgetUpdateWrapper, b, a, o)
    if up.planner.tree !== nothing
        up.planner.budget = max.(eps(Float64), up.planner._estimated_updated_budget)
    end
    return update(up.updater, b, a, o)
end

initialize_belief(bu::COBTSBudgetUpdateWrapper, dist) = initialize_belief(bu.updater, dist)
=#