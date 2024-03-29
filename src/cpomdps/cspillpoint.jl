struct SpillpointInjectionCPOMDP{P<:SpillpointInjectionPOMDP,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P
    constraint_budget::Float64 # discounted total volume of exited gas allowed
end 

# Option 0, amount penalty in POMDP, constraint on steps
"""
function SpillpointInjectionCPOMDP(;pomdp::P=SpillpointInjectionPOMDP(
        exited_reward_binary=0., 
        exited_reward_amount=-1000, 
        sat_noise_std = 0.01),
    constraint_budget::Float64 = 0.1, # discounted number of steps with allowed exit (aka exit probability)
    ) where {P<:SpillpointInjectionPOMDP}
    return SpillpointInjectionCPOMDP{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp, constraint_budget)
end

costs(::SpillpointInjectionCPOMDP, s, a, sp) = Float64[(sp.v_exited - s.v_exited) > eps(Float32)]
"""

# Option 1, no penalties in POMDP, constraint on steps
"""
function SpillpointInjectionCPOMDP(;pomdp::P=SpillpointInjectionPOMDP(
        exited_reward_amount=0.,
        exited_reward_binary=0., 
        sat_noise_std = 0.01),
    constraint_budget::Float64 = 0.1, 
    ) where {P<:SpillpointInjectionPOMDP}
    return SpillpointInjectionCPOMDP{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp, constraint_budget)
end

costs(::SpillpointInjectionCPOMDP, s, a, sp) = Float64[(sp.v_exited - s.v_exited) > eps(Float32)]
"""

# Option 2, no penalties in POMDP, constraint on amount

function SpillpointInjectionCPOMDP(;pomdp::P=SpillpointInjectionPOMDP(
        exited_reward_amount=0.,
        exited_reward_binary=0.,
        topsurface_std = 0.01,
        sat_noise_std = 0.01),
    constraint_budget::Float64 = 100., # Discounted amount of escaped gas allowed # FIXME
    ) where {P<:SpillpointInjectionPOMDP}
    return SpillpointInjectionCPOMDP{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp, constraint_budget)
end

CPOMDPs.costs(::SpillpointInjectionCPOMDP, s, a, sp) = [sp.v_exited - s.v_exited]

function CPOMDPs.costs(m::GenerativeBeliefCMDP{P}, b::Union{ParticleCollection, WeightedParticleBelief, SIRParticleBelief{SpillpointInjectionState}}, a) where {P <: SpillpointInjectionCPOMDP}
    ss = [rand(b) for i=1:100]
    sps = [gen(m.cpomdp, s, a).sp for s in ss]
    return Statistics.mean([costs(m.cpomdp, s, a, sp) for (s,sp) in zip(ss, sps)])
end

Base.convert(::Type{ParticleFilters.ParticleCollection{SpillpointPOMDP.SpillpointInjectionState}}, 
    s::SpillpointPOMDP.SIRParticleBelief{SpillpointPOMDP.SpillpointInjectionState}) = s.particle_collection

costs_limit(p::SpillpointInjectionCPOMDP) = [p.constraint_budget]
n_costs(::SpillpointInjectionCPOMDP) = 1
max_volume_diff(p::SpillpointInjectionCPOMDP) = maximum(p.pomdp.injection_rates) * p.pomdp.Δt
max_reward(p::SpillpointInjectionCPOMDP) = p.pomdp.trapped_reward * max_volume_diff(p)
min_reward(p::SpillpointInjectionCPOMDP) = minimum(p.pomdp.obs_rewards)

QMDP_V(pomdp::SpillpointInjectionPOMDP, s::SpillpointInjectionState, args...) = 0.1*pomdp.trapped_reward*(
    SpillpointPOMDP.trap_capacity(s.m, s.sr, lb=s.v_trapped, ub=0.3, rel_tol=1e-2, abs_tol=1e-3) - s.v_trapped)
QMDP_V(cpomdp, args...) = (QMDP_V(cpomdp.pomdp, args...), zeros(Float64, n_costs(cpomdp))) 
function QMDP_V(p::CPOMDPs.GenerativeBeliefCMDP{P}, b::Union{SIRParticleBelief{S}, ParticleFilters.ParticleCollection{S}}, args...) where {P<:SpillpointInjectionCPOMDP, S<:SpillpointInjectionState}
    remainings = []
    for s in particles(b)
        capacity = SpillpointPOMDP.trap_capacity(s.m, s.sr)
        remaining = max(0, capacity - s.v_trapped - s.v_exited)
        push!(remainings, remaining)
    end

    V = p.cpomdp.pomdp.trapped_reward*0.9*minimum(remainings)
    return (V, zeros(Float64, n_costs(p.cpomdp)))
end