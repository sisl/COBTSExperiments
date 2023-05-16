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
    println("hit option 2!")
    return SpillpointInjectionCPOMDP{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp, constraint_budget)
end

costs(::SpillpointInjectionCPOMDP, s, a, sp) = [sp.v_exited - s.v_exited]

Base.convert(::Type{ParticleFilters.ParticleCollection{SpillpointPOMDP.SpillpointInjectionState}}, 
    s::SpillpointPOMDP.SIRParticleBelief{SpillpointPOMDP.SpillpointInjectionState}) = s.particle_collection

costs_limit(p::SpillpointInjectionCPOMDP) = [p.constraint_budget]
n_costs(::SpillpointInjectionCPOMDP) = 1
max_volume_diff(p::SpillpointInjectionCPOMDP) = maximum(p.pomdp.injection_rates) * p.pomdp.Δt
max_reward(p::SpillpointInjectionCPOMDP) = p.pomdp.trapped_reward * max_volume_diff(p)
min_reward(p::SpillpointInjectionCPOMDP) = minimum(p.pomdp.obs_rewards)

QMDP_V(pomdp::SpillpointInjectionPOMDP, s::SpillpointInjectionState, args...) = 0.1*pomdp.trapped_reward*(
    CPOMDPExperiments.SpillpointPOMDP.trap_capacity(s.m, s.sr, lb=s.v_trapped, ub=0.3, rel_tol=1e-2, abs_tol=1e-3) - s.v_trapped)
QMDP_V(cpomdp, args...) = (QMDP_V(cpomdp.pomdp, args...), zeros(Float64, n_costs(cpomdp))) 
function QMDP_V(p::CPOMDPs.GenerativeBeliefCMDP{P}, s::ParticleFilters.ParticleCollection{S}, args...) where {P<:SpillpointInjectionCPOMDP, S<:SpillpointInjectionState}
    V = 0.
    ws = weights(s)
    for (part, w) in zip(particles(s),ws)
        V += QMDP_V(p.cpomdp, part, args...)[1] * w
    end
    V /= sum(ws)
    return (V, zeros(Float64, n_costs(p.cpomdp))) # replaces old weight_sum(particle collections) that was 1 

end