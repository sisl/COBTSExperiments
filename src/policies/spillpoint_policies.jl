### SpillpointCPOMDP Policies

struct SingleActionWrapper <: LowLevelPolicy
    action
end

function POMDPTools.action_info(p::SingleActionWrapper, b)
    return p.action, (;)
end

terminate(p::SingleActionWrapper, b) = Deterministic(true)

node_tag(p::SingleActionWrapper) = string(p.action)

mutable struct InferGeology{P} <: LowLevelPolicy
    problem::P
    Vinject
    monitor_actions::Array
    actions_performed::Array
    InferGeology(cpomdp::P, monitor_actions) where P = new{P}(cpomdp, 0.0, monitor_actions, [])
end

function CPOMDPs.reset!(p::InferGeology)
    p.monitor_actions = [p.actions_performed..., p.monitor_actions...]
    p.actions_performed = []
    p.Vinject = 0.0
end

function POMDPTools.action_info(p::InferGeology, b)
    # Inject at the highest rate possible until getting close to the target injection volume
    a = nothing

    if p.Vinject == 0.0
        p.Vinject = 1.0
        # Inject the safe amount
        remainings = []
        for i=1:100
            s = rand(b)
            capacity = SpillpointPOMDP.trap_capacity(s.m, s.sr)
            remaining = max(0, capacity - s.v_trapped - s.v_exited)
            push!(remainings, remaining)
        end

        if minimum(remainings) > 0.0
            a = (:inject, 0.9*minimum(remainings)/p.problem.pomdp.Δt)
            p.Vinject = 1.0
        end
    end

    # Use each of the monitor actions to reduct uncertainty
    if isnothing(a)
        a = popfirst!(p.monitor_actions)
        push!(p.actions_performed, a)
    end

    @assert !isnothing(a)
    return a, (;)
end

terminate(p::InferGeology, b) = Deterministic(isempty(p.monitor_actions))
node_tag(p::InferGeology) = "InferGeology"


mutable struct SafeFill{P} <: LowLevelPolicy
    problem::P
    should_stop::Bool
    isterm::Bool
    SafeFill(cpomdp::P) where P = new{P}(cpomdp, false, false)
end

function CPOMDPs.reset!(p::SafeFill)
    p.should_stop = false
    p.isterm = false
end

function POMDPTools.action_info(p::SafeFill, b)
    if !p.should_stop
        remainings = []
        for i=1:100
            s = rand(b)
            capacity = SpillpointPOMDP.trap_capacity(s.m, s.sr)
            remaining = max(0, capacity - s.v_trapped - s.v_exited)
            push!(remainings, remaining)
        end
        a = (:inject, 0.9*minimum(remainings)/p.problem.pomdp.Δt)
        p.should_stop=true
        return a, (;)
    else
        p.isterm = true
        return (:stop, 0.0), (;)
    end
end

terminate(p::SafeFill, b) = Deterministic(p.isterm)
node_tag(p::SafeFill) = "SafeFill"