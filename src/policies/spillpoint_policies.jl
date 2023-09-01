### SpillpointCPOMDP Policies

struct SingleActionWrapper <: LowLevelPolicy
    action
end

function POMDPTools.action_info(p::SingleActionWrapper, b)
    return p.action, (;)
end

terminate(p::SingleActionWrapper, b) = Deterministic(true)

node_tag(p::SingleActionWrapper) = "SingleActionWrapper"

mutable struct InferGeology{P} <: LowLevelPolicy
    problem::P
    Vinject
    monitor_actions::Array
    actions_performed::Array
    InferGeology(cpomdp::P, Vinject, monitor_actions) where P = new{P}(cpomdp, Vinject, monitor_actions, [])
end

function CPOMDPs.reset!(p::InferGeology)
    p.monitor_actions = [p.actions_performed..., p.monitor_actions...]
    p.actions_performed = []
end

function POMDPTools.action_info(p::InferGeology, b)
    s = rand(b)
    vinject_sofar = s.v_trapped + s.v_exited

    remaining_v = p.Vinject - vinject_sofar

    # Inject at the highest rate possible until getting close to the target injection volume
    a = nothing
    for rate in sort(p.problem.pomdp.injection_rates)
        if remaining_v < rate
            break
        else
            a = (:inject, rate)
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
    SafeFill(cpomdp::P) where P = new{P}(cpomdp, false)
end

CPOMDPs.reset!(p::SafeFill) = (p.should_stop = false)

function POMDPTools.action_info(p::SafeFill, b)
    for injection_rate in reverse(sort(p.problem.pomdp.injection_rates))
        all_good = true
        for i=1:10
            s = rand(b)
            sp, o, r = gen(p.problem, s, (:inject, injection_rate))
            if r < 0 
                all_good = false
                break
            end
        end
        if all_good
            return (:inject, injection_rate), (;)
        end
    end
    
    p.should_stop = true
    return (:inject, 0.0), (;)
end

terminate(p::SafeFill, b) = Deterministic(p.should_stop)
node_tag(p::SafeFill) = "SafeFill"