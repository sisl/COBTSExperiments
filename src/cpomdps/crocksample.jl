struct RockSampleCPOMDP{P<:RockSamplePOMDP,S,A,O} <: ConstrainPOMDPWrapper{P,S,A,O}
    pomdp::P 
    bad_rock_budget::Float64
end

function RockSampleCPOMDP(;pomdp::P=RockSamplePOMDP(), # default 0 incorrect_r, goes into cost
    bad_rock_budget::Float64=3.,
    ) where {P<:RockSamplePOMDP}
    return RockSampleCPOMDP{P, statetype(pomdp), actiontype(pomdp), obstype(pomdp)}(pomdp,bad_rock_budget)
end

function costs(p::RockSampleCPOMDP, s::RSState, a::Int)
    c = 0.
    if RockSample.next_position(s, a)[1] > p.pomdp.map_size[1]
        return [c]
    end
    if a == RockSample.BASIC_ACTIONS_DICT[:sample] && in(s.pos, p.pomdp.rocks_positions) # sample 
        rock_ind = findfirst(isequal(s.pos), p.pomdp.rocks_positions) # slow ?
        c += s.rocks[rock_ind] ? 0. : 1.
    end
    return [c]

end

costs_limit(pomdp::RockSampleCPOMDP) = [pomdp.bad_rock_budget]
n_costs(::RockSampleCPOMDP) = 1
min_reward(p::RockSampleCPOMDP) = p.pomdp.step_penalty + min(p.pomdp.bad_rock_penalty, p.pomdp.sensor_use_penalty)
max_reward(p::RockSampleCPOMDP) = p.pomdp.step_penalty + max(p.pomdp.exit_reward, p.pomdp.good_rock_reward)

Base.iterate(pomdp::RockSampleCPOMDP, i::Int=1) = Base.iterate(pomdp.pomdp, i)
POMDPTools.render(pomdp::RockSampleCPOMDP, step;
    viz_rock_state=true,
    viz_belief=true,
    pre_act_text=""
) = POMDPTools.render(pomdp.pomdp, step; viz_rock_state=viz_rock_state,viz_belief=viz_belief,pre_act_text=pre_act_text)