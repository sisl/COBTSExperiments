using Revise, COBTSExperiments, CPOMDPs, POMDPTools, Random, POMDPs

cpomdp = CLightDarkNew(cost_budget=0.1)
cmdp = UnderlyingCMDP(cpomdp)
rng=MersenneTwister(1234)

struct RandomNonTermPolicy <: POMDPs.Policy
    actions::Vector
    rng::AbstractRNG
end
RandomNonTermPolicy(actions::Vector;rng::AbstractRNG=GLOBAL_RNG) = RandomNonTermPolicy(actions,rng)
POMDPs.action(p::RandomNonTermPolicy, x) = rand(p.rng, p.actions)

normal_policy = RandomNonTermPolicy([-10,-5,-1,1,5,10]; rng=rng)
hist, R, C = run_cmdp_simulation(cmdp, normal_policy, 100;rng=rng)

rng=MersenneTwister(1234)
options_policy = RandomOptionsPolicy([Navigate(cmdp, goal) for goal in -50.:1:50.]; rng=rng)
hist2, R2, C2 = run_cmdp_simulation(cmdp, options_policy, 100; rng=rng)

#Extraction and plotting
random_y = [record[:s].y for record in hist]
options_y = [record[:s].y for record in hist2]
options_new=[record[:ai][:high][:new_option] for record in hist2]
options_goal=[record[:ai][:low][:goal] for record in hist2]
x = 1:100
using Plots
plt = plot(x,random_y,label="Random LL Policy")
plot!(x,options_y,label="Random Options Policy")
plot!(x,options_goal,label="Random Options Goals")
scatter!(x[options_new],options_y[options_new],label="")
savefig(plt,"scratch/figs/options_test.png")

