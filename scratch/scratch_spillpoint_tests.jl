using CPOMCPPlusExperiments
using Plots
using D3Trees
using Random

kwargs = Dict(:tree_queries=>100000, 
        :k_observation => 0.1,
        :alpha_action => 01/50)
c = 100.0 # 250
nu = 0.0
λ_test = [1.]


# basic pomcpow - lambda 0
pomdp = SoftConstraintPOMDPWrapper(CLightDark1D())
solver = CPOMCPPlusExperiments.POMCPOWSolver(;kwargs..., 
    criterion=CPOMCPPlusExperiments.POMCPOW.MaxUCB(c), 
    estimate_value=0.0,
    tree_in_info=true)
hist1, R1, C1, RC1 = run_pomdp_simulation(pomdp, solver)
R1
C1[1]
RC1
plot_lightdark_beliefs(hist1,"belief_l0.png")
inchrome(D3Tree(hist1[1][:tree]))
inchrome(D3Tree(hist1[5][:tree]))

# augmented pomcpow - lambda given
pomdp = SoftConstraintPOMDPWrapper(CLightDark1D();λ=λ_test)
solver = CPOMCPPlusExperiments.POMCPOWSolver(;kwargs..., 
    criterion=CPOMCPPlusExperiments.POMCPOW.MaxUCB(c), 
    estimate_value=0.0,
    tree_in_info=true)
hist2, R2, C2, RC2 = run_pomdp_simulation(pomdp, solver)
R2
C2[1]
RC2
plot_lightdark_beliefs(hist2,"belief_lgiven.png")

inchrome(D3Tree(hist2[1][:tree]))

# cpomcpow - fixed budget


kwargs = Dict(:tree_queries=>100000, 
        :k_observation => 0.1,
        :alpha_action => 01/100,
        :max_depth => 10)
c = 250.0 # 250
nu = 0.0
λ_test = [1.]
cpomdp = SoftConstraintPOMDPWrapper(CLightDark1D();λ=λ_test)
solver = CPOMCPPlusExperiments.CPOMCPOWSolver(;kwargs..., 
    criterion=CPOMCPPlusExperiments.CPOMCPOW.MaxCUCB(c, nu), 
    estimate_value=QMDP_V,
    tree_in_info=true,
    search_progress_info=true)

hist3, R3, C3, RC3 = run_cpomdp_simulation(cpomdp, solver)
R3
C3[1]
RC3
plot_lightdark_beliefs(hist3,"belief_constrained.png")

inchrome(D3Tree(hist3[1][:tree]))


extract_search_progress(search_info::NamedTuple) = (
    search_info[:v_best],
    [c[1] for c in search_info[:cv_best]],
    search_info[:v_taken],
    [c[1] for c in search_info[:cv_taken]],
    [c[1] for c in search_info[:lambda]],
)

v_best, cv_best, v_taken, cv_taken, lambdas = extract_search_progress(hist3[1])

plot(v_best)
plot(cv_best)
plot(v_taken)
plot(cv_taken)
plot(lambdas)



# check for working lambda ascent
kwargs[:tree_queries]=100000
cpomdp = SoftConstraintPOMDPWrapper(CLightDark1D(cost_budget=20.);λ=λ_test)
solver = CPOMCPPlusExperiments.CPOMCPOWSolver(;kwargs..., 
    criterion=CPOMCPPlusExperiments.CPOMCPOW.MaxCUCB(c, nu), 
    estimate_value=zero_V,
    tree_in_info=true,
    search_progress_info=true)
hist4, _, _, _ = run_cpomdp_simulation(cpomdp, solver)
v_best4 = hist4[1][:v_best]
cv_best4 = [c[1] for c in hist4[1][:cv_best]]
v_taken4 = hist4[1][:v_best]
cv_taken4 = [c[1] for c in hist4[1][:cv_taken]]
lambda4 = [c[1] for c in hist4[1][:lambda]]

plot(v_best4)
plot(cv_best4)
plot(v_taken4)
plot(cv_taken4)
plot(lambda4)

cpomdp2 = SoftConstraintPOMDPWrapper(CLightDark1D(cost_budget=200.);λ=λ_test)
solver2 = CPOMCPPlusExperiments.CPOMCPOWSolver(;kwargs..., 
    criterion=CPOMCPPlusExperiments.CPOMCPOW.MaxCUCB(c, nu), 
    estimate_value=QMDP_V,
    tree_in_info=true,
    search_progress_info=true)
hist5, _, _, _ = run_cpomdp_simulation(cpomdp2, solver2)
v_best5 = hist5[1][:v_best]
cv_best5 = [c[1] for c in hist5[1][:cv_best]]
v_taken5 = hist5[1][:v_best]
cv_taken5 = [c[1] for c in hist5[1][:cv_taken]]
lambda5 = [c[1] for c in hist5[1][:lambda]]

plot(v_best5)
plot(cv_best5)
plot(v_taken5)
plot(cv_taken5)
plot(lambda5)


######## multiple trials ######

nsims = 10
er_pomdp = ExperimentResults(nsims)
er_cpomdp = ExperimentResults(nsims)
pomdp = SoftConstraintPOMDPWrapper(CLightDark1D(cost_budget=20.);λ=λ_test)
for i=1:nsims
    println("i=$i")
    er_pomdp[i] = run_pomdp_simulation(pomdp, 
        CPOMCPPlusExperiments.POMCPOWSolver(;kwargs..., 
            criterion=CPOMCPPlusExperiments.POMCPOW.MaxUCB(c), 
            estimate_value=0.0,
            rng=MersenneTwister(i)))
    
    er_cpomdp[i] = run_cpomdp_simulation(pomdp, 
        CPOMCPPlusExperiments.CPOMCPOWSolver(;kwargs..., 
            criterion=CPOMCPPlusExperiments.CPOMCPOW.MaxCUCB(c, nu), 
            estimate_value=(args...)->(0.0, [0.0]),
            rng=MersenneTwister(i)))
end
Rp_m, Cp_m, RCp_m = mean(er_pomdp)
Rp_std, Cp_std, RCp_std = std(er_pomdp)
Rc_m, Cc_m, RCc_m = mean(er_cpomdp)
Rc_std, Cc_std, RCc_std = std(er_cpomdp)

### Lambda budget

lambdas = [0, 0.01, 0.1, 1, 10]
p = CLightDark1D(cost_budget=20.)
base_kwargs = Dict(:tree_queries=>10000, 
        :k_observation => 0.1,
        :alpha_action => 01/50)
c = 250.0
nu = 0.0

psolver_kwargs = Dict(:criterion=>CPOMCPPlusExperiments.POMCPOW.MaxUCB(c), 
    :estimate_value=>0.0)
csolver_kwargs = Dict(:criterion=>CPOMCPPlusExperiments.CPOMCPOW.MaxCUCB(c, nu), 
    :estimate_value=>(args...)->(0.0, [0.0]))

le = run_lambda_experiments(lambdas,
    p, 
    CPOMCPPlusExperiments.POMCPOWSolver, merge(base_kwargs,psolver_kwargs), 
    CPOMCPPlusExperiments.CPOMCPOWSolver, merge(base_kwargs,csolver_kwargs);
    nsims=5)

plot_lambdas(le;target_cost=30.,
    saveloc="ld_lambdas.png")