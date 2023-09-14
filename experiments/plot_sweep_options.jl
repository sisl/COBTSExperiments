using COBTSExperiments
using Plots; pgfplotsx()

nsims = 50
max_options = 32
experiment_runs = [
#    ("repeat_options", [4;collect(7:4:max_options)]),
#    ("goal_options", [4;collect(7:1:max_options)]),
    ("target_uncertainty", [4;collect(7:3:max_options)]),
#    ("random_3actions", [4;collect(7:4:max_options)]),
    ("random_6actions", [4;collect(7:4:max_options)]),
]

function get_rc_errors(ers::Vector{LightExperimentResults})
    means = [mean(er) for er in ers]
    stds = [std(er) for er in ers]
    R = [m[1] for m in means]
    C = [m[2][1] for m in means]
    errR = [s[1]/sqrt(nsims) for s in stds]
    errC = [s[2][1]/sqrt(nsims) for s in stds]
    return R, C, errR, errC
end

p = plot(
    plot(legend=true),
    plot(legend=false),
    layout=(2, 1),  # 2 rows, 1 column
    xlabel=["" "Number of Options"],
    ylabel=["Mean Reward" "Mean Cost"],
)
for (expname, ks) in experiment_runs
    ers = [COBTSExperiments.load_ler(
        "results/sweep_options/lightdark_COBTS_$(expname)_$(k)options_$(nsims)sims.jld2") for k in ks]
    R,C,errR,errC = get_rc_errors(ers)

    # plot means
    plot!(p[1], ks, R, yerr=errR, label=expname, legend=true, line=:auto)
    plot!(p[2], ks, C, yerr=errC, line=:auto)
end

er_cpft = [COBTSExperiments.load_ler("results/lightdark_cpft-noheur_100sims.jld2")]
R,C,errR,errC = get_rc_errors(er_cpft)
scatter!(p[1], [7], R, yerr=errR, label="CPFT-DPW", legend=true, line=:auto)
scatter!(p[2], [7], C, yerr=errC)

er_cpomcpow = [COBTSExperiments.load_ler("results/lightdark_cpomcpow_100sims.jld2")]
R,C,errR,errC = get_rc_errors(er_cpomcpow)
scatter!(p[1], [7], R, yerr=errR, label="CPOMCPOW", legend=true, line=:auto)
scatter!(p[2], [7], C, yerr=errC)

hline!(p[2], [0.1], color="black", linestyle=:dash)
annotate!(p[2], [(20, 0.13, text("Constraint Budget", :left, 10))])
#legend!(p)

!(isdir("results/figs/")) && mkpath("results/figs/")
savefig(p,"results/figs/sweep_options_$(nsims).tex")
savefig(p,"results/figs/sweep_options_$(nsims).pdf")