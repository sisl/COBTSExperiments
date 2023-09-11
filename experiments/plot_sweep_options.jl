using COBTSExperiments
using Plots; pgfplotsx()

nsims = 50
max_options = 40
experiment_runs = [
    ("repeat_options", [4;collect(7:4:max_options)]),
    ("goal_options", [4;collect(7:1:max_options)]),
    ("target_uncertainty", [4;collect(7:3:max_options)]),
    ("random_3actions", [4;collect(7:4:max_options)]),
    ("random_6actions", [4;collect(7:4:max_options)]),
]

p = plot(
    plot(legend=true),
    plot(legend=false),
    layout=(2, 1),  # 2 rows, 1 column
    xlabel=["" "Number of Options"],
    title=["Rs" "Cs"],
)
for (expname, ks) in experiment_runs
    ers = [COBTSExperiments.load_ler(
        "results/sweep_options/lightdark_COBTS_$(expname)_$(k)options_$(nsims)sims.jld2") for k in ks]
    means = [mean(er) for er in ers]
    stds = [std(er) for er in ers]
    y1 = [m[1] for m in means]
    y2 = [m[2][1] for m in means]

    # plot means
    plot!(p[1], ks, y1, label=expname, legend=true, line=:auto)
    plot!(p[2], ks, y2, line=:auto)
end
hline!(p[2], [0.1], color="black", linestyle=:dash)
annotate!(p[2], [(8, 0.09, text("Constraint Budget", :left, 10))])
#legend!(p)
savefig(p,"results/sweep_options_$(nsims).tex")
savefig(p,"results/sweep_options_$(nsims).pdf")