using COBTSExperiments
using Plots
using Infiltrator
# using LatexStrings

nsims = 20
max_options = 32
experiment_runs = [
    #("repeat_options", collect(4:4:max_options)),
    #("goal_options", collect(4:1:max_options)),
    #("target_uncertainty", collect(4:3:max_options)),
    ("random_3actions", collect(4:4:max_options)),
    ("random_6actions", collect(4:4:max_options)),
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
        "results/lightdark_COBTS_$(expname)_$(k)options_$(nsims)sims.jld2") for k in ks]
    means = [mean(er) for er in ers]
    stds = [std(er) for er in ers]
    y1 = [m[1] for m in means]
    y2 = [m[2][1] for m in means]

    # plot means
    plot!(p[1], ks, y1, label=expname, legend=true, line=:auto)
    plot!(p[2], ks, y2, line=:auto)

end
#legend!(p)
savefig(p,"results/sweep_options_$(nsims).png")