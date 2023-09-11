using COBTSExperiments
using Plots; pgfplotsx()

num_options = [2, 3, 4, 5] 
num_queries = [10,20, 50, 100,200, 500, 1000,2000,5000,10000,20000, 50000,100000]
nsims = 20

p = plot(
    plot(legend=true, xaxis=:log10),
    plot(legend=false, xaxis=:log10),
    layout=(2, 1),  # 2 rows, 1 column
    xlabel=["" "Number of Search Queries"],
    title=["Rs" "Cs"],
)
for k in num_options
    ers = [COBTSExperiments.load_ler(
        "results/sweep_queries/lightdark_cobts_$(nq)queries_$(k)options_$(nsims)sims.jld2") for nq in num_queries]
    means = [mean(er) for er in ers]
    stds = [std(er) for er in ers]
    y1 = [m[1] for m in means]
    y2 = [m[2][1] for m in means]

    # plot means
    plot!(p[1], num_queries, y1, label="K=$(k)", legend=true, line=:auto)
    plot!(p[2], num_queries, y2, line=:auto)
end
hline!(p[2], [0.1], color="black", linestyle=:dash)
annotate!(p[2], [(20, 0.09, text("Constraint Budget", :left, 10))])

#legend!(p)
savefig(p,"results/sweep_queries_$(nsims).tikz")
savefig(p,"results/sweep_queries_$(nsims).pdf")
