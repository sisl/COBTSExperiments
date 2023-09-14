using COBTSExperiments
using Plots; pgfplotsx()

num_queries = [10,20, 50, 100,200, 500, 1000,2000,5000]
nsims = 50

p = plot(
    plot(legend=false, xaxis=:log10),
    plot(legend=true, xaxis=:log10),
    layout=(2, 1),  # 2 rows, 1 column
    xlabel=["" "Number of Search Queries"],
    ylabel=["Mean Reward" "Mean Cost"],
)

function get_rc_errors(ers::Vector{LightExperimentResults})
    means = [mean(er) for er in ers]
    stds = [std(er) for er in ers]
    R = [m[1] for m in means]
    C = [m[2][1] for m in means]
    errR = [s[1]/sqrt(nsims) for s in stds]
    errC = [s[2][1]/sqrt(nsims) for s in stds]
    return R, C, errR, errC
end

# Cobets
for k in [3]
ers = [COBTSExperiments.load_ler(
    "results/sweep_queries/lightdark_cobts_$(nq)queries_$(k)options_$(nsims)sims.jld2") for nq in num_queries]
R,C,errR,errC = get_rc_errors(ers)
plot!(p[1], num_queries, R, yerr=errR, line=:auto)
plot!(p[2], num_queries, C, yerr=errC, label="COBeTS, K=$(k)", legend=true, line=:auto)
end

# CPFT
ers = [COBTSExperiments.load_ler(
    "results/sweep_queries/lightdark_cpft_$(nq)queries_$(nsims)sims.jld2") for nq in num_queries]
R,C,errR,errC = get_rc_errors(ers)
plot!(p[1], num_queries, R, yerr=errR, line=:auto)
plot!(p[2], num_queries, C, yerr=errC, label="CPFT-DPW", legend=true, line=:auto)

hline!(p[2], [0.1], color="black", label=nothing, linestyle=:dash)
annotate!(p[2], [(1000, 0.3, text("Constraint Budget", :left, 10))])

#legend!(p)

!(isdir("results/figs/")) && mkpath("results/figs/")
savefig(p,"results/figs/sweep_queries_$(nsims).tex")
savefig(p,"results/figs/sweep_queries_$(nsims).pdf")
