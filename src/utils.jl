### Experiment Results

mutable struct LightExperimentResults
    Rs::Vector{Float64}
    Cs::Vector{Vector{Float64}}
    RCs::Vector{Float64}
end

LightExperimentResults(num::Int) = LightExperimentResults(
    Array{Float64}(undef,num),
    Array{Vector{Float64}}(undef,num),
    Array{Float64}(undef,num),
    )

Base.getindex(X::LightExperimentResults, i::Int)	= (X.Rs[i],X.Cs[i],X.RCs[i])

function Base.setindex!(X::LightExperimentResults, v::Tuple{Vector{NamedTuple},Float64,Vector{Float64},Float64}, i::Int)
    X.Rs[i] = v[2]
    X.Cs[i] = v[3]
    X.RCs[i] = v[4]
end

mutable struct ExperimentResults
    hists::Vector{Vector{NamedTuple}}
    Rs::Vector{Float64}
    Cs::Vector{Vector{Float64}}
    RCs::Vector{Float64}
end

ExperimentResults(num::Int) = ExperimentResults(
    Array{Vector{NamedTuple}}(undef,num),
    Array{Float64}(undef,num),
    Array{Vector{Float64}}(undef,num),
    Array{Float64}(undef,num),
    )

Base.getindex(X::ExperimentResults, i::Int)	= (X.hists[i],X.Rs[i],X.Cs[i],X.RCs[i])

function Base.setindex!(X::ExperimentResults, v::Tuple{Vector{NamedTuple},Float64,Vector{Float64},Float64}, i::Int)
    X.hists[i] = v[1]
    X.Rs[i] = v[2]
    X.Cs[i] = v[3]
    X.RCs[i] = v[4]
end

mean(er::Union{LightExperimentResults,ExperimentResults}) = Statistics.mean(er.Rs), Statistics.mean(er.Cs), Statistics.mean(er.RCs)

function std(er::Union{LightExperimentResults,ExperimentResults};corrected::Bool=false)
    stdR = Statistics.std(er.Rs;corrected=corrected)
    stdC = Statistics.std(er.Cs;corrected=corrected)
    stdRC = Statistics.std(er.RCs;corrected=corrected)
    return stdR, stdC, stdRC
end

function print_and_save(er::Union{LightExperimentResults,ExperimentResults}, fileloc::String)
    l = length(er.Rs)
    mR, mC, mRC = mean(er)
    stdR, stdC, stdRC = std(er)
    println("R: $(mR) pm $(stdR ./ l)")
    println("C: $(mC) pm $(stdC ./ l)")
    println("RC: $(mRC) pm $(stdRC ./ l)")
    d = Dict(
        "R"=>er.Rs, "C"=> er.Cs, "RCs"=>er.RCs
    )
    FileIO.save(fileloc,d)
end

function load_and_print(fileloc::String)
    d = load(fileloc)
    er = LightExperimentResults(d["R"], d["C"], d["RCs"])
    l = length(er.Rs)
    mR, mC, mRC = mean(er)
    stdR, stdC, stdRC = std(er)
    println("R: $(mR) pm $(stdR ./ l)")
    println("C: $(mC) pm $(stdC ./ l)")
    println("RC: $(mRC) pm $(stdRC ./ l)")
    return er
end

### Other utils 

function plot_lightdark_beliefs(hist::Vector{NamedTuple},saveloc::Union{String,Nothing}=nothing )
    states = [h[:s].y for h in hist]
    beliefs = [h[:b] for h in hist]

    xpts = []
    ypts = []
    max_particles = 100
    for i=1:length(beliefs) 
        count = 0
        for s in beliefs[i].particles
            push!(xpts, i)
            push!(ypts, s.y)
            count += 1
            if count > max_particles
                break
            end
        end
    end

    scatter(xpts, ypts)
    scatter!(1:length(states), states)
    if !(saveloc == nothing)
        dir = dirname(saveloc)
        !isdir(dir) && mkpath(dir)
        savefig(saveloc)
    end
end

zero_V(p::POMDP, args...) = 0.
zero_V(p::CPOMDP, args...) = (0.0, zeros(Float64, n_costs(p)))
zero_V(p::MDP, args...) = 0.
zero_V(p::CMDP, args...) = (0.0, zeros(Float64, n_costs(p)))
QMDP_V(args...) = zero_V(args...) #default
function QMDP_V(p::SoftConstraintPOMDPWrapper, args...) 
    V, C = QMDP_V(p.cpomdp, args...)
    return V - λ⋅C
end


### 
struct SearchProgress
    v_best::Vector{Float64}
    cv_best::Vector{Float64}
    v_taken::Vector{Float64}
    cv_taken::Vector{Float64}
    lambda::Vector{Float64}
end
    
SearchProgress(search_info::NamedTuple) = SearchProgress(
    search_info[:v_best],
    [c[1] for c in search_info[:cv_best]],
    search_info[:v_taken],
    [c[1] for c in search_info[:cv_taken]],
    [c[1] for c in search_info[:lambda]],
)

# defaultclipping 

max_clip(cpomdp) = (max_reward(cpomdp) - min_reward(cpomdp))/(1-discount(cpomdp)) ./ costs_limit(cpomdp)