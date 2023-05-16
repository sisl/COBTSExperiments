using Pkg
Pkg.activate(".")
dev_packages = [
    # Remote Packages
    # Solvers
    # PackageSpec(url="https://github.com/sisl/CPOMDPs.jl"),
    
    # Problems 
    # PackageSpec(url="https://github.com/JuliaPOMDP/RockSample.jl", rev="4f8112b975d71c59ee6b67721bb1d41c06ad1334"),
    # PackageSpec(url="https://github.com/sisl/SpillpointPOMDP.jl") #, rev="976d171d370046f9991ce47d939c9256659a3404")

    # Local Packages
    PackageSpec(url=joinpath(@__DIR__, "CMCTS.jl")),
    PackageSpec(url=joinpath(@__DIR__, "COTS.jl")),
    PackageSpec(url=joinpath(@__DIR__, "CPOMDPs.jl")),

]

ci = haskey(ENV, "CI") && ENV["CI"] == "true"

if ci
    # remove "own" package when on CI
    pop!(dev_packages)
end

# Run dev altogether
# This is important that it's run together so there
# are no "expected pacakge X to be registered" errors.
Pkg.develop(dev_packages)