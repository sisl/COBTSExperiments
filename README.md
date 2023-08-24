### Installation Instructions

Experiments were run with [Julia 1.9.0](https://docs.julialang.org/en/v1.9/).

1. Install all recursive repository dependencies using [git submodules](https://git-scm.com/book/en/v2/Git-Tools-Submodules)  
    
    * With, for example, `git clone --recurse-submodules https://github.com/sisl/COBTSExperiments.git`  

2. Install packages for development within a local environment with `julia develop.jl`  
    
    * This will develop all non-registered packages, even if not changing them. This includes git submodule packages saved locally, as well as problem packages that default install to `~/.julia/dev/`. This is necessary to avoid "expected package X to be registered" errors when building from `Project.toml`...  

3. Install remaining packages from `Project.toml` with `] activate .` and `] instantiate`. 

### Experiment Instructions

Can run experiments in locally installed environment with, for example, `julia --project=. experiments/run_nav.jl`. 
