### Installation Instructions

1. Install packages for development within a local environment with `julia develop.jl`

    - Develop all non-registered packages, even if not changing them.

2. Install remaining packages from `Project.toml` with `] activate .` and `] instantiate`. 

### Experiment Instructions

Can run experiments in locally installed environment with, for example, `julia --project=. experiments/run_nav.jl`. 
