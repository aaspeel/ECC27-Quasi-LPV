# ECC27 — Quasi-LPV Control

Julia code for the paper **Safety and Stabilization of Quasi-Linear Parameter Varying Systems via Fixed-Point Theory** by Antoine ASPEEL.

Compares scheduled and common-gain feedback using JuMP, simulates the closed loop, and generates PDF figures.

## Run from terminal

With Julia installed, run from this directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. main.jl
```

This runs examples 1 and 2 and saves plots in `figures/`.

## Run from Julia REPL

Alternatively, start Julia from this directory and run in the REPL:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
include("main.jl")
```

## Files

- `control_algorithms.jl`: controller synthesis and simulation routines.
- `example_problem.jl`: three example systems.
- `plotting.jl`: plotting utilities.
- `closed_form_policy_plot.jl`: standalone scalar-policy plot; run with
  `julia --project=. closed_form_policy_plot.jl`.
