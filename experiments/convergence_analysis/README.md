# Convergence Analysis

Multi-run solver robustness testing with perturbed initial states.

## Purpose

Tests the nonlinear solver's convergence reliability by:
1. Running the nonlinear lane change scenario multiple times
2. Applying random perturbations to initial states
3. Collecting convergence statistics

## Usage

```julia
include("experiments/convergence_analysis/run.jl")

# Default: 10 runs with ±0.5 perturbation
results = run_convergence_analysis(verbose=true)

# Custom parameters
results = run_convergence_analysis(
    num_runs = 20,
    perturbation_scale = 0.3,
    T = 6,
    verbose = true
)
```

## Output

Returns a named tuple with:
- `results`: Array of individual run results
- `num_converged`: Count of successful convergences
- `convergence_rate`: Fraction that converged
- `iteration_stats`: Statistics on iteration counts
- `residual_stats`: Statistics on final residuals

## Configuration

Key parameters in `config.jl`:
- `num_runs`: Number of random trials
- `perturbation_scale`: Magnitude of state perturbations
- `T`, `Δt`, `R`: Lane change scenario parameters
