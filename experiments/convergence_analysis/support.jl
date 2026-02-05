#=
    Convergence Analysis - Support Functions

    Multi-run analysis utilities for testing solver convergence.
    Note: config.jl and nonlinear_lane_change modules must be included before this file.
=#

using Random: MersenneTwister
using Statistics: mean

"""
    perturb_initial_state(x0; rng, scale)

Add random perturbations to initial states.
"""
function perturb_initial_state(x0; rng, scale)
    return [x0[i] .+ scale .* randn(rng, length(x0[i])) for i in eachindex(x0)]
end

"""
    print_summary(result)

Print summary statistics for convergence analysis results.
"""
function print_summary(result)
    (; iterations, residuals, statuses, converged_count, config) = result

    println("\n" * "="^60)
    println("Convergence Analysis Summary")
    println("="^60)
    println("Configuration:")
    println("  num_runs = $(config.num_runs)")
    println("  max_iters = $(config.max_iters)")
    println("  perturb_scale = $(config.perturb_scale)")
    println()
    println("Results:")
    println("  Converged: $converged_count / $(config.num_runs)")
    println("  Iterations: min=$(minimum(iterations)), max=$(maximum(iterations)), mean=$(round(mean(iterations), digits=1))")
    println("  Final residuals: min=$(minimum(residuals)), max=$(maximum(residuals))")
    println()
    println("Status breakdown:")
    for status in unique(statuses)
        count_status = sum(statuses .== status)
        println("  $status: $count_status")
    end
    println("="^60)
end
