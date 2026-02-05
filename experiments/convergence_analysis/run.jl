#=
    Convergence Analysis Experiment

    Runs the nonlinear lane change example multiple times with perturbed initial
    states to validate solver robustness across different initial conditions.

    NOTE: Full iteration history tracking requires solver modifications.
    Currently tracks final metrics (iterations, residual, status) per run.
=#

using MixedHierarchyGames
using Graphs: SimpleDiGraph, add_edge!
using TrajectoryGamesBase: unflatten_trajectory
using SymbolicTracingUtils
using Random: MersenneTwister
using Statistics: mean, std
using LinearAlgebra: norm

# Include common utilities
include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
include(joinpath(@__DIR__, "..", "common", "collision_avoidance.jl"))
include(joinpath(@__DIR__, "..", "common", "trajectory_utils.jl"))

# ============================================================================
# Configuration
# ============================================================================

const DEFAULT_CONFIG = (
    R = 6.0,                    # turning radius
    T = 14,                     # time horizon
    Δt = 0.4,                   # time step
    num_runs = 11,              # number of runs with perturbed initial states
    max_iters = 200,            # max solver iterations per run
    perturb_scale = 0.1,        # perturbation scale for initial states
    seed = 1234,                # random seed for reproducibility
)

function default_initial_states(R)
    return [
        [-1.5R, R, 0.0, 2.0],      # P1 (LEADER)
        [-2.0R, R, 0.0, 2.0],      # P2 (FOLLOWER)
        [-R, 0.0, π/2, 1.523],     # P3 (LANE MERGER)
        [-2.5R, R, 0.0, 2.0],      # P4 (EXTRA PLAYER BEHIND P2)
    ]
end

# ============================================================================
# Problem Definition
# ============================================================================

function build_nonlinear_lane_change_solver(; R, T, Δt, max_iters, verbose=false)
    N = 4
    state_dim = 4   # [x, y, ψ, v]
    control_dim = 2 # [a, ω]

    # Hierarchy: P1 → P2 → P4 (P3 is Nash with all)
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)  # P1 leads P2
    add_edge!(G, 2, 4)  # P2 leads P4

    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)

    # Symbolic parameters for initial states
    θs = setup_problem_parameter_variables(fill(state_dim, N))

    # Cost functions
    function J₁(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs² = xs
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³ = xs
        (; xs,) = unflatten_trajectory(z₄, state_dim, control_dim)
        xs⁴ = xs

        control = 10 * sum(sum(u .^ 2) for u in us¹)
        collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
        velocity = sum((x¹[4] - 2.0)^2 for x¹ in xs¹)
        y_deviation = sum((x¹[2] - R)^2 for x¹ in xs¹)
        zero_heading = sum((x¹[3])^2 for x¹ in xs¹)

        control + collision + 5y_deviation + zero_heading + velocity
    end

    function J₂(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹ = xs
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³ = xs
        (; xs,) = unflatten_trajectory(z₄, state_dim, control_dim)
        xs⁴ = xs

        control = sum(sum(u .^ 2) for u in us²)
        collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
        velocity = sum((x²[4] - 2.0)^2 for x² in xs²)
        y_deviation = sum((x²[2] - R)^2 for x² in xs²)
        zero_heading = sum((x²[3])^2 for x² in xs²)

        control + collision + 5y_deviation + zero_heading + velocity
    end

    function J₃(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹ = xs
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs² = xs
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs,) = unflatten_trajectory(z₄, state_dim, control_dim)
        xs⁴ = xs

        tracking = 10 * sum((sum(x³[1:2] .^ 2) - R^2)^2 for x³ in xs³[2:div(T, 2)])
        control = sum(sum(u³ .^ 2) for u³ in us³)
        collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
        velocity = sum((x³[4] - 2.0)^2 for x³ in xs³)
        y_deviation = sum((x³[2] - R)^2 for x³ in xs³[div(T, 2):T])
        zero_heading = sum((x³[3])^2 for x³ in xs³[div(T, 2):T])

        tracking + control + collision + 5y_deviation + zero_heading + velocity
    end

    function J₄(z₁, z₂, z₃, z₄; θ=nothing)
        (; xs,) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹ = xs
        (; xs,) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs² = xs
        (; xs,) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³ = xs
        (; xs, us) = unflatten_trajectory(z₄, state_dim, control_dim)
        xs⁴, us⁴ = xs, us

        control = sum(sum(u .^ 2) for u in us⁴)
        collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
        velocity = sum((x⁴[4] - 2.0)^2 for x⁴ in xs⁴)
        y_deviation = sum((x⁴[2] - R)^2 for x⁴ in xs⁴)
        zero_heading = sum((x⁴[3])^2 for x⁴ in xs⁴)

        control + collision + y_deviation + zero_heading + velocity
    end

    Js = Dict{Int,Any}(1 => J₁, 2 => J₂, 3 => J₃, 4 => J₄)

    # Constraints: unicycle dynamics + initial condition
    function make_constraints(i)
        return function (zᵢ)
            dyn = mapreduce(vcat, 1:T) do t
                unicycle_dynamics(zᵢ, t; Δt, state_dim, control_dim)
            end
            (; xs,) = unflatten_trajectory(zᵢ, state_dim, control_dim)
            ic = xs[1] - θs[i]
            vcat(dyn, ic)
        end
    end

    gs = [make_constraints(i) for i in 1:N]

    verbose && @info "Building NonlinearSolver..." N T Δt
    solver = NonlinearSolver(
        G, Js, gs, primal_dims, θs, state_dim, control_dim;
        max_iters = max_iters, tol = 1e-6, verbose = false,
    )

    return solver, θs
end

# ============================================================================
# Initial Guess Generation
# ============================================================================

function generate_initial_guess(x0_base, R, T, Δt)
    # P1, P2, P4: straight trajectories
    x0_1, u0_1 = make_straight_traj(T, Δt; x0=x0_base[1])
    z0_1 = vcat([vcat(x0_1[t], u0_1[t]) for t in 1:T+1]...)

    x0_2, u0_2 = make_straight_traj(T, Δt; x0=x0_base[2])
    z0_2 = vcat([vcat(x0_2[t], u0_2[t]) for t in 1:T+1]...)

    # P3: unicycle trajectory (merging)
    x0_3, u0_3 = make_unicycle_traj(T, Δt; R, split=0.5, x0=x0_base[3])
    z0_3 = vcat([vcat(x0_3[t], u0_3[t]) for t in 1:T+1]...)

    x0_4, u0_4 = make_straight_traj(T, Δt; x0=x0_base[4])
    z0_4 = vcat([vcat(x0_4[t], u0_4[t]) for t in 1:T+1]...)

    return vcat(z0_1, z0_2, z0_3, z0_4)
end

# ============================================================================
# Multi-Run Experiment
# ============================================================================

function perturb_initial_state(x0; rng, scale)
    return [x0[i] .+ scale .* randn(rng, length(x0[i])) for i in eachindex(x0)]
end

"""
    run_convergence_analysis(; config=DEFAULT_CONFIG, verbose=false)

Run multiple solver iterations with perturbed initial states.

# Returns
Named tuple with:
- `iterations`: Vector of iteration counts per run
- `residuals`: Vector of final residuals per run
- `statuses`: Vector of solver statuses per run
- `converged_count`: Number of runs that converged
- `config`: Configuration used
"""
function run_convergence_analysis(; config=DEFAULT_CONFIG, verbose=false)
    (; R, T, Δt, num_runs, max_iters, perturb_scale, seed) = config

    rng = MersenneTwister(seed)
    x0_base = default_initial_states(R)
    z0_guess = generate_initial_guess(x0_base, R, T, Δt)

    verbose && @info "Building solver (one-time precomputation)..."
    solver, θs = build_nonlinear_lane_change_solver(; R, T, Δt, max_iters, verbose)

    iterations = Vector{Int}(undef, num_runs)
    residuals = Vector{Float64}(undef, num_runs)
    statuses = Vector{Symbol}(undef, num_runs)

    verbose && @info "Running $num_runs experiments..."
    for run_id in 1:num_runs
        x0_run = perturb_initial_state(x0_base; rng, scale=perturb_scale)
        params = Dict(i => x0_run[i] for i in 1:4)

        result = solve_raw(solver, params; initial_guess = z0_guess)

        iterations[run_id] = result.iterations
        residuals[run_id] = result.residual
        statuses[run_id] = result.status

        verbose && @info "Run $run_id: $(result.status), $(result.iterations) iters, residual=$(result.residual)"
    end

    converged_count = count(s -> s == :solved, statuses)

    return (; iterations, residuals, statuses, converged_count, config)
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
        count = sum(statuses .== status)
        println("  $status: $count")
    end
    println("="^60)
end

# ============================================================================
# Main Entry Point
# ============================================================================

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running convergence analysis with $(DEFAULT_CONFIG.num_runs) runs...")
    result = run_convergence_analysis(verbose=true)
    print_summary(result)
end
