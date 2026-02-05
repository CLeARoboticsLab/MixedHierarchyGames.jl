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

# Include experiment modules
include("config.jl")

# Include common utilities
include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
include(joinpath(@__DIR__, "..", "common", "collision_avoidance.jl"))
include(joinpath(@__DIR__, "..", "common", "trajectory_utils.jl"))

# Import nonlinear_lane_change config and support for reuse
include(joinpath(@__DIR__, "..", "nonlinear_lane_change", "config.jl"))
include(joinpath(@__DIR__, "..", "nonlinear_lane_change", "support.jl"))

include("support.jl")

"""
    build_solver(; R, T, Δt, max_iters, verbose)

Build the NonlinearSolver for the lane change problem (reuses nonlinear_lane_change setup).
"""
function build_solver(; R, T, Δt, max_iters, verbose=false)
    # Build hierarchy and cost functions (from nonlinear_lane_change)
    G = build_hierarchy()
    Js = make_cost_functions(STATE_DIM, CONTROL_DIM, T, R)

    # Dimensions
    primal_dim = (STATE_DIM + CONTROL_DIM) * (T + 1)
    primal_dims = fill(primal_dim, N)

    # Set up symbolic parameters
    θs = setup_problem_parameter_variables(fill(STATE_DIM, N))

    # Build constraints
    function make_constraints(i)
        return function (zᵢ)
            dyn = mapreduce(vcat, 1:T) do t
                unicycle_dynamics(zᵢ, t; Δt, state_dim=STATE_DIM, control_dim=CONTROL_DIM)
            end
            (; xs,) = unflatten_trajectory(zᵢ, STATE_DIM, CONTROL_DIM)
            ic = xs[1] - θs[i]
            vcat(dyn, ic)
        end
    end
    gs = [make_constraints(i) for i in 1:N]

    verbose && @info "Building NonlinearSolver..." N T Δt
    solver = NonlinearSolver(
        G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM;
        max_iters = max_iters, tol = TOLERANCE, verbose = false,
    )

    return solver
end

"""
    run_convergence_analysis(; config, verbose)

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
    z0_guess = build_initial_guess(x0_base, R, T, Δt)

    verbose && @info "Building solver (one-time precomputation)..."
    solver = build_solver(; R, T, Δt, max_iters, verbose)

    iterations = Vector{Int}(undef, num_runs)
    residuals = Vector{Float64}(undef, num_runs)
    statuses = Vector{Symbol}(undef, num_runs)

    verbose && @info "Running $num_runs experiments..."
    for run_id in 1:num_runs
        x0_run = perturb_initial_state(x0_base; rng, scale=perturb_scale)
        params = Dict(i => x0_run[i] for i in 1:N)

        result = solve_raw(solver, params; initial_guess = z0_guess)

        iterations[run_id] = result.iterations
        residuals[run_id] = result.residual
        statuses[run_id] = result.status

        verbose && @info "Run $run_id: $(result.status), $(result.iterations) iters, residual=$(result.residual)"
    end

    converged_count = count(s -> s == :solved, statuses)

    return (; iterations, residuals, statuses, converged_count, config)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running convergence analysis with $(DEFAULT_CONFIG.num_runs) runs...")
    result = run_convergence_analysis(verbose=true)
    print_summary(result)
end
