#=
    Nonlinear Lane Change Experiment

    Demonstrates a 4-player game with unicycle dynamics where vehicles
    perform lane change maneuvers on a highway with collision avoidance.

    Hierarchy: P1 → P2 → P4 (P1 leads P2, P2 leads P4; P3 is Nash)
=#

using MixedHierarchyGames
using TrajectoryGamesBase: unflatten_trajectory

# Include experiment modules
include("config.jl")

# Include common utilities (needed before support.jl)
include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
include(joinpath(@__DIR__, "..", "common", "collision_avoidance.jl"))
include(joinpath(@__DIR__, "..", "common", "trajectory_utils.jl"))

include("support.jl")

"""
    run_nonlinear_lane_change(; T, Δt, R, x0, max_iters, verbose)

Run the nonlinear lane change experiment with 4 vehicles.

# Arguments
- `T`: Time horizon (number of steps)
- `Δt`: Time step duration
- `R`: Turning radius / lane y-coordinate
- `x0`: Initial states for each player [x, y, ψ, v]
- `max_iters`: Maximum solver iterations
- `verbose`: Print detailed output

# Returns
Named tuple with solution trajectories and solver info.
"""
function run_nonlinear_lane_change(;
    T::Integer = DEFAULT_T,
    Δt::Real = DEFAULT_DT,
    R::Real = DEFAULT_R,
    x0::Vector{<:AbstractVector} = default_initial_states(R),
    max_iters::Integer = MAX_ITERS,
    verbose::Bool = false,
)
    # Build hierarchy and cost functions
    G = build_hierarchy()
    Js = make_cost_functions(STATE_DIM, CONTROL_DIM, T, R)

    # Dimensions
    primal_dim = (STATE_DIM + CONTROL_DIM) * (T + 1)
    primal_dims = fill(primal_dim, N)

    # Set up symbolic parameters for initial states
    θs = setup_problem_parameter_variables(fill(STATE_DIM, N))

    # Build constraints: unicycle dynamics + initial condition
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

    # Build initial guess
    z0_guess = build_initial_guess(x0, R, T, Δt)

    # Build and solve
    verbose && @info "Building NonlinearSolver..." N T Δt R
    solver = NonlinearSolver(
        G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM;
        max_iters = max_iters, tol = TOLERANCE, verbose = verbose,
    )

    verbose && @info "Solving..."
    parameter_values = Dict(i => x0[i] for i in 1:N)
    result = solve_raw(solver, parameter_values; initial_guess = z0_guess, verbose = verbose)

    # Extract per-player solutions
    z_sol = result.sol
    z_sols = Vector{Vector{Float64}}(undef, N)
    offs = 1
    for i in 1:N
        z_sols[i] = z_sol[offs:offs+primal_dim-1]
        offs += primal_dim
    end

    # Extract trajectories and compute costs
    trajectories = [unflatten_trajectory(z, STATE_DIM, CONTROL_DIM) for z in z_sols]
    costs = [Js[i](z_sols[1], z_sols[2], z_sols[3], z_sols[4]) for i in 1:N]

    if verbose
        @info "Solution found" status=result.status iterations=result.iterations residual=result.residual
        @info "Player costs" costs
    end

    return (;
        z_sol, z_sols, trajectories, costs,
        status = result.status,
        iterations = result.iterations,
        residual = result.residual,
        R, T, Δt,
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    result = run_nonlinear_lane_change(R = DEFAULT_R, verbose = true)
    println("\nExperiment completed with status: $(result.status)")
    println("Iterations: $(result.iterations), Residual: $(result.residual)")
end
