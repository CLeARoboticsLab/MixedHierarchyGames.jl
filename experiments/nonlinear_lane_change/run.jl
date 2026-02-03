#=
    Nonlinear Lane Change Experiment

    Demonstrates a 4-player game with unicycle dynamics where vehicles
    perform lane change maneuvers on a highway with collision avoidance.

    Hierarchy: P1 → P2 → P4 (P1 leads P2, P2 leads P4; P3 is Nash)
=#

using MixedHierarchyGames
using Graphs: SimpleDiGraph, add_edge!
using TrajectoryGamesBase: unflatten_trajectory
using SymbolicTracingUtils

# Include common utilities
include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
include(joinpath(@__DIR__, "..", "common", "collision_avoidance.jl"))
include(joinpath(@__DIR__, "..", "common", "trajectory_utils.jl"))

"""
    run_nonlinear_lane_change(; T=14, Δt=0.4, R=6.0, x0=default_x0, verbose=false)

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
    T::Integer = 14,
    Δt::Real = 0.4,
    R::Real = 6.0,
    x0::Vector{<:AbstractVector} = [
        [-1.5R, R, 0.0, 2.0],    # P1 (LEADER)
        [-2.0R, R, 0.0, 2.0],    # P2 (FOLLOWER of P1)
        [-R, 0.0, π/2, 1.523],   # P3 (LANE MERGER - Nash)
        [-2.5R, R, 0.0, 2.0],    # P4 (FOLLOWER of P2)
    ],
    max_iters::Integer = 100,
    verbose::Bool = false,
)
    N = 4
    state_dim = 4  # [x, y, ψ, v]
    control_dim = 2  # [a, ω]

    # Set up hierarchy: P1 → P2 → P4
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)  # P1 leads P2
    add_edge!(G, 2, 4)  # P2 leads P4
    # P3 is Nash (no edges)

    # Dimensions
    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)

    # Set up symbolic parameters for initial states
    backend = default_backend()
    θs = setup_problem_parameter_variables(backend, fill(state_dim, N))

    # Player objectives
    function J₁(z₁, z₂, z₃, z₄, θ)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs, us) = unflatten_trajectory(z₄, state_dim, control_dim)
        xs⁴, us⁴ = xs, us

        control = 10 * sum(sum(u .^ 2) for u in us¹)
        collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
        velocity = sum((x¹[4] - 2.0)^2 for x¹ in xs¹)
        y_deviation = sum((x¹[2] - R)^2 for x¹ in xs¹)
        zero_heading = sum((x¹[3])^2 for x¹ in xs¹)

        control + collision + 5 * y_deviation + zero_heading + velocity
    end

    function J₂(z₁, z₂, z₃, z₄, θ)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs, us) = unflatten_trajectory(z₄, state_dim, control_dim)
        xs⁴, us⁴ = xs, us

        control = sum(sum(u .^ 2) for u in us²)
        collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
        velocity = sum((x²[4] - 2.0)^2 for x² in xs²)
        y_deviation = sum((x²[2] - R)^2 for x² in xs²)
        zero_heading = sum((x²[3])^2 for x² in xs²)

        control + collision + 5 * y_deviation + zero_heading + velocity
    end

    function J₃(z₁, z₂, z₃, z₄, θ)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs, us) = unflatten_trajectory(z₄, state_dim, control_dim)
        xs⁴, us⁴ = xs, us

        # Track quarter circle for first half, then lane
        tracking = 10 * sum((sum(x³[1:2] .^ 2) - R^2)^2 for x³ in xs³[2:div(T, 2)])
        control = sum(sum(u³ .^ 2) for u³ in us³)
        collision = smooth_collision_all(xs¹, xs², xs³, xs⁴)
        velocity = sum((x³[4] - 2.0)^2 for x³ in xs³)
        y_deviation = sum((x³[2] - R)^2 for x³ in xs³[div(T, 2):T])
        zero_heading = sum((x³[3])^2 for x³ in xs³[div(T, 2):T])

        tracking + control + collision + 5 * y_deviation + zero_heading + velocity
    end

    function J₄(z₁, z₂, z₃, z₄, θ)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
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
            # Dynamics constraint
            dyn = mapreduce(vcat, 1:T) do t
                unicycle_dynamics(zᵢ, t; Δt, state_dim, control_dim)
            end
            # Initial condition constraint
            (; xs, us) = unflatten_trajectory(zᵢ, state_dim, control_dim)
            ic = xs[1] - θs[i]
            vcat(dyn, ic)
        end
    end

    gs = [make_constraints(i) for i in 1:N]

    # Build initial guess
    x0_1, u0_1 = make_straight_traj(T, Δt; x0 = x0[1])
    x0_2, u0_2 = make_straight_traj(T, Δt; x0 = x0[2])
    x0_3, u0_3 = make_unicycle_traj(T, Δt; R, split = 0.5, x0 = x0[3])
    x0_4, u0_4 = make_straight_traj(T, Δt; x0 = x0[4])

    z0_guess_1 = flatten_trajectory(x0_1, u0_1)
    z0_guess_2 = flatten_trajectory(x0_2, u0_2)
    z0_guess_3 = flatten_trajectory(x0_3, u0_3)
    z0_guess_4 = flatten_trajectory(x0_4, u0_4)
    z0_guess = vcat(z0_guess_1, z0_guess_2, z0_guess_3, z0_guess_4)

    # Build and solve
    verbose && @info "Building NonlinearSolver..." N T Δt R
    solver = NonlinearSolver(
        G, Js, gs, primal_dims, θs, state_dim, control_dim;
        max_iters = max_iters, tol = 1e-6, verbose = verbose,
    )

    verbose && @info "Solving..."
    parameter_values = Dict(i => x0[i] for i in 1:N)
    result = solve_raw(solver, parameter_values; initial_guess = z0_guess, verbose = verbose)

    # Extract per-player solutions
    z_sol = result.z_sol
    offs = 1
    z_sols = Vector{Vector{Float64}}(undef, N)
    for i in 1:N
        z_sols[i] = z_sol[offs:offs+primal_dim-1]
        offs += primal_dim
    end

    # Extract trajectories
    trajectories = map(z_sols) do z
        unflatten_trajectory(z, state_dim, control_dim)
    end

    # Compute costs
    costs = [Js[i](z_sols[1], z_sols[2], z_sols[3], z_sols[4], nothing) for i in 1:N]

    if verbose
        @info "Solution found" status = result.status iterations = result.iterations residual = result.residual
        @info "Player costs" costs
    end

    return (;
        z_sol,
        z_sols,
        trajectories,
        costs,
        status = result.status,
        iterations = result.iterations,
        residual = result.residual,
        R,
        T,
        Δt,
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    R = 6.0
    result = run_nonlinear_lane_change(R = R, verbose = true)
    println("\nExperiment completed with status: $(result.status)")
    println("Iterations: $(result.iterations), Residual: $(result.residual)")
end
