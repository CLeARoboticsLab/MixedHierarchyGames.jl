#=
    LQ Three Player Chain Experiment

    Demonstrates a linear-quadratic game with 3 players in a Stackelberg chain:
    P1 → P2 → P3 (P1 leads P2, P2 leads P3)

    This is the simplest example using single integrator dynamics and
    quadratic objectives.
=#

using MixedHierarchyGames
using Graphs: SimpleDiGraph, add_edge!
using TrajectoryGamesBase: unflatten_trajectory
using SymbolicTracingUtils

# Include common utilities
include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))

"""
    run_lq_three_player_chain(; T=3, Δt=0.5, x0=default_x0, verbose=false)

Run the LQ three-player chain experiment.

# Arguments
- `T`: Time horizon (number of steps)
- `Δt`: Time step duration
- `x0`: Initial states for each player (Vector of 2D positions)
- `verbose`: Print detailed output

# Returns
Named tuple with solution trajectories and solver info.
"""
function run_lq_three_player_chain(;
    T::Integer = 3,
    Δt::Real = 0.5,
    x0::Vector{<:AbstractVector} = [[0.0, 2.0], [2.0, 4.0], [6.0, 8.0]],
    verbose::Bool = false,
)
    N = 3
    state_dim = 2
    control_dim = 2

    # Set up hierarchy: P1 → P2 → P3
    G = SimpleDiGraph(N)
    add_edge!(G, 2, 1)  # P1 leads P2
    add_edge!(G, 2, 3)  # P2 leads P3

    # Dimensions
    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)

    # Set up symbolic parameters for initial states
    backend = default_backend()
    θs = setup_problem_parameter_variables(backend, fill(state_dim, N))

    # Player objectives
    function J₁(z₁, z₂, z₃, θi)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        # P1 wants to get close to P2's final position, minimize control
        0.5 * sum((xs¹[end] .- xs²[end]) .^ 2) + 0.05 * sum(sum(u .^ 2) for u in us¹)
    end

    function J₂(z₁, z₂, z₃, θi)
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        # P2 wants P1 and P3 to get to the origin
        sum((0.5 * (xs¹[end] .+ xs³[end])) .^ 2) + 0.05 * sum(sum(u .^ 2) for u in us²)
    end

    function J₃(z₁, z₂, z₃, θi)
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        # P3 wants to get close to P2's final position
        0.5 * sum((xs³[end] .- xs²[end]) .^ 2) +
        0.05 * sum(sum(u³ .^ 2) for u³ in us³) +
        0.05 * sum(sum(u² .^ 2) for u² in us²)
    end

    Js = Dict{Int,Any}(1 => J₁, 2 => J₂, 3 => J₃)

    # Constraints: dynamics + initial condition
    function make_constraints(i)
        return function (zᵢ)
            # Dynamics constraint
            dyn = mapreduce(vcat, 1:T) do t
                single_integrator_2d(zᵢ, t; Δt, state_dim, control_dim)
            end
            # Initial condition constraint
            (; xs, us) = unflatten_trajectory(zᵢ, state_dim, control_dim)
            ic = xs[1] - θs[i]
            vcat(dyn, ic)
        end
    end

    gs = [make_constraints(i) for i in 1:N]

    # Build and solve
    verbose && @info "Building NonlinearSolver..." N T Δt
    solver = NonlinearSolver(
        G, Js, gs, primal_dims, θs, state_dim, control_dim;
        max_iters = 50, tol = 1e-8, verbose = verbose,
    )

    verbose && @info "Solving..."
    parameter_values = Dict(i => x0[i] for i in 1:N)
    result = solve_raw(solver, parameter_values; verbose = verbose)

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
    costs = [Js[i](z_sols[1], z_sols[2], z_sols[3], nothing) for i in 1:N]

    if verbose
        @info "Solution found" status = result.status iterations = result.iterations
        @info "Player costs" costs
        for i in 1:N
            @info "Player $i trajectory" xs = trajectories[i].xs us = trajectories[i].us
        end
    end

    return (;
        z_sol,
        z_sols,
        trajectories,
        costs,
        status = result.status,
        iterations = result.iterations,
        residual = result.residual,
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    result = run_lq_three_player_chain(verbose = true)
    println("\nExperiment completed with status: $(result.status)")
end
