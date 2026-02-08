#=
    LQ Three Player Chain Experiment

    Demonstrates a linear-quadratic game with 3 players in a Stackelberg chain:
    P2 is the root leader, with P1 and P3 as followers.
    Hierarchy: P2 → P1, P2 → P3

    This is the simplest example using single integrator dynamics and
    quadratic objectives.
=#

using MixedHierarchyGames
using TrajectoryGamesBase: unflatten_trajectory
using LinearAlgebra: norm
using Plots

# Include experiment modules
include("config.jl")
include("support.jl")

# Include common utilities
include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))
include(joinpath(@__DIR__, "..", "common", "plotting.jl"))

"""
    run_lq_three_player_chain(; T, Δt, x0, verbose)

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
    T::Integer = DEFAULT_T,
    Δt::Real = DEFAULT_DT,
    x0::Vector{<:AbstractVector} = DEFAULT_X0,
    verbose::Bool = false,
    verify::Bool = false,
    plot::Bool = false,
    savepath::Union{Nothing,String} = nothing,
)
    # Build hierarchy and cost functions
    G = build_hierarchy()
    Js = make_cost_functions(STATE_DIM, CONTROL_DIM)

    # Dimensions
    primal_dim = (STATE_DIM + CONTROL_DIM) * (T + 1)
    primal_dims = fill(primal_dim, N)

    # Set up symbolic parameters for initial states
    θs = setup_problem_parameter_variables(fill(STATE_DIM, N))

    # Build constraints: dynamics + initial condition
    function make_constraints(i)
        return function (zᵢ)
            dyn = mapreduce(vcat, 1:T) do t
                single_integrator_2d(zᵢ, t; Δt, state_dim=STATE_DIM, control_dim=CONTROL_DIM)
            end
            (; xs,) = unflatten_trajectory(zᵢ, STATE_DIM, CONTROL_DIM)
            ic = xs[1] - θs[i]
            vcat(dyn, ic)
        end
    end
    gs = [make_constraints(i) for i in 1:N]

    # Build and solve
    verbose && @info "Building NonlinearSolver..." N T Δt
    solver = NonlinearSolver(
        G, Js, gs, primal_dims, θs, STATE_DIM, CONTROL_DIM;
        max_iters = MAX_ITERS, tol = TOLERANCE, verbose = verbose,
    )

    verbose && @info "Solving..."
    parameter_values = Dict(i => x0[i] for i in 1:N)
    result = solve_raw(solver, parameter_values; verbose = verbose)

    # Extract per-player solutions (primal portion only; sol may include duals)
    sol = result.sol
    sols = collect.(split_solution_vector(sol[1:sum(primal_dims)], primal_dims))

    # Extract trajectories and compute costs
    trajectories = [unflatten_trajectory(z, STATE_DIM, CONTROL_DIM) for z in sols]
    costs = [Js[i](sols[1], sols[2], sols[3]) for i in 1:N]

    if verbose
        @info "Solution found" status=result.status iterations=result.iterations
        @info "Player costs" costs
        for i in 1:N
            @info "Player $i trajectory" xs=trajectories[i].xs us=trajectories[i].us
        end
    end

    # Verify KKT conditions if requested
    kkt_residuals = nothing
    if verify
        kkt_residuals = verify_kkt_solution(solver, sol, θs, parameter_values; verbose=verbose)
        verbose && @info "KKT residual norm" norm=norm(kkt_residuals)
    end

    # Generate plots if requested
    plt = nothing
    if plot || savepath !== nothing
        plt = plot_trajectories_2d(
            trajectories;
            title = "LQ Three Player Chain (T=$T, Δt=$Δt)",
            labels = ["P1", "P2 (Leader)", "P3"],
            show = plot,
        )
        # Mark origin (target for leader's objective)
        scatter!(plt, [0.0], [0.0]; marker=:cross, ms=10, color=:black, label="Origin")

        if savepath !== nothing
            savefig(plt, savepath)
            verbose && @info "Plot saved to $savepath"
        end
    end

    return (;
        sol, sols, trajectories, costs,
        status = result.status,
        iterations = result.iterations,
        residual = result.residual,
        kkt_residuals,
        plt,
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    result = run_lq_three_player_chain(verbose = true)
    println("\nExperiment completed with status: $(result.status)")
end
