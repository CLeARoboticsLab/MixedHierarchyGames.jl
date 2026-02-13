#=
    Pursuer-Protector-VIP Experiment

    Multi-agent pursuit-protection game with 3 players:
    - Player 1: Pursuer - chases the VIP
    - Player 2: Protector - protects the VIP from pursuer (leader)
    - Player 3: VIP - tries to reach goal while staying near protector

    Hierarchy: P2 → P1, P2 → P3
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
    run_pursuer_protector_vip(; T, Δt, x0, x_goal, max_iters, verbose)

Run the pursuer-protector-VIP pursuit/defense scenario.

# Arguments
- `T`: Time horizon (number of steps)
- `Δt`: Time step duration
- `x0`: Initial states [pursuer, protector, VIP] as [x, y] positions
- `x_goal`: Goal position for VIP
- `max_iters`: Maximum solver iterations
- `verbose`: Print detailed output

# Returns
Named tuple with solution trajectories and solver info.
"""
function run_pursuer_protector_vip(;
    T::Integer = DEFAULT_T,
    Δt::Real = DEFAULT_DT,
    x0::Vector{<:AbstractVector} = DEFAULT_X0,
    x_goal::AbstractVector = DEFAULT_GOAL,
    max_iters::Integer = MAX_ITERS,
    verbose::Bool = false,
    verify::Bool = false,
    plot::Bool = false,
    savepath::Union{Nothing,String} = nothing,
)
    # Build hierarchy and cost functions
    G = build_hierarchy()
    Js = make_cost_functions(STATE_DIM, CONTROL_DIM, T, x_goal)

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
        max_iters = max_iters, tol = TOLERANCE, verbose = verbose, show_progress = true,
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
        @info "Final positions" pursuer=trajectories[1].xs[end] protector=trajectories[2].xs[end] vip=trajectories[3].xs[end]
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
        plt = plot_pursuit_game(
            trajectories;
            x_goal = x_goal,
            labels = ["Pursuer", "Protector (Leader)", "VIP"],
            show = plot,
            savepath = savepath,
        )
        verbose && savepath !== nothing && @info "Plot saved to $savepath"
    end

    return (;
        sol, sols, trajectories, costs,
        status = result.status,
        iterations = result.iterations,
        residual = result.residual,
        kkt_residuals,
        x_goal,
        plt,
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    result = run_pursuer_protector_vip(verbose = true)
    println("\nExperiment completed with status: $(result.status)")
    println("Iterations: $(result.iterations), Residual: $(result.residual)")
end
