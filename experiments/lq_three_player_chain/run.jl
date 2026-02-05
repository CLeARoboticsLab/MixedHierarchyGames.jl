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

# Include experiment modules
include("config.jl")
include("support.jl")

# Include common utilities
include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))

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
    costs = [Js[i](z_sols[1], z_sols[2], z_sols[3]) for i in 1:N]

    if verbose
        @info "Solution found" status=result.status iterations=result.iterations
        @info "Player costs" costs
        for i in 1:N
            @info "Player $i trajectory" xs=trajectories[i].xs us=trajectories[i].us
        end
    end

    return (;
        z_sol, z_sols, trajectories, costs,
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
