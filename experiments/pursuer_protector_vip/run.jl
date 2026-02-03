#=
    Pursuer-Protector-VIP Experiment

    Multi-agent pursuit-protection game with 3 players:
    - Player 1: Pursuer - chases the VIP
    - Player 2: Protector - protects the VIP from pursuer
    - Player 3: VIP - tries to reach goal while staying near protector

    Uses the same hierarchy as the LQ three-player chain:
    P1 → P2 → P3
=#

using MixedHierarchyGames
using Graphs: SimpleDiGraph, add_edge!
using TrajectoryGamesBase: unflatten_trajectory
using SymbolicTracingUtils

# Include common utilities
include(joinpath(@__DIR__, "..", "common", "dynamics.jl"))

"""
    run_pursuer_protector_vip(; T=20, Δt=0.1, x0=default_x0, x_goal=[0,0], verbose=false)

Run the pursuer-protector-VIP pursuit/defense scenario.

# Arguments
- `T`: Time horizon (number of steps)
- `Δt`: Time step duration
- `x0`: Initial states [pursuer, protector, VIP] as [x, y] positions
- `x_goal`: Goal position for VIP
- `verbose`: Print detailed output

# Returns
Named tuple with solution trajectories and solver info.
"""
function run_pursuer_protector_vip(;
    T::Integer = 20,
    Δt::Real = 0.1,
    x0::Vector{<:AbstractVector} = [
        [-5.0, 1.0],   # pursuer
        [-2.0, -2.5],  # protector
        [2.0, -4.0],   # VIP
    ],
    x_goal::AbstractVector = [0.0, 0.0],
    max_iters::Integer = 50,
    verbose::Bool = false,
)
    N = 3
    state_dim = 2  # [x, y] position
    control_dim = 2  # [vx, vy] velocity

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
    # Pursuer: chase VIP, lightly repulse protector, penalize control effort
    function J₁(z₁, z₂, z₃, θi)
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, us¹ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², _ = xs, us
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, _ = xs, us

        # Chase VIP (minimize distance to VIP)
        chase_vip = 2 * sum(sum((xs³[t] - xs¹[t]) .^ 2 for t in 1:T))
        # Avoid protector (maximize distance to protector)
        avoid_protector = -sum(sum((xs²[t] - xs¹[t]) .^ 2 for t in 1:T))
        # Control effort
        control = 1.25 * sum(sum(u .^ 2) for u in us¹)

        chase_vip + avoid_protector + control
    end

    # Protector: stay with VIP, pull VIP away from pursuer
    function J₂(z₁, z₂, z₃, θi)
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², us² = xs, us
        (; xs, us) = unflatten_trajectory(z₁, state_dim, control_dim)
        xs¹, _ = xs, us
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, _ = xs, us

        # Stay close to VIP
        stay_with_vip = 0.5 * sum(sum((xs³[t] - xs²[t]) .^ 2 for t in 1:T))
        # Keep VIP away from pursuer
        protect_vip = -sum(sum((xs³[t] - xs¹[t]) .^ 2 for t in 1:T))
        # Control effort
        control = 0.25 * sum(sum(u .^ 2) for u in us²)

        stay_with_vip + protect_vip + control
    end

    # VIP: reach goal, stay close to protector
    function J₃(z₁, z₂, z₃, θi)
        (; xs, us) = unflatten_trajectory(z₃, state_dim, control_dim)
        xs³, us³ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dim, control_dim)
        xs², _ = xs, us

        # Reach goal
        reach_goal = 10 * sum((xs³[end] .- x_goal) .^ 2)
        # Stay close to protector
        stay_with_protector = sum(sum((xs³[t] - xs²[t]) .^ 2 for t in 1:T))
        # Control effort
        control = 1.25 * sum(sum(u .^ 2) for u in us³)

        reach_goal + stay_with_protector + control
    end

    Js = Dict{Int,Any}(1 => J₁, 2 => J₂, 3 => J₃)

    # Constraints: single integrator dynamics + initial condition
    function make_constraints(i)
        return function (zᵢ)
            # Dynamics constraint (single integrator: x_{t+1} = x_t + Δt * u_t)
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
        max_iters = max_iters, tol = 1e-6, verbose = verbose,
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
        @info "Final positions" pursuer = trajectories[1].xs[end] protector = trajectories[2].xs[end] vip = trajectories[3].xs[end]
    end

    return (;
        z_sol,
        z_sols,
        trajectories,
        costs,
        status = result.status,
        iterations = result.iterations,
        residual = result.residual,
        x_goal,
    )
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    result = run_pursuer_protector_vip(verbose = true)
    println("\nExperiment completed with status: $(result.status)")
    println("Iterations: $(result.iterations), Residual: $(result.residual)")
end
