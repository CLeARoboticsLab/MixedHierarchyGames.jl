#=
    Main solver interface implementing TrajectoryGamesBase.solve_trajectory_game!
=#

"""
    TrajectoryGamesBase.solve_trajectory_game!(
        solver::QPSolver,
        game::HierarchyGame,
        initial_state;
        kwargs...
    )

Solve a QP (linear-quadratic) hierarchy game.

# Arguments
- `solver::QPSolver` - The QP solver instance
- `game::HierarchyGame` - The hierarchy game to solve
- `initial_state` - Initial state (BlockVector or Vector)

# Keyword Arguments
- `verbose::Bool=false` - Print debug info

# Returns
- `JointStrategy` over `OpenLoopStrategy`s for each player
"""
function TrajectoryGamesBase.solve_trajectory_game!(
    solver::QPSolver,
    game::HierarchyGame,
    initial_state;
    verbose::Bool = false,
    kwargs...
)
    # TODO: Implement
    # 1. Extract initial states per player from initial_state
    # 2. Call run_qp_solver with game parameters
    # 3. Convert solution to JointStrategy of OpenLoopStrategys
    error("Not implemented: solve_trajectory_game! for QPSolver")
end

"""
    TrajectoryGamesBase.solve_trajectory_game!(
        solver::NonlinearSolver,
        game::HierarchyGame,
        initial_state;
        kwargs...
    )

Solve a nonlinear hierarchy game.

# Arguments
- `solver::NonlinearSolver` - The nonlinear solver instance
- `game::HierarchyGame` - The hierarchy game to solve
- `initial_state` - Initial state (BlockVector or Vector)

# Keyword Arguments
- `initial_guess::Union{Nothing, Vector}=nothing` - Warm start
- `verbose::Bool=false` - Print iteration info

# Returns
- `JointStrategy` over `OpenLoopStrategy`s for each player
"""
function TrajectoryGamesBase.solve_trajectory_game!(
    solver::NonlinearSolver,
    game::HierarchyGame,
    initial_state;
    initial_guess::Union{Nothing, Vector} = nothing,
    verbose::Bool = false,
    kwargs...
)
    # TODO: Implement
    # 1. Extract initial states per player from initial_state
    # 2. Call run_nonlinear_solver with precomputed components
    # 3. Convert solution to JointStrategy of OpenLoopStrategys
    error("Not implemented: solve_trajectory_game! for NonlinearSolver")
end

#=
    Backend solver functions
=#

"""
    solve_with_path(πs::Dict, variables::Vector, θs::Dict, parameter_values::Dict; kwargs...)

Solve KKT system using PATH solver via ParametricMCPs.jl.

# Arguments
- `πs::Dict` - KKT conditions per player
- `variables::Vector` - All symbolic variables
- `θs::Dict` - Symbolic parameter variables per player
- `parameter_values::Dict` - Numerical parameter values per player

# Keyword Arguments
- `initial_guess::Union{Nothing, Vector}=nothing` - Warm start
- `verbose::Bool=false` - Print solver output

# Returns
Tuple of:
- `z_sol::Vector` - Solution vector
- `status::Symbol` - Solver status (:solved, :failed, etc.)
- `info` - PATH solver info
"""
function solve_with_path(
    πs::Dict,
    variables::Vector,
    θs::Dict,
    parameter_values::Dict;
    initial_guess::Union{Nothing, Vector} = nothing,
    verbose::Bool = false,
    kwargs...
)
    symbolic_type = eltype(variables)

    # Build KKT vector from all players
    F = Vector{symbolic_type}(vcat(collect(values(πs))...))

    # Bounds: unconstrained (MCP with -∞ to ∞)
    z_lower = fill(-Inf, length(F))
    z_upper = fill(Inf, length(F))

    # Order parameters by player index
    order = sort(collect(keys(πs)))
    all_θ_vec = vcat([θs[k] for k in order]...)
    all_param_vals_vec = vcat([parameter_values[k] for k in order]...)

    # Build parametric MCP
    parametric_mcp = ParametricMCPs.ParametricMCP(
        F, variables, all_θ_vec, z_lower, z_upper;
        compute_sensitivities = false
    )

    # Initial guess
    z0 = initial_guess !== nothing ? initial_guess : zeros(length(variables))

    # Solve
    z_sol, status, info = ParametricMCPs.solve(
        parametric_mcp,
        all_param_vals_vec;
        initial_guess = z0,
        verbose = verbose,
        cumulative_iteration_limit = 100000,
        proximal_perturbation = 1e-2,
        use_basics = true,
        use_start = true,
    )

    return z_sol, status, info
end

"""
    qp_game_linsolve(A, b; kwargs...)

Solve linear system Ax = b for QP games.

For LQ games, the KKT system is linear so this directly solves for the solution.

# Arguments
- `A` - System matrix (Jacobian of KKT conditions)
- `b` - Right-hand side

# Returns
- `x::Vector` - Solution vector
"""
function qp_game_linsolve(A, b; kwargs...)
    # Use backslash operator for direct solve
    # For larger systems, could use LinearSolve.jl with specific factorization
    return A \ b
end

#=
    Solution extraction utilities
=#

"""
    extract_trajectories(z_sol::Vector, dims::NamedTuple, T::Int, n_players::Int)

Extract state and control trajectories from flattened solution vector.

# Arguments
- `z_sol::Vector` - Flattened solution
- `dims::NamedTuple` - Dimension info per player
- `T::Int` - Time horizon
- `n_players::Int` - Number of players

# Returns
- `xs::Dict{Int, Vector{Vector}}` - State trajectories per player
- `us::Dict{Int, Vector{Vector}}` - Control trajectories per player
"""
function extract_trajectories(z_sol::Vector, dims::NamedTuple, T::Int, n_players::Int)
    xs = Dict{Int, Vector{Vector{Float64}}}()
    us = Dict{Int, Vector{Vector{Float64}}}()

    idx = 1
    for i in 1:n_players
        state_dim = dims.state_dims[i]
        control_dim = dims.control_dims[i]

        # Extract states: T+1 states (t=0 to t=T)
        xs[i] = Vector{Vector{Float64}}()
        for t in 1:(T+1)
            push!(xs[i], z_sol[idx:idx+state_dim-1])
            idx += state_dim
        end

        # Extract controls: T controls (t=0 to t=T-1)
        us[i] = Vector{Vector{Float64}}()
        for t in 1:T
            push!(us[i], z_sol[idx:idx+control_dim-1])
            idx += control_dim
        end
    end

    return xs, us
end

"""
    solution_to_joint_strategy(xs::Dict, us::Dict, n_players::Int)

Convert trajectory dictionaries to JointStrategy of OpenLoopStrategys.

# Arguments
- `xs::Dict` - State trajectories per player
- `us::Dict` - Control trajectories per player
- `n_players::Int` - Number of players

# Returns
- `JointStrategy` over `OpenLoopStrategy`s
"""
function solution_to_joint_strategy(xs::Dict, us::Dict, n_players::Int)
    substrategies = [OpenLoopStrategy(xs[i], us[i]) for i in 1:n_players]
    return JointStrategy(substrategies)
end
