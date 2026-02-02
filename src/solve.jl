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
    solve_with_path(mcp_problem, θ_values; kwargs...)

Solve KKT system using PATH solver via ParametricMCPs.jl.

# Arguments
- `mcp_problem` - The parametric MCP problem
- `θ_values::Vector` - Parameter values (initial states)

# Keyword Arguments
- `initial_guess::Union{Nothing, Vector}=nothing` - Warm start

# Returns
Named tuple with:
- `z::Vector` - Solution
- `info` - PATH solver info
"""
function solve_with_path(mcp_problem, θ_values; initial_guess = nothing, kwargs...)
    # TODO: Implement - wrap ParametricMCPs.solve
    error("Not implemented: solve_with_path")
end

"""
    qp_game_linsolve(M, b; kwargs...)

Custom linear solver for QP games using Newton step ∇F(z)δz = -F(z).

# Arguments
- `M` - System matrix (Jacobian)
- `b` - Right-hand side (-F(z))

# Returns
- `δz::Vector` - Newton step
"""
function qp_game_linsolve(M, b; kwargs...)
    # TODO: Implement
    error("Not implemented: qp_game_linsolve")
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
