#=
    Main solver interface implementing TrajectoryGamesBase.solve_trajectory_game!
=#

"""
    _extract_joint_strategy(sol, primal_dims, state_dim, control_dim)

Extract per-player trajectories from solution vector and build JointStrategy.
Shared helper used by both QPSolver and NonlinearSolver.
"""
function _extract_joint_strategy(sol::AbstractVector, primal_dims::Vector{Int}, state_dim::Int, control_dim::Int)
    N = length(primal_dims)
    substrategies = Vector{OpenLoopStrategy}(undef, N)

    offset = 1
    for i in 1:N
        zi = sol[offset:(offset + primal_dims[i] - 1)]
        offset += primal_dims[i]
        (; xs, us) = TrajectoryGamesBase.unflatten_trajectory(zi, state_dim, control_dim)
        substrategies[i] = OpenLoopStrategy(xs, us)
    end

    return JointStrategy(substrategies)
end

"""
    solve(solver::QPSolver, parameter_values::Dict; kwargs...)

Solve the QP hierarchy game with given parameter values (typically initial states).

Uses precomputed symbolic KKT conditions for efficiency.

# Arguments
- `solver::QPSolver` - The QP solver with precomputed components
- `parameter_values::Dict` - Numerical values for parameters (e.g., initial states per player)

# Keyword Arguments
- `verbose::Bool=false` - Print debug info
- `iteration_limit::Int=100000` - Maximum iterations for PATH solver (only used with :path)
- `proximal_perturbation::Float64=1e-2` - Proximal perturbation parameter (only used with :path)
- `use_basics::Bool=true` - Use basic solution from PATH (only used with :path)
- `use_start::Bool=true` - Use starting point from PATH (only used with :path)

# Returns
- `JointStrategy` containing `OpenLoopStrategy` for each player
"""
function solve(
    solver::QPSolver,
    parameter_values::Dict;
    verbose::Bool = false,
    iteration_limit::Int = 100000,
    proximal_perturbation::Float64 = 1e-2,
    use_basics::Bool = true,
    use_start::Bool = true,
)
    (; problem, solver_type, precomputed) = solver
    (; vars, πs_solve, parametric_mcp) = precomputed
    (; θs, primal_dims, state_dim, control_dim) = problem

    # Validate parameter_values
    _validate_parameter_values(parameter_values, θs)

    if solver_type == :linear
        sol, status = solve_qp_linear(parametric_mcp, θs, parameter_values; verbose)
    elseif solver_type == :path
        sol, status, _ = solve_with_path(
            parametric_mcp, θs, parameter_values;
            verbose, iteration_limit, proximal_perturbation, use_basics, use_start
        )
    else
        error("Unknown solver type: $solver_type. Use :linear or :path")
    end

    # Check for solver failure
    if status == :failed
        error("QPSolver failed to find a solution. The KKT system may be singular or ill-conditioned.")
    end

    return _extract_joint_strategy(sol, primal_dims, state_dim, control_dim)
end

"""
    solve_raw(solver::QPSolver, parameter_values::Dict; kwargs...)

Solve and return raw solution vector (for debugging/analysis).

# Keyword Arguments
- `verbose::Bool=false` - Print debug info
- `iteration_limit::Int=100000` - Maximum iterations for PATH solver (only used with :path)
- `proximal_perturbation::Float64=1e-2` - Proximal perturbation parameter (only used with :path)
- `use_basics::Bool=true` - Use basic solution from PATH (only used with :path)
- `use_start::Bool=true` - Use starting point from PATH (only used with :path)

# Returns
Named tuple with fields:
- `sol::Vector{Float64}` - Solution vector (concatenated player decision variables)
- `status::Symbol` - Solver status (`:solved` or `:failed`)
- `info` - PATH solver info (only for `:path` backend, `nothing` for `:linear`)
- `vars` - Symbolic variables from precomputation
"""
function solve_raw(
    solver::QPSolver,
    parameter_values::Dict;
    verbose::Bool = false,
    iteration_limit::Int = 100000,
    proximal_perturbation::Float64 = 1e-2,
    use_basics::Bool = true,
    use_start::Bool = true,
)
    (; problem, solver_type, precomputed) = solver
    (; vars, πs_solve, parametric_mcp) = precomputed
    (; θs) = problem

    if solver_type == :linear
        sol, status = solve_qp_linear(parametric_mcp, θs, parameter_values; verbose)
        info = nothing
    elseif solver_type == :path
        sol, status, info = solve_with_path(
            parametric_mcp, θs, parameter_values;
            verbose, iteration_limit, proximal_perturbation, use_basics, use_start
        )
    else
        error("Unknown solver type: $solver_type. Use :linear or :path")
    end

    return (; sol, status, info, vars)
end

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
- `initial_state` - Initial state per player (Dict{Int, Vector} or Vector of Vectors)

# Keyword Arguments
- `verbose::Bool=false` - Print debug info

# Returns
- `JointStrategy` containing `OpenLoopStrategy` for each player
"""
function TrajectoryGamesBase.solve_trajectory_game!(
    solver::QPSolver,
    game::HierarchyGame,
    initial_state;
    verbose::Bool = false,
    kwargs...
)
    # Convert initial_state to parameter_values Dict
    if initial_state isa Dict
        parameter_values = initial_state
    elseif initial_state isa AbstractVector && eltype(initial_state) <: AbstractVector
        # Vector of vectors → Dict
        parameter_values = Dict(i => initial_state[i] for i in 1:length(initial_state))
    else
        error("initial_state must be Dict{Int, Vector} or Vector of Vectors")
    end

    return solve(solver, parameter_values; verbose)
end

"""
    solve(solver::NonlinearSolver, parameter_values::Dict; kwargs...)

Solve the nonlinear hierarchy game with given parameter values (typically initial states).

Uses precomputed symbolic components for efficiency.

# Arguments
- `solver::NonlinearSolver` - The nonlinear solver with precomputed components
- `parameter_values::Dict` - Numerical values for parameters (e.g., initial states per player)

# Keyword Arguments
- `initial_guess::Union{Nothing, Vector}=nothing` - Warm start for the solver
- Additional options override solver.options

# Returns
- `JointStrategy` containing `OpenLoopStrategy` for each player
"""
function solve(
    solver::NonlinearSolver,
    parameter_values::Dict;
    initial_guess::Union{Nothing, Vector} = nothing,
    max_iters::Union{Nothing, Int} = nothing,
    tol::Union{Nothing, Float64} = nothing,
    verbose::Union{Nothing, Bool} = nothing,
    use_armijo::Union{Nothing, Bool} = nothing
)
    (; problem, precomputed, options) = solver
    (; θs, primal_dims, state_dim, control_dim, hierarchy_graph) = problem

    # Validate parameter_values
    _validate_parameter_values(parameter_values, θs)

    # Use options from solver unless overridden
    actual_max_iters = something(max_iters, options.max_iters)
    actual_tol = something(tol, options.tol)
    actual_verbose = something(verbose, options.verbose)
    actual_use_armijo = something(use_armijo, options.use_armijo)

    # Run the nonlinear solver
    result = run_nonlinear_solver(
        precomputed,
        parameter_values,
        hierarchy_graph;
        initial_guess = initial_guess,
        max_iters = actual_max_iters,
        tol = actual_tol,
        verbose = actual_verbose,
        use_armijo = actual_use_armijo
    )

    return _extract_joint_strategy(result.sol, primal_dims, state_dim, control_dim)
end

"""
    solve_raw(solver::NonlinearSolver, parameter_values::Dict; kwargs...)

Solve and return raw solution with convergence info (for debugging/analysis).

# Keyword Arguments
- `initial_guess::Union{Nothing, Vector}=nothing` - Warm start for the solver
- `max_iters::Int` - Maximum iterations (default from solver.options)
- `tol::Float64` - Convergence tolerance (default from solver.options)
- `verbose::Bool` - Print iteration info (default from solver.options)
- `use_armijo::Bool` - Use Armijo line search (default from solver.options)

# Returns
Named tuple with fields:
- `sol::Vector{Float64}` - Solution vector (concatenated player decision variables)
- `converged::Bool` - Whether solver converged to tolerance
- `iterations::Int` - Number of iterations taken
- `residual::Float64` - Final KKT residual norm
- `status::Symbol` - Solver status:
  - `:solved` - Converged successfully
  - `:solved_initial_point` - Initial guess was already a solution
  - `:max_iters_reached` - Did not converge within iteration limit
  - `:linear_solver_error` - Newton step computation failed
  - `:numerical_error` - NaN or Inf encountered
"""
function solve_raw(
    solver::NonlinearSolver,
    parameter_values::Dict;
    initial_guess::Union{Nothing, Vector} = nothing,
    max_iters::Union{Nothing, Int} = nothing,
    tol::Union{Nothing, Float64} = nothing,
    verbose::Union{Nothing, Bool} = nothing,
    use_armijo::Union{Nothing, Bool} = nothing
)
    (; problem, precomputed, options) = solver
    (; hierarchy_graph) = problem

    # Use options from solver unless overridden
    actual_max_iters = something(max_iters, options.max_iters)
    actual_tol = something(tol, options.tol)
    actual_verbose = something(verbose, options.verbose)
    actual_use_armijo = something(use_armijo, options.use_armijo)

    # Run the nonlinear solver
    result = run_nonlinear_solver(
        precomputed,
        parameter_values,
        hierarchy_graph;
        initial_guess = initial_guess,
        max_iters = actual_max_iters,
        tol = actual_tol,
        verbose = actual_verbose,
        use_armijo = actual_use_armijo
    )

    return result
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
- `initial_state` - Initial state per player (Dict{Int, Vector} or Vector of Vectors)

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
    # Convert initial_state to parameter_values Dict
    if initial_state isa Dict
        parameter_values = initial_state
    elseif initial_state isa AbstractVector && eltype(initial_state) <: AbstractVector
        # Vector of vectors → Dict
        parameter_values = Dict(i => initial_state[i] for i in 1:length(initial_state))
    else
        error("initial_state must be Dict{Int, Vector} or Vector of Vectors")
    end

    return solve(solver, parameter_values; initial_guess, verbose)
end

#=
    Backend solver functions
=#

"""
    solve_with_path(parametric_mcp, θs::Dict, parameter_values::Dict; kwargs...)

Solve KKT system using PATH solver via ParametricMCPs.jl with cached MCP.

# Arguments
- `parametric_mcp` - Precomputed ParametricMCP from QPSolver construction
- `θs::Dict` - Symbolic parameter variables per player
- `parameter_values::Dict` - Numerical parameter values per player

# Keyword Arguments
- `initial_guess::Union{Nothing, Vector}=nothing` - Warm start
- `verbose::Bool=false` - Print solver output
- `iteration_limit::Int=100000` - Maximum iterations for PATH solver
- `proximal_perturbation::Float64=1e-2` - Proximal perturbation parameter
- `use_basics::Bool=true` - Use basic solution from PATH
- `use_start::Bool=true` - Use starting point from PATH

# Returns
Tuple of:
- `sol::Vector` - Solution vector
- `status::Symbol` - Solver status (:solved, :failed, etc.)
- `info` - PATH solver info
"""
function solve_with_path(
    parametric_mcp,
    θs::Dict,
    parameter_values::Dict;
    initial_guess::Union{Nothing, Vector} = nothing,
    verbose::Bool = false,
    iteration_limit::Int = 100000,
    proximal_perturbation::Float64 = 1e-2,
    use_basics::Bool = true,
    use_start::Bool = true,
)
    # Order parameters by player index
    order = sort(collect(keys(θs)))
    all_param_vals_vec = reduce(vcat, (parameter_values[k] for k in order))

    # Initial guess
    n = size(parametric_mcp.jacobian_z!.result_buffer, 1)
    z0 = initial_guess !== nothing ? initial_guess : zeros(n)

    # Solve
    sol, status, info = ParametricMCPs.solve(
        parametric_mcp,
        all_param_vals_vec;
        initial_guess = z0,
        verbose = verbose,
        cumulative_iteration_limit = iteration_limit,
        proximal_perturbation = proximal_perturbation,
        use_basics = use_basics,
        use_start = use_start,
    )

    return sol, status, info
end

"""
    solve_with_path(πs::Dict, variables::Vector, θs::Dict, parameter_values::Dict; kwargs...)

Solve KKT system using PATH solver via ParametricMCPs.jl (builds MCP internally).

# Arguments
- `πs::Dict` - KKT conditions per player
- `variables::Vector` - All symbolic variables
- `θs::Dict` - Symbolic parameter variables per player
- `parameter_values::Dict` - Numerical parameter values per player

# Keyword Arguments
- `initial_guess::Union{Nothing, Vector}=nothing` - Warm start
- `verbose::Bool=false` - Print solver output
- `iteration_limit::Int=100000` - Maximum iterations for PATH solver
- `proximal_perturbation::Float64=1e-2` - Proximal perturbation parameter
- `use_basics::Bool=true` - Use basic solution from PATH
- `use_start::Bool=true` - Use starting point from PATH

# Returns
Tuple of:
- `sol::Vector` - Solution vector
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
    iteration_limit::Int = 100000,
    proximal_perturbation::Float64 = 1e-2,
    use_basics::Bool = true,
    use_start::Bool = true,
)
    symbolic_type = eltype(variables)

    # Build KKT vector from all players
    F = Vector{symbolic_type}(vcat(collect(values(πs))...))

    # Bounds: unconstrained (MCP with -∞ to ∞)
    z_lower = fill(-Inf, length(F))
    z_upper = fill(Inf, length(F))

    # Order parameters by player index
    order = sort(collect(keys(πs)))
    all_θ_vec = reduce(vcat, (θs[k] for k in order))

    # Build parametric MCP
    parametric_mcp = ParametricMCPs.ParametricMCP(
        F, variables, all_θ_vec, z_lower, z_upper;
        compute_sensitivities = false
    )

    # Delegate to cached version
    return solve_with_path(
        parametric_mcp, θs, parameter_values;
        initial_guess, verbose, iteration_limit, proximal_perturbation, use_basics, use_start
    )
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

"""
    solve_qp_linear(parametric_mcp, θs::Dict, parameter_values::Dict; kwargs...)

Solve LQ game KKT system using direct linear solve with cached MCP.

For LQ games with only equality constraints, the KKT system is linear: Jz = -F
where J is the Jacobian and F is the KKT residual.

# Arguments
- `parametric_mcp` - Precomputed ParametricMCP from QPSolver construction
- `θs::Dict` - Symbolic parameter variables per player
- `parameter_values::Dict` - Numerical parameter values per player

# Keyword Arguments
- `verbose::Bool=false` - Print solver output

# Returns
Tuple of:
- `sol::Vector` - Solution vector
- `status::Symbol` - Solver status (:solved or :failed)
"""
function solve_qp_linear(
    parametric_mcp,
    θs::Dict,
    parameter_values::Dict;
    verbose::Bool = false
)
    # Order parameters by player index
    order = sort(collect(keys(θs)))
    all_param_vals_vec = reduce(vcat, (parameter_values[k] for k in order))

    # Get Jacobian buffer from MCP (handles sparse structure correctly)
    n = size(parametric_mcp.jacobian_z!.result_buffer, 1)
    J = copy(parametric_mcp.jacobian_z!.result_buffer)
    F = zeros(n)

    # Evaluate at zero (for LQ, any point works since system is linear)
    z0 = zeros(n)
    parametric_mcp.f!(F, z0, all_param_vals_vec)
    parametric_mcp.jacobian_z!(J, z0, all_param_vals_vec)

    # Solve Jz = -F using sparse backslash (dispatches to appropriate factorization).
    # Note: No regularization is applied. For ill-conditioned systems, this may fail
    # or produce inaccurate results. See Phase 5 bead for planned Tikhonov regularization.
    try
        sol = J \ (-F)

        # Check for NaN/Inf in solution (can occur with near-singular matrices)
        if any(!isfinite, sol)
            verbose && @warn "Linear solve produced non-finite values (possible near-singular matrix)"
            return fill(NaN, n), :failed
        end

        # Check residual quality
        residual = norm(J * sol + F)
        if residual > 1e-6 * max(1.0, norm(F))
            verbose && @warn "Linear solve has unexpectedly high residual: $residual"
        end

        verbose && println("Linear solve successful (residual: $residual)")
        return sol, :solved
    catch e
        # Only catch expected linear algebra failures; rethrow programming errors
        if e isa SingularException || e isa LAPACKException
            verbose && @warn "Linear solve failed: $e"
            # Return NaN-filled vector to clearly signal invalid solution
            # (zeros could be a valid solution for some problems)
            return fill(NaN, n), :failed
        end
        rethrow()
    end
end

"""
    solve_qp_linear(πs::Dict, variables::Vector, θs::Dict, parameter_values::Dict; kwargs...)

Solve LQ game KKT system using direct linear solve (builds MCP internally).

For LQ games with only equality constraints, the KKT system is linear: Jz = -F
where J is the Jacobian and F is the KKT residual.

# Arguments
- `πs::Dict` - KKT conditions per player
- `variables::Vector` - All symbolic variables
- `θs::Dict` - Symbolic parameter variables per player
- `parameter_values::Dict` - Numerical parameter values per player

# Keyword Arguments
- `verbose::Bool=false` - Print solver output

# Returns
Tuple of:
- `sol::Vector` - Solution vector
- `status::Symbol` - Solver status (:solved or :failed)
"""
function solve_qp_linear(
    πs::Dict,
    variables::Vector,
    θs::Dict,
    parameter_values::Dict;
    verbose::Bool = false
)
    symbolic_type = eltype(variables)

    # Build KKT vector from all players
    F_sym = Vector{symbolic_type}(vcat(collect(values(πs))...))

    # Order parameters by player index
    order = sort(collect(keys(πs)))
    all_θ_vec = reduce(vcat, (θs[k] for k in order))

    # Build parametric MCP
    z_lower = fill(-Inf, length(F_sym))
    z_upper = fill(Inf, length(F_sym))

    parametric_mcp = ParametricMCPs.ParametricMCP(
        F_sym, variables, all_θ_vec, z_lower, z_upper;
        compute_sensitivities = false
    )

    # Delegate to cached version
    return solve_qp_linear(parametric_mcp, θs, parameter_values; verbose)
end

#=
    Input validation utilities
=#

"""
    _validate_parameter_values(parameter_values::Dict, θs::Dict)

Validate parameter_values against expected θs structure. Throws ArgumentError on mismatch.
"""
function _validate_parameter_values(parameter_values::Dict, θs::Dict)
    for (player, θ) in θs
        if !haskey(parameter_values, player)
            throw(ArgumentError("parameter_values is missing entry for player $player."))
        end
        expected_len = length(θ)
        actual_len = length(parameter_values[player])
        if actual_len != expected_len
            throw(ArgumentError("parameter_values[$player] has length $actual_len, expected $expected_len."))
        end
    end
end

#=
    Solution extraction utilities
=#

"""
    extract_trajectories(sol::Vector, dims::NamedTuple, T::Int, n_players::Int)

Extract state and control trajectories from flattened solution vector.

# Arguments
- `sol::Vector` - Flattened solution
- `dims::NamedTuple` - Dimension info per player
- `T::Int` - Time horizon
- `n_players::Int` - Number of players

# Returns
- `xs::Dict{Int, Vector{Vector}}` - State trajectories per player
- `us::Dict{Int, Vector{Vector}}` - Control trajectories per player
"""
function extract_trajectories(sol::Vector, dims::NamedTuple, T::Int, n_players::Int)
    xs = Dict{Int, Vector{Vector{Float64}}}()
    us = Dict{Int, Vector{Vector{Float64}}}()

    idx = 1
    for i in 1:n_players
        state_dim = dims.state_dims[i]
        control_dim = dims.control_dims[i]

        # Extract states: T+1 states (t=0 to t=T)
        xs[i] = Vector{Vector{Float64}}()
        for t in 1:(T+1)
            push!(xs[i], sol[idx:idx+state_dim-1])
            idx += state_dim
        end

        # Extract controls: T controls (t=0 to t=T-1)
        us[i] = Vector{Vector{Float64}}()
        for t in 1:T
            push!(us[i], sol[idx:idx+control_dim-1])
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
