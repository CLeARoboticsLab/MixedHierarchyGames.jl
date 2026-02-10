#=
    Types for MixedHierarchyGames solvers
=#

"""
    HierarchyGame

A trajectory game with hierarchical (Stackelberg) structure.

# Fields
- `game::TrajectoryGame` - The underlying trajectory game
- `hierarchy_graph::SimpleDiGraph` - DAG representing leader-follower relationships
  (edge i→j means player i is a leader of player j)
"""
struct HierarchyGame{TG<:TrajectoryGame, TH<:SimpleDiGraph}
    game::TG
    hierarchy_graph::TH
end

"""
    HierarchyProblem

Low-level problem specification for hierarchy games.
Stores cost functions, constraints, and symbolic variables.
Used by both QPSolver and NonlinearSolver.

# Fields
- `hierarchy_graph::SimpleDiGraph` - DAG of leader-follower relationships
- `Js::Dict` - Cost functions per player: Js[i](zs...; θ) → scalar
- `gs::Vector` - Constraint functions per player: gs[i](z) → Vector
- `primal_dims::Vector{Int}` - Decision variable dimension per player
- `θs::Dict` - Symbolic parameter variables per player
- `state_dim::Int` - State dimension per player (for trajectory extraction)
- `control_dim::Int` - Control dimension per player (for trajectory extraction)
"""
struct HierarchyProblem{TG<:SimpleDiGraph, TJ, TC, TP}
    hierarchy_graph::TG
    Js::TJ
    gs::TC
    primal_dims::Vector{Int}
    θs::TP
    state_dim::Int
    control_dim::Int
end

"""
    QPPrecomputed

Precomputed components for QPSolver, cached during construction for efficient repeated solves.

# Type Parameters
- `TV` - Type of problem variables (from setup_problem_variables)
- `TK` - Type of KKT conditions result (from get_qp_kkt_conditions)
- `TP` - Type of stripped policy constraints (Dict)
- `TM` - Type of parametric MCP (ParametricMCP)
- `TJ` - Type of Jacobian buffer (sparse or dense matrix)

# Fields
- `vars::TV` - Problem variables from setup_problem_variables
- `kkt_result::TK` - KKT conditions from get_qp_kkt_conditions
- `πs_solve::TP` - Stripped policy constraints for solving
- `parametric_mcp::TM` - Cached ParametricMCP for solving
- `J_buffer::TJ` - Pre-allocated Jacobian buffer for solve_qp_linear
- `F_buffer::Vector{Float64}` - Pre-allocated residual buffer for solve_qp_linear
- `z0_buffer::Vector{Float64}` - Pre-allocated zero vector for solve_qp_linear
"""
struct QPPrecomputed{TV, TK, TP, TM, TJ}
    vars::TV
    kkt_result::TK
    πs_solve::TP
    parametric_mcp::TM
    J_buffer::TJ
    F_buffer::Vector{Float64}
    z0_buffer::Vector{Float64}
end

"""
    QPSolver

Solver for quadratic programming hierarchy games (linear dynamics, quadratic costs).

# Fields
- `problem::HierarchyProblem` - The problem specification
- `solver_type::Symbol` - Solver backend (:linear or :path)
- `precomputed::QPPrecomputed` - Precomputed symbolic components (variables, KKT conditions)
"""
struct QPSolver{TP<:HierarchyProblem, TC<:QPPrecomputed}
    problem::TP
    solver_type::Symbol
    precomputed::TC
end

"""
    _validate_solver_inputs(hierarchy_graph, Js, gs, primal_dims, θs)

Validate inputs for solver constructors. Throws ArgumentError on invalid input.
Used by both QPSolver and NonlinearSolver.
"""
function _validate_solver_inputs(hierarchy_graph::SimpleDiGraph, Js::Dict, gs::Vector, primal_dims::Vector{Int}, θs::Dict)
    N = nv(hierarchy_graph)

    # Graph structure validation
    if has_self_loops(hierarchy_graph)
        throw(ArgumentError(
            "Hierarchy graph contains self-loops. Each player cannot be their own leader. " *
            "Check your add_edge! calls - ensure no edge (i, i) exists."
        ))
    end
    if is_cyclic(hierarchy_graph)
        throw(ArgumentError(
            "Hierarchy graph contains cycles. The hierarchy must be a DAG (directed acyclic graph). " *
            "Use Graphs.is_cyclic(G) to check, and Graphs.simplecycles(G) to find cycles."
        ))
    end

    # Single-parent validation: each player can have at most one leader
    for v in 1:N
        num_leaders = indegree(hierarchy_graph, v)
        if num_leaders > 1
            leaders = inneighbors(hierarchy_graph, v)
            throw(ArgumentError(
                "Player $v has multiple leaders ($(collect(leaders))). " *
                "The current implementation assumes each player has at most one parent in the hierarchy. " *
                "Consider restructuring your hierarchy graph."
            ))
        end
    end

    # Dimension consistency validation
    if length(primal_dims) != N
        throw(ArgumentError("Length of primal_dims ($(length(primal_dims))) must match number of players ($N)."))
    end
    if length(gs) != N
        throw(ArgumentError("Length of gs ($(length(gs))) must match number of players ($N)."))
    end

    # Check Js has all players
    for i in 1:N
        if !haskey(Js, i)
            throw(ArgumentError("Js is missing cost function for player $i."))
        end
    end

    # Check θs has all players
    for i in 1:N
        if !haskey(θs, i)
            throw(ArgumentError("θs is missing parameter variables for player $i."))
        end
    end
end

"""
    _build_parametric_mcp(πs::Dict, variables::Vector, θs::Dict)

Build a ParametricMCP from KKT conditions during QPSolver construction.

The MCP is cached in `precomputed` to avoid rebuilding it on each solve() call.
This provides significant performance benefits for repeated solves with different
parameter values (e.g., different initial states).

Note: We use ParametricMCPs primarily for its compiled f! and jacobian_z! functions.
The bounds infrastructure (-∞ to ∞) is unused since we only support equality constraints.
See task MixedHierarchyGames.jl-s6u for potential future optimization.
"""
function _build_parametric_mcp(πs::Dict, variables::Vector, θs::Dict)
    symbolic_type = eltype(variables)
    F_sym = Vector{symbolic_type}(vcat(collect(values(πs))...))

    # Order parameters by player index for consistency
    order = sort(collect(keys(πs)))
    all_θ_vec = reduce(vcat, (θs[k] for k in order))

    # Unconstrained bounds (equality-only KKT)
    z_lower = fill(-Inf, length(F_sym))
    z_upper = fill(Inf, length(F_sym))

    return ParametricMCPs.ParametricMCP(
        F_sym, variables, all_θ_vec, z_lower, z_upper;
        compute_sensitivities = false
    )
end

"""
    _verify_linear_system(mcp, n::Int, θs::Dict)

Verify that the KKT system is affine (constant Jacobian) as required for QP/LQ games.

QPSolver assumes the KKT system F(z) = 0 is affine in z, i.e., F(z) = Jz + b where J
is constant. This allows direct linear solve: z = -J⁻¹b. If the system is nonlinear,
the linear solve will produce incorrect results.

This check evaluates the Jacobian at two random points during construction. If they
differ, a warning is issued. The check runs once at construction time, not at solve time.
"""
function _verify_linear_system(mcp, n::Int, θs::Dict)
    # Check if jacobian buffer is available (defensive check)
    if !hasproperty(mcp.jacobian_z!, :result_buffer)
        @warn "Cannot verify linearity: jacobian buffer not available"
        return
    end

    # Generate random test points
    z1, z2 = randn(n), randn(n)

    # Create dummy parameter values (actual values don't affect linearity check)
    order = sort(collect(keys(θs)))
    θ_vals = reduce(vcat, (zeros(length(θs[k])) for k in order))

    # Allocate Jacobian buffers
    J1 = copy(mcp.jacobian_z!.result_buffer)
    J2 = copy(mcp.jacobian_z!.result_buffer)

    # Evaluate Jacobians
    mcp.jacobian_z!(J1, z1, θ_vals)
    mcp.jacobian_z!(J2, z2, θ_vals)

    # Check if Jacobians are equal (system is linear)
    if !isapprox(J1, J2; rtol=1e-10, atol=1e-12)
        @warn "KKT system Jacobian varies with z - system may not be linear (QP). " *
              "QPSolver assumes linear KKT conditions. Results may be incorrect."
    end
end

"""
    QPSolver(hierarchy_graph, Js, gs, primal_dims, θs, state_dim, control_dim; solver=:linear)

Construct a QPSolver from low-level problem components (matches original interface).

# Arguments
- `hierarchy_graph::SimpleDiGraph` - DAG of leader-follower relationships
- `Js::Dict` - Cost functions per player
- `gs::Vector` - Constraint functions per player
- `primal_dims::Vector{Int}` - Decision variable dimension per player
- `θs::Dict` - Symbolic parameter variables per player
- `state_dim::Int` - State dimension per player
- `control_dim::Int` - Control dimension per player

# Keyword Arguments
- `solver::Symbol=:linear` - Solver backend (:linear or :path)
"""
function QPSolver(
    hierarchy_graph::SimpleDiGraph,
    Js::Dict,
    gs::Vector,
    primal_dims::Vector{Int},
    θs::Dict,
    state_dim::Int,
    control_dim::Int;
    solver::Symbol = :linear,
    to::TimerOutput = TimerOutput()
)
    @timeit to "QPSolver construction" begin
        # Validate inputs
        _validate_solver_inputs(hierarchy_graph, Js, gs, primal_dims, θs)

        problem = HierarchyProblem(hierarchy_graph, Js, gs, primal_dims, θs, state_dim, control_dim)

        # Precompute symbolic variables and KKT conditions
        # Note: setup_problem_variables validates constraint function signatures internally
        vars = setup_problem_variables(hierarchy_graph, primal_dims, gs)

        @timeit to "KKT conditions" begin
            θ_all = reduce(vcat, (θs[k] for k in sort(collect(keys(θs)))))
            kkt_result = get_qp_kkt_conditions(
                hierarchy_graph, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys, vars.ws_z_indices;
                θ = θ_all, verbose = false
            )
            πs_solve = strip_policy_constraints(kkt_result.πs, hierarchy_graph, vars.zs, gs)
        end

        # Build and cache ParametricMCP for solving
        @timeit to "ParametricMCP build" begin
            parametric_mcp = _build_parametric_mcp(πs_solve, vars.all_variables, θs)
        end

        # Verify the system is linear (QP assumption) during construction
        @timeit to "linearity check" begin
            _verify_linear_system(parametric_mcp, length(vars.all_variables), θs)
        end

        # Pre-allocate solve buffers for solve_qp_linear
        n_vars = size(parametric_mcp.jacobian_z!.result_buffer, 1)
        J_buffer = copy(parametric_mcp.jacobian_z!.result_buffer)
        F_buffer = zeros(n_vars)
        z0_buffer = zeros(n_vars)

        precomputed = QPPrecomputed(vars, kkt_result, πs_solve, parametric_mcp,
                                    J_buffer, F_buffer, z0_buffer)
    end

    return QPSolver(problem, solver, precomputed)
end

"""
    QPSolver(game::HierarchyGame, Js, gs, primal_dims, θs, state_dim, control_dim; solver=:linear)

Construct a QPSolver from a HierarchyGame with explicit cost/constraint functions.

Uses the hierarchy graph from the game but allows custom Js and gs.
"""
function QPSolver(
    game::HierarchyGame,
    Js::Dict,
    gs::Vector,
    primal_dims::Vector{Int},
    θs::Dict,
    state_dim::Int,
    control_dim::Int;
    solver::Symbol = :linear,
    to::TimerOutput = TimerOutput()
)
    return QPSolver(game.hierarchy_graph, Js, gs, primal_dims, θs, state_dim, control_dim; solver, to)
end

"""
    NonlinearSolver

Solver for general nonlinear hierarchy games.

Uses iterative quasi-linear policy approximation with Armijo line search.

# Fields
- `problem::HierarchyProblem` - The problem specification
- `precomputed::NamedTuple` - Precomputed symbolic components from preoptimize_nonlinear_solver
- `options::NamedTuple` - Solver options (max_iters, tol, verbose, use_armijo)
"""
struct NonlinearSolver{TP<:HierarchyProblem, TC}
    problem::TP
    precomputed::TC
    options::NamedTuple
end

"""
    NonlinearSolver(hierarchy_graph, Js, gs, primal_dims, θs, state_dim, control_dim; kwargs...)

Construct a NonlinearSolver from low-level problem components.

# Arguments
- `hierarchy_graph::SimpleDiGraph` - DAG of leader-follower relationships
- `Js::Dict` - Cost functions per player: Js[i](z1, z2, ..., zN; θ) → scalar
- `gs::Vector` - Constraint functions per player: gs[i](z) → Vector
- `primal_dims::Vector{Int}` - Decision variable dimension per player
- `θs::Dict` - Symbolic parameter variables per player
- `state_dim::Int` - State dimension per player
- `control_dim::Int` - Control dimension per player

# Keyword Arguments
- `max_iters::Int=100` - Maximum iterations
- `tol::Float64=1e-6` - Convergence tolerance
- `verbose::Bool=false` - Print iteration info
- `use_armijo::Bool=true` - Use Armijo line search
"""
function NonlinearSolver(
    hierarchy_graph::SimpleDiGraph,
    Js::Dict,
    gs::Vector,
    primal_dims::Vector{Int},
    θs::Dict,
    state_dim::Int,
    control_dim::Int;
    max_iters::Int = 100,
    tol::Float64 = 1e-6,
    verbose::Bool = false,
    use_armijo::Bool = true,
    to::TimerOutput = TimerOutput()
)
    @timeit to "NonlinearSolver construction" begin
        # Validate inputs
        _validate_solver_inputs(hierarchy_graph, Js, gs, primal_dims, θs)

        # Create problem specification
        problem = HierarchyProblem(hierarchy_graph, Js, gs, primal_dims, θs, state_dim, control_dim)

        # Precompute symbolic components
        precomputed = preoptimize_nonlinear_solver(
            hierarchy_graph, Js, gs, primal_dims, θs;
            state_dim = state_dim,
            control_dim = control_dim,
            verbose = verbose,
            to = to
        )

        # Store solver options
        options = (; max_iters, tol, verbose, use_armijo)
    end

    return NonlinearSolver(problem, precomputed, options)
end

"""
    NonlinearSolver(game::HierarchyGame, Js, gs, primal_dims, θs, state_dim, control_dim; kwargs...)

Construct a NonlinearSolver from a HierarchyGame with explicit cost/constraint functions.
"""
function NonlinearSolver(
    game::HierarchyGame,
    Js::Dict,
    gs::Vector,
    primal_dims::Vector{Int},
    θs::Dict,
    state_dim::Int,
    control_dim::Int;
    to::TimerOutput = TimerOutput(),
    kwargs...
)
    return NonlinearSolver(game.hierarchy_graph, Js, gs, primal_dims, θs, state_dim, control_dim; to, kwargs...)
end
