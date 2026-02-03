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
    QPProblem

Low-level problem specification for QP hierarchy games.
Stores cost functions, constraints, and symbolic variables.

# Fields
- `hierarchy_graph::SimpleDiGraph` - DAG of leader-follower relationships
- `Js::Dict` - Cost functions per player: Js[i](zs...; θ) → scalar
- `gs::Vector` - Constraint functions per player: gs[i](z) → Vector
- `primal_dims::Vector{Int}` - Decision variable dimension per player
- `θs::Dict` - Symbolic parameter variables per player
- `state_dim::Int` - State dimension per player (for trajectory extraction)
- `control_dim::Int` - Control dimension per player (for trajectory extraction)
"""
struct QPProblem{TG<:SimpleDiGraph, TJ, TC, TP}
    hierarchy_graph::TG
    Js::TJ
    gs::TC
    primal_dims::Vector{Int}
    θs::TP
    state_dim::Int
    control_dim::Int
end

"""
    QPSolver

Solver for quadratic programming hierarchy games (linear dynamics, quadratic costs).

# Fields
- `problem::QPProblem` - The problem specification
- `solver_type::Symbol` - Solver backend (:linear or :path)
- `precomputed::Any` - Precomputed symbolic components (variables, KKT conditions)
"""
struct QPSolver{TP<:QPProblem, TC}
    problem::TP
    solver_type::Symbol
    precomputed::TC
end

"""
    _validate_qpsolver_inputs(hierarchy_graph, Js, gs, primal_dims, θs)

Validate inputs for QPSolver constructor. Throws ArgumentError on invalid input.
"""
function _validate_qpsolver_inputs(hierarchy_graph::SimpleDiGraph, Js::Dict, gs::Vector, primal_dims::Vector{Int}, θs::Dict)
    N = nv(hierarchy_graph)

    # Graph structure validation
    if has_self_loops(hierarchy_graph)
        throw(ArgumentError("Hierarchy graph contains self-loops. Each player cannot be their own leader."))
    end
    if is_cyclic(hierarchy_graph)
        throw(ArgumentError("Hierarchy graph contains cycles. The hierarchy must be a DAG (directed acyclic graph)."))
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
    solver::Symbol = :linear
)
    # Validate inputs
    _validate_qpsolver_inputs(hierarchy_graph, Js, gs, primal_dims, θs)

    problem = QPProblem(hierarchy_graph, Js, gs, primal_dims, θs, state_dim, control_dim)

    # Precompute symbolic variables and KKT conditions
    vars = setup_problem_variables(hierarchy_graph, primal_dims, gs)
    θ_all = vcat([θs[k] for k in sort(collect(keys(θs)))]...)
    kkt_result = get_qp_kkt_conditions(
        hierarchy_graph, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys;
        θ = θ_all, verbose = false
    )
    πs_solve = strip_policy_constraints(kkt_result.πs, hierarchy_graph, vars.zs, gs)

    precomputed = (; vars, kkt_result, πs_solve)

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
    solver::Symbol = :linear
)
    return QPSolver(game.hierarchy_graph, Js, gs, primal_dims, θs, state_dim, control_dim; solver)
end

"""
    NonlinearSolver

Solver for general nonlinear hierarchy games.

Uses iterative quasi-linear policy approximation with Armijo line search.

# Fields
- `horizon::Int` - Time horizon
- `dims::NamedTuple` - Problem dimensions per player
- `precomputed::Any` - Precomputed symbolic components (or nothing if not yet set up)
- `options::NamedTuple` - Solver options (max_iters, tol, verbose, etc.)
"""
struct NonlinearSolver{T}
    horizon::Int
    dims::NamedTuple
    precomputed::T
    options::NamedTuple
end

"""
    NonlinearSolver(game::HierarchyGame, horizon::Int; kwargs...)

Construct a NonlinearSolver for the given hierarchy game.

# Keyword Arguments
- `max_iters::Int=100` - Maximum iterations
- `tol::Float64=1e-6` - Convergence tolerance
- `verbose::Bool=false` - Print iteration info
- `use_armijo::Bool=true` - Use Armijo line search
"""
function NonlinearSolver(game::HierarchyGame, horizon::Int; kwargs...)
    # TODO: Implement - extract dimensions, preoptimize symbolic components
    error("Not implemented: NonlinearSolver constructor")
end
