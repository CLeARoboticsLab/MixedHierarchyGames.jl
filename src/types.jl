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
"""
struct QPProblem{TG<:SimpleDiGraph, TJ, TC, TP}
    hierarchy_graph::TG
    Js::TJ
    gs::TC
    primal_dims::Vector{Int}
    θs::TP
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
    QPSolver(hierarchy_graph, Js, gs, primal_dims, θs; solver=:linear)

Construct a QPSolver from low-level problem components (matches original interface).

# Arguments
- `hierarchy_graph::SimpleDiGraph` - DAG of leader-follower relationships
- `Js::Dict` - Cost functions per player
- `gs::Vector` - Constraint functions per player
- `primal_dims::Vector{Int}` - Decision variable dimension per player
- `θs::Dict` - Symbolic parameter variables per player

# Keyword Arguments
- `solver::Symbol=:linear` - Solver backend (:linear or :path)
"""
function QPSolver(
    hierarchy_graph::SimpleDiGraph,
    Js::Dict,
    gs::Vector,
    primal_dims::Vector{Int},
    θs::Dict;
    solver::Symbol = :linear
)
    problem = QPProblem(hierarchy_graph, Js, gs, primal_dims, θs)

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
    QPSolver(game::HierarchyGame, Js, gs, primal_dims, θs; solver=:linear)

Construct a QPSolver from a HierarchyGame with explicit cost/constraint functions.

Uses the hierarchy graph from the game but allows custom Js and gs.
"""
function QPSolver(
    game::HierarchyGame,
    Js::Dict,
    gs::Vector,
    primal_dims::Vector{Int},
    θs::Dict;
    solver::Symbol = :linear
)
    return QPSolver(game.hierarchy_graph, Js, gs, primal_dims, θs; solver)
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
