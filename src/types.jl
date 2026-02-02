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
    QPSolver

Solver for quadratic programming hierarchy games (linear dynamics, quadratic costs).

Uses direct KKT construction and PATH solver.

# Fields
- `horizon::Int` - Time horizon
- `dims::NamedTuple` - Problem dimensions per player
- `mcp_problem::Any` - Precomputed MCP problem representation (or nothing if not yet set up)
"""
struct QPSolver{T}
    horizon::Int
    dims::NamedTuple
    mcp_problem::T
end

"""
    QPSolver(game::HierarchyGame, horizon::Int)

Construct a QPSolver for the given hierarchy game.
"""
function QPSolver(game::HierarchyGame, horizon::Int)
    # TODO: Implement - extract dimensions, set up MCP problem
    error("Not implemented: QPSolver constructor")
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
