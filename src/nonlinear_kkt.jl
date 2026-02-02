#=
    KKT construction and solving for nonlinear hierarchy games
=#

"""
    setup_approximate_kkt_solver(
        hierarchy_graph::SimpleDiGraph,
        costs::Dict,
        dynamics,
        zs::Dict,
        λs::Dict,
        μs::Dict,
        θs::Dict;
        kwargs...
    )

Precompute symbolic KKT conditions and M/N matrix evaluation functions for non-LQ solver.

# Arguments
- `hierarchy_graph::SimpleDiGraph` - Hierarchy graph
- `costs::Dict` - Cost functions per player
- `dynamics` - System dynamics
- `zs::Dict` - Decision variables
- `λs::Dict` - Dynamics multipliers
- `μs::Dict` - Inequality multipliers
- `θs::Dict` - Parameter variables

# Returns
Named tuple with precomputed symbolic functions for:
- KKT evaluation
- Jacobian evaluation
- M and N matrix computation
"""
function setup_approximate_kkt_solver(
    hierarchy_graph::SimpleDiGraph,
    costs::Dict,
    dynamics,
    zs::Dict,
    λs::Dict,
    μs::Dict,
    θs::Dict;
    kwargs...
)
    # TODO: Implement
    error("Not implemented: setup_approximate_kkt_solver")
end

"""
    preoptimize_nonlq_solver(
        hierarchy_graph::SimpleDiGraph,
        costs::Dict,
        dynamics,
        T::Int,
        state_dims::Vector{Int},
        control_dims::Vector{Int};
        kwargs...
    )

Precompute all symbolic components for non-LQ solver.

This is called once before solving to build all the symbolic expressions
and compile them to efficient numerical functions.

# Arguments
- `hierarchy_graph::SimpleDiGraph` - Hierarchy graph
- `costs::Dict` - Cost functions
- `dynamics` - System dynamics
- `T::Int` - Time horizon
- `state_dims::Vector{Int}` - State dimensions
- `control_dims::Vector{Int}` - Control dimensions

# Returns
Named tuple containing all precomputed components needed by `run_nonlq_solver`
"""
function preoptimize_nonlq_solver(
    hierarchy_graph::SimpleDiGraph,
    costs::Dict,
    dynamics,
    T::Int,
    state_dims::Vector{Int},
    control_dims::Vector{Int};
    kwargs...
)
    # TODO: Implement
    error("Not implemented: preoptimize_nonlq_solver")
end

"""
    compute_K_evals(
        z_current::Vector,
        precomputed::NamedTuple,
        hierarchy_graph::SimpleDiGraph
    )

Evaluate K (policy) matrices numerically in reverse topological order.

# Arguments
- `z_current::Vector` - Current solution estimate
- `precomputed::NamedTuple` - Precomputed symbolic functions
- `hierarchy_graph::SimpleDiGraph` - Hierarchy graph

# Returns
- `K_evals::Dict` - Numerical K matrices per player
"""
function compute_K_evals(
    z_current::Vector,
    precomputed::NamedTuple,
    hierarchy_graph::SimpleDiGraph
)
    # TODO: Implement
    error("Not implemented: compute_K_evals")
end

"""
    run_nonlq_solver(
        precomputed::NamedTuple,
        initial_states::Dict,
        hierarchy_graph::SimpleDiGraph;
        initial_guess::Union{Nothing, Vector} = nothing,
        max_iters::Int = 100,
        tol::Float64 = 1e-6,
        verbose::Bool = false,
        use_armijo::Bool = true
    )

Iterative non-LQ solver using quasi-linear policy approximation.

Uses Armijo backtracking line search for step size selection.

# Arguments
- `precomputed::NamedTuple` - Precomputed symbolic components from `preoptimize_nonlq_solver`
- `initial_states::Dict` - Initial state for each player
- `hierarchy_graph::SimpleDiGraph` - Hierarchy graph

# Keyword Arguments
- `initial_guess::Vector` - Starting point (or nothing for zero initialization)
- `max_iters::Int=100` - Maximum iterations
- `tol::Float64=1e-6` - Convergence tolerance on KKT residual
- `verbose::Bool=false` - Print iteration info
- `use_armijo::Bool=true` - Use Armijo line search

# Returns
Named tuple containing:
- `z_sol::Vector` - Solution vector
- `xs::Dict` - State trajectories per player
- `us::Dict` - Control trajectories per player
- `converged::Bool` - Whether solver converged
- `iterations::Int` - Number of iterations taken
- `residual::Float64` - Final KKT residual norm
"""
function run_nonlq_solver(
    precomputed::NamedTuple,
    initial_states::Dict,
    hierarchy_graph::SimpleDiGraph;
    initial_guess::Union{Nothing, Vector} = nothing,
    max_iters::Int = 100,
    tol::Float64 = 1e-6,
    verbose::Bool = false,
    use_armijo::Bool = true
)
    # TODO: Implement
    error("Not implemented: run_nonlq_solver")
end

"""
    armijo_backtracking_linesearch(
        f_eval::Function,
        z::Vector,
        δz::Vector,
        f_z::Vector;
        α_init::Float64 = 1.0,
        β::Float64 = 0.5,
        σ::Float64 = 1e-4,
        max_iters::Int = 20
    )

Armijo backtracking line search for step size selection.

# Arguments
- `f_eval::Function` - Function evaluating residual at a point
- `z::Vector` - Current point
- `δz::Vector` - Search direction
- `f_z::Vector` - Residual at current point

# Keyword Arguments
- `α_init::Float64=1.0` - Initial step size
- `β::Float64=0.5` - Step size reduction factor
- `σ::Float64=1e-4` - Sufficient decrease parameter
- `max_iters::Int=20` - Maximum line search iterations

# Returns
- `α::Float64` - Selected step size
"""
function armijo_backtracking_linesearch(
    f_eval::Function,
    z::Vector,
    δz::Vector,
    f_z::Vector;
    α_init::Float64 = 1.0,
    β::Float64 = 0.5,
    σ::Float64 = 1e-4,
    max_iters::Int = 20
)
    # TODO: Implement
    error("Not implemented: armijo_backtracking_linesearch")
end
