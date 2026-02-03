#=
    Utility functions for MixedHierarchyGames
=#

#=
    Graph topology utilities
=#

"""
    has_leader(G::SimpleDiGraph, node::Int)

Check if node has any incoming edges (i.e., has a leader).
"""
function has_leader(G::SimpleDiGraph, node::Int)
    return !isempty(inneighbors(G, node))
end

"""
    is_root(G::SimpleDiGraph, node::Int)

Check if node has no incoming edges (i.e., is a root/top-level leader).
"""
function is_root(G::SimpleDiGraph, node::Int)
    return isempty(inneighbors(G, node))
end

"""
    is_leaf(G::SimpleDiGraph, node::Int)

Check if node has no outgoing edges (i.e., is a leaf/pure follower).
"""
function is_leaf(G::SimpleDiGraph, node::Int)
    return isempty(outneighbors(G, node))
end

"""
    get_roots(G::SimpleDiGraph)

Return all root nodes (nodes with no incoming edges).
"""
function get_roots(G::SimpleDiGraph)
    return [v for v in vertices(G) if is_root(G, v)]
end

"""
    get_all_leaders(G::SimpleDiGraph, node::Int)

Return all ancestors of node up to root (all direct and indirect leaders).
"""
function get_all_leaders(G::SimpleDiGraph, node::Int)
    leaders = Int[]
    to_visit = collect(inneighbors(G, node))
    while !isempty(to_visit)
        leader = popfirst!(to_visit)
        push!(leaders, leader)
        append!(to_visit, inneighbors(G, leader))
    end
    return leaders
end

"""
    get_all_followers(G::SimpleDiGraph, node::Int)

Return all descendants of node (entire subtree of followers).
"""
function get_all_followers(G::SimpleDiGraph, node::Int)
    followers = Int[]
    to_visit = collect(outneighbors(G, node))
    while !isempty(to_visit)
        follower = popfirst!(to_visit)
        push!(followers, follower)
        append!(to_visit, outneighbors(G, follower))
    end
    return followers
end

#=
    Symbolic variable naming utilities
=#

"""
    make_symbolic_variable(name::Symbol, player::Int, indices...)

Construct consistent symbol names for optimization variables.

# Arguments
- `name::Symbol` - Variable type (:z, :u, :x, :λ, :ψ, :μ, :M, :N, :K, :θ)
- `player::Int` - Player index
- `indices...` - Additional indices (time step, dimension, etc.)

# Returns
- `Symbolics.Num` - Symbolic variable with consistent naming
"""
function make_symbolic_variable(name::Symbol, player::Int, indices...)
    # Build symbol string like "z_1_2_3" for player 1, indices 2, 3
    idx_str = join(vcat(player, collect(indices)), "_")
    sym_name = Symbol(name, "_", idx_str)
    return Symbolics.variable(sym_name)
end

"""
    make_symbolic_vector(name::Symbol, player::Int, dim::Int; start_idx::Int=1)

Create a vector of symbolic variables.

# Arguments
- `name::Symbol` - Variable type
- `player::Int` - Player index
- `dim::Int` - Vector dimension

# Returns
- `Vector{Symbolics.Num}` - Vector of symbolic variables
"""
function make_symbolic_vector(name::Symbol, player::Int, dim::Int; start_idx::Int=1)
    return [make_symbolic_variable(name, player, i) for i in start_idx:(start_idx + dim - 1)]
end

"""
    make_symbolic_vector(name::Symbol, leader::Int, follower::Int, dim::Int; start_idx::Int=1)

Create a vector of symbolic variables with leader-follower indexing.
Creates variables like μ_1_2_1, μ_1_2_2, ... for leader=1, follower=2.

# Arguments
- `name::Symbol` - Variable type
- `leader::Int` - Leader player index
- `follower::Int` - Follower player index
- `dim::Int` - Vector dimension

# Returns
- `Vector{Symbolics.Num}` - Vector of symbolic variables
"""
function make_symbolic_vector(name::Symbol, leader::Int, follower::Int, dim::Int; start_idx::Int=1)
    return [make_symbolic_variable(name, leader, follower, i) for i in start_idx:(start_idx + dim - 1)]
end

"""
    make_symbolic_matrix(name::Symbol, player::Int, rows::Int, cols::Int)

Create a matrix of symbolic variables.

# Arguments
- `name::Symbol` - Variable type
- `player::Int` - Player index
- `rows::Int` - Number of rows
- `cols::Int` - Number of columns

# Returns
- `Matrix{Symbolics.Num}` - Matrix of symbolic variables
"""
function make_symbolic_matrix(name::Symbol, player::Int, rows::Int, cols::Int)
    return [make_symbolic_variable(name, player, i, j) for i in 1:rows, j in 1:cols]
end

#=
    Solution validation utilities
=#

"""
    evaluate_kkt_residuals(
        πs::Dict,
        all_variables::Vector,
        z_sol::Vector,
        θs::Dict,
        parameter_values::Dict;
        tol::Float64 = 1e-6,
        verbose::Bool = false,
        should_enforce::Bool = false
    )

Evaluate symbolic KKT conditions at the numerical solution.

Substitutes numerical values and computes residual norms to verify solution validity.

# Arguments
- `πs::Dict{Int, Any}` - Symbolic KKT conditions per player
- `all_variables::Vector` - All symbolic decision variables
- `z_sol::Vector` - Numerical solution vector
- `θs::Dict{Int, Any}` - Symbolic parameter variables per player
- `parameter_values::Dict{Int, Vector}` - Numerical values for parameters

# Keyword Arguments
- `tol::Float64=1e-6` - Tolerance for checking satisfaction
- `verbose::Bool=false` - Print residual details
- `should_enforce::Bool=false` - Assert if conditions not satisfied

# Returns
- `π_eval::Vector{Float64}` - Evaluated residual vector
"""
function evaluate_kkt_residuals(
    πs::Dict,
    all_variables::Vector,
    z_sol::Vector,
    θs::Dict,
    parameter_values::Dict;
    tol::Float64 = 1e-6,
    verbose::Bool = false,
    should_enforce::Bool = false
)
    # TODO(Phase F): Implement for nonlinear solver debugging/validation
    error("evaluate_kkt_residuals is not yet implemented. Planned for nonlinear solver phase.")
end

#=
    Trajectory utilities
=#

"""
    flatten_trajectory(xs::Vector{Vector{T}}, us::Vector{Vector{T}}) where T

Flatten state and control trajectories into a single vector.
"""
function flatten_trajectory(xs::Vector{Vector{T}}, us::Vector{Vector{T}}) where T
    return vcat(vcat(xs...), vcat(us...))
end

# Note: unflatten_trajectory is imported from TrajectoryGamesBase
# It uses interleaved format: [x0, u0, x1, u1, ...]
