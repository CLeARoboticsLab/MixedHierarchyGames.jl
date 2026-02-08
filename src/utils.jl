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
    Player ordering utilities
=#

"""
    ordered_player_indices(d::Dict)

Return the keys of `d` as a sorted vector, providing a canonical player ordering.

This is a convenience wrapper around `sort(collect(keys(d)))` used throughout the
codebase to iterate over player-indexed dictionaries in a deterministic order.
"""
ordered_player_indices(d::Dict) = sort(collect(keys(d)))

#=
    Solution validation utilities
=#

"""
    evaluate_kkt_residuals(
        πs::Dict,
        all_variables::Vector,
        sol::Vector,
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
- `sol::Vector` - Numerical solution vector
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
    sol::Vector,
    θs::Dict,
    parameter_values::Dict;
    tol::Float64 = 1e-6,
    verbose::Bool = false,
    should_enforce::Bool = false
)
    # Order players consistently by index
    order = sort(collect(keys(πs)))

    # Concatenate all KKT conditions into a single vector
    all_πs = if isempty(order)
        eltype(all_variables)[]
    else
        vcat([πs[ii] for ii in order]...)
    end

    # Build combined variable vector: [decision vars; parameters]
    θ_order = sort(collect(keys(θs)))
    θ_vec = if isempty(θ_order)
        eltype(all_variables)[]
    else
        vcat([θs[ii] for ii in θ_order]...)
    end
    all_vars_and_params = vcat(all_variables, θ_vec)

    # Handle empty case
    if isempty(all_πs)
        return Float64[]
    end

    # Compile symbolic expressions to numerical function
    π_fn! = SymbolicTracingUtils.build_function(all_πs, all_vars_and_params; in_place=true)

    # Prepare parameter values
    param_vals_vec = if isempty(θ_order)
        Float64[]
    else
        vcat([parameter_values[ii] for ii in θ_order]...)
    end

    # Evaluate KKT residuals
    π_eval = zeros(Float64, length(all_πs))
    π_fn!(π_eval, vcat(sol, param_vals_vec))

    residual_norm = norm(π_eval)

    if verbose
        println("\n" * "="^20 * " KKT Residuals " * "="^20)
        for (idx, val) in enumerate(π_eval)
            if abs(val) >= tol
                println("  π[$idx] = $val")
            end
        end
        println("All KKT conditions satisfied (< $tol)? ", all(abs.(π_eval) .< tol))
        println("‖π‖₂ = ", residual_norm)
        println("="^55)
    end

    if should_enforce
        @assert residual_norm < tol "KKT conditions not satisfied. ‖π‖₂ = $residual_norm > $tol"
    end

    return π_eval
end

"""
    verify_kkt_solution(solver, sol::Vector, θs::Dict, parameter_values::Dict; kwargs...)

Verify that a solution satisfies the KKT conditions.
The solver must have `precomputed.setup_info.πs` and related fields (NonlinearSolver).

This is a convenience wrapper that extracts the necessary data from the solver
and calls `evaluate_kkt_residuals`.

Note: For NonlinearSolver, this function evaluates the KKT conditions with the
actual K (policy derivative) values computed at the solution. The residuals
should be small for a properly converged solution.

# Arguments
- `solver` - The solver used to find the solution (NonlinearSolver)
- `sol::Vector` - The solution vector (from `result.sol`)
- `θs::Dict` - Symbolic parameter variables per player
- `parameter_values::Dict` - Numerical values for parameters

# Keyword Arguments
- `tol::Float64=1e-6` - Tolerance for checking satisfaction
- `verbose::Bool=false` - Print residual details
- `should_enforce::Bool=false` - Assert if conditions not satisfied

# Returns
- `π_eval::Vector{Float64}` - Evaluated residual vector

# Example
```julia
result = solve_raw(solver, parameter_values)
residuals = verify_kkt_solution(solver, result.sol, θs, parameter_values; verbose=true)
```
"""
function verify_kkt_solution(
    solver,
    sol::Vector,
    θs::Dict,
    parameter_values::Dict;
    tol::Float64 = 1e-6,
    verbose::Bool = false,
    should_enforce::Bool = false
)
    # Extract data from solver's precomputed structures
    πs = solver.precomputed.setup_info.πs
    zs = solver.precomputed.problem_vars.zs
    G = solver.precomputed.setup_info.graph
    gs = solver.problem.gs
    problem_vars = solver.precomputed.problem_vars
    setup_info = solver.precomputed.setup_info

    # Get augmented variables (includes K matrix symbols)
    all_augmented_variables = solver.precomputed.all_augmented_variables

    # Compute K (policy) matrix values at the solution
    all_K_vec, _ = compute_K_evals(sol, problem_vars, setup_info)

    # Build augmented solution vector: [primal solution; K values]
    sol_augmented = vcat(sol, all_K_vec)

    # Strip policy constraints (keep only stationarity + equality constraints)
    πs_eval = strip_policy_constraints(πs, G, zs, gs)

    # Evaluate KKT residuals with augmented variables
    return evaluate_kkt_residuals(
        πs_eval, all_augmented_variables, sol_augmented, θs, parameter_values;
        tol = tol, verbose = verbose, should_enforce = should_enforce
    )
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
