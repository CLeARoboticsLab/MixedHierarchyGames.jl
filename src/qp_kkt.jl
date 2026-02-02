#=
    KKT construction for QP (Linear-Quadratic) hierarchy games
=#

"""
    get_lq_kkt_conditions(
        hierarchy_graph::SimpleDiGraph,
        costs::Dict,
        dynamics,
        zs::Dict,
        λs::Dict,
        μs::Dict,
        θs::Dict;
        inequality_constraints::Dict = Dict()
    )

Construct KKT conditions for all players in reverse topological order.

Handles both leaf and non-leaf players differently:
- Leaf players: Standard KKT (stationarity + constraints)
- Non-leaf players: KKT with policy constraints from followers

# Arguments
- `hierarchy_graph::SimpleDiGraph` - DAG of leader-follower relationships
- `costs::Dict{Int, Function}` - Cost function for each player
- `dynamics` - System dynamics function
- `zs::Dict` - Decision variables per player
- `λs::Dict` - Dynamics constraint multipliers
- `μs::Dict` - Inequality constraint multipliers
- `θs::Dict` - Parameter variables (initial states)
- `inequality_constraints::Dict` - Inequality constraints per player

# Returns
Named tuple containing:
- `πs::Dict` - KKT conditions per player
- `Ks::Dict` - Policy matrices per player (for non-leaf players)
- `all_kkt::Vector` - Flattened KKT conditions
"""
function get_lq_kkt_conditions(
    hierarchy_graph::SimpleDiGraph,
    costs::Dict,
    dynamics,
    zs::Dict,
    λs::Dict,
    μs::Dict,
    θs::Dict;
    inequality_constraints::Dict = Dict()
)
    # TODO: Implement - build KKT conditions in reverse topological order
    error("Not implemented: get_lq_kkt_conditions")
end

"""
    strip_policy_constraints(πs::Dict, hierarchy_graph::SimpleDiGraph)

Remove policy constraint rows from KKT conditions, keeping only stationarity + constraints.

Used for solving the reduced KKT system.

# Arguments
- `πs::Dict` - Full KKT conditions per player
- `hierarchy_graph::SimpleDiGraph` - Hierarchy graph

# Returns
- `πs_stripped::Dict` - KKT conditions without policy constraint rows
"""
function strip_policy_constraints(πs::Dict, hierarchy_graph::SimpleDiGraph)
    # TODO: Implement
    error("Not implemented: strip_policy_constraints")
end

"""
    run_lq_solver(
        hierarchy_graph::SimpleDiGraph,
        costs::Dict,
        dynamics,
        T::Int,
        state_dims::Vector{Int},
        control_dims::Vector{Int},
        initial_states::Dict;
        kwargs...
    )

Main LQ solver that orchestrates KKT construction and solving using PATH solver.

# Arguments
- `hierarchy_graph::SimpleDiGraph` - Hierarchy graph
- `costs::Dict` - Cost functions
- `dynamics` - System dynamics
- `T::Int` - Time horizon
- `state_dims::Vector{Int}` - State dimensions
- `control_dims::Vector{Int}` - Control dimensions
- `initial_states::Dict` - Initial state for each player

# Keyword Arguments
- `inequality_constraints::Dict` - Inequality constraints
- `verbose::Bool=false` - Print debug info

# Returns
Named tuple containing:
- `z_sol::Vector` - Solution vector
- `xs::Dict` - State trajectories per player
- `us::Dict` - Control trajectories per player
- `info::NamedTuple` - Additional solver info
"""
function run_lq_solver(
    hierarchy_graph::SimpleDiGraph,
    costs::Dict,
    dynamics,
    T::Int,
    state_dims::Vector{Int},
    control_dims::Vector{Int},
    initial_states::Dict;
    kwargs...
)
    # TODO: Implement
    error("Not implemented: run_lq_solver")
end
