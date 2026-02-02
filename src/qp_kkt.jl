#=
    KKT construction for QP (Linear-Quadratic) hierarchy games
=#

"""
    get_qp_kkt_conditions(
        G::SimpleDiGraph,
        Js::Dict,
        zs,
        λs,
        μs::Dict,
        gs,
        ws::Dict,
        ys::Dict;
        θ = nothing,
        verbose::Bool = false
    )

Construct KKT conditions for all players in reverse topological order.

Handles both leaf and non-leaf players differently:
- Leaf players: Standard KKT (stationarity + constraints)
- Non-leaf players: KKT with policy constraints from followers

# Arguments
- `G::SimpleDiGraph` - DAG of leader-follower relationships
- `Js::Dict{Int, Function}` - Cost function for each player: Js[i](zs...; θ) → scalar
- `zs` - Decision variables per player (Dict or Vector)
- `λs` - Lagrange multipliers per player (Dict or Vector)
- `μs::Dict{Tuple{Int,Int}, Any}` - Policy constraint multipliers for (leader, follower) pairs
- `gs` - Constraint functions per player: gs[i](z) → Vector
- `ws::Dict` - Remaining variables (for policy constraints)
- `ys::Dict` - Information variables (leader decisions)
- `θ` - Parameter variables (optional)
- `verbose::Bool` - Print debug info

# Returns
Named tuple containing:
- `πs::Dict` - KKT conditions per player
- `Ms::Dict` - M matrices for followers (from Mw + Ny = 0)
- `Ns::Dict` - N matrices for followers
- `Ks::Dict` - Policy matrices K = M \\ N
"""
function get_qp_kkt_conditions(
    G::SimpleDiGraph,
    Js::Dict,
    zs,
    λs,
    μs::Dict,
    gs,
    ws::Dict,
    ys::Dict;
    θ = nothing,
    verbose::Bool = false
)
    N = nv(G)

    # Convert zs/λs to Dict if needed
    zs_dict = zs isa Dict ? zs : Dict(i => zs[i] for i in 1:length(zs))
    λs_dict = λs isa Dict ? λs : Dict(i => λs[i] for i in 1:length(λs))

    # Output containers
    πs = Dict{Int, Any}()
    Ms = Dict{Int, Any}()
    Ns = Dict{Int, Any}()
    Ks = Dict{Int, Any}()

    # Process in reverse topological order (leaves first)
    order = reverse(topological_sort_by_dfs(G))

    for ii in order
        zi = zs_dict[ii]
        zi_size = length(zi)

        # Build Lagrangian: L = J - λ'g
        # Cost function signature: Js[i](zs...; θ) or Js[i](zs...)
        all_zs = [zs_dict[j] for j in 1:N]
        Lᵢ = Js[ii](all_zs...; θ=θ) - λs_dict[ii]' * gs[ii](zi)

        if is_leaf(G, ii)
            # Leaf player: stationarity + constraints
            stationarity = Symbolics.gradient(Lᵢ, zi)
            constraints = gs[ii](zi)
            πs[ii] = vcat(stationarity, constraints)
        else
            # Leader: add follower policy constraint terms to Lagrangian
            for jj in get_all_followers(G, ii)
                if haskey(μs, (ii, jj)) && haskey(Ks, jj)
                    # Policy constraint: zⱼ = Kⱼ * yⱼ
                    # Extract the part of K corresponding to leader i's variables
                    extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))
                    Φⱼ = -extractor * Ks[jj] * ys[jj]
                    Lᵢ -= μs[(ii, jj)]' * (zs_dict[jj] - Φⱼ)
                end
            end

            # Build KKT: stationarity w.r.t own vars + follower vars + policy constraints
            πᵢ = Symbolics.gradient(Lᵢ, zi)

            for jj in get_all_followers(G, ii)
                # Stationarity w.r.t follower variables
                πᵢ = vcat(πᵢ, Symbolics.gradient(Lᵢ, zs_dict[jj]))

                # Policy constraint
                if haskey(Ks, jj)
                    extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))
                    Φⱼ = -extractor * Ks[jj] * ys[jj]
                    πᵢ = vcat(πᵢ, zs_dict[jj] - Φⱼ)
                end
            end

            # Own constraints
            πᵢ = vcat(πᵢ, gs[ii](zi))
            πs[ii] = πᵢ
        end

        # Compute M, N, K for followers (players with leaders)
        if has_leader(G, ii)
            Ms[ii] = Symbolics.jacobian(πs[ii], ws[ii])
            Ns[ii] = Symbolics.jacobian(πs[ii], ys[ii])
            Ks[ii] = Ms[ii] \ Ns[ii]
        end

        verbose && println("Player $ii: $(length(πs[ii])) KKT conditions")
    end

    (; πs, Ms, Ns, Ks)
end

"""
    strip_policy_constraints(πs::Dict, hierarchy_graph::SimpleDiGraph, zs::Dict, gs)

Remove policy constraint rows from KKT conditions, keeping only stationarity + constraints.

Used for solving the reduced KKT system.

# Arguments
- `πs::Dict` - Full KKT conditions per player
- `hierarchy_graph::SimpleDiGraph` - Hierarchy graph
- `zs::Dict` - Decision variables per player
- `gs` - Constraint functions per player

# Returns
- `πs_stripped::Dict` - KKT conditions without policy constraint rows
"""
function strip_policy_constraints(πs::Dict, hierarchy_graph::SimpleDiGraph, zs::Dict, gs)
    N = nv(hierarchy_graph)
    πs_stripped = Dict{Int, Any}()

    for ii in 1:N
        if is_leaf(hierarchy_graph, ii)
            # Leaf players: KKT unchanged (stationarity + constraints)
            πs_stripped[ii] = πs[ii]
        else
            # Leader: KKT has stationarity (own + followers) + policy constraints + own constraints
            # We need to strip the policy constraint rows

            zi_size = length(zs[ii])
            constraint_size = length(gs[ii](zs[ii]))
            followers = get_all_followers(hierarchy_graph, ii)

            # Count follower variable dimensions and policy constraint rows
            follower_var_dims = sum(length(zs[jj]) for jj in followers; init=0)
            policy_constraint_rows = sum(length(zs[jj]) for jj in followers; init=0)

            # Original structure: [stationarity_own, stationarity_followers, policy_constraints, own_constraints]
            # Sizes: [zi_size, follower_var_dims, policy_constraint_rows, constraint_size]

            stationarity_rows = zi_size + follower_var_dims

            # Keep: stationarity rows + own constraints
            # Strip: policy constraint rows (in the middle)
            keep_front = 1:stationarity_rows
            keep_back = (stationarity_rows + policy_constraint_rows + 1):length(πs[ii])

            πs_stripped[ii] = vcat(πs[ii][keep_front], πs[ii][keep_back])
        end
    end

    return πs_stripped
end

"""
    run_qp_solver(
        hierarchy_graph::SimpleDiGraph,
        costs::Dict,
        dynamics,
        T::Int,
        state_dims::Vector{Int},
        control_dims::Vector{Int},
        initial_states::Dict;
        kwargs...
    )

Main QP solver that orchestrates KKT construction and solving using PATH solver.

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
function run_qp_solver(
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
    error("Not implemented: run_qp_solver")
end
