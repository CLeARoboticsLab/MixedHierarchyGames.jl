#=
    Problem setup functions for constructing symbolic variables
=#

"""
    setup_problem_parameter_variables(num_params_per_player::Vector{Int})

Create symbolic parameter variables (θ) for each player's initial state.

# Arguments
- `num_params_per_player::Vector{Int}` - Number of parameters for each player

# Returns
- `θs::Dict{Int, Vector{Num}}` - Dictionary mapping player index to their parameter variables
"""
function setup_problem_parameter_variables(num_params_per_player::Vector{Int})
    θs = Dict{Int, Any}()
    for idx in 1:length(num_params_per_player)
        θs[idx] = make_symbolic_vector(:θ, idx, num_params_per_player[idx])
    end
    return θs
end

"""
    setup_problem_variables(
        graph::SimpleDiGraph,
        primal_dims::Vector{Int},
        gs::Vector
    )

Construct all symbolic variables needed for the KKT system.

# Arguments
- `graph::SimpleDiGraph` - DAG of leader-follower relationships
- `primal_dims::Vector{Int}` - Decision variable dimension for each player
- `gs::Vector` - Constraint functions for each player (gs[i](z) returns constraints)

# Returns
Named tuple containing:
- `zs::Dict` - Decision variables per player
- `λs::Dict` - Lagrange multipliers for constraints per player
- `μs::Dict` - Policy constraint multipliers (leader, follower) pairs
- `ys::Dict` - Information vectors (leader decisions visible to each player)
- `ws::Dict` - Remaining variables for policy constraints
- `all_variables::Vector` - Flattened vector of all variables
"""
function setup_problem_variables(
    graph::SimpleDiGraph,
    primal_dims::Vector{Int},
    gs::Vector
)
    N = nv(graph)

    # Create decision variables for each player
    zs = Dict{Int, Any}()
    for i in 1:N
        zs[i] = make_symbolic_vector(:z, i, primal_dims[i])
    end

    # Create Lagrange multipliers based on constraint dimensions
    λs = Dict{Int, Any}()
    for i in 1:N
        constraint_dim = length(gs[i](zs[i]))
        λs[i] = make_symbolic_vector(:λ, i, constraint_dim)
    end

    # Create policy constraint multipliers for leader-follower pairs
    μs = Dict{Tuple{Int, Int}, Any}()
    for i in 1:N
        followers = get_all_followers(graph, i)
        for j in followers
            μs[(i, j)] = make_symbolic_vector(:μ, i * 10 + j, primal_dims[j])
        end
    end

    # Information vectors: ys[i] contains decisions of all leaders of i
    ys = Dict{Int, Any}()
    for i in 1:N
        leaders = get_all_leaders(graph, i)
        if isempty(leaders)
            ys[i] = Symbolics.Num[]
        else
            ys[i] = vcat([zs[l] for l in leaders]...)
        end
    end

    # Remaining variables for policy constraints
    ws = Dict{Int, Any}()
    for i in 1:N
        leaders = get_all_leaders(graph, i)
        # ws[i] starts with own variables
        ws[i] = copy(zs[i])
        # Add non-leader, non-self variables
        for j in 1:N
            if j != i && !(j in leaders)
                ws[i] = vcat(ws[i], zs[j])
            end
        end
        # Add λs for self and followers
        ws[i] = vcat(ws[i], λs[i])
        for j in get_all_followers(graph, i)
            ws[i] = vcat(ws[i], λs[j])
        end
        # Add μs for follower policies
        for j in get_all_followers(graph, i)
            if haskey(μs, (i, j))
                ws[i] = vcat(ws[i], μs[(i, j)])
            end
        end
    end

    # Flatten all variables
    all_variables = vcat(
        vcat([zs[i] for i in 1:N]...),
        vcat([λs[i] for i in 1:N]...),
        (isempty(μs) ? Symbolics.Num[] : vcat(collect(values(μs))...))
    )

    (; zs, λs, μs, ys, ws, all_variables)
end

"""
    construct_augmented_variables(zs, Ks, hierarchy_graph)

Build augmented variable list including symbolic K matrices for optimized solving.

# Arguments
- `zs::Dict` - Decision variables per player
- `Ks::Dict` - Policy matrices per player
- `hierarchy_graph::SimpleDiGraph` - Hierarchy graph

# Returns
- `augmented_vars::Vector` - Variables augmented with K matrix entries
"""
function construct_augmented_variables(zs, Ks, hierarchy_graph)
    # TODO: Implement
    error("Not implemented: construct_augmented_variables")
end
