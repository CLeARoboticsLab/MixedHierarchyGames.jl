#=
    Problem setup functions for constructing symbolic variables
=#

#=
    Symbol naming utilities
=#

# Player variables:     :z, :λ, :θ, :u, :x, :M, :N, :K
const PLAYER_SYMBOLS = (:z, :λ, :θ, :u, :x, :M, :N, :K)

# Pair variables (leader-follower): :μ
const PAIR_SYMBOLS = (:μ,)

const ALL_SYMBOLS = (PLAYER_SYMBOLS..., PAIR_SYMBOLS...)

"""
    make_symbol(name::Symbol, player::Int)

Create a symbol for a player variable. Valid names: $PLAYER_SYMBOLS

Returns `Symbol("name^player")`, e.g., `Symbol("z^1")`.
"""
function make_symbol(name::Symbol, player::Int)
    name in PLAYER_SYMBOLS || throw(ArgumentError(
        "Symbol :$name is not a player variable. Use one of: $PLAYER_SYMBOLS"))
    return Symbol(name, "^", player)
end

"""
    make_symbol(name::Symbol, leader::Int, follower::Int)

Create a symbol for a leader-follower pair variable. Valid names: $PAIR_SYMBOLS

Returns `Symbol("name^(leader-follower)")`, e.g., `Symbol("μ^(1-2)")`.
"""
function make_symbol(name::Symbol, leader::Int, follower::Int)
    name in PAIR_SYMBOLS || throw(ArgumentError(
        "Symbol :$name is not a pair variable. Use one of: $PAIR_SYMBOLS"))
    return Symbol(name, "^(", leader, "-", follower, ")")
end

#=
    Symbolic variable creation
=#

"Return a fresh SymbolicsBackend instance for symbolic tracing."
default_backend() = SymbolicTracingUtils.SymbolicsBackend()

"""
    make_symbolic_vector(name::Symbol, player::Int, dim::Int; backend=default_backend())

Create a vector of `dim` symbolic variables for a player.

Valid names: $PLAYER_SYMBOLS
"""
function make_symbolic_vector(name::Symbol, player::Int, dim::Int; backend=default_backend())
    sym = make_symbol(name, player)
    return SymbolicTracingUtils.make_variables(backend, sym, dim)
end

"""
    make_symbolic_vector(name::Symbol, leader::Int, follower::Int, dim::Int; backend=default_backend())

Create a vector of `dim` symbolic variables for a leader-follower pair.

Valid names: $PAIR_SYMBOLS
"""
function make_symbolic_vector(name::Symbol, leader::Int, follower::Int, dim::Int; backend=default_backend())
    sym = make_symbol(name, leader, follower)
    return SymbolicTracingUtils.make_variables(backend, sym, dim)
end

"""
    make_symbolic_matrix(name::Symbol, player::Int, rows::Int, cols::Int; backend=default_backend())

Create a `rows × cols` matrix of symbolic variables for a player.

Valid names: $PLAYER_SYMBOLS
"""
function make_symbolic_matrix(name::Symbol, player::Int, rows::Int, cols::Int; backend=default_backend())
    sym = make_symbol(name, player)
    vars = SymbolicTracingUtils.make_variables(backend, sym, rows * cols)
    return reshape(vars, rows, cols)
end

#=
    Problem variable setup
=#

"""
    setup_problem_parameter_variables(num_params_per_player::Vector{Int}; backend=default_backend())

Create symbolic parameter variables (θ) for each player's initial state.

# Arguments
- `num_params_per_player::Vector{Int}` - Number of parameters for each player
- `backend` - SymbolicTracingUtils backend (default: SymbolicsBackend)

# Returns
- `θs::Dict{Int, Vector}` - Dictionary mapping player index to their parameter variables
"""
function setup_problem_parameter_variables(num_params_per_player::Vector{Int}; backend=default_backend())
    return Dict(idx => make_symbolic_vector(:θ, idx, num_params_per_player[idx]; backend)
                for idx in 1:length(num_params_per_player))
end

"""
    _validate_constraint_functions(gs::Vector, zs::Dict)

Validate that constraint functions have correct signatures, return Vectors,
and only depend on the player's own decision variables (decoupled constraints).

Called during problem setup to catch errors early with clear messages.

# Arguments
- `gs::Vector` - Constraint functions per player: gs[i](z) → Vector
- `zs::Dict` - Decision variables per player (used to test function signatures)

# Throws
- `ArgumentError` if gs[i] has wrong signature, doesn't return AbstractVector,
  or contains variables from other players (coupled constraint)
"""
function _validate_constraint_functions(gs::Vector, zs::Dict)
    # Collect all variables from all players for coupled constraint detection
    all_zs_flat = Set(Iterators.flatten(values(zs)))

    for (i, g) in enumerate(gs)
        try
            result = g(zs[i])
            if !(result isa AbstractVector)
                throw(ArgumentError("gs[$i] must return a Vector, got $(typeof(result))"))
            end

            # Check for coupled constraints: ensure result only uses variables from zs[i]
            if !isempty(result)
                allowed_vars = Set(zs[i])
                for constraint in result
                    constraint_vars = Symbolics.get_variables(constraint)
                    for var in constraint_vars
                        if var in all_zs_flat && !(var in allowed_vars)
                            throw(ArgumentError(
                                "gs[$i] contains coupled constraint: references variable " *
                                "from another player. Coupled constraints (gs[i] depending " *
                                "on zs[j] for j ≠ i) are not supported. " *
                                "See README.md for details."
                            ))
                        end
                    end
                end
            end
        catch e
            if e isa MethodError
                throw(ArgumentError("gs[$i] has wrong signature. Expected gs[$i](z::Vector). Error: $e"))
            end
            rethrow()
        end
    end
end

"""
    setup_problem_variables(
        graph::SimpleDiGraph,
        primal_dims::Vector{Int},
        gs::Vector;
        backend=default_backend()
    )

Construct all symbolic variables needed for the KKT system.

# Arguments
- `graph::SimpleDiGraph` - DAG of leader-follower relationships
- `primal_dims::Vector{Int}` - Decision variable dimension for each player
- `gs::Vector` - Constraint functions for each player (gs[i](z) returns constraints)
- `backend` - SymbolicTracingUtils backend (default: SymbolicsBackend)

# Returns
Named tuple containing:
- `zs::Dict` - Decision variables per player
- `λs::Dict` - Lagrange multipliers for constraints per player
- `μs::Dict` - Policy constraint multipliers (leader, follower) pairs
- `ys::Dict` - Information vectors (leader decisions visible to each player)
- `ws::Dict` - Remaining variables for policy constraints
- `ws_z_indices::Dict` - Index mapping: ws_z_indices[i][j] gives range where zs[j] appears in ws[i]
- `all_variables::Vector` - Flattened vector of all variables
"""
function setup_problem_variables(
    graph::SimpleDiGraph,
    primal_dims::Vector{Int},
    gs::Vector;
    backend=default_backend()
)
    N = nv(graph)

    # Create decision variables for each player
    zs = Dict(i => make_symbolic_vector(:z, i, primal_dims[i]; backend) for i in 1:N)

    # Validate constraint function signatures before using them
    _validate_constraint_functions(gs, zs)

    # Create Lagrange multipliers based on constraint dimensions
    λs = Dict(i => make_symbolic_vector(:λ, i, length(gs[i](zs[i])); backend) for i in 1:N)

    # Create policy constraint multipliers for leader-follower pairs
    μs = Dict((i, j) => make_symbolic_vector(:μ, i, j, primal_dims[j]; backend)
              for i in 1:N for j in get_all_followers(graph, i))

    # Information vectors: ys[i] contains decisions of all leaders of i
    ys = Dict{Int, Vector{Symbolics.Num}}()
    for i in 1:N
        leaders = get_all_leaders(graph, i)
        ys[i] = isempty(leaders) ? eltype(zs[1])[] : vcat([zs[l] for l in leaders]...)
    end

    # Remaining variables for policy constraints
    # ws[i] contains variables in player i's KKT conditions that are NOT in ys[i].
    # Structure (deterministic ordering by player index):
    #   ws[i] = [zs[i], zs[non-leaders in index order], λs[i], λs[followers], μs[pairs]]
    #
    # The ordering is critical for the implicit function theorem: from Mw + Ny = 0,
    # we get dw/dy = -M⁻¹N = -K. The policy extractor then selects the relevant
    # portion of K * y for policy constraints.
    ws = Dict{Int, Vector{Symbolics.Num}}()

    # Explicit index tracking for policy extractors (avoids fragile position assumptions).
    # ws_z_indices[i][j] gives the range where zs[j] appears in ws[i].
    # This is used in get_qp_kkt_conditions to build extraction matrices.
    ws_z_indices = Dict{Int, Dict{Int, UnitRange{Int}}}()

    for i in 1:N
        leaders = get_all_leaders(graph, i)
        ws_z_indices[i] = Dict{Int, UnitRange{Int}}()

        # ws[i] starts with the player's own decision variable
        ws[i] = copy(zs[i])
        ws_z_indices[i][i] = 1:primal_dims[i]
        offset = primal_dims[i]

        # Add non-leader, non-self variables (in player index order for consistency)
        for j in 1:N
            if j != i && !(j in leaders)
                ws[i] = vcat(ws[i], zs[j])
                ws_z_indices[i][j] = (offset + 1):(offset + primal_dims[j])
                offset += primal_dims[j]
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

    # Flatten all variables (in player index order)
    all_variables = vcat(
        vcat([zs[i] for i in 1:N]...),
        vcat([λs[i] for i in 1:N]...),
        (isempty(μs) ? eltype(zs[1])[] : vcat(collect(values(μs))...))
    )

    (; zs, λs, μs, ys, ws, ws_z_indices, all_variables)
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
    # TODO(Phase F): Implement for nonlinear solver K matrix handling
    error("construct_augmented_variables is not yet implemented. Planned for nonlinear solver phase.")
end
