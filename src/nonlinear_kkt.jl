#=
    KKT construction and solving for nonlinear hierarchy games

    Key difference from QP: Instead of computing K = M \ N symbolically (which can cause
    exponential expression growth), we create symbolic M and N functions and evaluate
    K = M \ N numerically at each iteration.
=#

# Use Julia's built-in `something(x, default)` for value-or-default pattern
# Note: something() returns the first non-nothing value, so something(x, default)
# is equivalent to isnothing(x) ? default : x

# Line search constants for run_nonlinear_solver.
# Passed to armijo_backtracking/geometric_reduction from src/linesearch.jl.
# Note: These differ from the linesearch module defaults (20 iters).
const LINESEARCH_MAX_ITERS = 10
const LINESEARCH_BACKTRACK_FACTOR = 0.5

"""
    check_convergence(residual, tol; verbose=false, iteration=nothing)

Check whether the solver has converged based on the KKT residual norm.

Returns a named tuple `(converged, status)` where:
- `converged::Bool` - whether the residual is below tolerance
- `status::Symbol` - `:solved`, `:not_converged`, or `:numerical_error`

# Arguments
- `residual::Real` - Current KKT residual norm
- `tol::Real` - Convergence tolerance

# Keyword Arguments
- `verbose::Bool=false` - Print convergence info
- `iteration::Union{Nothing,Int}=nothing` - Current iteration number (for verbose output)
"""
function check_convergence(residual, tol; verbose::Bool=false, iteration=nothing)
    # Guard against NaN/Inf
    if !isfinite(residual)
        verbose && @warn "Residual contains NaN or Inf values"
        return (; converged=false, status=:numerical_error)
    end

    if verbose
        iter_str = isnothing(iteration) ? "" : "Iteration $iteration: "
        @info "$(iter_str)residual = $residual"
    end

    if residual < tol
        return (; converged=true, status=:solved)
    end

    return (; converged=false, status=:not_converged)
end

"""
    compute_newton_step(linsolver, jacobian, neg_residual)

Solve the Newton step linear system `jacobian * δz = neg_residual`.

Uses the provided LinearSolve solver instance for the factorization and solve.
Handles singular matrix errors gracefully by returning `success=false`.

# Arguments
- `linsolver` - Initialized LinearSolve solver (mutated in place)
- `jacobian` - Jacobian matrix (∇F)
- `neg_residual` - Negative residual vector (-F)

# Returns
Named tuple `(step, success)` where:
- `step::Vector` - Newton step direction δz (undefined if `success=false`)
- `success::Bool` - Whether the linear solve succeeded
"""
function compute_newton_step(linsolver, jacobian, neg_residual)
    linsolver.A = jacobian
    linsolver.b = neg_residual
    try
        solution = solve!(linsolver)
        success = SciMLBase.successful_retcode(solution) || solution.retcode === SciMLBase.ReturnCode.Default
        return (; step=solution.u, success)
    catch e
        if e isa SingularException || e isa LAPACKException
            return (; step=neg_residual, success=false)
        end
        rethrow()
    end
end

"""
    perform_linesearch(residual_norm_fn, z_est, δz, current_residual_norm; use_armijo=true)

Perform backtracking line search to select step size for Newton update.

When `use_armijo=true`, backtracks from α=1.0 by halving until the trial point
has a smaller residual norm than the current point, or max iterations are reached.
When `use_armijo=false`, returns α=1.0 (full Newton step).

# Arguments
- `residual_norm_fn` - Function `z_trial -> Float64` returning residual norm at trial point
- `z_est::Vector` - Current iterate
- `δz::Vector` - Newton step direction
- `current_residual_norm::Float64` - Residual norm at current iterate

# Keyword Arguments
- `use_armijo::Bool=true` - Whether to perform backtracking line search

# Returns
- `α::Float64` - Selected step size
"""
function perform_linesearch(residual_norm_fn, z_est, δz, current_residual_norm;
                            use_armijo::Bool=true)
    α = 1.0

    if !use_armijo
        return α
    end

    for _ in 1:LINESEARCH_MAX_ITERS
        z_trial = z_est .+ α .* δz
        trial_residual_norm = residual_norm_fn(z_trial)

        if trial_residual_norm < current_residual_norm
            break
        end
        α *= LINESEARCH_BACKTRACK_FACTOR
    end

    return α
end

"""
    _construct_augmented_variables(ii, all_variables, K_syms, G)

Build augmented variable list for player ii including follower K matrices.

For computing M and N that depend on follower policies, we need to include
symbolic K matrices in the variable list.

# Arguments
- `ii::Int` - Player index
- `all_variables::Vector` - Base symbolic variables
- `K_syms::Dict` - Symbolic K matrices per player
- `G::SimpleDiGraph` - Hierarchy graph

# Returns
- `augmented::Vector` - Variables including follower K matrix entries
"""
function _construct_augmented_variables(ii, all_variables, K_syms, G)
    # Leaf players don't need augmentation
    if is_leaf(G, ii)
        return all_variables
    end

    augmented = copy(all_variables)

    # Add K matrices for all followers of ii (in BFS order)
    for jj in BFSIterator(G, ii)
        if ii == jj
            continue
        end
        if has_leader(G, jj) && !isempty(K_syms[jj])
            # Flatten K matrix and append
            augmented = vcat(augmented, reshape(K_syms[jj], :))
        end
    end

    return augmented
end

"""
    setup_approximate_kkt_solver(
        G::SimpleDiGraph,
        Js::Dict,
        zs::Dict,
        λs::Dict,
        μs::Dict,
        gs::Vector,
        ws::Dict,
        ys::Dict,
        θs::Dict,
        all_variables::Vector,
        backend;
        verbose::Bool = false
    )

Precompute symbolic KKT conditions and M/N matrix evaluation functions for nonlinear solver.

Unlike QP KKT construction which computes K = M \\ N symbolically, this creates
compiled functions for evaluating M and N numerically, avoiding expression blowup.

# Arguments
- `G::SimpleDiGraph` - Hierarchy graph
- `Js::Dict` - Cost functions per player
- `zs::Dict` - Decision variables per player
- `λs::Dict` - Lagrange multipliers per player
- `μs::Dict` - Policy constraint multipliers
- `gs::Vector` - Constraint functions per player
- `ws::Dict` - Remaining variables (policy output)
- `ys::Dict` - Information variables (policy input)
- `θs::Dict` - Parameter variables per player
- `all_variables::Vector` - All symbolic variables
- `backend` - SymbolicTracingUtils backend

# Keyword Arguments
- `verbose::Bool=false` - Print debug info
- `cse::Bool=false` - Enable Common Subexpression Elimination during symbolic compilation.
  CSE can dramatically reduce construction time and memory for problems with redundant
  symbolic structure (e.g., quadratic costs), but may slightly increase per-solve runtime.
  Recommended only when construction time is a bottleneck and you can tolerate slightly
  slower solve times. Default: false for maximum runtime performance.

# Returns
Tuple of:
- `all_augmented_variables::Vector` - Variables including K matrix symbols
- `setup_info::NamedTuple` - Contains:
  - `graph` - Hierarchy graph
  - `πs` - KKT conditions per player
  - `K_syms` - Symbolic K matrices per player
  - `M_fns` - Compiled M matrix evaluation functions
  - `N_fns` - Compiled N matrix evaluation functions
  - `π_sizes` - KKT condition sizes per player
"""
function setup_approximate_kkt_solver(
    G::SimpleDiGraph,
    Js::Dict,
    zs::Dict,
    λs::Dict,
    μs::Dict,
    gs::Vector,
    ws::Dict,
    ys::Dict,
    θs::Dict,
    all_variables::Vector,
    backend;
    verbose::Bool = false,
    cse::Bool = false
)
    N = nv(G)
    reverse_order = reverse(topological_sort_by_dfs(G))

    # Output containers
    π_sizes = Dict{Int, Int}()
    K_syms = Dict{Int, Union{Matrix{Symbolics.Num}, Vector{Symbolics.Num}}}()
    πs = Dict{Int, Any}()
    M_fns = Dict{Int, Function}()
    N_fns = Dict{Int, Function}()
    augmented_variables = Dict{Int, Vector{Symbolics.Num}}()

    # First pass: create symbolic K matrices for all followers
    for ii in 1:N
        if has_leader(G, ii)
            # Create symbolic K matrix: dimensions (length(ws[ii]), length(ys[ii]))
            K_size = length(ws[ii]) * length(ys[ii])
            K_syms[ii] = reshape(
                SymbolicTracingUtils.make_variables(backend, make_symbol(:K, ii), K_size),
                length(ws[ii]),
                length(ys[ii])
            )
        else
            K_syms[ii] = eltype(all_variables)[]  # Empty for roots
        end
    end

    # Second pass: build KKT conditions and M/N functions
    for ii in reverse_order
        # Compute π_sizes (total KKT conditions for player ii)
        # Structure: [grad_self | grad_f1 | policy_f1 | ... | own_constraints]
        π_sizes[ii] = length(gs[ii](zs[ii]))  # Own constraints
        for jj in BFSIterator(G, ii)
            π_sizes[ii] += length(zs[jj])  # Gradient w.r.t. each player in subtree
            if ii != jj
                π_sizes[ii] += length(zs[jj])  # Policy constraint for each follower
            end
        end

        # Build Lagrangian using symbolic K matrices
        # Get θ for this player (flatten if needed)
        θ_order = ordered_player_indices(θs)
        θ_all = vcat([θs[k] for k in θ_order]...)

        all_zs = [zs[j] for j in 1:N]
        # Cost function signature: Js[i](zs...; θ) with θ as keyword argument
        Lᵢ = Js[ii](all_zs...; θ=θ_all) - λs[ii]' * gs[ii](zs[ii])

        # Add follower policy constraint terms using symbolic K
        for jj in BFSIterator(G, ii)
            if ii == jj
                continue
            end

            # Policy constraint: zⱼ = Φⱼ where Φⱼ = -E * K[jj] * ys[jj]
            # E extracts the zⱼ portion from the full policy response
            zi_size = length(zs[ii])
            extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))
            Φⱼ = -extractor * K_syms[jj] * ys[jj]

            if haskey(μs, (ii, jj))
                Lᵢ -= μs[(ii, jj)]' * (zs[jj] - Φⱼ)
            end
        end

        # Build KKT conditions as block vectors for leaders.
        # Using mortar to preserve interleaved block structure:
        # [grad_self | grad_f1 | policy_f1 | grad_f2 | policy_f2 | ... | own_constraints]
        # This matches the pattern in get_qp_kkt_conditions (qp_kkt.jl) and enables
        # strip_policy_constraints to use the BlockVector path directly.
        πᵢ = Vector{Symbolics.Num}[]
        for jj in BFSIterator(G, ii)
            # Stationarity gradient
            push!(πᵢ, Symbolics.gradient(Lᵢ, zs[jj]))

            # Policy constraint (for followers only)
            if ii != jj
                zi_size = length(zs[ii])
                extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))
                Φⱼ = -extractor * K_syms[jj] * ys[jj]
                push!(πᵢ, zs[jj] - Φⱼ)
            end
        end
        push!(πᵢ, gs[ii](zs[ii]))  # Own constraints

        if is_leaf(G, ii)
            πs[ii] = vcat(πᵢ...)
        else
            πs[ii] = mortar(πᵢ)
        end

        # Compute M and N functions for followers (players with leaders)
        if has_leader(G, ii)
            # Build augmented variable list (includes follower K matrices)
            augmented_variables[ii] = _construct_augmented_variables(ii, all_variables, K_syms, G)

            # collect() ensures plain Vector for Symbolics.jacobian (BlockVector unsupported)
            πs_flat = collect(πs[ii])
            Mᵢ = Symbolics.jacobian(πs_flat, ws[ii])
            Nᵢ = Symbolics.jacobian(πs_flat, ys[ii])

            # Compile to functions
            M_fns[ii] = SymbolicTracingUtils.build_function(Mᵢ, augmented_variables[ii]; in_place=false, backend_options=(; cse))
            N_fns[ii] = SymbolicTracingUtils.build_function(Nᵢ, augmented_variables[ii]; in_place=false, backend_options=(; cse))
        else
            augmented_variables[ii] = all_variables
        end

        verbose && println("Player $ii: $(π_sizes[ii]) KKT conditions, augmented vars: $(length(get(augmented_variables, ii, [])))")
    end

    # Build full augmented variable list
    all_K_syms_vec = vcat([reshape(something(K_syms[ii], eltype(all_variables)[]), :) for ii in 1:N]...)
    all_augmented_variables = vcat(all_variables, all_K_syms_vec)

    return all_augmented_variables, (; graph=G, πs, K_syms, M_fns, N_fns, π_sizes)
end

"""
    preoptimize_nonlinear_solver(
        hierarchy_graph::SimpleDiGraph,
        Js::Dict,
        gs::Vector,
        primal_dims::Vector{Int},
        θs::Dict;
        state_dim::Int = 2,
        control_dim::Int = 2,
        backend = default_backend(),
        verbose::Bool = false
    )

Precompute all symbolic components for nonlinear solver.

This is called once before solving to build all the symbolic expressions
and compile them to efficient numerical functions.

# Arguments
- `hierarchy_graph::SimpleDiGraph` - Hierarchy graph
- `Js::Dict` - Cost functions per player
- `gs::Vector` - Constraint functions per player
- `primal_dims::Vector{Int}` - Primal variable dimension per player
- `θs::Dict` - Parameter variables per player

# Keyword Arguments
- `state_dim::Int=2` - State dimension (for trajectory extraction)
- `control_dim::Int=2` - Control dimension (for trajectory extraction)
- `backend` - SymbolicTracingUtils backend
- `verbose::Bool=false` - Print debug info
- `cse::Bool=false` - Enable Common Subexpression Elimination during symbolic compilation.
  CSE can dramatically reduce construction time and memory for problems with redundant
  symbolic structure (e.g., quadratic costs), but may slightly increase per-solve runtime.
  Recommended only when construction time is a bottleneck and you can tolerate slightly
  slower solve times. Default: false for maximum runtime performance.

# Returns
Named tuple containing:
- `problem_vars` - Problem variables (zs, λs, μs, ws, ys, all_variables)
- `setup_info` - Setup info from setup_approximate_kkt_solver
- `mcp_obj` - ParametricMCP object for residual evaluation
- `linsolver` - LinearSolve problem for iterative solving
- `all_variables` - All symbolic variables
- `all_augmented_variables` - Variables including K matrices
- `F_sym` - Symbolic KKT residual vector
- `π_sizes_trimmed` - Trimmed KKT sizes per player
- `state_dim` - State dimension
- `control_dim` - Control dimension
"""
function preoptimize_nonlinear_solver(
    hierarchy_graph::SimpleDiGraph,
    Js::Dict,
    gs::Vector,
    primal_dims::Vector{Int},
    θs::Dict;
    state_dim::Int = 2,
    control_dim::Int = 2,
    backend = default_backend(),
    verbose::Bool = false,
    cse::Bool = false,
    to::TimerOutput = TimerOutput()
)
    N = nv(hierarchy_graph)

    # Setup symbolic variables
    @timeit to "variable setup" begin
        problem_vars = setup_problem_variables(hierarchy_graph, primal_dims, gs; backend)
        all_variables = problem_vars.all_variables
        zs = problem_vars.zs
        λs = problem_vars.λs
        μs = problem_vars.μs
        ws = problem_vars.ws
        ys = problem_vars.ys
    end

    # Setup approximate KKT solver (creates symbolic M/N functions)
    @timeit to "approximate KKT setup" begin
        all_augmented_variables, setup_info = setup_approximate_kkt_solver(
            hierarchy_graph, Js, zs, λs, μs, gs, ws, ys, θs,
            all_variables, backend;
            verbose, cse
        )

        πs = setup_info.πs
        K_syms = setup_info.K_syms

        # Build flattened K symbols vector for use as parameters
        all_K_syms_vec = vcat([reshape(something(K_syms[ii], eltype(all_variables)[]), :) for ii in 1:N]...)

        # Build parameter vector (θ values + K matrix values)
        θ_order = ordered_player_indices(θs)
        θ_syms_flat = vcat([θs[k] for k in θ_order]...)
        all_param_syms_vec = vcat(θ_syms_flat, all_K_syms_vec)

        # Strip policy constraints for MCP construction
        πs_solve = strip_policy_constraints(πs, hierarchy_graph, zs, gs)
        π_sizes_trimmed = Dict(ii => length(πs_solve[ii]) for ii in keys(πs_solve))

        # Build MCP function vector
        π_order = ordered_player_indices(πs_solve)
        F_sym = Symbolics.Num.(vcat([πs_solve[k] for k in π_order]...))
    end

    # Build ParametricMCP
    @timeit to "ParametricMCP build" begin
        z_lower = fill(-Inf, length(F_sym))
        z_upper = fill(Inf, length(F_sym))

        verbose && @info "Preoptimization: $(length(all_variables)) variables, $(length(F_sym)) conditions"

        params_syms_vec = Symbolics.Num.(all_param_syms_vec)
        mcp_obj = ParametricMCPs.ParametricMCP(
            F_sym, all_variables, params_syms_vec, z_lower, z_upper;
            compute_sensitivities = false
        )
    end

    # Initialize linear solver
    @timeit to "linear solver init" begin
        F_size = length(F_sym)
        linear_solve_algorithm = LinearSolve.UMFPACKFactorization()
        linsolver = init(LinearProblem(spzeros(F_size, F_size), zeros(F_size)), linear_solve_algorithm)
    end

    return (;
        problem_vars,
        setup_info,
        mcp_obj,
        linsolver,
        all_variables,
        all_augmented_variables,
        F_sym,
        π_sizes_trimmed,
        state_dim,
        control_dim,
        backend
    )
end

"""
    _build_augmented_z_est(ii, z_est, K_evals, graph, follower_cache, buffer_cache)

Build augmented z vector for player ii including follower K evaluations.

# Arguments
- `ii::Int` - Player index
- `z_est::Vector` - Current z estimate
- `K_evals::Dict` - Numerical K matrices per player
- `graph::SimpleDiGraph` - Hierarchy graph
- `follower_cache::Dict` - Cache for follower lists
- `buffer_cache::Dict` - Cache for augmented buffers

# Returns
- `augmented_z::Vector` - z_est augmented with follower K values
"""
function _build_augmented_z_est(ii, z_est, K_evals, graph, follower_cache, buffer_cache)
    # Get cached follower list
    followers = get!(follower_cache, ii) do
        collect(BFSIterator(graph, ii))[2:end]  # Exclude self
    end

    # Compute required length
    aug_len = length(z_est)
    for jj in followers
        kj = K_evals[jj]
        aug_len += isnothing(kj) ? 0 : length(kj)
    end

    # Get or resize buffer
    buf = get!(buffer_cache, ii) do
        Vector{Float64}(undef, aug_len)
    end
    if length(buf) != aug_len
        resize!(buf, aug_len)
    end

    # Fill buffer: [z_est, K_f1[:], K_f2[:], ...]
    copyto!(buf, 1, z_est, 1, length(z_est))
    offset = length(z_est) + 1
    for jj in followers
        kj = K_evals[jj]
        if isnothing(kj)
            continue
        end
        flat = reshape(kj, :)
        copyto!(buf, offset, flat, 1, length(flat))
        offset += length(flat)
    end

    return buf
end

"""
    compute_K_evals(z_current, problem_vars, setup_info; use_sparse=false, use_inplace_ksolve=false)

Evaluate K (policy) matrices numerically in reverse topological order.

Note: This function is NOT thread-safe. The precomputed M_fns and N_fns contain
shared result buffers that would cause data races if called concurrently.
For multi-threaded use, each thread needs its own solver instance.
See Phase 6 for planned thread-safety improvements.

# Arguments
- `z_current::Vector` - Current solution estimate
- `problem_vars::NamedTuple` - Problem variables (from setup_problem_variables)
- `setup_info::NamedTuple` - Setup info (from setup_approximate_kkt_solver)

# Keyword Arguments
- `use_sparse::Bool=false` - If true, use sparse LU factorization for M\\N solve.
  Beneficial for large M matrices (>100 rows) with structural sparsity from the
  KKT system. For small matrices, dense solve is faster due to sparse overhead.
- `use_inplace_ksolve::Bool=false` - If true, use `ldiv!(K_buf, lu(M), N)` instead
  of `M \\ N`. May reduce allocations for large problems.

# Returns
Tuple of:
- `all_K_vec::Vector` - Concatenated K matrix values for all players
- `info::NamedTuple` - Contains M_evals, N_evals, K_evals, status
  - `status` is `:ok` or `:singular_matrix`
"""
function compute_K_evals(
    z_current::Vector,
    problem_vars::NamedTuple,
    setup_info::NamedTuple;
    use_sparse::Bool=false,
    use_inplace_ksolve::Bool=false
)
    ws = problem_vars.ws
    ys = problem_vars.ys
    zs = problem_vars.zs
    M_fns = setup_info.M_fns
    N_fns = setup_info.N_fns
    π_sizes = setup_info.π_sizes
    graph = setup_info.graph

    M_evals = Dict{Int, Union{Matrix{Float64}, Nothing}}()
    N_evals = Dict{Int, Union{Matrix{Float64}, Nothing}}()
    K_evals = Dict{Int, Union{Matrix{Float64}, Nothing}}()

    # Caches to reduce allocations
    follower_cache = Dict{Int, Vector{Int}}()
    buffer_cache = Dict{Int, Vector{Float64}}()

    status = :ok

    # Process in reverse topological order (leaves first)
    for ii in reverse(topological_sort_by_dfs(graph))
        if has_leader(graph, ii)
            # Build augmented z with follower K evaluations
            augmented_z = _build_augmented_z_est(ii, z_current, K_evals, graph, follower_cache, buffer_cache)

            # Evaluate M and N
            M_raw = M_fns[ii](augmented_z)
            N_raw = N_fns[ii](augmented_z)

            # Reshape to correct dimensions
            M_evals[ii] = reshape(M_raw, π_sizes[ii], length(ws[ii]))
            N_evals[ii] = reshape(N_raw, π_sizes[ii], length(ys[ii]))

            # Solve K = M \ N with singular matrix protection
            K_evals[ii] = _solve_K(M_evals[ii], N_evals[ii], ii; use_sparse, use_inplace_ksolve)
            if any(isnan, K_evals[ii])
                status = :singular_matrix
            end
        else
            M_evals[ii] = nothing
            N_evals[ii] = nothing
            K_evals[ii] = nothing
        end
    end

    # Concatenate all K values into single vector
    N = nv(graph)
    all_K_vec = vcat([reshape(something(K_evals[ii], Float64[]), :) for ii in 1:N]...)

    return all_K_vec, (; M_evals, N_evals, K_evals, status)
end

"""
    _solve_K(M, N, player_idx; use_sparse=false, use_inplace_ksolve=false)

Solve `K = M \\ N` with protection against singular or ill-conditioned M matrices.

When `use_sparse=true`, converts M to sparse format before solving, which can be
beneficial for large M matrices (>100 rows) with structural sparsity from the KKT system.

When `use_inplace_ksolve=true`, uses `ldiv!(K_buffer, lu(M), N)` to write the result
directly into a pre-allocated buffer, avoiding the allocation of a new K matrix.
May reduce GC pressure for large problems. `use_sparse` takes precedence if both are true.

Returns a NaN-filled matrix (same size as expected K) if M is singular or
severely ill-conditioned, with a warning.
"""
function _solve_K(M::Matrix{Float64}, N::Matrix{Float64}, player_idx::Int;
                  use_sparse::Bool=false, use_inplace_ksolve::Bool=false)
    try
        K = if use_sparse
            sparse(M) \ N
        elseif use_inplace_ksolve && size(M, 1) == size(M, 2)
            # ldiv! with lu requires square M; non-square M falls back to backslash
            K_buf = similar(N)
            ldiv!(K_buf, lu(M), N)
            K_buf
        else
            M \ N
        end

        # Check for NaN/Inf in result (can occur with near-singular matrices)
        if any(!isfinite, K)
            @warn "K evaluation for player $player_idx produced non-finite values (near-singular M)"
            return fill(NaN, size(K))
        end

        return K
    catch e
        if e isa SingularException || e isa LAPACKException
            @warn "Singular M matrix for player $player_idx: $e. Using NaN fallback."
            n_rows = size(N, 1)
            n_cols = size(N, 2)
            return fill(NaN, n_rows, n_cols)
        end
        rethrow()
    end
end

"""
    run_nonlinear_solver(
        precomputed::NamedTuple,
        initial_states::Dict,
        hierarchy_graph::SimpleDiGraph;
        initial_guess::Union{Nothing, Vector{Float64}} = nothing,
        max_iters::Int = 100,
        tol::Float64 = 1e-6,
        verbose::Bool = false,
        linesearch_method::Symbol = :geometric,
        recompute_policy_in_linesearch::Bool = true,
        use_sparse::Bool = false,
        use_inplace_ksolve::Bool = false,
        show_progress::Bool = false,
        to::TimerOutput = TimerOutput()
    )

Orchestrates the Newton iteration loop for solving nonlinear hierarchy games.

Each iteration: evaluate the KKT residual, check convergence, compute a Newton
step via [`compute_newton_step`](@ref), and select a step size via configurable
line search. Convergence is checked by [`check_convergence`](@ref).

# Arguments
- `precomputed::NamedTuple` - Precomputed symbolic components from [`preoptimize_nonlinear_solver`](@ref)
- `initial_states::Dict` - Initial state for each player (keyed by player index)
- `hierarchy_graph::SimpleDiGraph` - Player hierarchy graph

# Keyword Arguments
- `initial_guess::Union{Nothing, Vector{Float64}}=nothing` - Starting point (zero-initialized if `nothing`)
- `max_iters::Int=100` - Maximum Newton iterations
- `tol::Float64=1e-6` - Convergence tolerance on KKT residual norm
- `verbose::Bool=false` - Print per-iteration convergence info
- `linesearch_method::Symbol=:geometric` - Line search method (:armijo, :geometric, or :constant)
- `recompute_policy_in_linesearch::Bool=true` - Recompute K matrices at each line search trial step. Set to `false` for ~1.6x speedup (reuses K from current Newton iteration).
- `use_sparse::Bool=false` - Use sparse LU for M\\N solve (beneficial for large problems)
- `use_inplace_ksolve::Bool=false` - Use in-place `ldiv!` for K = M\\N solve
- `show_progress::Bool=false` - Display iteration progress table (iter, residual, step size, time)
- `to::TimerOutput=TimerOutput()` - Timer for profiling solver phases

# Returns
Named tuple `(; sol, converged, iterations, residual, status)`:
- `sol::Vector{Float64}` - Solution vector
- `converged::Bool` - Whether the solver reached the tolerance
- `iterations::Int` - Number of iterations performed
- `residual::Float64` - Final KKT residual norm
- `status::Symbol` - One of `:solved`, `:solved_initial_point`, `:max_iters_reached`,
  `:linear_solver_error`, `:line_search_failed`, `:numerical_error`
"""
function run_nonlinear_solver(
    precomputed::NamedTuple,
    initial_states::Dict,
    hierarchy_graph::SimpleDiGraph;
    initial_guess::Union{Nothing, Vector{Float64}} = nothing,
    max_iters::Int = 100,
    tol::Float64 = 1e-6,
    verbose::Bool = false,
    linesearch_method::Symbol = :geometric,
    recompute_policy_in_linesearch::Bool = true,
    use_sparse::Bool = false,
    use_inplace_ksolve::Bool = false,
    show_progress::Bool = false,
    to::TimerOutput = TimerOutput()
)
    # Unpack precomputed components
    problem_vars = precomputed.problem_vars
    setup_info = precomputed.setup_info
    mcp_obj = precomputed.mcp_obj
    linsolver = precomputed.linsolver
    all_variables = precomputed.all_variables

    # Build parameter values vector
    θs_order = ordered_player_indices(initial_states)
    θ_vals_vec = vcat([initial_states[k] for k in θs_order]...)

    # Initialize z estimate
    z_est = something(initial_guess, zeros(length(all_variables)))
    if length(z_est) < length(all_variables)
        verbose && @info "Padding initial guess from $(length(z_est)) to $(length(all_variables))"
        z_est = vcat(z_est, zeros(length(all_variables) - length(z_est)))
    end

    # Solver state
    num_iterations = 0
    residual_norm = Inf
    status = :max_iters_reached

    # Allocate buffers
    n = length(all_variables)
    F_eval = zeros(n)
    ∇F = copy(mcp_obj.jacobian_z!.result_buffer)
    z_trial = Vector{Float64}(undef, n)

    # Pre-allocate param_vec buffer: [θ_vals_vec; all_K_vec]
    # Size is determined from the MCP's parameter_dimension (set during preoptimize)
    θ_len = length(θ_vals_vec)
    param_vec = Vector{Float64}(undef, mcp_obj.parameter_dimension)
    copyto!(param_vec, 1, θ_vals_vec, 1, θ_len)

    # Helper: compute parameters (θ, K) for a given z, reusing param_vec buffer
    function params_for_z!(z)
        all_K_vec, _ = compute_K_evals(z, problem_vars, setup_info; use_sparse, use_inplace_ksolve)
        copyto!(param_vec, θ_len + 1, all_K_vec, 1, length(all_K_vec))
        return param_vec, all_K_vec
    end

    # Progress tracking
    t_start = time()
    if show_progress
        println("┌──────────────────────────────────────────────────────────────┐")
        println("│  iter      residual          α         time                  │")
        println("├──────────────────────────────────────────────────────────────┤")
    end

    # Main iteration loop
    α = NaN  # track step size for progress display
    while true
        # Evaluate K matrices at current z
        @timeit to "compute K evals" begin
            param_vec, all_K_vec = params_for_z!(z_est)
        end

        # Evaluate residual and check convergence
        @timeit to "residual evaluation" begin
            mcp_obj.f!(F_eval, z_est, param_vec)
            residual_norm = norm(F_eval)
        end

        conv = check_convergence(residual_norm, tol; verbose, iteration=num_iterations)

        if conv.status == :numerical_error
            status = :numerical_error
            break
        end

        if conv.converged
            status = num_iterations > 0 ? :solved : :solved_initial_point
            break
        end

        if num_iterations >= max_iters
            break
        end

        num_iterations += 1

        # Solve linearized system: ∇F * δz = -F
        @timeit to "Jacobian evaluation" begin
            mcp_obj.jacobian_z!(∇F, z_est, param_vec)
        end

        @timeit to "Newton step" begin
            newton_result = compute_newton_step(linsolver, ∇F, -F_eval)
        end

        if !newton_result.success
            verbose && @warn "Linear solve failed"
            status = :linear_solver_error
            break
        end

        δz = newton_result.step

        # Line search for step size
        @timeit to "line search" begin
            # Residual function closure that optionally recomputes K at each trial point
            function residual_at_trial(z)
                param_trial = if recompute_policy_in_linesearch
                    first(params_for_z!(z))
                else
                    param_vec
                end
                F_trial = similar(F_eval)
                mcp_obj.f!(F_trial, z, param_trial)
                return F_trial
            end

            if linesearch_method == :armijo
                α = armijo_backtracking(residual_at_trial, z_est, δz, 1.0;
                    rho=LINESEARCH_BACKTRACK_FACTOR, max_iters=LINESEARCH_MAX_ITERS)
            elseif linesearch_method == :geometric
                α = geometric_reduction(residual_at_trial, z_est, δz, 1.0;
                    rho=LINESEARCH_BACKTRACK_FACTOR, max_iters=LINESEARCH_MAX_ITERS)
            elseif linesearch_method == :constant
                α = 1.0
            else
                error("Unknown linesearch_method: $linesearch_method")
            end
        end

        # Update estimate (in-place to avoid allocation)
        @. z_est += α * δz

        # Progress display after iteration update
        if show_progress
            elapsed = time() - t_start
            iter_str = lpad(num_iterations, 4)
            res_str = lpad(string(residual_norm), 14)
            α_str = lpad(string(round(α; digits=4)), 8)
            t_str = lpad(string(round(elapsed; digits=2)) * "s", 9)
            println("│  iter $iter_str  residual $res_str  α $α_str  time $t_str │")
        end

        # Guard against NaN/Inf in solution
        if any(!isfinite, z_est)
            verbose && @warn "Solution contains NaN or Inf values after update, terminating"
            status = :numerical_error
            break
        end
    end

    # Progress summary
    if show_progress
        elapsed = time() - t_start
        println("└──────────────────────────────────────────────────────────────┘")
        status_str = status in (:solved, :solved_initial_point) ? "Converged" : "Did not converge"
        println("  $status_str in $num_iterations iterations ($(round(elapsed; digits=2))s), final residual: $residual_norm")
    end

    return (;
        sol = z_est,
        converged = status in (:solved, :solved_initial_point),
        iterations = num_iterations,
        residual = residual_norm,
        status
    )
end

