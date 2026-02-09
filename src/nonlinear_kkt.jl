#=
    KKT construction and solving for nonlinear hierarchy games

    Key difference from QP: Instead of computing K = M \ N symbolically (which can cause
    exponential expression growth), we create symbolic M and N functions and evaluate
    K = M \ N numerically at each iteration.
=#

# Use Julia's built-in `something(x, default)` for value-or-default pattern
# Note: something() returns the first non-nothing value, so something(x, default)
# is equivalent to isnothing(x) ? default : x

# Line search constants for run_nonlinear_solver
# Note: These differ from armijo_backtracking_linesearch defaults (20 iters).
# See Bead 1 for planned unification of line search implementations.
const LINESEARCH_MAX_ITERS = 10
const LINESEARCH_BACKTRACK_FACTOR = 0.5

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

# Returns
Tuple of:
- `all_augmented_variables::Vector` - Variables including K matrix symbols
- `setup_info::NamedTuple` - Contains:
  - `graph` - Hierarchy graph
  - `πs` - KKT conditions per player
  - `K_syms` - Symbolic K matrices per player
  - `M_fns` - Compiled M matrix evaluation functions (out-of-place)
  - `N_fns` - Compiled N matrix evaluation functions (out-of-place)
  - `M_fns!` - Compiled M matrix evaluation functions (in-place)
  - `N_fns!` - Compiled N matrix evaluation functions (in-place)
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
    verbose::Bool = false
)
    N = nv(G)
    reverse_order = reverse(topological_sort_by_dfs(G))

    # Output containers
    π_sizes = Dict{Int, Int}()
    K_syms = Dict{Int, Union{Matrix{Symbolics.Num}, Vector{Symbolics.Num}}}()
    πs = Dict{Int, Vector{Symbolics.Num}}()
    M_fns = Dict{Int, Function}()
    N_fns = Dict{Int, Function}()
    M_fns! = Dict{Int, Function}()
    N_fns! = Dict{Int, Function}()
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
        # Structure: [grad_self | grad_followers | policy_constraints | own_constraints]
        π_sizes[ii] = length(gs[ii](zs[ii]))  # Own constraints
        for jj in BFSIterator(G, ii)
            π_sizes[ii] += length(zs[jj])  # Gradient w.r.t. each player in subtree
            if ii != jj
                π_sizes[ii] += length(zs[jj])  # Policy constraint for each follower
            end
        end

        # Build Lagrangian using symbolic K matrices
        # Get θ for this player (flatten if needed)
        θ_order = sort(collect(keys(θs)))
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

        # Build KKT conditions (accumulator holds vectors, vcat flattens to Num[])
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
        πs[ii] = vcat(πᵢ...)

        # Compute M and N functions for followers (players with leaders)
        if has_leader(G, ii)
            # Build augmented variable list (includes follower K matrices)
            augmented_variables[ii] = _construct_augmented_variables(ii, all_variables, K_syms, G)

            # Compute Jacobians
            Mᵢ = Symbolics.jacobian(πs[ii], ws[ii])
            Nᵢ = Symbolics.jacobian(πs[ii], ys[ii])

            # Compile to functions (out-of-place for baseline)
            M_fns[ii] = SymbolicTracingUtils.build_function(Mᵢ, augmented_variables[ii]; in_place=false)
            N_fns[ii] = SymbolicTracingUtils.build_function(Nᵢ, augmented_variables[ii]; in_place=false)

            # Compile in-place variants for Strategy A (write into pre-allocated buffers)
            M_fns![ii] = SymbolicTracingUtils.build_function(Mᵢ, augmented_variables[ii]; in_place=true)
            N_fns![ii] = SymbolicTracingUtils.build_function(Nᵢ, augmented_variables[ii]; in_place=true)
        else
            augmented_variables[ii] = all_variables
        end

        verbose && println("Player $ii: $(π_sizes[ii]) KKT conditions, augmented vars: $(length(get(augmented_variables, ii, [])))")
    end

    # Build full augmented variable list
    all_K_syms_vec = vcat([reshape(something(K_syms[ii], eltype(all_variables)[]), :) for ii in 1:N]...)
    all_augmented_variables = vcat(all_variables, all_K_syms_vec)

    return all_augmented_variables, (; graph=G, πs, K_syms, M_fns, N_fns, M_fns!, N_fns!, π_sizes)
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
            verbose
        )

        πs = setup_info.πs
        K_syms = setup_info.K_syms

        # Build flattened K symbols vector for use as parameters
        all_K_syms_vec = vcat([reshape(something(K_syms[ii], eltype(all_variables)[]), :) for ii in 1:N]...)

        # Build parameter vector (θ values + K matrix values)
        θ_order = sort(collect(keys(θs)))
        θ_syms_flat = vcat([θs[k] for k in θ_order]...)
        all_param_syms_vec = vcat(θ_syms_flat, all_K_syms_vec)

        # Strip policy constraints for MCP construction
        πs_solve = strip_policy_constraints(πs, hierarchy_graph, zs, gs)
        π_sizes_trimmed = Dict(ii => length(πs_solve[ii]) for ii in keys(πs_solve))

        # Build MCP function vector
        π_order = sort(collect(keys(πs_solve)))
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
    compute_K_evals(z_current::Vector, problem_vars::NamedTuple, setup_info::NamedTuple)

Evaluate K (policy) matrices numerically in reverse topological order.

Note: This function is NOT thread-safe. The precomputed M_fns and N_fns contain
shared result buffers that would cause data races if called concurrently.
For multi-threaded use, each thread needs its own solver instance.
See Phase 6 for planned thread-safety improvements.

# Arguments
- `z_current::Vector` - Current solution estimate
- `problem_vars::NamedTuple` - Problem variables (from setup_problem_variables)
- `setup_info::NamedTuple` - Setup info (from setup_approximate_kkt_solver)

# Returns
Tuple of:
- `all_K_vec::Vector` - Concatenated K matrix values for all players
- `info::NamedTuple` - Contains M_evals, N_evals, K_evals
"""
function compute_K_evals(
    z_current::Vector,
    problem_vars::NamedTuple,
    setup_info::NamedTuple;
    inplace_MN::Bool = false,
    inplace_ldiv::Bool = false,
    inplace_lu::Bool = false
)
    ws = problem_vars.ws
    ys = problem_vars.ys
    zs = problem_vars.zs
    M_fns_oop = setup_info.M_fns
    N_fns_oop = setup_info.N_fns
    π_sizes = setup_info.π_sizes
    graph = setup_info.graph

    M_evals = Dict{Int, Union{Matrix{Float64}, Nothing}}()
    N_evals = Dict{Int, Union{Matrix{Float64}, Nothing}}()
    K_evals = Dict{Int, Union{Matrix{Float64}, Nothing}}()

    # Caches to reduce allocations
    follower_cache = Dict{Int, Vector{Int}}()
    buffer_cache = Dict{Int, Vector{Float64}}()

    # Pre-allocated buffers for in-place strategies
    # These are allocated once per call and reused across players
    M_buf = Dict{Int, Matrix{Float64}}()
    N_buf = Dict{Int, Matrix{Float64}}()
    K_buf = Dict{Int, Matrix{Float64}}()
    M_scratch = Dict{Int, Matrix{Float64}}()  # For lu! (destroys input)

    # Process in reverse topological order (leaves first)
    for ii in reverse(topological_sort_by_dfs(graph))
        if has_leader(graph, ii)
            # Build augmented z with follower K evaluations
            augmented_z = _build_augmented_z_est(ii, z_current, K_evals, graph, follower_cache, buffer_cache)

            m_rows = π_sizes[ii]
            m_cols = length(ws[ii])
            n_cols = length(ys[ii])

            if inplace_MN
                # Strategy A: In-place M/N evaluation
                M_fns_ip = setup_info.var"M_fns!"
                N_fns_ip = setup_info.var"N_fns!"

                # Allocate flat buffers for in-place evaluation
                if !haskey(M_buf, ii)
                    M_buf[ii] = Matrix{Float64}(undef, m_rows, m_cols)
                    N_buf[ii] = Matrix{Float64}(undef, m_rows, n_cols)
                end

                # In-place functions write into flat arrays
                M_flat = Vector{Float64}(undef, m_rows * m_cols)
                N_flat = Vector{Float64}(undef, m_rows * n_cols)
                M_fns_ip[ii](M_flat, augmented_z)
                N_fns_ip[ii](N_flat, augmented_z)

                M_evals[ii] = reshape(M_flat, m_rows, m_cols)
                N_evals[ii] = reshape(N_flat, m_rows, n_cols)
            else
                # Baseline: out-of-place M/N evaluation
                M_raw = M_fns_oop[ii](augmented_z)
                N_raw = N_fns_oop[ii](augmented_z)

                M_evals[ii] = reshape(M_raw, m_rows, m_cols)
                N_evals[ii] = reshape(N_raw, m_rows, n_cols)
            end

            # Solve K = M \ N
            # Note: M is π_sizes[ii] × length(ws[ii]), which may be non-square.
            # For square M, lu/lu! is appropriate.
            # For non-square M, backslash uses QR internally — we can't use lu.
            is_square = m_rows == m_cols

            if inplace_lu && is_square
                # Strategy C: lu! + ldiv! (in-place LU factorization + in-place solve)
                if !haskey(K_buf, ii)
                    K_buf[ii] = Matrix{Float64}(undef, m_cols, n_cols)
                    M_scratch[ii] = Matrix{Float64}(undef, m_rows, m_cols)
                end
                copyto!(M_scratch[ii], M_evals[ii])
                lu_M = lu!(M_scratch[ii])
                ldiv!(K_buf[ii], lu_M, N_evals[ii])
                K_evals[ii] = K_buf[ii]
            elseif inplace_ldiv && is_square
                # Strategy B: ldiv! with allocating lu (in-place solve only)
                if !haskey(K_buf, ii)
                    K_buf[ii] = Matrix{Float64}(undef, m_cols, n_cols)
                end
                lu_M = lu(M_evals[ii])
                ldiv!(K_buf[ii], lu_M, N_evals[ii])
                K_evals[ii] = K_buf[ii]
            else
                # Baseline: allocating backslash (handles rectangular M via QR)
                K_evals[ii] = M_evals[ii] \ N_evals[ii]
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

    return all_K_vec, (; M_evals, N_evals, K_evals)
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
        use_armijo::Bool = true
    )

Iterative nonlinear solver using quasi-linear policy approximation.

Uses Armijo backtracking line search for step size selection.

# Arguments
- `precomputed::NamedTuple` - Precomputed symbolic components from `preoptimize_nonlinear_solver`
- `initial_states::Dict` - Initial state for each player (parameter values)
- `hierarchy_graph::SimpleDiGraph` - Hierarchy graph

# Keyword Arguments
- `initial_guess::Vector` - Starting point (or nothing for zero initialization)
- `max_iters::Int=100` - Maximum iterations
- `tol::Float64=1e-6` - Convergence tolerance on KKT residual
- `verbose::Bool=false` - Print iteration info
- `use_armijo::Bool=true` - Use Armijo line search

# Returns
Named tuple containing:
- `sol::Vector` - Solution vector
- `converged::Bool` - Whether solver converged
- `iterations::Int` - Number of iterations taken
- `residual::Float64` - Final KKT residual norm
- `status::Symbol` - Solver status (:solved, :max_iters_reached, :linear_solver_error)
"""
function run_nonlinear_solver(
    precomputed::NamedTuple,
    initial_states::Dict,
    hierarchy_graph::SimpleDiGraph;
    initial_guess::Union{Nothing, Vector{Float64}} = nothing,
    max_iters::Int = 100,
    tol::Float64 = 1e-6,
    verbose::Bool = false,
    use_armijo::Bool = true,
    to::TimerOutput = TimerOutput(),
    inplace_MN::Bool = false,
    inplace_ldiv::Bool = false,
    inplace_lu::Bool = false
)
    # Unpack precomputed components
    problem_vars = precomputed.problem_vars
    setup_info = precomputed.setup_info
    mcp_obj = precomputed.mcp_obj
    linsolver = precomputed.linsolver
    all_variables = precomputed.all_variables

    # Build parameter values vector
    θs_order = sort(collect(keys(initial_states)))
    θ_vals_vec = vcat([initial_states[k] for k in θs_order]...)

    # Initialize z estimate
    z_est = something(initial_guess, zeros(length(all_variables)))
    if length(z_est) < length(all_variables)
        verbose && @info "Padding initial guess from $(length(z_est)) to $(length(all_variables))"
        z_est = vcat(z_est, zeros(length(all_variables) - length(z_est)))
    end

    # Solver state
    num_iterations = 0
    convergence_criterion = Inf
    status = :in_progress
    converged = false

    # Allocate buffers
    n = length(all_variables)
    F_eval = zeros(n)
    ∇F = copy(mcp_obj.jacobian_z!.result_buffer)

    # Helper: compute parameters (θ, K) for a given z
    function params_for_z(z)
        all_K_vec, _ = compute_K_evals(z, problem_vars, setup_info; inplace_MN, inplace_ldiv, inplace_lu)
        return vcat(θ_vals_vec, all_K_vec), all_K_vec
    end

    # Main iteration loop
    while true
        # Evaluate K matrices at current z
        @timeit to "compute K evals" begin
            param_vec, all_K_vec = params_for_z(z_est)
        end

        # Check convergence
        @timeit to "residual evaluation" begin
            mcp_obj.f!(F_eval, z_est, param_vec)
            convergence_criterion = norm(F_eval)
        end

        # Guard against NaN/Inf in residual computation
        if !isfinite(convergence_criterion)
            verbose && @warn "Residual contains NaN or Inf values, terminating"
            status = :numerical_error
            break
        end

        verbose && @info "Iteration $num_iterations: residual = $convergence_criterion"

        if convergence_criterion < tol
            status = num_iterations > 0 ? :solved : :solved_initial_point
            converged = true
            break
        end

        if num_iterations >= max_iters
            status = :max_iters_reached
            break
        end

        num_iterations += 1

        # Solve linearized system: ∇F * δz = -F
        @timeit to "Jacobian evaluation" begin
            mcp_obj.jacobian_z!(∇F, z_est, param_vec)
        end

        @timeit to "Newton step" begin
            linsolver.A = ∇F
            linsolver.b = -F_eval
            solution = solve!(linsolver)
        end

        if !SciMLBase.successful_retcode(solution) && solution.retcode !== SciMLBase.ReturnCode.Default
            verbose && @warn "Linear solve failed: $(solution.retcode)"
            status = :linear_solver_error
            break
        end

        δz = solution.u

        # Line search for step size
        @timeit to "line search" begin
            α = 1.0
            F_eval_current_norm = norm(F_eval)

            if use_armijo
                for _ in 1:LINESEARCH_MAX_ITERS
                    z_trial = z_est .+ α .* δz
                    param_trial, _ = params_for_z(z_trial)
                    mcp_obj.f!(F_eval, z_trial, param_trial)

                    if norm(F_eval) < F_eval_current_norm
                        break
                    end
                    α *= LINESEARCH_BACKTRACK_FACTOR
                end
            end
        end

        # Update estimate (in-place to avoid allocation)
        @. z_est += α * δz

        # Guard against NaN/Inf in solution
        if any(!isfinite, z_est)
            verbose && @warn "Solution contains NaN or Inf values after update, terminating"
            status = :numerical_error
            break
        end
    end

    return (;
        sol = z_est,
        converged,
        iterations = num_iterations,
        residual = convergence_criterion,
        status
    )
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
    # Merit function: ϕ(z) = ||f(z)||²
    ϕ_0 = norm(f_z)^2

    # For Newton-like methods, the directional derivative is approximately -2*||f||²
    # Armijo condition: ϕ(z + αδz) ≤ ϕ(z) + σ * α * ∇ϕ'δz
    # With ∇ϕ'δz ≈ -2*||f||², condition becomes: ϕ_new ≤ ϕ_0 * (1 - 2*σ*α)

    α = α_init
    for _ in 1:max_iters
        z_new = z .+ α .* δz
        f_new = f_eval(z_new)
        ϕ_new = norm(f_new)^2

        # Sufficient decrease condition
        if ϕ_new <= ϕ_0 + σ * α * (-2 * ϕ_0)
            return α
        end

        # Backtrack
        α *= β
    end

    # Signal failure if no sufficient decrease found
    @warn "Armijo line search failed to find sufficient decrease after $max_iters iterations"
    return 0.0
end
