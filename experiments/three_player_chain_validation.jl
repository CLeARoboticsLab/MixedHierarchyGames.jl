#=
    Three-Player Chain Validation Script

    Validates that the new QPSolver produces identical results to the old
    `run_lq_solver` implementation for a three-player Stackelberg chain game.

    Hierarchy structure (edges point from leader to follower):
        P2 (root leader)
        ├── P1 (follower of P2)
        └── P3 (follower of P2)

    Usage:
        julia --project experiments/three_player_chain_validation.jl
=#

using MixedHierarchyGames
using Graphs: SimpleDiGraph, add_edge!, nv, inneighbors, outneighbors, indegree, outdegree, vertices, topological_sort, BFSIterator
using SymbolicTracingUtils
using Symbolics
using TrajectoryGamesBase: unflatten_trajectory
using LinearAlgebra: norm, I
using ParametricMCPs: ParametricMCPs
using TimerOutputs

# ============================================================================
# OLD SOLVER COMPONENTS (from legacy/)
# These are the reference implementations we're validating against
# ============================================================================

#--- graph_utils.jl ---
function old_has_leader(graph::SimpleDiGraph, node::Int)
    return !old_is_root(graph, node)
end

function old_is_root(graph::SimpleDiGraph, node::Int)
    return iszero(indegree(graph, node))
end

function old_get_roots(graph::SimpleDiGraph)
    return [v for v in vertices(graph) if old_is_root(graph, v)]
end

function old_get_all_leaders(graph::SimpleDiGraph, node::Int)
    parents_path = []
    parents = inneighbors(graph, node)
    while !isempty(parents)
        parent = only(parents)
        push!(parents_path, parent)
        parents = inneighbors(graph, parent)
    end
    return reverse(parents_path)
end

function old_get_all_followers(graph::SimpleDiGraph, node)
    all_children = outneighbors(graph, node)
    children = all_children
    has_next_layer = !isempty(children)
    while has_next_layer
        grandchildren = []
        for child in children
            for grandchild in outneighbors(graph, child)
                push!(all_children, grandchild)
                push!(grandchildren, grandchild)
            end
        end
        children = grandchildren
        has_next_layer = !isempty(children)
    end
    return all_children
end

function old_is_leaf(graph::SimpleDiGraph, node::Int)
    return outdegree(graph, node) == 0
end

#--- make_symbolic_variables.jl ---
function make_symbolic_variable(args...)
    variable_name = args[1]
    time = string(last(args))
    time_str = ""
    num_items = length(args)
    @assert variable_name in [:x, :u, :λ, :ψ, :μ, :z, :M, :N, :Φ, :K, :θ]
    variable_name_str = string(variable_name)
    if variable_name in [:x] && num_items == 2
        return Symbol(variable_name_str * time_str)
    elseif variable_name in [:θ] && num_items == 2
        return Symbol(variable_name_str * "^" * string(args[2]))
    elseif variable_name in [:u, :λ, :z, :M, :N, :K, :θ] && num_items == 3
       return Symbol(variable_name_str * "^" * string(args[2]) * time_str)
    elseif variable_name in [:ψ, :μ] && num_items == 4
        return Symbol(variable_name_str * "^(" * string(args[2]) * "-" * string(args[3]) * ")" * time_str)
    elseif variable_name in [:z] && num_items > 3
        indices = join(string.(args[2:num_items-1]), ",")
        return Symbol(variable_name_str * "^(" * indices * ")" *time_str)
    else
        error("Invalid format has number of args $(num_items) for $args.")
    end
end

#--- solve_kkt_conditions.jl (just solve_with_path) ---
function old_solve_with_path(πs, variables, θs, parameter_values)
    symbolic_type = eltype(variables)
    F = Vector{symbolic_type}([
        vcat(collect(values(πs))...)...
    ])
    z̲ = fill(-Inf, length(F));
    z̅ = fill(Inf, length(F))
    order = sort(collect(keys(πs)))
    all_θ_vec = vcat([θs[k] for k in order]...)
    all_param_vals_vec = vcat([parameter_values[k] for k in order]...)
    parametric_mcp = ParametricMCPs.ParametricMCP(F, variables, all_θ_vec, z̲, z̅; compute_sensitivities = false)
    z_sol, status, info = ParametricMCPs.solve(
        parametric_mcp,
        all_param_vals_vec;
        initial_guess = zeros(length(variables)),
        verbose = false,
        cumulative_iteration_limit = 100000,
        proximal_perturbation = 1e-2,
        use_basics = true,
        use_start = true,
    )
    return z_sol, status, info
end

#--- general_kkt_construction.jl (old KKT construction) ---
function old_get_lq_kkt_conditions(G::SimpleDiGraph,
    Js::Dict{Int, Any},
    zs,
    λs,
    μs::Dict{Tuple{Int, Int}, Any},
    gs,
    ws::Dict{Int, Any},
    ys::Dict{Int, Any},
    θ;
    verbose = false,
    to = TimerOutput())

    Ms = Dict{Int, Any}()
    Ns = Dict{Int, Any}()
    Ks = Dict{Int, Any}()
    πs = Dict{Int, Any}()
    Φs = Dict{Int, Any}()

    order = reverse(topological_sort(G))

    for ii in order
        zi_size = length(zs[ii])
        Lᵢ = Js[ii](zs..., θ) - λs[ii]' * gs[ii](zs[ii])

        if old_is_leaf(G, ii)
            @timeit to "[KKT Conditions] Leaf" begin
                πs[ii] = vcat(Symbolics.gradient(Lᵢ, zs[ii]),
                              gs[ii](zs[ii]))
            end
        else
            @timeit to "[KKT Conditions] Non-Leaf" begin
                for jj in BFSIterator(G, ii)
                    if ii == jj
                        continue
                    end
                    πⱼ = πs[jj]
                    extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))
                    @timeit to "[KKT Conditions][Non-Leaf][Symbolic M '\' N]" begin
                        Φʲ = - extractor * Ks[jj] * ys[jj]
                    end
                    Lᵢ -= μs[(ii, jj)]' * (zs[jj] - Φʲ)
                end
            end
        end

        πᵢ = []
        for jj in BFSIterator(G, ii)
            @timeit to "[KKT Conditions] Compute πᵢ" begin
                πᵢ = vcat(πᵢ, Symbolics.gradient(Lᵢ, zs[jj]))
                if ii != jj
                    extractor = hcat(I(zi_size), zeros(zi_size, length(ws[jj]) - zi_size))
                    Φʲ = - extractor * Ks[jj] * ys[jj]
                    πᵢ = vcat(πᵢ, zs[jj] - Φʲ)
                end
            end
        end
        πᵢ = vcat(πᵢ, gs[ii](zs[ii]))
        πs[ii] = πᵢ

        if old_has_leader(G, ii)
            @timeit to "[KKT Conditions] Compute M and N for follower $ii" begin
                Ms[ii] = Symbolics.jacobian(πs[ii], ws[ii])
                Ns[ii] = Symbolics.jacobian(πs[ii], ys[ii])
                Ks[ii] = Ms[ii] \ Ns[ii]
            end
        end
    end

    return πs, Ms, Ns, (;K_evals = nothing)
end

function old_strip_policy_constraints(πs, G, zs, gs)
    πs_stripped = Dict{Int, Any}()
    for ii in 1:nv(G)
        πᵢ = πs[ii]
        parts = Any[]
        idx = 1
        for jj in BFSIterator(G, ii)
            len_z = length(zs[jj])
            push!(parts, πᵢ[idx:(idx + len_z - 1)])
            idx += len_z
            if ii != jj
                idx += len_z
            end
        end
        len_g = length(gs[ii](zs[ii]))
        push!(parts, πᵢ[idx:(idx + len_g - 1)])
        idx += len_g
        @assert idx - 1 == length(πᵢ) "strip_policy_constraints: unexpected π length for player $ii."
        πs_stripped[ii] = vcat(parts...)
    end
    return πs_stripped
end

#--- automatic_solver.jl (old variable setup and solver) ---
function old_setup_problem_parameter_variables(backend, num_params_per_player; verbose = false)
    θs = Dict{Int, Any}()
    for idx in 1:length(num_params_per_player)
        θs[idx] = SymbolicTracingUtils.make_variables(backend, make_symbolic_variable(:θ, idx), num_params_per_player[idx])
    end
    return θs
end

function old_setup_problem_variables(H, graph, primal_dimension_per_player, gs; backend = SymbolicTracingUtils.SymbolicsBackend(), verbose = false)
    N = nv(graph)

    zs = [SymbolicTracingUtils.make_variables(
        backend,
        make_symbolic_variable(:z, i, H),
        primal_dimension_per_player,
    ) for i in 1:N]

    λs = [SymbolicTracingUtils.make_variables(
        backend,
        make_symbolic_variable(:λ, i, H),
        length(gs[i](zs[i])),
    ) for i in 1:N]

    μs = Dict{Tuple{Int, Int}, Any}()
    ws = Dict{Int, Any}()
    ys = Dict{Int, Any}()
    for i in 1:N
        leaders = old_get_all_leaders(graph, i)
        ys[i] = vcat(zs[leaders]...)
        ws[i] = zs[i]

        for jj in 1:N
            if jj in leaders || jj == i
                continue
            end
            ws[i] = vcat(ws[i], zs[jj])
        end

        for jj in BFSIterator(graph, i)
            ws[i] = vcat(ws[i], λs[jj])
        end

        followers = old_get_all_followers(graph, i)
        for j in followers
            μs[(i, j)] = SymbolicTracingUtils.make_variables(
                backend,
                make_symbolic_variable(:μ, i, j, H),
                primal_dimension_per_player,
            )
            ws[i] = vcat(ws[i], μs[(i, j)])
        end
    end

    temp = vcat(collect(values(μs))...)
    all_variables = vcat(vcat(zs...), vcat(λs...))
    if !isempty(temp)
        all_variables = vcat(all_variables, vcat(collect(values(μs))...))
    end

    (; all_variables, zs, λs, μs, ys, ws)
end

function old_run_lq_solver(H, graph, primal_dimension_per_player, Js, gs, θs, parameter_values; verbose = false)
    N = nv(graph)
    (; all_variables, zs, λs, μs, ws, ys) = old_setup_problem_variables(H, graph, primal_dimension_per_player, gs; verbose)
    πs, _, _, _ = old_get_lq_kkt_conditions(graph, Js, zs, λs, μs, gs, ws, ys, θs)

    temp = vcat(collect(values(μs))...)
    all_variables = vcat(vcat(zs...), vcat(λs...))
    if !isempty(temp)
        all_variables = vcat(all_variables, vcat(collect(values(μs))...))
    end
    πs_solve = old_strip_policy_constraints(πs, graph, zs, gs)
    z_sol, status, info = old_solve_with_path(πs_solve, all_variables, θs, parameter_values)

    z_sol, status, info, all_variables, (; πs, zs, λs, μs, θs)
end

# ============================================================================
# THREE-PLAYER PROBLEM SETUP
# ============================================================================

function get_three_player_openloop_lq_problem(T=10, Δt=0.5; verbose = false, backend=SymbolicTracingUtils.SymbolicsBackend())
    N = 3

    # Set up the information structure:
    # P2 is leader of P1, and P2 is leader of P3
    # Edges point from the node with followers TO the follower
    G = SimpleDiGraph(N);
    add_edge!(G, 2, 1);  # P2 -> P1 (P2 leads P1)
    add_edge!(G, 2, 3);  # P2 -> P3 (P2 leads P3)

    H = 1
    Hp1 = H+1

    flatten(vs) = collect(Iterators.flatten(vs))

    state_dimension = 2
    control_dimension = 2

    x_dim = state_dimension * (T+1)
    u_dim = control_dimension * (T+1)
    aggre_state_dimension = x_dim * N
    aggre_control_dimension = u_dim * N
    total_dimension = aggre_state_dimension + aggre_control_dimension
    primal_dimension_per_player = x_dim + u_dim

    num_params_per_player = repeat([state_dimension], N)
    θs = old_setup_problem_parameter_variables(backend, num_params_per_player)

    problem_dims = (;
        state_dimension,
        control_dimension,
        x_dim,
        u_dim,
        aggre_state_dimension,
        aggre_control_dimension,
        total_dimension,
        primal_dimension_per_player,
    )

    # Player objectives
    # Note: Old solver uses positional θ, new solver uses keyword θ
    # We define both signatures
    function J₁(z₁, z₂, z₃, θi)
        (; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
        xs¹, us¹ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
        xs², us² = xs, us
        0.5*sum((xs¹[end] .- xs²[end]) .^ 2) + 0.05*sum(sum(u .^ 2) for u in us¹)
    end
    J₁(z₁, z₂, z₃; θ=nothing) = J₁(z₁, z₂, z₃, θ)

    function J₂(z₁, z₂, z₃, θi)
        (; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
        xs³, us³ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
        xs², us² = xs, us
        (; xs, us) = unflatten_trajectory(z₁, state_dimension, control_dimension)
        xs¹, us¹ = xs, us
        sum((0.5*(xs¹[end] .+ xs³[end])) .^ 2) + 0.05*sum(sum(u .^ 2) for u in us²)
    end
    J₂(z₁, z₂, z₃; θ=nothing) = J₂(z₁, z₂, z₃, θ)

    function J₃(z₁, z₂, z₃, θi)
        (; xs, us) = unflatten_trajectory(z₃, state_dimension, control_dimension)
        xs³, us³ = xs, us
        (; xs, us) = unflatten_trajectory(z₂, state_dimension, control_dimension)
        xs², us² = xs, us
        0.5*sum((xs³[end] .- xs²[end]) .^ 2) + 0.05*sum(sum(u³ .^ 2) for u³ in us³) + 0.05*sum(sum(u² .^ 2) for u² in us²)
    end
    J₃(z₁, z₂, z₃; θ=nothing) = J₃(z₁, z₂, z₃, θ)

    Js = Dict{Int, Any}(
        1 => J₁,
        2 => J₂,
        3 => J₃,
    )

    # Dynamics constraints
    function single_integrator_dynamics(z, t)
        (; xs, us) = unflatten_trajectory(z, state_dimension, control_dimension)
        x = xs[t]
        u = us[t]
        xp1 = xs[t+1]
        return xp1 - x - Δt*u
    end

    make_ic_constraint(i) = function (zᵢ)
        (; xs, us) = unflatten_trajectory(zᵢ, state_dimension, control_dimension)
        x1 = xs[1]
        return x1 - θs[i]
    end

    dynamics_constraint(zᵢ) =
        mapreduce(vcat, 1:T) do t
            single_integrator_dynamics(zᵢ, t)
        end

    gs = [function (zᵢ)
        vcat(dynamics_constraint(zᵢ), make_ic_constraint(i)(zᵢ))
    end for i in 1:N]

    return N, G, H, problem_dims, Js, gs, θs, backend
end

# ============================================================================
# VALIDATION
# ============================================================================

"""
    run_validation(; T=3, Δt=0.5, verbose=false)

Run the three-player chain validation test.

Compares solutions from:
1. Old solver: `old_run_lq_solver` (reference implementation)
2. New solver: `run_qp_solver` from `MixedHierarchyGames`

Returns true if solutions match within tolerance.
"""
function run_validation(; T=3, Δt=0.5, tol=1e-10, verbose=false)
    println("=" ^ 60)
    println("Three-Player Chain Validation Test")
    println("=" ^ 60)
    println()

    # Get the problem setup
    N, G, H, problem_dims, Js, gs, θs, backend = get_three_player_openloop_lq_problem(T, Δt; verbose, backend=SymbolicTracingUtils.SymbolicsBackend())

    primal_dimension_per_player = problem_dims.primal_dimension_per_player

    # Initial states for each player
    parameter_values = [
        [0.0, 2.0],   # P1 initial state
        [2.0, 4.0],   # P2 initial state
        [6.0, 8.0]    # P3 initial state
    ]

    println("Problem configuration:")
    println("  - Number of players: $N")
    println("  - Hierarchy: P2 -> P1, P2 -> P3 (P2 is leader)")
    println("  - Time horizon: T=$T, Δt=$Δt")
    println("  - Primal dimension per player: $primal_dimension_per_player")
    println()

    # ----------------------------------------------------------------
    # Run OLD solver
    # ----------------------------------------------------------------
    println("Running OLD solver (reference implementation)...")
    z_sol_old, status_old, info_old, all_vars_old, vars_old = old_run_lq_solver(
        H, G, primal_dimension_per_player, Js, gs, θs, parameter_values;
        verbose=verbose
    )
    println("  Status: $status_old")
    println("  Solution norm: $(norm(z_sol_old))")
    println()

    # ----------------------------------------------------------------
    # Run NEW solver
    # ----------------------------------------------------------------
    println("Running NEW solver (MixedHierarchyGames._run_qp_solver)...")

    # Convert parameter_values to Dict format expected by new solver
    param_dict = Dict(i => parameter_values[i] for i in 1:N)

    # Convert gs to Vector format
    gs_vec = [gs[i] for i in 1:N]

    # Convert primal_dims to Vector
    primal_dims_vec = fill(primal_dimension_per_player, N)

    # Note: _run_qp_solver is internal (not exported), so we use the module-qualified name
    result_new = MixedHierarchyGames._run_qp_solver(
        G,
        Js,
        gs_vec,
        primal_dims_vec,
        θs,
        param_dict;
        solver=:linear,
        verbose=verbose
    )

    z_sol_new = result_new.sol
    status_new = result_new.status

    println("  Status: $status_new")
    println("  Solution norm: $(norm(z_sol_new))")
    println()

    # ----------------------------------------------------------------
    # Compare solutions
    # ----------------------------------------------------------------
    println("Comparing solutions...")

    diff_norm = norm(z_sol_old - z_sol_new)
    max_diff = maximum(abs.(z_sol_old - z_sol_new))

    println("  Solution difference (L2 norm): $diff_norm")
    println("  Maximum element difference: $max_diff")
    println("  Tolerance: $tol")
    println()

    # Check if solutions match
    solutions_match = diff_norm < tol

    if solutions_match
        println("VALIDATION PASSED: Solutions match within tolerance ($tol)")
    else
        println("VALIDATION FAILED: Solutions differ beyond tolerance")
        println()
        println("Detailed comparison (first 10 elements):")
        for i in 1:min(10, length(z_sol_old))
            println("  [$i] old=$(z_sol_old[i]), new=$(z_sol_new[i]), diff=$(abs(z_sol_old[i] - z_sol_new[i]))")
        end
        if length(z_sol_old) > 10
            println("  ... (showing first 10 elements)")
        end
    end
    println()

    # ----------------------------------------------------------------
    # Additional validation info
    # ----------------------------------------------------------------
    println("Solution dimensions:")
    println("  Old solution length: $(length(z_sol_old))")
    println("  New solution length: $(length(z_sol_new))")

    if length(z_sol_old) != length(z_sol_new)
        println("  WARNING: Solution dimensions do not match!")
    end

    println()
    println("=" ^ 60)
    println("Validation " * (solutions_match ? "PASSED" : "FAILED"))
    println("=" ^ 60)

    return solutions_match
end

# Run validation when script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    # Use 1e-6 tolerance since we're comparing two different numerical solvers
    # (PATH vs direct linear solve), which have different numerical characteristics.
    # The solutions should be functionally identical for game-theoretic purposes.
    success = run_validation(; T=3, Δt=0.5, tol=1e-6, verbose=false)
    exit(success ? 0 : 1)
end
