#=
    Benchmark: Adaptive sparse M\N solve across game structures

    Compares :never vs :always vs :auto across:
    1. Nash game (flat, no hierarchy) — all players are leaves
    2. 3-player Stackelberg chain (1→2→3)
    3. 5-player Stackelberg chain (1→2→3→4→5)

    For each structure, reports:
    - Per-player M matrix sizes and sparsity
    - Per-solve time for each mode
    - Whether :auto picks the right strategy per player
=#

using Dates
using Graphs: SimpleDiGraph, add_edge!, nv, topological_sort_by_dfs
using LinearAlgebra: norm
using SparseArrays: sparse, nnz
using Printf
using MixedHierarchyGames:
    preoptimize_nonlinear_solver,
    compute_K_evals,
    setup_problem_parameter_variables,
    has_leader,
    is_leaf
using TrajectoryGamesBase: unflatten_trajectory

#= Problem builders =#

function make_nash_game(; N=3, T=3, state_dim=2, control_dim=2)
    G = SimpleDiGraph(N)
    primal_dim = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))

    function make_cost(idx, goal)
        (zs...; θ=nothing) -> begin
            z = zs[idx]
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
    end

    goals = [[Float64(i), Float64(i)] for i in 1:N]
    Js = Dict(i => make_cost(i, goals[i]) for i in 1:N)

    function make_g(idx)
        z -> begin
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = [xs[t+1] - xs[t] - us[t] for t in 1:T]
            push!(constraints, xs[1] - θs[idx])
            vcat(constraints...)
        end
    end

    gs = [make_g(i) for i in 1:N]
    (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N)
end

function make_chain(; N=3, T=3, state_dim=2, control_dim=2)
    G = SimpleDiGraph(N)
    for i in 1:(N-1)
        add_edge!(G, i, i+1)
    end
    primal_dim = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))

    function make_cost(idx, goal)
        (zs...; θ=nothing) -> begin
            z = zs[idx]
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
    end

    goals = [vcat(fill(Float64(i), min(2, state_dim)), zeros(max(0, state_dim - 2))) for i in 1:N]
    Js = Dict(i => make_cost(i, goals[i]) for i in 1:N)

    function make_g(idx)
        z -> begin
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                # Simple integrator: first control_dim states get control input
                x_next = copy(xs[t])
                for d in 1:min(control_dim, state_dim)
                    x_next[d] += us[t][d]
                end
                push!(constraints, xs[t+1] - x_next)
            end
            push!(constraints, xs[1] - θs[idx])
            vcat(constraints...)
        end
    end

    gs = [make_g(i) for i in 1:N]
    (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N)
end

#= Benchmark runner =#

function benchmark_problem(label, prob; n_warmup=5, n_trials=200)
    println("\n", "="^70)
    println("  $label  (N=$(prob.N), T=$(prob.T), s=$(prob.state_dim), c=$(prob.control_dim))")
    println("="^70)

    precomputed = preoptimize_nonlinear_solver(
        prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
        state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
    )
    z_current = randn(length(precomputed.all_variables))

    # Per-player analysis
    _, info_ref = compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info; use_sparse=:never)

    println("\n  Per-player M matrix analysis:")
    println("  ", "-"^66)
    @printf("  %-8s %-10s %-12s %-14s %-10s %-10s\n",
        "Player", "Role", "M size", "nnz/total", "Sparsity", "Auto mode")
    println("  ", "-"^66)

    for ii in 1:prob.N
        M = info_ref.M_evals[ii]
        role = if !has_leader(prob.G, ii)
            "root"
        elseif is_leaf(prob.G, ii)
            "leaf"
        else
            "mid"
        end

        if isnothing(M)
            @printf("  %-8d %-10s %-12s %-14s %-10s %-10s\n",
                ii, role, "n/a", "n/a", "n/a", "n/a (no M\\N)")
        else
            total = length(M)
            nz = count(!iszero, M)
            sparsity = 1.0 - nz / total
            auto_mode = is_leaf(prob.G, ii) ? "dense" : "sparse"
            @printf("  %-8d %-10s %-12s %-14s %-10.3f %-10s\n",
                ii, role, "$(size(M, 1))×$(size(M, 2))", "$nz/$total", sparsity, auto_mode)
        end
    end

    # Timing benchmark
    modes = [:never, :always, :auto]
    times = Dict{Symbol, Float64}()

    for mode in modes
        # Warmup
        for _ in 1:n_warmup
            compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info; use_sparse=mode)
        end

        # Timed trials
        t = @elapsed for _ in 1:n_trials
            compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info; use_sparse=mode)
        end
        times[mode] = t / n_trials
    end

    println("\n  Timing (per compute_K_evals call, $n_trials trials):")
    println("  ", "-"^50)
    @printf("  %-10s %12s %12s\n", "Mode", "Time (μs)", "vs :never")
    println("  ", "-"^50)
    t_never = times[:never]
    for mode in modes
        t = times[mode]
        speedup = t_never / t
        @printf("  %-10s %12.1f %11.2fx\n", mode, t * 1e6, speedup)
    end

    # Verify numerical equivalence
    K_never, _ = compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info; use_sparse=:never)
    K_always, _ = compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info; use_sparse=:always)
    K_auto, _ = compute_K_evals(z_current, precomputed.problem_vars, precomputed.setup_info; use_sparse=:auto)

    err_always = norm(K_always - K_never) / max(norm(K_never), 1.0)
    err_auto = norm(K_auto - K_never) / max(norm(K_never), 1.0)
    println("\n  Numerical equivalence (relative to :never):")
    @printf("  :always error = %.2e\n", err_always)
    @printf("  :auto   error = %.2e\n", err_auto)

    return times
end

#= Main =#

println("Adaptive Sparse M\\N Solve Benchmark")
println("Julia ", VERSION, " on ", Sys.MACHINE)
println("Date: ", Dates.now())

results = Dict{String, Dict{Symbol, Float64}}()

# 1. Nash game (all leaves, no hierarchy)
results["Nash 3P"] = benchmark_problem("Nash game (3 players, no hierarchy)",
    make_nash_game(N=3, T=3, state_dim=2, control_dim=2))

# 2. 3-player Stackelberg chain
results["Chain 3P"] = benchmark_problem("Stackelberg chain (1→2→3)",
    make_chain(N=3, T=3, state_dim=2, control_dim=2))

# 3. 5-player Stackelberg chain
results["Chain 5P"] = benchmark_problem("Stackelberg chain (1→2→3→4→5)",
    make_chain(N=5, T=3, state_dim=2, control_dim=2))

# 4. 3-player chain with larger dimensions (closer to real problems)
results["Chain 3P large"] = benchmark_problem("Stackelberg chain 3P (T=5, s=4, c=2)",
    make_chain(N=3, T=5, state_dim=4, control_dim=2))

# Summary table
println("\n\n", "="^70)
println("  SUMMARY TABLE")
println("="^70)
@printf("\n  %-20s %12s %12s %12s %12s\n",
    "Structure", ":never (μs)", ":always (μs)", ":auto (μs)", ":auto vs best")
println("  ", "-"^68)

for label in sort(collect(keys(results)))
    times = results[label]
    t_never = times[:never] * 1e6
    t_always = times[:always] * 1e6
    t_auto = times[:auto] * 1e6
    best = min(t_never, t_always)
    ratio = t_auto / best
    @printf("  %-20s %12.1f %12.1f %12.1f %11.2fx\n",
        label, t_never, t_always, t_auto, ratio)
end

println("\n  Interpretation:")
println("  - :auto should match or beat both :always and :never across all structures")
println("  - Nash game: :auto → all dense (leaves only) → should match :never")
println("  - Chain: :auto → sparse for leaders, dense for leaf → best of both")
println("  - ratio < 1.0 means :auto is faster than the best of :never/:always")
println("  - ratio ≈ 1.0 means :auto matches the best strategy")
