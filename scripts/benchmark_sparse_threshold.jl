#=
    Benchmark: sparse threshold heuristic for :auto mode

    Compares old :auto (graph-position-based) vs new :auto (size-based threshold)
    vs :always and :never baselines.

    Usage: julia --project=. scripts/benchmark_sparse_threshold.jl
=#

using MixedHierarchyGames
using MixedHierarchyGames:
    preoptimize_nonlinear_solver,
    compute_K_evals,
    setup_problem_parameter_variables,
    run_nonlinear_solver

using Graphs: SimpleDiGraph, add_edge!
using TrajectoryGamesBase: unflatten_trajectory
using Statistics: median

# ── Problem builders ────────────────────────────────────────────────────

function make_three_player_chain(; T=3, state_dim=2, control_dim=2)
    N = 3
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)
    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))
    function make_cost(player_idx, goal)
        function cost(zs...; θ=nothing)
            z = zs[player_idx]
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        return cost
    end
    goals = [[Float64(i), Float64(i)] for i in 1:N]
    Js = Dict(i => make_cost(i, goals[i]) for i in 1:N)
    function make_dynamics(player_idx)
        function dynamics_constraint(z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                push!(constraints, xs[t+1] - xs[t] - us[t])
            end
            push!(constraints, xs[1] - θs[player_idx])
            return vcat(constraints...)
        end
        return dynamics_constraint
    end
    gs = [make_dynamics(i) for i in 1:N]
    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N)
end

function make_four_player_chain(; T=5, state_dim=4, control_dim=2)
    N = 4
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)
    add_edge!(G, 3, 4)
    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))
    function make_cost(player_idx, goal)
        function cost(zs...; θ=nothing)
            z = zs[player_idx]
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end
        return cost
    end
    goals = [[Float64(i), Float64(i), 0.0, 0.0] for i in 1:N]
    Js = Dict(i => make_cost(i, goals[i]) for i in 1:N)
    function make_dynamics(player_idx)
        function dynamics_constraint(z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                x_next = copy(xs[t])
                x_next[1] += us[t][1]
                x_next[2] += us[t][2]
                push!(constraints, xs[t+1] - x_next)
            end
            push!(constraints, xs[1] - θs[player_idx])
            return vcat(constraints...)
        end
        return dynamics_constraint
    end
    gs = [make_dynamics(i) for i in 1:N]
    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N)
end

# ── Benchmark helpers ───────────────────────────────────────────────────

function benchmark_solve(prob; n_solves=20, use_sparse=:auto, sparse_threshold=50)
    precomputed = preoptimize_nonlinear_solver(
        prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
        state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
    )
    initial_states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

    # Warmup
    run_nonlinear_solver(precomputed, initial_states, prob.G;
        use_sparse, sparse_threshold, max_iters=5)

    times = Float64[]
    allocs = Int[]
    for _ in 1:n_solves
        a = @allocated t = @elapsed run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            use_sparse, sparse_threshold, max_iters=20
        )
        push!(times, t)
        push!(allocs, a)
    end
    return (; median_time=median(times), median_alloc=median(allocs))
end

# ── Main ────────────────────────────────────────────────────────────────

println("=" ^ 70)
println("Sparse Threshold Heuristic Benchmark")
println("=" ^ 70)

problems = [
    ("3-player chain (T=3, s=2, c=2)", make_three_player_chain()),
    ("4-player chain (T=5, s=4, c=2)", make_four_player_chain()),
]

modes = [
    (:never, 50, "dense only (:never)"),
    (:auto, 50, "new :auto (threshold=50)"),
    (:always, 50, "sparse always (:always)"),
]

for (label, prob) in problems
    println("\n── $label ──")

    # Show M matrix sizes
    precomputed = preoptimize_nonlinear_solver(
        prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
        state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
    )
    z = zeros(length(precomputed.all_variables))
    _, info = compute_K_evals(z, precomputed.problem_vars, precomputed.setup_info)
    for ii in 1:prob.N
        M = info.M_evals[ii]
        if !isnothing(M)
            rows = size(M, 1)
            sparse_decision = rows >= 50 ? "SPARSE" : "DENSE"
            println("  Player $ii: M is $(size(M)) → $sparse_decision under threshold=50")
        end
    end

    for (mode, threshold, desc) in modes
        result = benchmark_solve(prob; use_sparse=mode, sparse_threshold=threshold)
        t_ms = result.median_time * 1000
        a_kb = result.median_alloc / 1024
        println("  $desc: $(round(t_ms, digits=2))ms, $(round(a_kb, digits=1))KB")
    end
end

println("\n" * "=" ^ 70)
println("Done.")
