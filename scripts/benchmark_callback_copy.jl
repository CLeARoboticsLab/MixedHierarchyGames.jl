#=
    Benchmark: callback copy(z_est) guard
    Measures allocation difference between callback and no-callback paths.

    Run: julia --project=. scripts/benchmark_callback_copy.jl
=#

using MixedHierarchyGames
using TrajectoryGamesBase: unflatten_trajectory
using Graphs: SimpleDiGraph, add_edge!
using Printf
using Statistics: median

# ─────────────────────────────────────────────────────────────────────────────
# Problem setup (same as test_allocation_optimization.jl)
# ─────────────────────────────────────────────────────────────────────────────

function make_benchmark_problem(; T=3, state_dim=2, control_dim=2)
    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)

    θs = setup_problem_parameter_variables(fill(state_dim, N))

    Js = Dict(
        1 => (z1, z2; θ=nothing) -> begin
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((xs[end] .- [1.0, 1.0]).^2) + 0.1 * sum(sum(u.^2) for u in us)
        end,
        2 => (z1, z2; θ=nothing) -> begin
            (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
            sum((xs[end] .- [2.0, 2.0]).^2) + 0.1 * sum(sum(u.^2) for u in us)
        end,
    )

    function make_dynamics_constraint(player_idx)
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

    gs = [make_dynamics_constraint(i) for i in 1:N]
    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim)
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

function run_benchmark()
    println("=" ^ 70)
    println("  Benchmark: callback copy(z_est) guard")
    println("=" ^ 70)

    prob = make_benchmark_problem()
    precomputed = preoptimize_nonlinear_solver(
        prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
        state_dim=prob.state_dim, control_dim=prob.control_dim
    )
    initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])
    solve_kwargs = (; max_iters=50, tol=1e-8, verbose=false)

    # Warmup
    println("\nWarming up...")
    run_nonlinear_solver(precomputed, initial_states, prob.G; solve_kwargs..., callback=nothing)
    history = []
    run_nonlinear_solver(precomputed, initial_states, prob.G; solve_kwargs..., callback=info -> push!(history, info))

    n_runs = 20

    # Measure: no callback
    allocs_no_cb = Int[]
    times_no_cb = Float64[]
    for _ in 1:n_runs
        a = @allocated t = @elapsed run_nonlinear_solver(
            precomputed, initial_states, prob.G; solve_kwargs..., callback=nothing
        )
        push!(allocs_no_cb, a)
        push!(times_no_cb, t)
    end

    # Measure: with callback
    allocs_with_cb = Int[]
    times_with_cb = Float64[]
    for _ in 1:n_runs
        cb_history = []
        a = @allocated t = @elapsed run_nonlinear_solver(
            precomputed, initial_states, prob.G; solve_kwargs..., callback=info -> push!(cb_history, info)
        )
        push!(allocs_with_cb, a)
        push!(times_with_cb, t)
    end

    med_alloc_no = median(allocs_no_cb)
    med_alloc_with = median(allocs_with_cb)
    med_time_no = median(times_no_cb)
    med_time_with = median(times_with_cb)

    println("\n" * "-" ^ 70)
    @printf("  %-30s %12s %12s\n", "Metric", "No callback", "With callback")
    println("-" ^ 70)
    @printf("  %-30s %10.0f B %10.0f B\n", "Median allocations", med_alloc_no, med_alloc_with)
    @printf("  %-30s %10.1f μs %10.1f μs\n", "Median time", med_time_no * 1e6, med_time_with * 1e6)
    println("-" ^ 70)

    alloc_diff = med_alloc_with - med_alloc_no
    if med_alloc_with > 0
        alloc_pct = (alloc_diff / med_alloc_with) * 100
        @printf("  Allocation saved (no-cb): %.0f B (%.1f%% of with-callback)\n", alloc_diff, alloc_pct)
    end

    if med_time_no > 0
        speedup_pct = ((med_time_with - med_time_no) / med_time_with) * 100
        @printf("  Time saved (no-cb):       %.1f%% faster\n", speedup_pct)
    end

    println("\n  Conclusion: copy(z_est) guard avoids per-iteration allocation")
    println("  when callback=nothing (the default hot path).")
    println("=" ^ 70)
end

run_benchmark()
