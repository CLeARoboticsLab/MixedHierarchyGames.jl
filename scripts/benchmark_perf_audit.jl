#=
    Comprehensive Performance Audit Benchmark
    Covers: QPSolver vs NonlinearSolver, small→large, shallow→deep
    Metrics: construction time, solve time, per-component times, allocations

    Run: julia --project=. debug/benchmark_perf_audit.jl
    Run specific section: julia --project=. debug/benchmark_perf_audit.jl [section]
      Sections: all, construction, solve, components, allocations, scaling
=#

using MixedHierarchyGames
using TrajectoryGamesBase: unflatten_trajectory
using Graphs: SimpleDiGraph, add_edge!, nv
using LinearAlgebra: norm, dot
using Statistics: median, mean, std
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# Problem Builders
# ─────────────────────────────────────────────────────────────────────────────

"""Build a chain hierarchy: P1→P2→...→PN (maximum depth)."""
function build_chain(; N=3, T=3, state_dim=2, control_dim=2)
    G = SimpleDiGraph(N)
    for i in 1:(N-1)
        add_edge!(G, i, i+1)
    end
    _build_problem(G, N, T, state_dim, control_dim)
end

"""Build a star hierarchy: P1 is leader of P2,P3,...,PN (minimum depth, maximum width)."""
function build_star(; N=3, T=3, state_dim=2, control_dim=2)
    G = SimpleDiGraph(N)
    for i in 2:N
        add_edge!(G, 1, i)
    end
    _build_problem(G, N, T, state_dim, control_dim)
end

"""Build a flat Nash game: no hierarchy edges (all independent)."""
function build_nash(; N=3, T=3, state_dim=2, control_dim=2)
    G = SimpleDiGraph(N)
    _build_problem(G, N, T, state_dim, control_dim)
end

function _build_problem(G, N, T, state_dim, control_dim)
    primal_dim = (state_dim + control_dim) * (T + 1)
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

# ─────────────────────────────────────────────────────────────────────────────
# Timing Utilities
# ─────────────────────────────────────────────────────────────────────────────

function bench(f, n_warmup, n_runs)
    for _ in 1:n_warmup; f(); end
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed f()
        push!(times, t)
    end
    return times
end

function bench_alloc(f, n_warmup, n_runs)
    for _ in 1:n_warmup; f(); end
    allocs = Int[]
    for _ in 1:n_runs
        a = @allocated f()
        push!(allocs, a)
    end
    return allocs
end

function fmt_time(seconds)
    if seconds < 1e-3
        @sprintf("%8.1fμs", seconds * 1e6)
    elseif seconds < 1.0
        @sprintf("%8.2fms", seconds * 1e3)
    else
        @sprintf("%8.2fs ", seconds)
    end
end

function fmt_alloc(bytes)
    if bytes < 1024
        @sprintf("%6dB ", bytes)
    elseif bytes < 1024^2
        @sprintf("%6.1fKB", bytes / 1024)
    else
        @sprintf("%6.1fMB", bytes / 1024^2)
    end
end

function print_stat_row(label, times; extra="")
    med = median(times)
    mn = minimum(times)
    mx = maximum(times)
    σ = length(times) > 1 ? std(times) : 0.0
    @printf("  %-40s  %s  %s  %s  %s %s\n",
        label, fmt_time(med), fmt_time(mn), fmt_time(mx), fmt_time(σ), extra)
end

# ─────────────────────────────────────────────────────────────────────────────
# Problem Configurations
# ─────────────────────────────────────────────────────────────────────────────

const CONFIGS = [
    # (builder, kwargs, label, solver_types, n_runs)
    # --- Small problems ---
    (build_chain, (N=2, T=3, state_dim=2, control_dim=2), "chain 2P (s=2,T=3)", [:qp, :nl], 500),
    (build_star,  (N=2, T=3, state_dim=2, control_dim=2), "star  2P (s=2,T=3)", [:qp, :nl], 500),
    # --- Medium problems ---
    (build_chain, (N=3, T=3, state_dim=2, control_dim=2), "chain 3P (s=2,T=3)", [:qp, :nl], 300),
    (build_star,  (N=3, T=3, state_dim=2, control_dim=2), "star  3P (s=2,T=3)", [:qp, :nl], 300),
    (build_nash,  (N=3, T=3, state_dim=2, control_dim=2), "nash  3P (s=2,T=3)", [:qp, :nl], 300),
    # --- Larger state ---
    (build_chain, (N=3, T=5, state_dim=4, control_dim=2), "chain 3P (s=4,T=5)", [:nl],      200),
    (build_star,  (N=3, T=5, state_dim=4, control_dim=2), "star  3P (s=4,T=5)", [:nl],      200),
    # --- Deep hierarchy ---
    (build_chain, (N=4, T=5, state_dim=4, control_dim=2), "chain 4P (s=4,T=5)", [:nl],      100),
    (build_chain, (N=5, T=3, state_dim=2, control_dim=2), "chain 5P (s=2,T=3)", [:nl],      100),
    # --- Wide hierarchy ---
    (build_star,  (N=5, T=3, state_dim=2, control_dim=2), "star  5P (s=2,T=3)", [:nl],      100),
]

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Construction Time
# ─────────────────────────────────────────────────────────────────────────────

function run_construction_benchmark()
    println("\n" * "=" ^ 80)
    println("  SECTION 1: Construction Time (solver creation)")
    println("=" ^ 80)
    @printf("\n  %-40s  %10s  %10s  %10s  %10s\n",
        "Problem", "median", "min", "max", "σ")
    println("  " * "─" ^ 90)

    for (builder, kwargs, label, solver_types, _) in CONFIGS
        prob = builder(; kwargs...)
        states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        if :qp in solver_types
            t = bench(() -> QPSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim
            ), 1, 5)
            print_stat_row("QP:  $label", t)
        end

        if :nl in solver_types
            t = bench(() -> NonlinearSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim;
                max_iters=100, tol=1e-6
            ), 1, 5)
            print_stat_row("NL:  $label", t)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Solve Time
# ─────────────────────────────────────────────────────────────────────────────

function run_solve_benchmark()
    println("\n" * "=" ^ 80)
    println("  SECTION 2: Solve Time (solve_raw + solve)")
    println("=" ^ 80)

    # --- solve_raw ---
    println("\n  ── solve_raw() ──")
    @printf("  %-40s  %10s  %10s  %10s  %10s %s\n",
        "Problem", "median", "min", "max", "σ", "iters/status")
    println("  " * "─" ^ 100)

    for (builder, kwargs, label, solver_types, n_runs) in CONFIGS
        prob = builder(; kwargs...)
        states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        if :qp in solver_types
            solver = QPSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim)
            t = bench(() -> solve_raw(solver, states), 5, n_runs)
            raw = solve_raw(solver, states)
            extra = @sprintf("i=%d %s", raw.iterations, raw.status)
            print_stat_row("QP:  $label", t; extra)
        end

        if :nl in solver_types
            solver = NonlinearSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim;
                max_iters=100, tol=1e-6)
            t = bench(() -> solve_raw(solver, states), 5, n_runs)
            raw = solve_raw(solver, states)
            extra = @sprintf("i=%d %s", raw.iterations, raw.status)
            print_stat_row("NL:  $label", t; extra)
        end
    end

    # --- solve (with trajectory extraction) ---
    println("\n  ── solve() (includes trajectory extraction) ──")
    @printf("  %-40s  %10s  %10s  %10s  %10s\n",
        "Problem", "median", "min", "max", "σ")
    println("  " * "─" ^ 90)

    for (builder, kwargs, label, solver_types, n_runs) in CONFIGS
        prob = builder(; kwargs...)
        states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        if :nl in solver_types
            solver = NonlinearSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim;
                max_iters=100, tol=1e-6)
            t = bench(() -> solve(solver, states), 5, n_runs)
            print_stat_row("NL:  $label", t)
        end
    end

    # --- solve_raw with regularization ---
    println("\n  ── solve_raw() with regularization=1e-8 ──")
    @printf("  %-40s  %10s  %10s  %10s  %10s %s\n",
        "Problem", "median", "min", "max", "σ", "overhead")
    println("  " * "─" ^ 100)

    for (builder, kwargs, label, solver_types, n_runs) in CONFIGS
        if :nl in solver_types
            prob = builder(; kwargs...)
            states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

            solver_base = NonlinearSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim;
                max_iters=100, tol=1e-6)
            t_base = bench(() -> solve_raw(solver_base, states), 5, n_runs)

            solver_reg = NonlinearSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim;
                max_iters=100, tol=1e-6, regularization=1e-8)
            t_reg = bench(() -> solve_raw(solver_reg, states), 5, n_runs)

            overhead = (median(t_reg) / median(t_base) - 1) * 100
            extra = @sprintf("%.1f%%", overhead)
            print_stat_row("NL:  $label", t_reg; extra)
        end
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Per-Component Timing (via callback + TimerOutputs)
# ─────────────────────────────────────────────────────────────────────────────

function run_component_benchmark()
    println("\n" * "=" ^ 80)
    println("  SECTION 3: Per-Iteration Component Breakdown (NonlinearSolver)")
    println("=" ^ 80)

    # Select a few representative problems
    component_configs = [
        (build_chain, (N=2, T=3, state_dim=2, control_dim=2), "chain 2P (s=2,T=3)", 200),
        (build_chain, (N=3, T=3, state_dim=2, control_dim=2), "chain 3P (s=2,T=3)", 200),
        (build_chain, (N=3, T=5, state_dim=4, control_dim=2), "chain 3P (s=4,T=5)", 100),
        (build_star,  (N=3, T=3, state_dim=2, control_dim=2), "star  3P (s=2,T=3)", 200),
        (build_chain, (N=5, T=3, state_dim=2, control_dim=2), "chain 5P (s=2,T=3)", 50),
    ]

    for (builder, kwargs, label, n_runs) in component_configs
        prob = builder(; kwargs...)
        states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        using_timer = isdefined(MixedHierarchyGames, :TimerOutputs)

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            max_iters=100, tol=1e-6)

        # Warmup
        solve_raw(solver, states)

        # Collect iteration-level data via callback
        all_residuals = Vector{Float64}[]
        all_step_sizes = Vector{Float64}[]

        for _ in 1:n_runs
            residuals = Float64[]
            step_sizes = Float64[]
            callback = info -> begin
                push!(residuals, info.residual)
                push!(step_sizes, info.step_size)
            end
            solve_raw(solver, states; callback=callback)
            push!(all_residuals, residuals)
            push!(all_step_sizes, step_sizes)
        end

        println("\n  ── $label ──")
        n_iters = length(all_residuals[1])
        @printf("    Iterations: %d\n", n_iters)
        if n_iters > 0
            @printf("    Initial residual:  %.6e\n", all_residuals[1][1])
            @printf("    Final residual:    %.6e\n", all_residuals[1][end])
            @printf("    Convergence rate:  %.2fx per iter (geometric mean)\n",
                n_iters > 1 ? (all_residuals[1][end] / all_residuals[1][1])^(1/(n_iters-1)) : NaN)
            @printf("    Step sizes: %s\n",
                join([@sprintf("%.3f", s) for s in all_step_sizes[1]], ", "))
        end

        # Time individual components using separate timed calls
        # (1) K evaluation
        raw_result = solve_raw(solver, states)
        total_t = bench(() -> solve_raw(solver, states), 3, n_runs)

        # (2) Time with callback (measures callback overhead)
        dummy_cb = info -> nothing
        t_with_cb = bench(() -> solve_raw(solver, states; callback=dummy_cb), 3, n_runs)

        cb_overhead = (median(t_with_cb) / median(total_t) - 1) * 100
        @printf("    Solve time:        %s (median, n=%d)\n", fmt_time(median(total_t)), n_runs)
        @printf("    Callback overhead: %.1f%%\n", cb_overhead)
        @printf("    Time per iter:     %s\n", fmt_time(median(total_t) / max(n_iters, 1)))
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Allocation Profiling
# ─────────────────────────────────────────────────────────────────────────────

function run_allocation_benchmark()
    println("\n" * "=" ^ 80)
    println("  SECTION 4: Allocation Profiling")
    println("=" ^ 80)

    println("\n  ── solve_raw() allocations ──")
    @printf("  %-40s  %10s  %10s  %10s\n",
        "Problem", "median", "min", "max")
    println("  " * "─" ^ 75)

    for (builder, kwargs, label, solver_types, n_runs) in CONFIGS
        if :nl in solver_types
            prob = builder(; kwargs...)
            states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

            solver = NonlinearSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim;
                max_iters=100, tol=1e-6)

            # warmup
            solve_raw(solver, states)

            allocs = bench_alloc(() -> solve_raw(solver, states), 3, min(n_runs, 50))
            med = median(allocs)
            mn = minimum(allocs)
            mx = maximum(allocs)
            @printf("  NL:  %-36s  %s  %s  %s\n",
                label, fmt_alloc(med), fmt_alloc(mn), fmt_alloc(mx))
        end

        if :qp in solver_types
            prob = builder(; kwargs...)
            states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

            solver = QPSolver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                prob.state_dim, prob.control_dim)

            solve_raw(solver, states)

            allocs = bench_alloc(() -> solve_raw(solver, states), 3, min(n_runs, 50))
            med = median(allocs)
            mn = minimum(allocs)
            mx = maximum(allocs)
            @printf("  QP:  %-36s  %s  %s  %s\n",
                label, fmt_alloc(med), fmt_alloc(mn), fmt_alloc(mx))
        end
    end

    # Per-function allocation breakdown
    println("\n  ── Per-function allocations (3P chain, s=2, T=3) ──")
    prob = build_chain(N=3, T=3, state_dim=2, control_dim=2)
    states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)
    solver = NonlinearSolver(
        prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
        prob.state_dim, prob.control_dim;
        max_iters=100, tol=1e-6)

    # Warmup
    solve_raw(solver, states)
    solve(solver, states)

    a_raw = @allocated solve_raw(solver, states)
    a_full = @allocated solve(solver, states)
    a_extraction = a_full - a_raw

    @printf("    solve_raw():           %s\n", fmt_alloc(a_raw))
    @printf("    solve() total:         %s\n", fmt_alloc(a_full))
    @printf("    trajectory extraction: %s (%.1f%%)\n",
        fmt_alloc(a_extraction), a_extraction / a_full * 100)

    # With regularization
    solver_reg = NonlinearSolver(
        prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
        prob.state_dim, prob.control_dim;
        max_iters=100, tol=1e-6, regularization=1e-8)
    solve_raw(solver_reg, states)
    a_reg = @allocated solve_raw(solver_reg, states)
    @printf("    solve_raw(reg=1e-8):   %s (+%.1f%%)\n",
        fmt_alloc(a_reg), (a_reg / a_raw - 1) * 100)

    # With callback
    cb = info -> nothing
    solve_raw(solver, states; callback=cb)
    a_cb = @allocated solve_raw(solver, states; callback=cb)
    @printf("    solve_raw(callback):   %s (+%.1f%%)\n",
        fmt_alloc(a_cb), (a_cb / a_raw - 1) * 100)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Scaling Analysis
# ─────────────────────────────────────────────────────────────────────────────

function run_scaling_benchmark()
    println("\n" * "=" ^ 80)
    println("  SECTION 5: Scaling Analysis")
    println("=" ^ 80)

    # --- Scale with N (players) ---
    println("\n  ── Scale with N (chain, s=2, c=2, T=3) ──")
    @printf("  %-8s  %12s  %12s  %12s  %6s\n",
        "N", "construct", "solve_raw", "alloc", "iters")
    println("  " * "─" ^ 55)

    for N in [2, 3, 4, 5]
        prob = build_chain(N=N, T=3, state_dim=2, control_dim=2)
        states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        t_construct = bench(() -> NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim; max_iters=100, tol=1e-6
        ), 1, 3)

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim; max_iters=100, tol=1e-6)

        n_runs = N <= 3 ? 300 : 50
        t_solve = bench(() -> solve_raw(solver, states), 5, n_runs)
        solve_raw(solver, states)
        a = @allocated solve_raw(solver, states)
        raw = solve_raw(solver, states)

        @printf("  N=%-5d  %s  %s  %s  %6d\n",
            N, fmt_time(median(t_construct)), fmt_time(median(t_solve)), fmt_alloc(a), raw.iterations)
    end

    # --- Scale with T (time horizon) ---
    println("\n  ── Scale with T (chain 3P, s=2, c=2) ──")
    @printf("  %-8s  %12s  %12s  %12s  %6s\n",
        "T", "construct", "solve_raw", "alloc", "iters")
    println("  " * "─" ^ 55)

    for T in [3, 5, 8, 12]
        prob = build_chain(N=3, T=T, state_dim=2, control_dim=2)
        states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        t_construct = bench(() -> NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim; max_iters=100, tol=1e-6
        ), 1, 3)

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim; max_iters=100, tol=1e-6)

        n_runs = T <= 5 ? 200 : 50
        t_solve = bench(() -> solve_raw(solver, states), 5, n_runs)
        solve_raw(solver, states)
        a = @allocated solve_raw(solver, states)
        raw = solve_raw(solver, states)

        @printf("  T=%-5d  %s  %s  %s  %6d\n",
            T, fmt_time(median(t_construct)), fmt_time(median(t_solve)), fmt_alloc(a), raw.iterations)
    end

    # --- Scale with state_dim ---
    println("\n  ── Scale with state_dim (chain 3P, c=2, T=3) ──")
    @printf("  %-8s  %12s  %12s  %12s  %6s\n",
        "s", "construct", "solve_raw", "alloc", "iters")
    println("  " * "─" ^ 55)

    for s in [2, 4, 6, 8]
        prob = build_chain(N=3, T=3, state_dim=s, control_dim=2)
        states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        t_construct = bench(() -> NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim; max_iters=100, tol=1e-6
        ), 1, 3)

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim; max_iters=100, tol=1e-6)

        n_runs = s <= 4 ? 200 : 50
        t_solve = bench(() -> solve_raw(solver, states), 5, n_runs)
        solve_raw(solver, states)
        a = @allocated solve_raw(solver, states)
        raw = solve_raw(solver, states)

        @printf("  s=%-5d  %s  %s  %s  %6d\n",
            s, fmt_time(median(t_construct)), fmt_time(median(t_solve)), fmt_alloc(a), raw.iterations)
    end

    # --- Chain vs Star vs Nash (same N) ---
    println("\n  ── Topology comparison (N=3, s=2, c=2, T=3) ──")
    @printf("  %-10s  %12s  %12s  %12s  %6s\n",
        "Topology", "construct", "solve_raw", "alloc", "iters")
    println("  " * "─" ^ 55)

    for (builder, topo_name) in [(build_chain, "chain"), (build_star, "star"), (build_nash, "nash")]
        prob = builder(N=3, T=3, state_dim=2, control_dim=2)
        states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        t_construct = bench(() -> NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim; max_iters=100, tol=1e-6
        ), 1, 3)

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim; max_iters=100, tol=1e-6)

        t_solve = bench(() -> solve_raw(solver, states), 5, 300)
        solve_raw(solver, states)
        a = @allocated solve_raw(solver, states)
        raw = solve_raw(solver, states)

        @printf("  %-8s  %s  %s  %s  %6d\n",
            topo_name, fmt_time(median(t_construct)), fmt_time(median(t_solve)), fmt_alloc(a), raw.iterations)
    end

    # --- QP vs NL on same problem ---
    println("\n  ── QPSolver vs NonlinearSolver (same problem, chain) ──")
    @printf("  %-20s  %12s  %12s  %12s  %6s\n",
        "Solver / Problem", "construct", "solve_raw", "alloc", "iters")
    println("  " * "─" ^ 65)

    for N in [2, 3]
        prob = build_chain(N=N, T=3, state_dim=2, control_dim=2)
        states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        # QP
        t_qp_c = bench(() -> QPSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim), 1, 3)
        qp_solver = QPSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim)
        t_qp_s = bench(() -> solve_raw(qp_solver, states), 5, 300)
        solve_raw(qp_solver, states)
        a_qp = @allocated solve_raw(qp_solver, states)
        raw_qp = solve_raw(qp_solver, states)

        @printf("  QP  %dP chain        %s  %s  %s  %6d\n",
            N, fmt_time(median(t_qp_c)), fmt_time(median(t_qp_s)), fmt_alloc(a_qp), raw_qp.iterations)

        # NL
        t_nl_c = bench(() -> NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim; max_iters=100, tol=1e-6), 1, 3)
        nl_solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim; max_iters=100, tol=1e-6)
        t_nl_s = bench(() -> solve_raw(nl_solver, states), 5, 300)
        solve_raw(nl_solver, states)
        a_nl = @allocated solve_raw(nl_solver, states)
        raw_nl = solve_raw(nl_solver, states)

        @printf("  NL  %dP chain        %s  %s  %s  %6d\n",
            N, fmt_time(median(t_nl_c)), fmt_time(median(t_nl_s)), fmt_alloc(a_nl), raw_nl.iterations)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    section = length(ARGS) >= 1 ? ARGS[1] : "all"

    branch = try
        strip(read(`git branch --show-current`, String))
    catch
        "unknown"
    end

    println("=" ^ 80)
    println("  COMPREHENSIVE PERFORMANCE AUDIT")
    println("  Branch: $branch")
    println("  Julia:  $(VERSION)")
    println("  Time:   $(Dates.format(Dates.now(), "yyyy-mm-dd HH:MM:SS"))")
    println("=" ^ 80)

    using Dates

    if section in ["all", "construction"]
        run_construction_benchmark()
    end
    if section in ["all", "solve"]
        run_solve_benchmark()
    end
    if section in ["all", "components"]
        run_component_benchmark()
    end
    if section in ["all", "allocations"]
        run_allocation_benchmark()
    end
    if section in ["all", "scaling"]
        run_scaling_benchmark()
    end

    println("\n" * "=" ^ 80)
    println("  AUDIT COMPLETE")
    println("=" ^ 80)
end

# Need Dates for timestamp
using Dates

main()
