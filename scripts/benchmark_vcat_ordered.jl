#=
    Benchmark: reduce(vcat) vs vcat_ordered! (copyto!)

    Measures:
      1. Micro-benchmark: isolated reduce(vcat) vs vcat_ordered!
      2. End-to-end: QP solve with LQ 3-player chain experiment

    Run:  julia --project=. scripts/benchmark_vcat_ordered.jl
=#

using MixedHierarchyGames
using TrajectoryGamesBase: unflatten_trajectory
using Graphs: SimpleDiGraph, add_edge!
using Statistics: median, std
using Printf

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
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

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: Micro-benchmark — reduce(vcat) vs vcat_ordered!
# ─────────────────────────────────────────────────────────────────────────────

println("=" ^ 72)
println("  Micro-benchmark: reduce(vcat) vs vcat_ordered!")
println("=" ^ 72)

for (label, sizes) in [
    ("2 players, 4-elem each", Dict(1 => 4, 2 => 4)),
    ("3 players, 8-elem each", Dict(1 => 8, 2 => 8, 3 => 8)),
    ("3 players, mixed sizes", Dict(1 => 2, 2 => 8, 3 => 4)),
    ("5 players, 16-elem each", Dict(1 => 16, 2 => 16, 3 => 16, 4 => 16, 5 => 16)),
]
    d = Dict(k => randn(n) for (k, n) in sizes)
    order = MixedHierarchyGames.ordered_player_indices(d)
    total_len = sum(length(d[k]) for k in order)
    buf = Vector{Float64}(undef, total_len)

    # Baseline: reduce(vcat)
    baseline_f() = reduce(vcat, (d[k] for k in order))
    # New: vcat_ordered!
    new_f() = MixedHierarchyGames.vcat_ordered!(buf, d, order)

    # Verify identical output
    expected = baseline_f()
    new_f()
    @assert buf == expected "Output mismatch!"

    n_warmup = 1000
    n_runs = 5000

    t_old = bench(baseline_f, n_warmup, n_runs)
    t_new = bench(new_f, n_warmup, n_runs)
    a_old = bench_alloc(baseline_f, 100, 500)
    a_new = bench_alloc(new_f, 100, 500)

    med_old = median(t_old)
    med_new = median(t_new)
    speedup = med_old / med_new
    alloc_old = median(a_old)
    alloc_new = median(a_new)

    println("\n  $label (total_len=$total_len)")
    @printf("    reduce(vcat):   %s  alloc: %s\n", fmt_time(med_old), fmt_alloc(Int(alloc_old)))
    @printf("    vcat_ordered!:  %s  alloc: %s\n", fmt_time(med_new), fmt_alloc(Int(alloc_new)))
    @printf("    Speedup: %.1fx   Alloc reduction: %.0f%%\n",
        speedup, (1 - alloc_new / alloc_old) * 100)
end

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: End-to-end QP solve (LQ 3-player chain)
# ─────────────────────────────────────────────────────────────────────────────

println("\n" * "=" ^ 72)
println("  End-to-end: QP solve — LQ 3-player chain")
println("=" ^ 72)

module ExLQ
    using MixedHierarchyGames
    using TrajectoryGamesBase: unflatten_trajectory
    using Graphs: SimpleDiGraph, add_edge!

    include(joinpath(@__DIR__, "..", "experiments", "common", "dynamics.jl"))
    include(joinpath(@__DIR__, "..", "experiments", "lq_three_player_chain", "config.jl"))
    include(joinpath(@__DIR__, "..", "experiments", "lq_three_player_chain", "support.jl"))

    function setup(; T=DEFAULT_T, Δt=DEFAULT_DT)
        G = build_hierarchy()
        Js = make_cost_functions(STATE_DIM, CONTROL_DIM)
        primal_dim = (STATE_DIM + CONTROL_DIM) * (T + 1)
        primal_dims = fill(primal_dim, N)
        θs = setup_problem_parameter_variables(fill(STATE_DIM, N))

        function _make_constraints(i)
            return function (zᵢ)
                dyn = mapreduce(vcat, 1:T) do t
                    single_integrator_2d(zᵢ, t; Δt, state_dim=STATE_DIM, control_dim=CONTROL_DIM)
                end
                (; xs,) = unflatten_trajectory(zᵢ, STATE_DIM, CONTROL_DIM)
                ic = xs[1] - θs[i]
                vcat(dyn, ic)
            end
        end
        gs = [_make_constraints(i) for i in 1:N]
        params = Dict(i => DEFAULT_X0[i] for i in 1:N)

        return (; G, Js, gs, primal_dims, θs, params,
                  N, T, state_dim=STATE_DIM, control_dim=CONTROL_DIM)
    end
end

exp = ExLQ.setup()

# Build the QP solver (one-time)
qp_solver = QPSolver(exp.G, exp.Js, exp.gs, exp.primal_dims, exp.θs,
                      exp.state_dim, exp.control_dim)

# QP solve benchmark
solve_f() = solve(qp_solver, exp.params)

# Warmup
for _ in 1:5; solve_f(); end

# Time
n_warmup = 50
n_runs = 500

t_solve = bench(solve_f, n_warmup, n_runs)
a_solve = bench_alloc(solve_f, 10, 100)

med_solve = median(t_solve)
min_solve = minimum(t_solve)
max_solve = maximum(t_solve)
σ_solve = std(t_solve)
med_alloc = median(a_solve)

println("\n  QP Solve (vcat_ordered! in place):")
@printf("    Median: %s   Min: %s   Max: %s   σ: %s\n",
    fmt_time(med_solve), fmt_time(min_solve), fmt_time(max_solve), fmt_time(σ_solve))
@printf("    Allocations (median): %s\n", fmt_alloc(Int(med_alloc)))

println("\n  Note: Compare these end-to-end results against the baseline from")
println("  the perf audit benchmark (before this optimization).")
println("  Expected impact: 1-3%% improvement on per-solve time.")
println()
