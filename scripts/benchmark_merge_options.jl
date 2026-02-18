#!/usr/bin/env julia
# Benchmark script for _merge_options short-circuit optimization
# Usage: julia --project=. scripts/benchmark_merge_options.jl

using MixedHierarchyGames

opts = NonlinearSolverOptions(max_iters=42, tol=1e-8)

# Benchmark functions to prevent dead-code elimination
function bench_no_overrides(opts, N)
    s = 0
    for _ in 1:N
        r = MixedHierarchyGames._merge_options(opts)
        s += r.max_iters
    end
    return s
end

function bench_all_nothing(opts, N)
    s = 0
    for _ in 1:N
        r = MixedHierarchyGames._merge_options(opts;
            max_iters=nothing, tol=nothing, verbose=nothing,
            linesearch_method=nothing, recompute_policy_in_linesearch=nothing,
            use_sparse=nothing, show_progress=nothing, regularization=nothing)
        s += r.max_iters
    end
    return s
end

function bench_with_override(opts, N)
    s = 0
    for _ in 1:N
        r = MixedHierarchyGames._merge_options(opts; max_iters=42)
        s += r.max_iters
    end
    return s
end

N = 1_000_000

# Warmup
bench_no_overrides(opts, 1000)
bench_all_nothing(opts, 1000)
bench_with_override(opts, 1000)

# Benchmark
t_none = @elapsed bench_no_overrides(opts, N)
t_all_nothing = @elapsed bench_all_nothing(opts, N)
t_override = @elapsed bench_with_override(opts, N)

println("_merge_options benchmark (N=$N calls)")
println("=" ^ 50)
println("No overrides (short-circuit): $(round(t_none*1e9/N, digits=1)) ns/call")
println("All-nothing kwargs (short-circuit): $(round(t_all_nothing*1e9/N, digits=1)) ns/call")
println("With override (constructor):  $(round(t_override*1e9/N, digits=1)) ns/call")
println()
println("Speedup vs constructor (no overrides): $(round(t_override/t_none, digits=1))x")
println("Speedup vs constructor (all nothing):  $(round(t_override/t_all_nothing, digits=1))x")
