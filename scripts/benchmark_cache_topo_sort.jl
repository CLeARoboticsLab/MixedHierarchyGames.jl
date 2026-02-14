#=
    Benchmark: Cache topological sort (Perf T3-4)

    Measures the cost of topological_sort_by_dfs on hierarchy graphs of
    various sizes, demonstrating that caching eliminates redundant calls
    in compute_K_evals (called once per Newton iteration).

    Run: julia --project=. scripts/benchmark_cache_topo_sort.jl
=#

using Graphs: SimpleDiGraph, add_edge!, topological_sort_by_dfs
using Statistics: median
using Printf

function build_chain_graph(n)
    G = SimpleDiGraph(n)
    for i in 1:(n-1)
        add_edge!(G, i, i + 1)
    end
    G
end

function build_star_graph(n)
    G = SimpleDiGraph(n)
    for i in 2:n
        add_edge!(G, 1, i)
    end
    G
end

function bench_topo_sort(G, n_warmup, n_runs)
    for _ in 1:n_warmup
        reverse(topological_sort_by_dfs(G))
    end
    times_ns = Float64[]
    for _ in 1:n_runs
        t = @elapsed reverse(topological_sort_by_dfs(G))
        push!(times_ns, t * 1e9)
    end
    median(times_ns)
end

function main()
    println("=" ^ 60)
    println("Benchmark: Cache topological sort (Perf T3-4)")
    println("=" ^ 60)
    println()

    sizes = [3, 5, 10, 20, 50]
    n_warmup = 1000
    n_runs = 10_000

    println("Chain graphs (P1 -> P2 -> ... -> Pn):")
    println("-" ^ 50)
    @printf("  %-8s  %-15s  %-20s\n", "N", "topo_sort (ns)", "per-iter savings")
    for n in sizes
        G = build_chain_graph(n)
        t_ns = bench_topo_sort(G, n_warmup, n_runs)
        # compute_K_evals is called once per Newton iteration (typ. 5-20 iters)
        savings_10 = t_ns * 10
        @printf("  %-8d  %-15.1f  %-20s\n", n, t_ns,
                @sprintf("%.1f ns (10 iters)", savings_10))
    end

    println()
    println("Star graphs (P1 -> {P2, ..., Pn}):")
    println("-" ^ 50)
    @printf("  %-8s  %-15s  %-20s\n", "N", "topo_sort (ns)", "per-iter savings")
    for n in sizes
        G = build_star_graph(n)
        t_ns = bench_topo_sort(G, n_warmup, n_runs)
        savings_10 = t_ns * 10
        @printf("  %-8d  %-15.1f  %-20s\n", n, t_ns,
                @sprintf("%.1f ns (10 iters)", savings_10))
    end

    println()
    println("Summary:")
    println("  - topological_sort_by_dfs cost: O(V+E), ~100-1000 ns for typical games")
    println("  - Previously called once in setup + once per Newton iteration")
    println("  - Now computed once in setup and cached in setup_info NamedTuple")
    println("  - Impact: negligible for typical 3-5 player games, measurable for scaling")
end

main()
