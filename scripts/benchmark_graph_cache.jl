#!/usr/bin/env julia
"""
Benchmark: Graph traversal caching in setup_problem_variables.

Compares setup time with and without caching get_all_followers/get_all_leaders.
The "without cache" baseline is simulated by calling the traversal functions
directly in the same pattern as the original code.
"""

using Graphs: SimpleDiGraph, add_edge!, nv
using MixedHierarchyGames: get_all_followers, get_all_leaders, _build_graph_caches

function build_chain_graph(N)
    G = SimpleDiGraph(N)
    for i in 1:(N-1)
        add_edge!(G, i, i + 1)
    end
    return G
end

function build_star_graph(N)
    G = SimpleDiGraph(N)
    for i in 2:N
        add_edge!(G, 1, i)
    end
    return G
end

"""Simulate the uncached pattern: call get_all_followers/get_all_leaders inline."""
function uncached_traversals(G)
    N = nv(G)
    # Pattern from original code: 5 traversal sites
    # Site 1: μs comprehension - followers for each i
    for i in 1:N
        get_all_followers(G, i)
    end
    # Site 2: ys loop - leaders for each i
    for i in 1:N
        get_all_leaders(G, i)
    end
    # Site 3: ws loop - leaders for each i (again)
    for i in 1:N
        get_all_leaders(G, i)
    end
    # Site 4: ws loop - followers for λs
    for i in 1:N
        get_all_followers(G, i)
    end
    # Site 5: ws loop - followers for μs
    for i in 1:N
        get_all_followers(G, i)
    end
end

"""Simulate the cached pattern: build cache once, do dict lookups."""
function cached_traversals(G)
    N = nv(G)
    followers_cache, leaders_cache = _build_graph_caches(G)
    # All 5 sites now use dict lookups
    for i in 1:N; followers_cache[i]; end
    for i in 1:N; leaders_cache[i]; end
    for i in 1:N; leaders_cache[i]; end
    for i in 1:N; followers_cache[i]; end
    for i in 1:N; followers_cache[i]; end
end

function benchmark_one(label, G, n_warmup=3, n_trials=50)
    # Warmup
    for _ in 1:n_warmup
        uncached_traversals(G)
        cached_traversals(G)
    end

    # Benchmark uncached
    t_uncached = zeros(n_trials)
    for k in 1:n_trials
        t_uncached[k] = @elapsed uncached_traversals(G)
    end

    # Benchmark cached
    t_cached = zeros(n_trials)
    for k in 1:n_trials
        t_cached[k] = @elapsed cached_traversals(G)
    end

    median_uncached = sort(t_uncached)[n_trials ÷ 2] * 1e6  # μs
    median_cached = sort(t_cached)[n_trials ÷ 2] * 1e6
    speedup = median_uncached / median_cached

    println("  $label (N=$(nv(G))): uncached=$(round(median_uncached, digits=1))μs, " *
            "cached=$(round(median_cached, digits=1))μs, speedup=$(round(speedup, digits=2))x")
    return (; label, N=nv(G), median_uncached, median_cached, speedup)
end

println("=" ^ 70)
println("Graph Traversal Cache Benchmark")
println("=" ^ 70)

results = []
for N in [3, 5, 8, 10, 15, 20]
    println("\nN = $N players:")
    push!(results, benchmark_one("chain", build_chain_graph(N)))
    push!(results, benchmark_one("star ", build_star_graph(N)))
end

println("\n" * "=" ^ 70)
println("Summary: Speedup by graph size")
println("-" ^ 70)
println("  N  | Chain Speedup | Star Speedup")
println("-" ^ 70)
for N in [3, 5, 8, 10, 15, 20]
    chain = filter(r -> r.N == N && startswith(r.label, "chain"), results)
    star = filter(r -> r.N == N && startswith(r.label, "star"), results)
    if !isempty(chain) && !isempty(star)
        println("  $(lpad(N, 2)) |    $(lpad(round(chain[1].speedup, digits=2), 6))x   |   $(lpad(round(star[1].speedup, digits=2), 6))x")
    end
end
println("=" ^ 70)
