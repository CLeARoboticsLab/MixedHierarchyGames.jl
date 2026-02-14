#!/usr/bin/env julia
# Benchmark: Single vcat vs repeated vcat for ws construction
#
# Usage: julia --project=. scripts/benchmark_ws_vcat.jl

using Graphs
using Graphs: SimpleDiGraph, add_edge!
using Symbolics
using MixedHierarchyGames: setup_problem_variables, get_all_leaders, get_all_followers,
    default_backend, make_symbolic_vector

"""Simulate the OLD approach: repeated vcat in a loop."""
function ws_repeated_vcat(graph, zs, λs, μs, primal_dims)
    N = Graphs.nv(graph)
    ws = Dict{Int, Vector{Symbolics.Num}}()
    for i in 1:N
        leaders = get_all_leaders(graph, i)
        ws[i] = copy(zs[i])
        for j in 1:N
            if j != i && !(j in leaders)
                ws[i] = vcat(ws[i], zs[j])
            end
        end
        ws[i] = vcat(ws[i], λs[i])
        for j in get_all_followers(graph, i)
            ws[i] = vcat(ws[i], λs[j])
        end
        for j in get_all_followers(graph, i)
            if haskey(μs, (i, j))
                ws[i] = vcat(ws[i], μs[(i, j)])
            end
        end
    end
    return ws
end

"""Simulate the NEW approach: collect pieces, single reduce(vcat, ...)."""
function ws_single_vcat(graph, zs, λs, μs, primal_dims)
    N = Graphs.nv(graph)
    ws = Dict{Int, Vector{Symbolics.Num}}()
    for i in 1:N
        leaders = get_all_leaders(graph, i)
        pieces = Vector{Symbolics.Num}[]
        push!(pieces, zs[i])
        for j in 1:N
            if j != i && !(j in leaders)
                push!(pieces, zs[j])
            end
        end
        push!(pieces, λs[i])
        for j in get_all_followers(graph, i)
            push!(pieces, λs[j])
        end
        for j in get_all_followers(graph, i)
            if haskey(μs, (i, j))
                push!(pieces, μs[(i, j)])
            end
        end
        ws[i] = reduce(vcat, pieces)
    end
    return ws
end

function build_test_case(N, primal_dim, constraint_dim)
    # Build a chain graph: 1→2→3→...→N
    G = SimpleDiGraph(N)
    for i in 1:(N-1)
        add_edge!(G, i, i + 1)
    end

    backend = default_backend()
    primal_dims = fill(primal_dim, N)
    zs = Dict(i => make_symbolic_vector(:z, i, primal_dim; backend) for i in 1:N)
    λs = Dict(i => make_symbolic_vector(:λ, i, constraint_dim; backend) for i in 1:N)
    μs = Dict((i, j) => make_symbolic_vector(:μ, i, j, primal_dim; backend)
              for i in 1:N for j in get_all_followers(G, i))

    return G, zs, λs, μs, primal_dims
end

function run_benchmark()
    println("=" ^ 60)
    println("Benchmark: ws construction — repeated vcat vs single vcat")
    println("=" ^ 60)

    for (N, pdim, cdim) in [(3, 10, 5), (4, 20, 10), (5, 30, 15), (6, 40, 20)]
        G, zs, λs, μs, primal_dims = build_test_case(N, pdim, cdim)

        # Warmup
        ws_repeated_vcat(G, zs, λs, μs, primal_dims)
        ws_single_vcat(G, zs, λs, μs, primal_dims)

        # Benchmark
        n_iters = 100
        t_old = @elapsed for _ in 1:n_iters
            ws_repeated_vcat(G, zs, λs, μs, primal_dims)
        end
        t_new = @elapsed for _ in 1:n_iters
            ws_single_vcat(G, zs, λs, μs, primal_dims)
        end

        # Verify correctness
        ws_old = ws_repeated_vcat(G, zs, λs, μs, primal_dims)
        ws_new = ws_single_vcat(G, zs, λs, μs, primal_dims)
        for i in 1:N
            @assert isequal(ws_old[i], ws_new[i]) "Mismatch for player $i"
        end

        speedup = t_old / t_new
        μs_old = 1e6 * t_old / n_iters
        μs_new = 1e6 * t_new / n_iters
        println("\nN=$N, primal_dim=$pdim, constraint_dim=$cdim:")
        println("  Old (repeated vcat): $(round(μs_old, digits=1))μs")
        println("  New (single vcat):   $(round(μs_new, digits=1))μs")
        println("  Speedup: $(round(speedup, digits=2))x")
    end

    # Also benchmark allocation counts
    println("\n" * "=" ^ 60)
    println("Allocation comparison (N=5, pdim=30, cdim=15):")
    G, zs, λs, μs, primal_dims = build_test_case(5, 30, 15)

    allocs_old = @allocated ws_repeated_vcat(G, zs, λs, μs, primal_dims)
    allocs_new = @allocated ws_single_vcat(G, zs, λs, μs, primal_dims)

    println("  Old allocations: $(allocs_old) bytes")
    println("  New allocations: $(allocs_new) bytes")
    println("  Reduction: $(round(100 * (1 - allocs_new / allocs_old), digits=1))%")
    println("=" ^ 60)
end

run_benchmark()
