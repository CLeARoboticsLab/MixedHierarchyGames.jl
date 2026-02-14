#!/usr/bin/env julia
#
# Benchmark: BFS collect optimization
# Compares collect(BFSIterator(g, v))[2:end] vs manual iteration
# Run with: julia --project=. scripts/benchmark_bfs_collect.jl

using Graphs

println("=" ^ 60)
println("Benchmark: BFS follower list collection")
println("=" ^ 60)

# Old approach: collect then slice
function bfs_followers_old(g, root)
    return collect(BFSIterator(g, root))[2:end]
end

# New approach: manual iteration, skip self
function bfs_followers_new(g, root)
    result = Int[]
    for v in BFSIterator(g, root)
        v == root && continue
        push!(result, v)
    end
    return result
end

# Build test graphs
function make_chain(n)
    g = SimpleDiGraph(n)
    for i in 1:(n-1)
        add_edge!(g, i, i+1)
    end
    return g
end

function bench(f, args...; warmup=100, trials=10_000)
    # Warmup
    for _ in 1:warmup
        f(args...)
    end
    # Measure
    times = Float64[]
    allocs = Int[]
    for _ in 1:trials
        t0 = time_ns()
        a = @allocated f(args...)
        t1 = time_ns()
        push!(times, Float64(t1 - t0))
        push!(allocs, a)
    end
    med_time = sort(times)[length(times) ÷ 2]
    med_alloc = sort(allocs)[length(allocs) ÷ 2]
    return (; time_ns=med_time, alloc_bytes=med_alloc)
end

println("\n--- Chain graphs ---")
for n in [3, 5, 10, 20]
    g = make_chain(n)
    root = 1

    # Verify equivalence
    old = bfs_followers_old(g, root)
    new = bfs_followers_new(g, root)
    @assert old == new "Results differ for chain($n)!"

    b_old = bench(bfs_followers_old, g, root)
    b_new = bench(bfs_followers_new, g, root)

    speedup = b_old.time_ns / max(b_new.time_ns, 1)
    alloc_ratio = b_old.alloc_bytes / max(b_new.alloc_bytes, 1)

    println("Chain($n): old=$(round(b_old.time_ns, digits=0))ns/$(b_old.alloc_bytes)B  " *
            "new=$(round(b_new.time_ns, digits=0))ns/$(b_new.alloc_bytes)B  " *
            "speedup=$(round(speedup, digits=2))x  alloc_ratio=$(round(alloc_ratio, digits=2))x")
end

println("\n--- Type stability bonus ---")
g = make_chain(5)
old_result = bfs_followers_old(g, 1)
new_result = bfs_followers_new(g, 1)
println("Old type: $(typeof(old_result))  (element type: $(eltype(old_result)))")
println("New type: $(typeof(new_result))  (element type: $(eltype(new_result)))")

println("\n--- Summary ---")
println("The manual iteration approach:")
println("  1. Avoids intermediate full-array allocation from collect()")
println("  2. Avoids array slicing [2:end] copy")
println("  3. Produces Vector{Int} instead of Vector{Any} (type-stable)")
println("  4. Impact is per-player, first-iteration only (cached thereafter)")
