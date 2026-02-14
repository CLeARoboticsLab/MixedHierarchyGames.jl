#=
    Benchmark: Extractor Matrix Caching in get_qp_kkt_conditions

    Measures the time to build KKT conditions for 2-player and 3-player games.
    The cache eliminates redundant _build_extractor calls (one per follower
    per leader, saving ~50% of extractor construction).

    Run:  julia --project=. scripts/benchmark_extractor_cache.jl
=#

using MixedHierarchyGames
using MixedHierarchyGames: get_qp_kkt_conditions, setup_problem_variables, _build_extractor
using Graphs: SimpleDiGraph, add_edge!
using Statistics: median, mean, std
using Printf

function bench(f, n_warmup, n_runs)
    for _ in 1:n_warmup; f(); end
    times = Float64[]
    for _ in 1:n_runs
        t = @elapsed f()
        push!(times, t)
    end
    return times
end

function fmt_time(seconds)
    if seconds < 1e-3
        @sprintf("%8.1fμs", seconds * 1e6)
    elseif seconds < 1.0
        @sprintf("%8.2fms", seconds * 1e3)
    else
        @sprintf("%8.3fs", seconds)
    end
end

# ─────────────────────────────────────────────────────────────────────────────
# Setup: 2-player Stackelberg
# ─────────────────────────────────────────────────────────────────────────────
function setup_2player()
    G = SimpleDiGraph(2)
    add_edge!(G, 1, 2)
    primal_dims = [4, 4]
    gs = [z -> [z[1] + z[2] - 1.0, z[3] - z[4]], z -> [z[1] - z[2], z[3] + z[4] - 2.0]]
    vars = setup_problem_variables(G, primal_dims, gs)
    Js = Dict(
        1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + 0.5 * sum(vars.zs[2].^2),
        2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
    )
    return G, Js, vars, gs
end

# ─────────────────────────────────────────────────────────────────────────────
# Setup: 3-player chain (1→2→3)
# ─────────────────────────────────────────────────────────────────────────────
function setup_3player_chain()
    G = SimpleDiGraph(3)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)
    primal_dims = [4, 3, 3]
    gs = [
        z -> [z[1] - 1.0, z[2] + z[3]],
        z -> [z[1] + z[2], z[3] - 1.0],
        z -> [z[1] - z[2]],
    ]
    vars = setup_problem_variables(G, primal_dims, gs)
    Js = Dict(
        1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2),
        2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2) + sum(vars.zs[3].^2),
        3 => (zs...; θ=nothing) -> sum(vars.zs[3].^2),
    )
    return G, Js, vars, gs
end

# ─────────────────────────────────────────────────────────────────────────────
# Setup: 3-player star (1→2, 1→3)
# ─────────────────────────────────────────────────────────────────────────────
function setup_3player_star()
    G = SimpleDiGraph(3)
    add_edge!(G, 1, 2)
    add_edge!(G, 1, 3)
    primal_dims = [4, 3, 3]
    gs = [
        z -> [z[1] - 1.0, z[2] + z[3]],
        z -> [z[1] + z[2]],
        z -> [z[1] - z[2]],
    ]
    vars = setup_problem_variables(G, primal_dims, gs)
    Js = Dict(
        1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2) + sum(vars.zs[3].^2),
        2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        3 => (zs...; θ=nothing) -> sum(vars.zs[3].^2),
    )
    return G, Js, vars, gs
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: _build_extractor alone
# ─────────────────────────────────────────────────────────────────────────────
function bench_build_extractor()
    println("\n─── _build_extractor micro-benchmark ───")
    indices = 1:10
    total_len = 30
    times = bench(() -> _build_extractor(indices, total_len), 100, 1000)
    med = median(times)
    println("  _build_extractor(1:10, 30): $(fmt_time(med)) median ($(length(times)) runs)")
    return med
end

# ─────────────────────────────────────────────────────────────────────────────
# Benchmark: get_qp_kkt_conditions
# ─────────────────────────────────────────────────────────────────────────────
function bench_kkt_conditions(name, setup_fn; n_warmup=2, n_runs=5)
    G, Js, vars, gs = setup_fn()
    times = bench(
        () -> get_qp_kkt_conditions(
            G, Js, vars.zs, vars.λs, vars.μs, gs,
            vars.ws, vars.ys, vars.ws_z_indices
        ),
        n_warmup, n_runs
    )
    med = median(times)
    println(@sprintf("  %-30s %s median (%d runs, std=%s)", name, fmt_time(med), n_runs, fmt_time(std(times))))

    # Also measure allocations
    G2, Js2, vars2, gs2 = setup_fn()
    for _ in 1:2  # warmup
        get_qp_kkt_conditions(G2, Js2, vars2.zs, vars2.λs, vars2.μs, gs2, vars2.ws, vars2.ys, vars2.ws_z_indices)
    end
    alloc = @allocated get_qp_kkt_conditions(G2, Js2, vars2.zs, vars2.λs, vars2.μs, gs2, vars2.ws, vars2.ys, vars2.ws_z_indices)
    println(@sprintf("  %-30s %d bytes allocated", "", alloc))

    return med
end

function main()
    println("=" ^ 60)
    println("  Extractor Matrix Cache Benchmark")
    println("  $(Dates.now())")
    println("=" ^ 60)

    extractor_time = bench_build_extractor()

    println("\n─── get_qp_kkt_conditions (with extractor caching) ───")
    t2p = bench_kkt_conditions("2-player Stackelberg", setup_2player)
    t3c = bench_kkt_conditions("3-player chain (1→2→3)", setup_3player_chain)
    t3s = bench_kkt_conditions("3-player star (1→{2,3})", setup_3player_star)

    println("\n─── Summary ───")
    println("  Single _build_extractor call: $(fmt_time(extractor_time))")
    println("  2-player KKT construction:    $(fmt_time(t2p))")
    println("  3-player chain KKT:           $(fmt_time(t3c))")
    println("  3-player star KKT:            $(fmt_time(t3s))")
    println()
    println("  Cache saves 1 _build_extractor call per follower per leader iteration.")
    println("  For 3-player chain: 3 calls saved (P2 once for P1, P3 once for P1, P3 once for P2)")
    println("  For 3-player star:  2 calls saved (P2 once for P1, P3 once for P1)")
    println("=" ^ 60)
end

using Dates
main()
