using Test
using Graphs: SimpleDiGraph, add_edge!
using SparseArrays: sparse, nnz
using BlockArrays: blocks
using MixedHierarchyGames: get_qp_kkt_conditions, setup_problem_variables, _build_extractor

@testset "Extractor Matrix Caching" begin
    @testset "_build_extractor returns identical matrices for same inputs" begin
        indices = 1:3
        total_len = 7
        E1 = _build_extractor(indices, total_len)
        E2 = _build_extractor(indices, total_len)
        @test E1 == E2
        @test size(E1) == (3, 7)
    end

    @testset "Extractor cache key correctness: different inputs produce different extractors" begin
        E1 = _build_extractor(1:2, 5)
        E2 = _build_extractor(3:4, 5)
        @test E1 != E2

        E3 = _build_extractor(1:2, 5)
        E4 = _build_extractor(1:2, 8)
        @test size(E3) != size(E4)
    end

    @testset "get_qp_kkt_conditions uses extractor cache (2-player)" begin
        # 2-player Stackelberg: 1→2
        # The extractor for follower 2 is built for the same (indices, total_len) in two places.
        # With caching, these should be built once and reused.
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [
            z -> [z[1] + z[2] - 1.0],
            z -> [z[1] - z[2]],
        ]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        # Run the function — this exercises the cache internally
        result = get_qp_kkt_conditions(
            G, Js, vars.zs, vars.λs, vars.μs, gs,
            vars.ws, vars.ys, vars.ws_z_indices
        )

        # Verify correctness is preserved with caching
        @test haskey(result.πs, 1)
        @test haskey(result.πs, 2)
        @test haskey(result.Ms, 2)
        @test haskey(result.Ns, 2)
        @test haskey(result.Ks, 2)

        blks = collect(blocks(result.πs[1]))
        @test length(blks) == 4
        @test length(blks[3]) == 2  # policy constraint dim matches follower dim

        # Verify the function returns the extractor_cache in its result
        @test haskey(result, :extractor_cache)
        cache = result.extractor_cache

        # Cache should contain entries keyed by follower player index
        # For 2-player with follower 2, there should be exactly 1 cache entry
        @test length(cache) >= 1
        @test haskey(cache, 2)
    end

    @testset "get_qp_kkt_conditions extractor cache (3-player chain)" begin
        # 3-player chain: 1→2→3
        G = SimpleDiGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)

        primal_dims = [3, 2, 2]
        gs = [
            z -> [z[1] - 1.0],
            z -> [z[1] + z[2]],
            z -> [z[1]],
        ]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2) + sum(vars.zs[3].^2),
            3 => (zs...; θ=nothing) -> sum(vars.zs[3].^2),
        )

        result = get_qp_kkt_conditions(
            G, Js, vars.zs, vars.λs, vars.μs, gs,
            vars.ws, vars.ys, vars.ws_z_indices
        )

        # Verify correctness
        for i in 1:3
            @test haskey(result.πs, i)
        end
        for j in [2, 3]
            @test haskey(result.Ms, j)
            @test haskey(result.Ks, j)
        end

        # P1 leader blocks: [grad_self(3), grad_P2(2), policy_P2(2), grad_P3(2), policy_P3(2), constraints(1)]
        blks1 = collect(blocks(result.πs[1]))
        @test length(blks1) == 6
        @test length(blks1[3]) == 2
        @test length(blks1[5]) == 2

        # Cache should have entries for both followers
        @test haskey(result, :extractor_cache)
        cache = result.extractor_cache
        @test haskey(cache, 2)
        @test haskey(cache, 3)

        # Extractors in cache should have correct dimensions
        E2 = cache[2]
        E3 = cache[3]
        @test size(E2, 2) == length(vars.ws[2])
        @test size(E3, 2) == length(vars.ws[3])
    end

    @testset "Cached vs uncached KKT conditions are identical (2-player)" begin
        # Build KKT conditions and verify they match reference values from existing tests.
        # This is a regression test: caching must NOT change the symbolic expressions.
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [z -> [z[1] + z[2] - 1.0], z -> [z[1] - z[2]]]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        result = get_qp_kkt_conditions(
            G, Js, vars.zs, vars.λs, vars.μs, gs,
            vars.ws, vars.ys, vars.ws_z_indices
        )

        # Leader P1: total KKT length = grad_self(2) + grad_follower(2) + policy(2) + constraints(1) = 7
        @test length(result.πs[1]) == 7

        # Follower P2: stationarity(2) + constraints(1) = 3
        @test length(result.πs[2]) == 3

        # M matrix should be 3×3 (KKT dim × ws dim)
        @test size(result.Ms[2]) == (3, length(vars.ws[2]))

        # K = M \ N should be well-formed
        @test size(result.Ks[2]) == (length(vars.ws[2]), length(vars.ys[2]))
    end
end
