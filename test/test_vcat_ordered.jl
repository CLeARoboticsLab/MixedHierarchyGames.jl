using Test
using MixedHierarchyGames

@testset "vcat_ordered! - in-place ordered concatenation" begin
    @testset "basic correctness matches reduce(vcat)" begin
        # Two players with different-length parameter vectors
        d = Dict(1 => [1.0, 2.0, 3.0], 2 => [4.0, 5.0])
        order = MixedHierarchyGames.ordered_player_indices(d)
        expected = reduce(vcat, (d[k] for k in order))

        total_len = sum(length(d[k]) for k in order)
        buf = Vector{Float64}(undef, total_len)
        MixedHierarchyGames.vcat_ordered!(buf, d, order)

        @test buf == expected
        @test buf == [1.0, 2.0, 3.0, 4.0, 5.0]
    end

    @testset "three players, non-sequential keys" begin
        d = Dict(3 => [9.0], 1 => [1.0, 2.0], 5 => [5.0, 6.0, 7.0])
        order = MixedHierarchyGames.ordered_player_indices(d)
        expected = reduce(vcat, (d[k] for k in order))

        total_len = sum(length(d[k]) for k in order)
        buf = Vector{Float64}(undef, total_len)
        MixedHierarchyGames.vcat_ordered!(buf, d, order)

        @test buf == expected
        # order is [1, 3, 5], so result is [1.0, 2.0, 9.0, 5.0, 6.0, 7.0]
        @test buf == [1.0, 2.0, 9.0, 5.0, 6.0, 7.0]
    end

    @testset "single player" begin
        d = Dict(1 => [10.0, 20.0, 30.0])
        order = MixedHierarchyGames.ordered_player_indices(d)
        expected = reduce(vcat, (d[k] for k in order))

        buf = Vector{Float64}(undef, 3)
        MixedHierarchyGames.vcat_ordered!(buf, d, order)

        @test buf == expected
    end

    @testset "overwrites previous buffer contents" begin
        d = Dict(1 => [1.0, 2.0], 2 => [3.0])
        order = MixedHierarchyGames.ordered_player_indices(d)

        buf = fill(999.0, 3)
        MixedHierarchyGames.vcat_ordered!(buf, d, order)

        @test buf == [1.0, 2.0, 3.0]
    end

    @testset "returns the buffer" begin
        d = Dict(1 => [1.0])
        order = MixedHierarchyGames.ordered_player_indices(d)
        buf = Vector{Float64}(undef, 1)
        result = MixedHierarchyGames.vcat_ordered!(buf, d, order)
        @test result === buf
    end

    @testset "zero allocations on repeated calls" begin
        d = Dict(1 => [1.0, 2.0, 3.0], 2 => [4.0, 5.0])
        order = MixedHierarchyGames.ordered_player_indices(d)
        total_len = sum(length(d[k]) for k in order)
        buf = Vector{Float64}(undef, total_len)

        # Warm up
        MixedHierarchyGames.vcat_ordered!(buf, d, order)

        # Measure allocations
        allocs = @allocated MixedHierarchyGames.vcat_ordered!(buf, d, order)
        @test allocs == 0
    end
end
