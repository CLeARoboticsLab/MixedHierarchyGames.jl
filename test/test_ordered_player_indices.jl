using Test
using MixedHierarchyGames: ordered_player_indices

@testset "ordered_player_indices" begin
    @testset "returns sorted keys from Dict{Int}" begin
        d = Dict(3 => "c", 1 => "a", 2 => "b")
        @test ordered_player_indices(d) == [1, 2, 3]
    end

    @testset "single-element dict" begin
        d = Dict(5 => [1.0, 2.0])
        @test ordered_player_indices(d) == [5]
    end

    @testset "empty dict" begin
        d = Dict{Int, Vector{Float64}}()
        @test ordered_player_indices(d) == Int[]
    end

    @testset "already-sorted keys" begin
        d = Dict(1 => :a, 2 => :b, 3 => :c)
        @test ordered_player_indices(d) == [1, 2, 3]
    end

    @testset "non-contiguous keys" begin
        d = Dict(10 => "x", 2 => "y", 7 => "z")
        @test ordered_player_indices(d) == [2, 7, 10]
    end
end
