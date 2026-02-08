using Test
using Graphs: SimpleDiGraph, add_edge!
using MixedHierarchyGames: is_root, is_leaf, get_roots, get_all_leaders, get_all_followers,
    ordered_player_indices

@testset "Graph Utilities" begin
    @testset "Chain hierarchy: 1→2→3" begin
        G = SimpleDiGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)

        # Root detection
        @test is_root(G, 1) == true
        @test is_root(G, 2) == false
        @test is_root(G, 3) == false
        @test get_roots(G) == [1]

        # Leaf detection
        @test is_leaf(G, 1) == false
        @test is_leaf(G, 2) == false
        @test is_leaf(G, 3) == true

        # Leaders (ancestors)
        @test get_all_leaders(G, 1) == []
        @test get_all_leaders(G, 2) == [1]
        @test Set(get_all_leaders(G, 3)) == Set([2, 1])

        # Followers (descendants)
        @test Set(get_all_followers(G, 1)) == Set([2, 3])
        @test get_all_followers(G, 2) == [3]
        @test get_all_followers(G, 3) == []
    end

    @testset "Tree hierarchy: 1→{2,3}" begin
        G = SimpleDiGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 1, 3)

        # Root detection
        @test is_root(G, 1) == true
        @test is_root(G, 2) == false
        @test is_root(G, 3) == false
        @test get_roots(G) == [1]

        # Leaf detection
        @test is_leaf(G, 1) == false
        @test is_leaf(G, 2) == true
        @test is_leaf(G, 3) == true

        # Leaders
        @test get_all_leaders(G, 1) == []
        @test get_all_leaders(G, 2) == [1]
        @test get_all_leaders(G, 3) == [1]

        # Followers
        @test Set(get_all_followers(G, 1)) == Set([2, 3])
        @test get_all_followers(G, 2) == []
        @test get_all_followers(G, 3) == []
    end

    @testset "Mixed hierarchy: P1 Nash with P2→P3 Stackelberg" begin
        # P1 and P2 are Nash (no edges between them)
        # P2 leads P3
        G = SimpleDiGraph(3)
        add_edge!(G, 2, 3)

        # Roots (both P1 and P2 are roots)
        @test is_root(G, 1) == true
        @test is_root(G, 2) == true
        @test is_root(G, 3) == false
        @test Set(get_roots(G)) == Set([1, 2])

        # Leaves
        @test is_leaf(G, 1) == true  # P1 has no followers
        @test is_leaf(G, 2) == false
        @test is_leaf(G, 3) == true

        # Leaders/Followers
        @test get_all_leaders(G, 3) == [2]
        @test get_all_followers(G, 2) == [3]
        @test get_all_followers(G, 1) == []
    end

    @testset "ordered_player_indices" begin
        @testset "returns sorted keys from Dict" begin
            d = Dict(3 => "c", 1 => "a", 2 => "b")
            @test ordered_player_indices(d) == [1, 2, 3]
        end

        @testset "single-element Dict" begin
            d = Dict(5 => [1.0, 2.0])
            @test ordered_player_indices(d) == [5]
        end

        @testset "empty Dict" begin
            d = Dict{Int, Any}()
            @test ordered_player_indices(d) == Int[]
        end

        @testset "already sorted keys" begin
            d = Dict(1 => :a, 2 => :b, 3 => :c)
            @test ordered_player_indices(d) == [1, 2, 3]
        end

        @testset "non-contiguous keys" begin
            d = Dict(10 => "x", 2 => "y", 7 => "z")
            @test ordered_player_indices(d) == [2, 7, 10]
        end
    end

    @testset "Diamond hierarchy: 1→{2,3}→4" begin
        G = SimpleDiGraph(4)
        add_edge!(G, 1, 2)
        add_edge!(G, 1, 3)
        add_edge!(G, 2, 4)
        add_edge!(G, 3, 4)

        # Root and leaf
        @test get_roots(G) == [1]
        @test is_leaf(G, 4) == true
        @test is_leaf(G, 2) == false
        @test is_leaf(G, 3) == false

        # P4 has two leaders (P2 and P3), and transitively P1
        leaders_of_4 = get_all_leaders(G, 4)
        @test 2 in leaders_of_4 || 3 in leaders_of_4  # At least one direct leader
        @test 1 in leaders_of_4  # Transitive leader

        # P1 has all others as followers
        @test Set(get_all_followers(G, 1)) == Set([2, 3, 4])
    end
end
