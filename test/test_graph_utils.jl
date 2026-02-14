using Test
using Graphs: SimpleDiGraph, add_edge!
using MixedHierarchyGames: is_root, is_leaf, has_leader, get_roots, get_all_leaders, get_all_followers

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

        # Has leader detection
        @test has_leader(G, 1) == false
        @test has_leader(G, 2) == true
        @test has_leader(G, 3) == true

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

        # Has leader detection
        @test has_leader(G, 1) == false
        @test has_leader(G, 2) == true
        @test has_leader(G, 3) == true

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

    @testset "has_leader consistency with is_root" begin
        # has_leader should always be the negation of is_root
        G = SimpleDiGraph(4)
        add_edge!(G, 1, 2)
        add_edge!(G, 1, 3)
        add_edge!(G, 2, 4)
        add_edge!(G, 3, 4)

        for v in 1:4
            @test has_leader(G, v) == !is_root(G, v)
        end

        # Return types must be Bool
        @test is_root(G, 1) isa Bool
        @test is_leaf(G, 1) isa Bool
        @test has_leader(G, 1) isa Bool
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
