using Test
using Symbolics: Num  # Only for type checking
using MixedHierarchyGames: make_symbolic_vector, make_symbolic_matrix, make_symbol
using MixedHierarchyGames: PLAYER_SYMBOLS, PAIR_SYMBOLS

@testset "Symbolic Variable Utilities" begin
    @testset "make_symbol validation" begin
        # Player symbols work with 2 args
        @test make_symbol(:z, 1) == Symbol("z^1")
        @test make_symbol(:θ, 2) == Symbol("θ^2")
        @test make_symbol(:λ, 3) == Symbol("λ^3")

        # Pair symbols work with 3 args
        @test make_symbol(:μ, 1, 2) == Symbol("μ^(1-2)")

        # Player symbol with 3 args throws
        @test_throws ArgumentError make_symbol(:z, 1, 2)

        # Pair symbol with 2 args throws
        @test_throws ArgumentError make_symbol(:μ, 1)

        # Unknown symbol throws
        @test_throws ArgumentError make_symbol(:unknown, 1)
    end

    @testset "make_symbolic_vector for player variables" begin
        for name in PLAYER_SYMBOLS
            vec = make_symbolic_vector(name, 1, 4)
            @test length(vec) == 4
            @test all(x -> x isa Num, vec)
            @test occursin(string(name), string(vec[1]))
        end
    end

    @testset "make_symbolic_vector for pair variables" begin
        for name in PAIR_SYMBOLS
            vec = make_symbolic_vector(name, 1, 2, 3)
            @test length(vec) == 3
            @test all(x -> x isa Num, vec)
            @test occursin("$(name)^(1-2)", string(vec[1]))
        end
    end

    @testset "make_symbolic_vector validation" begin
        # Player symbol with pair signature throws
        @test_throws ArgumentError make_symbolic_vector(:z, 1, 2, 3)

        # Pair symbol with player signature throws
        @test_throws ArgumentError make_symbolic_vector(:μ, 1, 3)
    end

    @testset "make_symbolic_matrix dimensions" begin
        mat = make_symbolic_matrix(:K, 1, 2, 3)
        @test size(mat) == (2, 3)
        @test all(x -> x isa Num, mat)
        @test occursin("K^1", string(mat[1, 1]))
    end
end
