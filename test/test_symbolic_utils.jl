using Test
using Symbolics
using MixedHierarchyGames: make_symbolic_variable, make_symbolic_vector, make_symbolic_matrix

@testset "Symbolic Variable Utilities" begin
    @testset "make_symbolic_variable naming" begin
        # Single index
        v1 = make_symbolic_variable(:z, 1)
        @test string(v1) == "z_1"

        # Multiple indices
        v2 = make_symbolic_variable(:z, 1, 2, 3)
        @test string(v2) == "z_1_2_3"

        # Different variable types
        @test string(make_symbolic_variable(:θ, 2)) == "θ_2"
        @test string(make_symbolic_variable(:λ, 1, 5)) == "λ_1_5"
        @test string(make_symbolic_variable(:u, 3, 2, 1)) == "u_3_2_1"
    end

    @testset "make_symbolic_vector dimensions" begin
        vec = make_symbolic_vector(:x, 1, 4)
        @test length(vec) == 4
        @test string(vec[1]) == "x_1_1"
        @test string(vec[4]) == "x_1_4"

        # With custom start index
        vec2 = make_symbolic_vector(:u, 2, 3; start_idx=5)
        @test length(vec2) == 3
        @test string(vec2[1]) == "u_2_5"
        @test string(vec2[3]) == "u_2_7"
    end

    @testset "make_symbolic_matrix dimensions" begin
        mat = make_symbolic_matrix(:K, 1, 2, 3)
        @test size(mat) == (2, 3)
        @test string(mat[1, 1]) == "K_1_1_1"
        @test string(mat[2, 3]) == "K_1_2_3"
    end

    @testset "symbolic variables are Symbolics.Num type" begin
        v = make_symbolic_variable(:test, 1)
        @test v isa Symbolics.Num

        vec = make_symbolic_vector(:test, 1, 2)
        @test all(x -> x isa Symbolics.Num, vec)

        mat = make_symbolic_matrix(:test, 1, 2, 2)
        @test all(x -> x isa Symbolics.Num, mat)
    end
end
