using Test
using MixedHierarchyGames: MixedHierarchyGames, AbstractMixedHierarchyGameSolver, QPSolver, NonlinearSolver

#=
    Tests for the unified solver interface.

    Owns: abstract type hierarchy and internal conversion utilities.
    Vector-of-Vectors parameter passing behavioral tests are in test_flexible_callsite.jl.
=#

@testset "Unified Solver Interface" begin

    @testset "AbstractMixedHierarchyGameSolver type hierarchy" begin
        @testset "QPSolver is a subtype of AbstractMixedHierarchyGameSolver" begin
            @test QPSolver <: AbstractMixedHierarchyGameSolver
        end

        @testset "NonlinearSolver is a subtype of AbstractMixedHierarchyGameSolver" begin
            @test NonlinearSolver <: AbstractMixedHierarchyGameSolver
        end
    end

    @testset "Type conversion: _to_parameter_dict" begin
        # Test the internal conversion function directly
        to_dict = MixedHierarchyGames._to_parameter_dict

        # Dict passthrough
        d = Dict(1 => [1.0, 2.0], 2 => [3.0, 4.0])
        @test to_dict(d) === d  # Same object (no copy)

        # Vector-of-Vectors conversion
        vv = [[1.0, 2.0], [3.0, 4.0]]
        result = to_dict(vv)
        @test result isa Dict
        @test result[1] == [1.0, 2.0]
        @test result[2] == [3.0, 4.0]

        # Error on invalid input
        @test_throws ArgumentError to_dict(42)
        @test_throws ArgumentError to_dict([1.0, 2.0])  # flat vector, not vector-of-vectors
    end
end
