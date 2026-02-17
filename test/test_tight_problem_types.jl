using Test
using MixedHierarchyGames: HierarchyProblem
using Graphs: SimpleDiGraph

@testset "Tight HierarchyProblem Type Parameters" begin
    @testset "Type parameter bounds reject AbstractDict subtypes that aren't Dict" begin
        # If bounds are tightened to Dict (not AbstractDict), then a custom
        # AbstractDict subtype should be rejected at construction.
        # We create a minimal AbstractDict subtype to test this.
        struct _TestAbstractDict <: AbstractDict{Int, Function}
            d::Dict{Int, Function}
        end
        Base.iterate(d::_TestAbstractDict) = iterate(d.d)
        Base.iterate(d::_TestAbstractDict, s) = iterate(d.d, s)
        Base.length(d::_TestAbstractDict) = length(d.d)

        G = SimpleDiGraph(1)
        gs = [z -> z]
        primal_dims = [2]
        θs = Dict(1 => [1.0])

        # With tight bounds (Dict, not AbstractDict), this should throw MethodError
        @test_throws MethodError HierarchyProblem(
            G, _TestAbstractDict(Dict(1 => identity)), gs, primal_dims, θs, 1, 1
        )
    end

    @testset "Type parameter bounds reject AbstractVector subtypes that aren't Vector" begin
        # If bounds are tightened to Vector (not AbstractVector), then a custom
        # AbstractVector subtype should be rejected for gs.
        struct _TestAbstractVec <: AbstractVector{Function}
            v::Vector{Function}
        end
        Base.size(v::_TestAbstractVec) = size(v.v)
        Base.getindex(v::_TestAbstractVec, i::Int) = v.v[i]

        G = SimpleDiGraph(1)
        Js = Dict(1 => identity)
        primal_dims = [2]
        θs = Dict(1 => [1.0])

        # With tight bounds (Vector, not AbstractVector), this should throw MethodError
        @test_throws MethodError HierarchyProblem(
            G, Js, _TestAbstractVec([z -> z]), primal_dims, θs, 1, 1
        )
    end

    @testset "θs type parameter bound rejects non-Dict AbstractDict" begin
        struct _TestAbstractDict2 <: AbstractDict{Int, Vector{Float64}}
            d::Dict{Int, Vector{Float64}}
        end
        Base.iterate(d::_TestAbstractDict2) = iterate(d.d)
        Base.iterate(d::_TestAbstractDict2, s) = iterate(d.d, s)
        Base.length(d::_TestAbstractDict2) = length(d.d)

        G = SimpleDiGraph(1)
        Js = Dict(1 => identity)
        gs = [z -> z]
        primal_dims = [2]

        # With tight bounds (Dict, not AbstractDict), this should throw MethodError
        @test_throws MethodError HierarchyProblem(
            G, Js, gs, primal_dims, _TestAbstractDict2(Dict(1 => [1.0])), 1, 1
        )
    end

    @testset "Dict and Vector still accepted (positive test)" begin
        G = SimpleDiGraph(1)
        prob = HierarchyProblem(G, Dict(1 => identity), [z -> z], [2], Dict(1 => [1.0]), 1, 1)
        @test prob isa HierarchyProblem
        @test typeof(prob.Js) <: Dict
        @test typeof(prob.gs) <: Vector
        @test typeof(prob.θs) <: Dict
    end
end
