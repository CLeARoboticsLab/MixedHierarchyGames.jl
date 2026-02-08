using Test
using Graphs: SimpleDiGraph, add_edge!
using MixedHierarchyGames: HierarchyProblem, QPPrecomputed, NonlinearSolver, HierarchyGame, QPSolver

# make_θ helper is provided by testing_utils.jl (included in runtests.jl)

@testset "Type Parameter Bounds" begin
    @testset "HierarchyProblem" begin
        @testset "rejects non-AbstractDict for Js" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)
            # Js should be a Dict (AbstractDict), not a Vector
            @test_throws MethodError HierarchyProblem(G, [1, 2], [identity], [2, 2], Dict(), 1, 1)
        end

        @testset "rejects non-AbstractVector for gs" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)
            # gs should be a Vector (AbstractVector), not a Dict
            @test_throws MethodError HierarchyProblem(G, Dict(), Dict(1 => identity), [2, 2], Dict(), 1, 1)
        end

        @testset "rejects non-AbstractDict for θs" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)
            # θs should be a Dict (AbstractDict), not a Vector
            @test_throws MethodError HierarchyProblem(G, Dict(), [identity], [2, 2], [1, 2], 1, 1)
        end

        @testset "accepts valid types" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)
            prob = HierarchyProblem(G, Dict(1 => identity), [identity, identity], [2, 2], Dict(1 => [1.0]), 1, 1)
            @test prob isa HierarchyProblem
            @test prob isa HierarchyProblem{<:SimpleDiGraph, <:AbstractDict, <:AbstractVector, <:AbstractDict}
        end
    end

    @testset "QPPrecomputed" begin
        @testset "rejects non-NamedTuple for vars" begin
            # vars should be a NamedTuple, not a String
            # Use a plain NamedTuple for the other fields (parametric_mcp bound prevents easy testing)
            @test_throws MethodError QPPrecomputed("not_a_namedtuple", (;), Dict(), "fake_mcp")
        end

        @testset "rejects non-NamedTuple for kkt_result" begin
            @test_throws MethodError QPPrecomputed((;), 42, Dict(), "fake_mcp")
        end

        @testset "rejects non-AbstractDict for πs_solve" begin
            @test_throws MethodError QPPrecomputed((;), (;), [1, 2], "fake_mcp")
        end

        @testset "type bounds enforced in real QPSolver" begin
            # Verify that a real QPSolver produces QPPrecomputed with correct type params
            G = SimpleDiGraph(1)
            primal_dims = [4]
            θ_vec = make_θ(1, 1)
            θs = Dict(1 => θ_vec)
            gs = [z -> [z[1] - θ_vec[1]]]
            Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

            solver = QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
            pc = solver.precomputed

            @test pc isa QPPrecomputed{<:NamedTuple, <:NamedTuple, <:AbstractDict}
        end
    end

    @testset "NonlinearSolver" begin
        @testset "rejects non-NamedTuple for precomputed" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)
            prob = HierarchyProblem(G, Dict(1 => identity), [identity, identity], [2, 2], Dict(1 => [1.0]), 1, 1)
            # precomputed should be a NamedTuple, not a String
            @test_throws MethodError NonlinearSolver(prob, "not_a_namedtuple", (;))
        end

        @testset "accepts valid types" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)
            prob = HierarchyProblem(G, Dict(1 => identity), [identity, identity], [2, 2], Dict(1 => [1.0]), 1, 1)
            solver = NonlinearSolver(prob, (;), (;))
            @test solver isa NonlinearSolver
            @test solver isa NonlinearSolver{<:HierarchyProblem, <:NamedTuple}
        end
    end
end
