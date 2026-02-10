using Test
using MixedHierarchyGames: HierarchyProblem, QPPrecomputed, NonlinearSolver
using Graphs: SimpleDiGraph

@testset "Type Parameter Bounds" begin
    @testset "HierarchyProblem rejects non-Dict Js" begin
        G = SimpleDiGraph(1)
        # Pass a non-Dict (Vector) for Js — should be rejected by type bound
        @test_throws MethodError HierarchyProblem(G, [1, 2], [z -> z], [2], Dict(1 => [1.0]), 1, 1)
    end

    @testset "HierarchyProblem rejects non-Vector gs" begin
        G = SimpleDiGraph(1)
        # Pass a non-Vector (Dict) for gs — should be rejected by type bound
        @test_throws MethodError HierarchyProblem(G, Dict(1 => identity), Dict(1 => identity), [2], Dict(1 => [1.0]), 1, 1)
    end

    @testset "HierarchyProblem rejects non-Dict θs" begin
        G = SimpleDiGraph(1)
        # Pass a non-Dict (Vector) for θs — should be rejected by type bound
        @test_throws MethodError HierarchyProblem(G, Dict(1 => identity), [z -> z], [2], [1.0], 1, 1)
    end

    @testset "HierarchyProblem accepts valid Dict/Vector types" begin
        G = SimpleDiGraph(1)
        # Valid construction: Dict for Js, Vector for gs, Dict for θs
        prob = HierarchyProblem(G, Dict(1 => identity), [z -> z], [2], Dict(1 => [1.0]), 1, 1)
        @test prob isa HierarchyProblem
    end

    @testset "QPPrecomputed rejects non-Dict πs_solve" begin
        # Pass a non-Dict (Vector) for πs_solve — should be rejected by type bound
        @test_throws MethodError QPPrecomputed((;), (;), [1, 2], nothing, nothing, Float64[], Float64[])
    end

    @testset "QPPrecomputed accepts valid Dict πs_solve" begin
        qp = QPPrecomputed((;), (;), Dict(1 => [1.0]), nothing, nothing, Float64[], Float64[])
        @test qp isa QPPrecomputed
    end

    @testset "NonlinearSolver rejects non-NamedTuple precomputed" begin
        G = SimpleDiGraph(1)
        prob = HierarchyProblem(G, Dict(1 => identity), [z -> z], [2], Dict(1 => [1.0]), 1, 1)
        # Pass a Dict for precomputed — should be rejected since it needs NamedTuple
        @test_throws MethodError NonlinearSolver(prob, Dict(:a => 1), (;))
    end

    @testset "NonlinearSolver accepts valid NamedTuple precomputed" begin
        G = SimpleDiGraph(1)
        prob = HierarchyProblem(G, Dict(1 => identity), [z -> z], [2], Dict(1 => [1.0]), 1, 1)
        solver = NonlinearSolver(prob, (;), (;))
        @test solver isa NonlinearSolver
    end
end
