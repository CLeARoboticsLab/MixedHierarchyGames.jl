using Test
using Graphs: SimpleDiGraph, add_edge!
using Symbolics: Num
using MixedHierarchyGames: MixedHierarchyGames, AbstractHierarchySolver, QPSolver, NonlinearSolver
using MixedHierarchyGames: solve, solve_raw
using TrajectoryGamesBase: JointStrategy

# make_θ helper is provided by testing_utils.jl (included in runtests.jl)

@testset "Unified Solver Interface" begin

    @testset "AbstractHierarchySolver type hierarchy" begin
        @testset "QPSolver is a subtype of AbstractHierarchySolver" begin
            @test QPSolver <: AbstractHierarchySolver
        end

        @testset "NonlinearSolver is a subtype of AbstractHierarchySolver" begin
            @test NonlinearSolver <: AbstractHierarchySolver
        end
    end

    @testset "solve() accepts Vector-of-Vectors initial state" begin
        @testset "QPSolver with Vector-of-Vectors" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)
            state_dim = 1
            control_dim = 1
            primal_dims = [4, 4]  # T=2: (state+control)*T per player

            θ1_vec = make_θ(1, state_dim)
            θ2_vec = make_θ(2, state_dim)
            θs = Dict(1 => θ1_vec, 2 => θ2_vec)
            gs = [
                z -> [z[1] - θ1_vec[1]],
                z -> [z[1] - θ2_vec[1]],
            ]
            Js = Dict(
                1 => (z1, z2; θ=nothing) -> sum(z1.^2),
                2 => (z1, z2; θ=nothing) -> sum(z2.^2),
            )

            solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

            # Current Dict interface still works
            result_dict = solve(solver, Dict(1 => [1.0], 2 => [2.0]))
            @test result_dict isa JointStrategy

            # NEW: Vector-of-Vectors also works
            result_vec = solve(solver, [[1.0], [2.0]])
            @test result_vec isa JointStrategy

            # Results should be identical
            for i in 1:2
                @test result_dict.substrategies[i].xs ≈ result_vec.substrategies[i].xs atol=1e-10
                @test result_dict.substrategies[i].us ≈ result_vec.substrategies[i].us atol=1e-10
            end
        end

        @testset "NonlinearSolver with Vector-of-Vectors" begin
            G = SimpleDiGraph(1)
            T = 2
            state_dim = 1
            control_dim = 1
            primal_dims = [control_dim * T]

            θ_vec = make_θ(1, state_dim)
            θs = Dict(1 => θ_vec)
            gs = [z -> Num[]]

            function J1(z1; θ=nothing)
                x0 = θs[1][1]
                x1 = x0 + z1[1]
                x2 = x1 + z1[2]
                return x1^2 + x2^2 + z1[1]^2 + z1[2]^2
            end
            Js = Dict(1 => J1)

            solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

            # Dict interface
            result_dict = solve(solver, Dict(1 => [1.0]))
            @test result_dict isa JointStrategy

            # NEW: Vector-of-Vectors interface
            result_vec = solve(solver, [[1.0]])
            @test result_vec isa JointStrategy

            # Results should match
            @test result_dict.substrategies[1].xs ≈ result_vec.substrategies[1].xs atol=1e-10
            @test result_dict.substrategies[1].us ≈ result_vec.substrategies[1].us atol=1e-10
        end
    end

    @testset "solve_raw() accepts Vector-of-Vectors initial state" begin
        @testset "QPSolver solve_raw with Vector-of-Vectors" begin
            G = SimpleDiGraph(1)
            primal_dims = [4]
            state_dim = 1
            control_dim = 1

            θ_vec = make_θ(1, 1)
            θs = Dict(1 => θ_vec)
            gs = [z -> [z[1] - θ_vec[1]]]
            Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

            solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

            result_dict = solve_raw(solver, Dict(1 => [1.0]))
            result_vec = solve_raw(solver, [[1.0]])

            @test result_dict.status == :solved
            @test result_vec.status == :solved
            @test result_dict.sol ≈ result_vec.sol atol=1e-10
        end

        @testset "NonlinearSolver solve_raw with Vector-of-Vectors" begin
            G = SimpleDiGraph(1)
            T = 2
            state_dim = 1
            control_dim = 1
            primal_dims = [control_dim * T]

            θ_vec = make_θ(1, state_dim)
            θs = Dict(1 => θ_vec)
            gs = [z -> Num[]]

            function J1_raw(z1; θ=nothing)
                x0 = θs[1][1]
                x1 = x0 + z1[1]
                x2 = x1 + z1[2]
                return x1^2 + x2^2 + z1[1]^2 + z1[2]^2
            end
            Js = Dict(1 => J1_raw)

            solver = NonlinearSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

            result_dict = solve_raw(solver, Dict(1 => [1.0]))
            result_vec = solve_raw(solver, [[1.0]])

            @test result_dict.status == :solved
            @test result_vec.status == :solved
            @test result_dict.sol ≈ result_vec.sol atol=1e-10
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
