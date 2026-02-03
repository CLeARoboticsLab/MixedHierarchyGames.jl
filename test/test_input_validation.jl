using Test
using Graphs: SimpleDiGraph, add_edge!, add_vertex!
using MixedHierarchyGames: QPSolver, solve, QPPrecomputed

# make_θ helper is provided by testing_utils.jl (included in runtests.jl)

"""
Run function with timeout. Returns (result, timed_out::Bool).
If timed out, result is nothing.
"""
function with_timeout(f, timeout_sec::Real)
    result = Channel{Any}(1)
    task = @async put!(result, f())

    if timedwait(() -> isready(result), timeout_sec) == :timed_out
        # Try to interrupt the task
        try
            schedule(task, InterruptException(); error=true)
        catch
        end
        return (nothing, true)
    end

    return (take!(result), false)
end

@testset "Input Validation" begin
    @testset "Graph structure validation" begin
        @testset "Rejects cyclic graph" begin
            # Create a cycle: 1 → 2 → 1
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)
            add_edge!(G, 2, 1)

            primal_dims = [2, 2]
            θ1_vec = make_θ(1, 1)
            θ2_vec = make_θ(2, 1)
            θs = Dict(1 => θ1_vec, 2 => θ2_vec)
            gs = [z -> [z[1] - θ1_vec[1]], z -> [z[1] - θ2_vec[1]]]
            Js = Dict(
                1 => (z1, z2; θ=nothing) -> sum(z1.^2),
                2 => (z1, z2; θ=nothing) -> sum(z2.^2),
            )

            # Use timeout to prevent hanging if validation fails
            (result, timed_out) = with_timeout(5.0) do
                try
                    QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
                    return :no_error
                catch e
                    return e
                end
            end

            @test !timed_out  # Fails if validation didn't catch the cycle
            @test result isa ArgumentError
        end

        @testset "Rejects self-loop" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 1)  # Self-loop

            primal_dims = [2, 2]
            θ1_vec = make_θ(1, 1)
            θ2_vec = make_θ(2, 1)
            θs = Dict(1 => θ1_vec, 2 => θ2_vec)
            gs = [z -> [z[1] - θ1_vec[1]], z -> [z[1] - θ2_vec[1]]]
            Js = Dict(
                1 => (z1, z2; θ=nothing) -> sum(z1.^2),
                2 => (z1, z2; θ=nothing) -> sum(z2.^2),
            )

            # Use timeout to prevent hanging if validation fails
            (result, timed_out) = with_timeout(5.0) do
                try
                    QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
                    return :no_error
                catch e
                    return e
                end
            end

            @test !timed_out  # Fails if validation didn't catch the self-loop
            @test result isa ArgumentError
        end
    end

    @testset "Dimension consistency validation" begin
        @testset "Rejects mismatched primal_dims length" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)

            primal_dims = [2, 2, 2]  # 3 elements for 2-player game
            θ1_vec = make_θ(1, 1)
            θ2_vec = make_θ(2, 1)
            θs = Dict(1 => θ1_vec, 2 => θ2_vec)
            gs = [z -> [z[1]], z -> [z[1]]]
            Js = Dict(
                1 => (z1, z2; θ=nothing) -> sum(z1.^2),
                2 => (z1, z2; θ=nothing) -> sum(z2.^2),
            )

            @test_throws ArgumentError QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
        end

        @testset "Rejects mismatched gs length" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)

            primal_dims = [2, 2]
            θ1_vec = make_θ(1, 1)
            θ2_vec = make_θ(2, 1)
            θs = Dict(1 => θ1_vec, 2 => θ2_vec)
            gs = [z -> [z[1]]]  # Only 1 constraint function for 2 players
            Js = Dict(
                1 => (z1, z2; θ=nothing) -> sum(z1.^2),
                2 => (z1, z2; θ=nothing) -> sum(z2.^2),
            )

            @test_throws ArgumentError QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
        end

        @testset "Rejects missing player in Js" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)

            primal_dims = [2, 2]
            θ1_vec = make_θ(1, 1)
            θ2_vec = make_θ(2, 1)
            θs = Dict(1 => θ1_vec, 2 => θ2_vec)
            gs = [z -> [z[1]], z -> [z[1]]]
            Js = Dict(1 => (z1, z2; θ=nothing) -> sum(z1.^2))  # Missing player 2

            @test_throws ArgumentError QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
        end

        @testset "Rejects missing player in θs" begin
            G = SimpleDiGraph(2)
            add_edge!(G, 1, 2)

            primal_dims = [2, 2]
            θ1_vec = make_θ(1, 1)
            θs = Dict(1 => θ1_vec)  # Missing player 2
            gs = [z -> [z[1]], z -> [z[1]]]
            Js = Dict(
                1 => (z1, z2; θ=nothing) -> sum(z1.^2),
                2 => (z1, z2; θ=nothing) -> sum(z2.^2),
            )

            @test_throws ArgumentError QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
        end
    end

    @testset "Parameter validation in solve()" begin
        # Setup a valid solver first
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [4, 4]
        state_dim = 1
        control_dim = 1

        θ1_vec = make_θ(1, 1)
        θ2_vec = make_θ(2, 1)
        θs = Dict(1 => θ1_vec, 2 => θ2_vec)
        gs = [z -> [z[1] - θ1_vec[1]], z -> [z[1] - θ2_vec[1]]]
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2),
        )

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

        @testset "Rejects missing player in parameter_values" begin
            parameter_values = Dict(1 => [1.0])  # Missing player 2
            @test_throws ArgumentError solve(solver, parameter_values)
        end

        @testset "Rejects wrong parameter dimension" begin
            parameter_values = Dict(1 => [1.0, 2.0], 2 => [3.0])  # Player 1 has wrong dim
            @test_throws ArgumentError solve(solver, parameter_values)
        end
    end

    @testset "Constraint function validation" begin
        @testset "Rejects constraint function with wrong signature" begin
            G = SimpleDiGraph(1)
            primal_dims = [2]
            θ_vec = make_θ(1, 1)
            θs = Dict(1 => θ_vec)
            # Wrong signature: takes two arguments instead of one
            gs = [(z, extra) -> [z[1]]]
            Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

            @test_throws ArgumentError QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
        end

        @testset "Rejects constraint function returning scalar" begin
            G = SimpleDiGraph(1)
            primal_dims = [2]
            θ_vec = make_θ(1, 1)
            θs = Dict(1 => θ_vec)
            # Returns scalar instead of Vector
            gs = [z -> z[1]]
            Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

            @test_throws ArgumentError QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
        end

        @testset "Accepts valid constraint function" begin
            G = SimpleDiGraph(1)
            primal_dims = [4]
            θ_vec = make_θ(1, 1)
            θs = Dict(1 => θ_vec)
            # Valid: takes one argument, returns Vector
            gs = [z -> [z[1] - θ_vec[1]]]
            Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

            solver = QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
            @test solver isa QPSolver
        end
    end

    @testset "QPPrecomputed struct" begin
        @testset "Solver precomputed field is QPPrecomputed type" begin
            G = SimpleDiGraph(1)
            primal_dims = [4]
            θ_vec = make_θ(1, 1)
            θs = Dict(1 => θ_vec)
            gs = [z -> [z[1] - θ_vec[1]]]
            Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

            solver = QPSolver(G, Js, gs, primal_dims, θs, 1, 1)

            @test solver.precomputed isa QPPrecomputed
        end

        @testset "QPPrecomputed has required fields" begin
            G = SimpleDiGraph(1)
            primal_dims = [4]
            θ_vec = make_θ(1, 1)
            θs = Dict(1 => θ_vec)
            gs = [z -> [z[1] - θ_vec[1]]]
            Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

            solver = QPSolver(G, Js, gs, primal_dims, θs, 1, 1)
            precomputed = solver.precomputed

            @test hasproperty(precomputed, :vars)
            @test hasproperty(precomputed, :kkt_result)
            @test hasproperty(precomputed, :πs_solve)
            @test hasproperty(precomputed, :parametric_mcp)
        end
    end
end
