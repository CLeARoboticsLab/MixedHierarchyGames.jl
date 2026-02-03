using Test
using Graphs: SimpleDiGraph, add_edge!, nv
using Symbolics: Num  # Only for type checking
using MixedHierarchyGames: setup_problem_parameter_variables, setup_problem_variables

@testset "Problem Setup" begin
    @testset "setup_problem_parameter_variables" begin
        # 3 players with different parameter dimensions (e.g., initial state dims)
        num_params = [4, 4, 4]  # Each player has 4-dim state

        θs = setup_problem_parameter_variables(num_params)

        # Should return Dict with entry per player
        @test θs isa Dict
        @test length(θs) == 3
        @test haskey(θs, 1) && haskey(θs, 2) && haskey(θs, 3)

        # Each θ should have correct dimension
        @test length(θs[1]) == 4
        @test length(θs[2]) == 4
        @test length(θs[3]) == 4

        # Should be symbolic variables
        @test all(x -> x isa Num, θs[1])
    end

    @testset "setup_problem_variables structure" begin
        # Simple 2-player Stackelberg: 1→2
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        # Each player has 6-dim decision variable (e.g., state + control over horizon)
        primal_dims = [6, 6]

        # Simple constraint functions (dynamics placeholder)
        gs = [
            z -> zeros(Num, 2),  # P1: 2 constraints
            z -> zeros(Num, 2),  # P2: 2 constraints
        ]

        result = setup_problem_variables(G, primal_dims, gs)

        # Should return named tuple or struct with zs, λs, μs, ys, ws
        @test haskey(result, :zs) || hasproperty(result, :zs)
        @test haskey(result, :λs) || hasproperty(result, :λs)
        @test haskey(result, :ys) || hasproperty(result, :ys)
        @test haskey(result, :ws) || hasproperty(result, :ws)
    end

    @testset "setup_problem_variables dimensions" begin
        # 3-player chain: 1→2→3
        G = SimpleDiGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)

        primal_dims = [4, 4, 4]
        constraint_dims = [2, 2, 2]

        gs = [
            z -> zeros(Num, constraint_dims[i]) for i in 1:3
        ]

        result = setup_problem_variables(G, primal_dims, gs)

        # zs should have correct primal dimensions
        @test length(result.zs[1]) == primal_dims[1]
        @test length(result.zs[2]) == primal_dims[2]
        @test length(result.zs[3]) == primal_dims[3]

        # λs should have correct constraint dimensions
        @test length(result.λs[1]) == constraint_dims[1]
        @test length(result.λs[2]) == constraint_dims[2]
        @test length(result.λs[3]) == constraint_dims[3]
    end

    @testset "information vectors (ys) follow hierarchy" begin
        # Chain: 1→2→3
        # P1's info: just state (no leaders)
        # P2's info: state + P1's decision
        # P3's info: state + P1's decision + P2's decision
        G = SimpleDiGraph(3)
        add_edge!(G, 1, 2)
        add_edge!(G, 2, 3)

        primal_dims = [4, 4, 4]
        gs = [z -> zeros(Num, 2) for _ in 1:3]

        result = setup_problem_variables(G, primal_dims, gs)

        # P1 (root): ys should be empty or just parameters
        @test length(result.ys[1]) == 0

        # P2: ys should include P1's decisions
        @test length(result.ys[2]) == primal_dims[1]

        # P3: ys should include P1's and P2's decisions
        @test length(result.ys[3]) == primal_dims[1] + primal_dims[2]
    end

    @testset "mixed hierarchy info vectors" begin
        # P1 Nash with P2, P2→P3
        G = SimpleDiGraph(3)
        add_edge!(G, 2, 3)  # Only P2→P3

        primal_dims = [4, 4, 4]
        gs = [z -> zeros(Num, 2) for _ in 1:3]

        result = setup_problem_variables(G, primal_dims, gs)

        # P1 (root, Nash): no leaders
        @test length(result.ys[1]) == 0

        # P2 (root, Nash): no leaders
        @test length(result.ys[2]) == 0

        # P3 (follower of P2): has P2's decisions
        @test length(result.ys[3]) == primal_dims[2]
    end
end
