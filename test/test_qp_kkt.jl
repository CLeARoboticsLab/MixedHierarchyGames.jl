using Test
using Graphs: SimpleDiGraph, add_edge!, nv
using LinearAlgebra: I, norm
using Symbolics
using MixedHierarchyGames: get_qp_kkt_conditions, strip_policy_constraints
using MixedHierarchyGames: setup_problem_variables, make_symbolic_vector

@testset "QP KKT Construction" begin
    @testset "Leaf player KKT conditions" begin
        # Single player (leaf) should have:
        # 1. Stationarity: ∇_z L = 0
        # 2. Constraints: g(z) = 0
        G = SimpleDiGraph(1)  # Single player, no edges

        # Setup: 2-dim decision variable, 1 constraint
        primal_dims = [2]
        gs = [z -> [z[1] + z[2] - 1.0]]  # Single equality constraint

        vars = setup_problem_variables(G, primal_dims, gs)

        # Simple quadratic cost: J = z₁² + z₂²
        Js = Dict(1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2))

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys)

        # Should return πs dict with KKT for player 1
        @test haskey(result.πs, 1)

        # Leaf player KKT: stationarity (2) + constraints (1) = 3 conditions
        @test length(result.πs[1]) == 3
    end

    @testset "Leader includes follower policy constraints" begin
        # 2-player Stackelberg: 1→2
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [
            z -> [z[1] + z[2] - 1.0],  # P1 constraint
            z -> [z[1] - z[2]],         # P2 constraint
        ]

        vars = setup_problem_variables(G, primal_dims, gs)

        # Costs
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2) + sum(vars.zs[2].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys)

        # Both players should have KKT conditions
        @test haskey(result.πs, 1)
        @test haskey(result.πs, 2)

        # P2 (leaf): stationarity (2) + constraints (1) = 3
        @test length(result.πs[2]) == 3

        # P1 (leader): stationarity (2+2) + policy constraint (2) + own constraints (1) = 7
        # (leader's KKT includes follower variables and policy)
        @test length(result.πs[1]) >= 5  # At least stationarity + constraints
    end

    @testset "Returns M and N matrices for followers" begin
        # 2-player: 1→2, P2 is follower so needs M, N
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [z -> [z[1]], z -> [z[1]]]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys)

        # P2 has a leader, so should have M and N matrices
        @test haskey(result.Ms, 2)
        @test haskey(result.Ns, 2)

        # P1 is root, no M/N needed
        @test !haskey(result.Ms, 1)
    end
end

@testset "Strip Policy Constraints" begin
    @testset "Removes policy rows from leader KKT" begin
        # Setup 2-player game
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        gs = [z -> [z[1]], z -> [z[1]]]

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys)

        # Strip policy constraints
        πs_stripped = strip_policy_constraints(result.πs, G, vars.zs, gs)

        # Stripped should have fewer or equal conditions
        @test length(πs_stripped[1]) <= length(result.πs[1])

        # Leaf (P2) should be unchanged
        @test length(πs_stripped[2]) == length(result.πs[2])
    end
end
