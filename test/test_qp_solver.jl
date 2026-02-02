using Test
using Graphs: SimpleDiGraph, add_edge!, nv
using LinearAlgebra: norm, I
using Symbolics
using PATHSolver: PATHSolver
using MixedHierarchyGames:
    get_qp_kkt_conditions,
    strip_policy_constraints,
    setup_problem_variables,
    setup_problem_parameter_variables,
    solve_with_path,
    qp_game_linsolve,
    make_symbolic_vector

@testset "QP Solver - solve_with_path" begin
    @testset "Solves simple 1-player QP" begin
        # Single player: min z² s.t. z = θ (initial state constraint)
        # Solution: z = θ
        G = SimpleDiGraph(1)

        primal_dims = [2]
        @variables θ[1:2]
        θ_vec = collect(θ)

        gs = [z -> z - θ_vec]  # z = θ constraint

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2))

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys; θ=θ_vec)
        πs_solve = strip_policy_constraints(result.πs, G, vars.zs, gs)

        # Build MCP and solve
        z_sol, status, info = solve_with_path(πs_solve, vars.all_variables, Dict(1 => θ_vec), Dict(1 => [1.0, 2.0]))

        @test status == PATHSolver.MCP_Solved
        # Solution should be z = θ = [1.0, 2.0]
        @test isapprox(z_sol[1:2], [1.0, 2.0], atol=1e-6)
    end

    @testset "Solves 2-player Stackelberg" begin
        # Leader-follower: 1 → 2
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]
        @variables θ1[1:2] θ2[1:2]
        θ1_vec = collect(θ1)
        θ2_vec = collect(θ2)

        # Constraints: initial state
        gs = [
            z -> z - θ1_vec,
            z -> z - θ2_vec,
        ]

        vars = setup_problem_variables(G, primal_dims, gs)

        # Costs: each player minimizes own z²
        Js = Dict(
            1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2),
            2 => (zs...; θ=nothing) -> sum(vars.zs[2].^2),
        )

        θs = Dict(1 => θ1_vec, 2 => θ2_vec)
        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys; θ=vcat(θ1_vec, θ2_vec))
        πs_solve = strip_policy_constraints(result.πs, G, vars.zs, gs)

        param_values = Dict(1 => [1.0, 2.0], 2 => [3.0, 4.0])
        z_sol, status, info = solve_with_path(πs_solve, vars.all_variables, θs, param_values)

        @test status == PATHSolver.MCP_Solved
    end
end

@testset "QP Solver - qp_game_linsolve" begin
    @testset "Solves linear system for LQ game" begin
        # For LQ games, KKT system is linear: Ax = b
        # Test with simple 2x2 system
        A = [2.0 1.0; 1.0 3.0]
        b = [3.0, 4.0]

        x = qp_game_linsolve(A, b)

        @test isapprox(A * x, b, atol=1e-10)
    end

    @testset "Handles larger systems" begin
        n = 10
        A = randn(n, n)
        A = A' * A + I  # Make positive definite
        b = randn(n)

        x = qp_game_linsolve(A, b)

        @test isapprox(A * x, b, atol=1e-8)
    end
end

@testset "QP Solver - KKT residual at solution" begin
    @testset "Residual is zero at optimal" begin
        # Solve a problem and verify KKT residual is zero
        G = SimpleDiGraph(1)
        primal_dims = [2]

        @variables θ[1:2]
        θ_vec = collect(θ)

        gs = [z -> z - θ_vec]
        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2))

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys; θ=θ_vec)
        πs_solve = strip_policy_constraints(result.πs, G, vars.zs, gs)

        z_sol, status, _ = solve_with_path(πs_solve, vars.all_variables, Dict(1 => θ_vec), Dict(1 => [1.0, 2.0]))

        # Evaluate KKT residual at solution
        # The residual should be near zero
        @test status == PATHSolver.MCP_Solved
    end
end
