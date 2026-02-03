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
    solve_qp_linear,
    qp_game_linsolve,
    run_qp_solver,
    make_symbolic_vector,
    QPSolver

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

@testset "QP Solver - run_qp_solver (linear)" begin
    @testset "Solves single player problem" begin
        G = SimpleDiGraph(1)
        primal_dims = [2]

        @variables θ[1:2]
        θ_vec = collect(θ)
        θs = Dict(1 => θ_vec)

        gs = [z -> z - θ_vec]

        # Cost: minimize z²
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        parameter_values = Dict(1 => [1.0, 2.0])

        result = run_qp_solver(G, Js, gs, primal_dims, θs, parameter_values; solver=:linear)

        @test result.status == :solved
        @test isapprox(result.z_sol[1:2], [1.0, 2.0], atol=1e-6)
    end

    @testset "Solves 2-player Stackelberg" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]

        @variables θ1[1:2] θ2[1:2]
        θ1_vec = collect(θ1)
        θ2_vec = collect(θ2)
        θs = Dict(1 => θ1_vec, 2 => θ2_vec)

        gs = [
            z -> z - θ1_vec,
            z -> z - θ2_vec,
        ]

        # Each player minimizes own z²
        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2),
        )

        parameter_values = Dict(1 => [1.0, 2.0], 2 => [3.0, 4.0])

        result = run_qp_solver(G, Js, gs, primal_dims, θs, parameter_values; solver=:linear)

        @test result.status == :solved
        # Check that solution satisfies constraints
        @test isapprox(result.z_sol[1:2], [1.0, 2.0], atol=1e-6)
        @test isapprox(result.z_sol[3:4], [3.0, 4.0], atol=1e-6)
    end

    @testset "Returns vars and kkt_result" begin
        G = SimpleDiGraph(1)
        primal_dims = [2]

        @variables θ[1:2]
        θ_vec = collect(θ)
        θs = Dict(1 => θ_vec)
        gs = [z -> z - θ_vec]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))
        parameter_values = Dict(1 => [1.0, 2.0])

        result = run_qp_solver(G, Js, gs, primal_dims, θs, parameter_values)

        @test haskey(result.vars, :zs)
        @test haskey(result.vars, :λs)
        @test haskey(result.kkt_result, :πs)
    end
end

@testset "QP Solver - run_qp_solver (PATH)" begin
    @testset "Solves with PATH solver" begin
        G = SimpleDiGraph(1)
        primal_dims = [2]

        @variables θ[1:2]
        θ_vec = collect(θ)
        θs = Dict(1 => θ_vec)
        gs = [z -> z - θ_vec]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))
        parameter_values = Dict(1 => [1.0, 2.0])

        result = run_qp_solver(G, Js, gs, primal_dims, θs, parameter_values; solver=:path)

        @test result.status == PATHSolver.MCP_Solved
        @test isapprox(result.z_sol[1:2], [1.0, 2.0], atol=1e-6)
    end
end

@testset "QPSolver struct interface" begin
    @testset "Constructor precomputes KKT" begin
        G = SimpleDiGraph(1)
        primal_dims = [2]

        @variables θ[1:2]
        θ_vec = collect(θ)
        θs = Dict(1 => θ_vec)
        gs = [z -> z - θ_vec]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs)

        @test solver.problem isa MixedHierarchyGames.QPProblem
        @test solver.solver_type == :linear
        @test haskey(solver.precomputed, :vars)
        @test haskey(solver.precomputed, :πs_solve)
    end

    @testset "solve() uses precomputed components" begin
        G = SimpleDiGraph(1)
        primal_dims = [2]

        @variables θ[1:2]
        θ_vec = collect(θ)
        θs = Dict(1 => θ_vec)
        gs = [z -> z - θ_vec]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs)
        result = MixedHierarchyGames.solve(solver, Dict(1 => [1.0, 2.0]))

        @test result.status == :solved
        @test isapprox(result.z_sol[1:2], [1.0, 2.0], atol=1e-6)
    end

    @testset "solve() with different parameter values" begin
        G = SimpleDiGraph(1)
        primal_dims = [2]

        @variables θ[1:2]
        θ_vec = collect(θ)
        θs = Dict(1 => θ_vec)
        gs = [z -> z - θ_vec]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs)

        # Solve with different initial states
        result1 = MixedHierarchyGames.solve(solver, Dict(1 => [1.0, 2.0]))
        result2 = MixedHierarchyGames.solve(solver, Dict(1 => [5.0, 6.0]))

        @test isapprox(result1.z_sol[1:2], [1.0, 2.0], atol=1e-6)
        @test isapprox(result2.z_sol[1:2], [5.0, 6.0], atol=1e-6)
    end

    @testset "2-player Stackelberg with QPSolver struct" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]

        @variables θ1[1:2] θ2[1:2]
        θ1_vec = collect(θ1)
        θ2_vec = collect(θ2)
        θs = Dict(1 => θ1_vec, 2 => θ2_vec)

        gs = [
            z -> z - θ1_vec,
            z -> z - θ2_vec,
        ]

        Js = Dict(
            1 => (z1, z2; θ=nothing) -> sum(z1.^2),
            2 => (z1, z2; θ=nothing) -> sum(z2.^2),
        )

        solver = QPSolver(G, Js, gs, primal_dims, θs)
        result = MixedHierarchyGames.solve(solver, Dict(1 => [1.0, 2.0], 2 => [3.0, 4.0]))

        @test result.status == :solved
        @test isapprox(result.z_sol[1:2], [1.0, 2.0], atol=1e-6)
        @test isapprox(result.z_sol[3:4], [3.0, 4.0], atol=1e-6)
    end

    @testset "PATH solver option" begin
        G = SimpleDiGraph(1)
        primal_dims = [2]

        @variables θ[1:2]
        θ_vec = collect(θ)
        θs = Dict(1 => θ_vec)
        gs = [z -> z - θ_vec]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs; solver=:path)
        result = MixedHierarchyGames.solve(solver, Dict(1 => [1.0, 2.0]))

        @test result.status == PATHSolver.MCP_Solved
        @test isapprox(result.z_sol[1:2], [1.0, 2.0], atol=1e-6)
    end
end
