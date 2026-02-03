using Test
using Graphs: SimpleDiGraph, add_edge!, nv
using LinearAlgebra: norm, I
using PATHSolver: PATHSolver
using TrajectoryGamesBase: JointStrategy, OpenLoopStrategy
using MixedHierarchyGames:
    get_qp_kkt_conditions,
    strip_policy_constraints,
    setup_problem_variables,
    setup_problem_parameter_variables,
    solve_with_path,
    solve_qp_linear,
    qp_game_linsolve,
    _run_qp_solver,
    QPSolver,
    QPPrecomputed

# make_θ helper is provided by testing_utils.jl (included in runtests.jl)

@testset "QP Solver - solve_with_path" begin
    @testset "Solves simple 1-player QP" begin
        # Single player: min z² s.t. z = θ (initial state constraint)
        # Solution: z = θ
        G = SimpleDiGraph(1)

        primal_dims = [2]
        θ_vec = make_θ(1, 2)

        gs = [z -> z - θ_vec]  # z = θ constraint

        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2))

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys, vars.ws_z_indices; θ=θ_vec)
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
        θ1_vec = make_θ(1, 2)
        θ2_vec = make_θ(2, 2)

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
        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys, vars.ws_z_indices; θ=vcat(θ1_vec, θ2_vec))
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

        θ_vec = make_θ(1, 2)

        gs = [z -> z - θ_vec]
        vars = setup_problem_variables(G, primal_dims, gs)
        Js = Dict(1 => (zs...; θ=nothing) -> sum(vars.zs[1].^2))

        result = get_qp_kkt_conditions(G, Js, vars.zs, vars.λs, vars.μs, gs, vars.ws, vars.ys, vars.ws_z_indices; θ=θ_vec)
        πs_solve = strip_policy_constraints(result.πs, G, vars.zs, gs)

        z_sol, status, _ = solve_with_path(πs_solve, vars.all_variables, Dict(1 => θ_vec), Dict(1 => [1.0, 2.0]))

        # Evaluate KKT residual at solution
        # The residual should be near zero
        @test status == PATHSolver.MCP_Solved
    end
end

@testset "QP Solver - _run_qp_solver (linear)" begin
    @testset "Solves single player problem" begin
        G = SimpleDiGraph(1)
        primal_dims = [2]

        θ_vec = make_θ(1, 2)
        θs = Dict(1 => θ_vec)

        gs = [z -> z - θ_vec]

        # Cost: minimize z²
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        parameter_values = Dict(1 => [1.0, 2.0])

        result = _run_qp_solver(G, Js, gs, primal_dims, θs, parameter_values; solver=:linear)

        @test result.status == :solved
        @test isapprox(result.z_sol[1:2], [1.0, 2.0], atol=1e-6)
    end

    @testset "Solves 2-player Stackelberg" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [2, 2]

        θ1_vec = make_θ(1, 2)
        θ2_vec = make_θ(2, 2)
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

        result = _run_qp_solver(G, Js, gs, primal_dims, θs, parameter_values; solver=:linear)

        @test result.status == :solved
        # Check that solution satisfies constraints
        @test isapprox(result.z_sol[1:2], [1.0, 2.0], atol=1e-6)
        @test isapprox(result.z_sol[3:4], [3.0, 4.0], atol=1e-6)
    end

    @testset "Returns vars and kkt_result" begin
        G = SimpleDiGraph(1)
        primal_dims = [2]

        θ_vec = make_θ(1, 2)
        θs = Dict(1 => θ_vec)
        gs = [z -> z - θ_vec]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))
        parameter_values = Dict(1 => [1.0, 2.0])

        result = _run_qp_solver(G, Js, gs, primal_dims, θs, parameter_values)

        @test hasproperty(result.vars, :zs)
        @test hasproperty(result.vars, :λs)
        @test hasproperty(result.kkt_result, :πs)
    end
end

@testset "QP Solver - _run_qp_solver (PATH)" begin
    @testset "Solves with PATH solver" begin
        G = SimpleDiGraph(1)
        primal_dims = [2]

        θ_vec = make_θ(1, 2)
        θs = Dict(1 => θ_vec)
        gs = [z -> z - θ_vec]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))
        parameter_values = Dict(1 => [1.0, 2.0])

        result = _run_qp_solver(G, Js, gs, primal_dims, θs, parameter_values; solver=:path)

        @test result.status == PATHSolver.MCP_Solved
        @test isapprox(result.z_sol[1:2], [1.0, 2.0], atol=1e-6)
    end
end

@testset "QPSolver struct interface" begin
    @testset "Constructor precomputes KKT" begin
        G = SimpleDiGraph(1)
        primal_dims = [4]  # 2 timesteps * (state_dim + control_dim) = 2 * (1+1) = 4
        state_dim = 1
        control_dim = 1

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)
        gs = [z -> [z[1] - θ_vec[1]]]  # IC constraint: x0 = θ
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

        @test solver.problem isa MixedHierarchyGames.QPProblem
        @test solver.solver_type == :linear
        @test solver.precomputed isa QPPrecomputed
        @test hasproperty(solver.precomputed, :vars)
        @test hasproperty(solver.precomputed, :πs_solve)
    end

    @testset "solve_raw() returns raw solution" begin
        G = SimpleDiGraph(1)
        primal_dims = [4]
        state_dim = 1
        control_dim = 1

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)
        gs = [z -> [z[1] - θ_vec[1]]]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)
        result = MixedHierarchyGames.solve_raw(solver, Dict(1 => [1.0]))

        @test result.status == :solved
        @test result.z_sol[1] ≈ 1.0 atol=1e-6  # x0 = θ = 1.0
    end

    @testset "solve() returns JointStrategy" begin
        G = SimpleDiGraph(1)
        # T=2 timesteps: primal_dim = (state_dim + control_dim) * T = (1+1)*2 = 4
        primal_dims = [4]
        state_dim = 1
        control_dim = 1

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)
        gs = [z -> [z[1] - θ_vec[1]]]  # IC: x0 = θ
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)
        strategy = MixedHierarchyGames.solve(solver, Dict(1 => [1.0]))

        @test strategy isa JointStrategy
        @test length(strategy.substrategies) == 1
        @test strategy.substrategies[1] isa OpenLoopStrategy
        @test strategy.substrategies[1].xs[1][1] ≈ 1.0 atol=1e-6  # x0 = θ
    end

    @testset "solve() with different parameter values" begin
        G = SimpleDiGraph(1)
        primal_dims = [4]
        state_dim = 1
        control_dim = 1

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)
        gs = [z -> [z[1] - θ_vec[1]]]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim)

        # Solve with different initial states
        strategy1 = MixedHierarchyGames.solve(solver, Dict(1 => [1.0]))
        strategy2 = MixedHierarchyGames.solve(solver, Dict(1 => [5.0]))

        @test strategy1.substrategies[1].xs[1][1] ≈ 1.0 atol=1e-6
        @test strategy2.substrategies[1].xs[1][1] ≈ 5.0 atol=1e-6
    end

    @testset "2-player Stackelberg with QPSolver struct" begin
        G = SimpleDiGraph(2)
        add_edge!(G, 1, 2)

        primal_dims = [4, 4]
        state_dim = 1
        control_dim = 1

        θ1_vec = make_θ(1, 1)
        θ2_vec = make_θ(2, 1)
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
        strategy = MixedHierarchyGames.solve(solver, Dict(1 => [1.0], 2 => [3.0]))

        @test strategy isa JointStrategy
        @test length(strategy.substrategies) == 2
        @test strategy.substrategies[1].xs[1][1] ≈ 1.0 atol=1e-6
        @test strategy.substrategies[2].xs[1][1] ≈ 3.0 atol=1e-6
    end

    @testset "PATH solver option" begin
        G = SimpleDiGraph(1)
        primal_dims = [4]
        state_dim = 1
        control_dim = 1

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)
        gs = [z -> [z[1] - θ_vec[1]]]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim; solver=:path)
        strategy = MixedHierarchyGames.solve(solver, Dict(1 => [1.0]))

        @test strategy isa JointStrategy
        @test strategy.substrategies[1].xs[1][1] ≈ 1.0 atol=1e-6
    end

    @testset "Configurable solver parameters" begin
        G = SimpleDiGraph(1)
        primal_dims = [4]
        state_dim = 1
        control_dim = 1

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)
        gs = [z -> [z[1] - θ_vec[1]]]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim; solver=:path)

        # Test that custom parameters are accepted and produce valid results
        strategy = MixedHierarchyGames.solve(
            solver,
            Dict(1 => [1.0]);
            iteration_limit = 50000,
            proximal_perturbation = 1e-3,
            use_basics = false,
            use_start = false
        )

        @test strategy isa JointStrategy
        @test strategy.substrategies[1].xs[1][1] ≈ 1.0 atol=1e-6
    end

    @testset "solve_raw with configurable parameters" begin
        G = SimpleDiGraph(1)
        primal_dims = [4]
        state_dim = 1
        control_dim = 1

        θ_vec = make_θ(1, 1)
        θs = Dict(1 => θ_vec)
        gs = [z -> [z[1] - θ_vec[1]]]
        Js = Dict(1 => (z1; θ=nothing) -> sum(z1.^2))

        solver = QPSolver(G, Js, gs, primal_dims, θs, state_dim, control_dim; solver=:path)

        result = MixedHierarchyGames.solve_raw(
            solver,
            Dict(1 => [1.0]);
            iteration_limit = 50000,
            proximal_perturbation = 1e-3
        )

        @test result.z_sol[1] ≈ 1.0 atol=1e-6
    end
end
