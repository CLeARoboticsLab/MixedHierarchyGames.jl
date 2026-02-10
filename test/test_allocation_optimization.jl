using Test
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: norm
using MixedHierarchyGames:
    QPSolver,
    NonlinearSolver,
    solve,
    solve_raw,
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    setup_problem_parameter_variables,
    default_backend

using TrajectoryGamesBase: unflatten_trajectory, JointStrategy

# Helper: create a 2-player Stackelberg problem for QPSolver testing
function make_qp_test_problem()
    G = SimpleDiGraph(2)
    add_edge!(G, 1, 2)

    state_dim = 2
    control_dim = 2
    T = 3
    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = [primal_dim, primal_dim]

    θs = setup_problem_parameter_variables([state_dim, state_dim])

    Js = Dict(
        1 => (z1, z2; θ=nothing) -> begin
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((xs[end] .- [1.0, 0.0]).^2) + 0.1 * sum(sum(u.^2) for u in us)
        end,
        2 => (z1, z2; θ=nothing) -> begin
            (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
            sum((xs[end] .- [0.0, 1.0]).^2) + 0.1 * sum(sum(u.^2) for u in us)
        end,
    )

    function make_qp_dynamics(player_idx)
        function dynamics_constraint(z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                push!(constraints, xs[t+1] - xs[t] - us[t])
            end
            push!(constraints, xs[1] - θs[player_idx])
            return vcat(constraints...)
        end
        return dynamics_constraint
    end

    gs = [make_qp_dynamics(i) for i in 1:2]

    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim)
end

# Helper: create a 2-player nonlinear problem for NonlinearSolver testing
function make_nonlinear_test_problem(; T=3, state_dim=2, control_dim=2)
    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    primal_dim = (state_dim + control_dim) * (T + 1)
    primal_dims = fill(primal_dim, N)

    θs = setup_problem_parameter_variables(fill(state_dim, N))

    Js = Dict(
        1 => (z1, z2; θ=nothing) -> begin
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((xs[end] .- [1.0, 1.0]).^2) + 0.1 * sum(sum(u.^2) for u in us)
        end,
        2 => (z1, z2; θ=nothing) -> begin
            (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
            sum((xs[end] .- [2.0, 2.0]).^2) + 0.1 * sum(sum(u.^2) for u in us)
        end,
    )

    function make_dynamics_constraint(player_idx)
        function dynamics_constraint(z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                push!(constraints, xs[t+1] - xs[t] - us[t])
            end
            push!(constraints, xs[1] - θs[player_idx])
            return vcat(constraints...)
        end
        return dynamics_constraint
    end

    gs = [make_dynamics_constraint(i) for i in 1:N]

    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N)
end

@testset "Allocation Optimization - Correctness" begin

    @testset "QPSolver: repeated solves produce identical results" begin
        prob = make_qp_test_problem()
        solver = QPSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                         prob.state_dim, prob.control_dim)

        params1 = Dict(1 => [0.0, 0.0], 2 => [1.0, 0.5])
        params2 = Dict(1 => [0.5, -0.3], 2 => [-1.0, 0.8])
        params3 = Dict(1 => [0.0, 0.0], 2 => [1.0, 0.5])  # Same as params1

        # Solve three times - first and third should be identical
        result1 = solve_raw(solver, params1)
        result2 = solve_raw(solver, params2)
        result3 = solve_raw(solver, params3)

        @test result1.status == :solved
        @test result2.status == :solved
        @test result3.status == :solved

        # First and third solve with same params must produce identical results
        @test isapprox(result1.sol, result3.sol, atol=1e-14)

        # Second solve with different params should produce different results
        @test !isapprox(result1.sol, result2.sol, atol=1e-2)
    end

    @testset "QPSolver: solve buffers do not corrupt across calls" begin
        prob = make_qp_test_problem()
        solver = QPSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                         prob.state_dim, prob.control_dim)

        # Solve many times to exercise buffer reuse
        results = []
        param_sets = [
            Dict(1 => [Float64(i), Float64(-i)], 2 => [Float64(i+1), Float64(i-1)])
            for i in 1:10
        ]

        for params in param_sets
            r = solve_raw(solver, params)
            push!(results, r)
            @test r.status == :solved
        end

        # Re-solve first params and verify identical result
        r_check = solve_raw(solver, param_sets[1])
        @test isapprox(results[1].sol, r_check.sol, atol=1e-14)
    end

    @testset "NonlinearSolver: repeated solves produce identical results" begin
        prob = make_nonlinear_test_problem()
        solver = NonlinearSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                                prob.state_dim, prob.control_dim;
                                max_iters=100, tol=1e-8)

        params1 = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])
        params2 = Dict(1 => [1.0, -1.0], 2 => [0.0, 0.0])
        params3 = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])  # Same as params1

        result1 = solve_raw(solver, params1)
        result2 = solve_raw(solver, params2)
        result3 = solve_raw(solver, params3)

        @test result1.converged
        @test result3.converged

        # Same params should produce identical results
        @test isapprox(result1.sol, result3.sol, atol=1e-10)
        @test result1.iterations == result3.iterations
    end

    @testset "NonlinearSolver: line search buffer reuse correctness" begin
        prob = make_nonlinear_test_problem()
        solver = NonlinearSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                                prob.state_dim, prob.control_dim;
                                max_iters=50, tol=1e-8, use_armijo=true)

        # Multiple solves to test line search buffer reuse
        param_sets = [
            Dict(1 => [Float64(i)*0.1, Float64(-i)*0.1], 2 => [Float64(i+1)*0.1, Float64(i)*0.1])
            for i in 1:5
        ]

        results = [solve_raw(solver, p) for p in param_sets]

        # Re-solve first and verify identical
        r_check = solve_raw(solver, param_sets[1])
        @test isapprox(results[1].sol, r_check.sol, atol=1e-10)
        @test results[1].iterations == r_check.iterations
    end

    @testset "NonlinearSolver: param_vec buffer reuse correctness" begin
        prob = make_nonlinear_test_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Run solver twice with same inputs
        result1 = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=50, tol=1e-8, verbose=false
        )
        result2 = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=50, tol=1e-8, verbose=false
        )

        @test result1.converged
        @test result2.converged
        @test isapprox(result1.sol, result2.sol, atol=1e-10)
        @test result1.iterations == result2.iterations
        @test isapprox(result1.residual, result2.residual, atol=1e-14)
    end

    @testset "QPSolver: solve returns JointStrategy with correct structure" begin
        prob = make_qp_test_problem()
        solver = QPSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                         prob.state_dim, prob.control_dim)

        params = Dict(1 => [0.0, 0.0], 2 => [1.0, 0.5])
        strategy = solve(solver, params)

        @test strategy isa JointStrategy
        @test length(strategy.substrategies) == 2

        # Initial states should match parameters
        @test isapprox(strategy.substrategies[1].xs[1], [0.0, 0.0], atol=1e-6)
        @test isapprox(strategy.substrategies[2].xs[1], [1.0, 0.5], atol=1e-6)
    end

    @testset "NonlinearSolver: solve returns JointStrategy with correct structure" begin
        prob = make_nonlinear_test_problem()
        solver = NonlinearSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                                prob.state_dim, prob.control_dim;
                                max_iters=100, tol=1e-8)

        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])
        strategy = solve(solver, params)

        @test strategy isa JointStrategy
        @test length(strategy.substrategies) == 2

        # Initial states should match parameters
        @test isapprox(strategy.substrategies[1].xs[1], [0.0, 0.0], atol=1e-6)
        @test isapprox(strategy.substrategies[2].xs[1], [0.5, 0.5], atol=1e-6)
    end
end
