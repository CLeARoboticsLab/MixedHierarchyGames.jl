using Test
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: norm
using TimerOutputs: TimerOutput, ncalls, flatten
using MixedHierarchyGames:
    QPSolver,
    NonlinearSolver,
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    solve_qp_linear,
    setup_problem_variables,
    setup_problem_parameter_variables,
    get_qp_kkt_conditions,
    strip_policy_constraints,
    default_backend,
    enable_timing!,
    disable_timing!
using TrajectoryGamesBase: TrajectoryGamesBase, JointStrategy, unflatten_trajectory

#=
    Test Helpers: Small problem for timer testing
=#

"""
Create a minimal 2-player Stackelberg problem for timer tests.
P1 -> P2 (leader-follower), simple integrator dynamics.
"""
function make_timer_test_problem(; T=2, state_dim=2, control_dim=2)
    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    primal_dim_per_player = state_dim * (T + 1) + control_dim * (T + 1)
    primal_dims = fill(primal_dim_per_player, N)

    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

    function J1(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        sum((xs[end] .- [1.0, 1.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J2(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        sum((xs[end] .- [2.0, 2.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    Js = Dict(1 => J1, 2 => J2)

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

@testset "TimerOutputs Integration" begin
    prob = make_timer_test_problem()
    param_values = Dict(1 => [0.0, 0.0], 2 => [1.0, 1.0])

    # Enable timing for tests that verify timing sections are recorded
    enable_timing!()

    @testset "QPSolver construction records timing sections" begin
        to = TimerOutput()
        solver = QPSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            to=to
        )
        # Should have recorded construction timing
        @test haskey(to, "QPSolver construction")
        inner = to["QPSolver construction"]
        @test haskey(inner, "KKT conditions")
        @test haskey(inner, "ParametricMCP build")
        @test haskey(inner, "linearity check")
    end

    @testset "QPSolver solve records timing sections" begin
        to_construct = TimerOutput()
        solver = QPSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            to=to_construct
        )

        to_solve = TimerOutput()
        result = solve(solver, param_values; to=to_solve)

        @test haskey(to_solve, "QPSolver solve")
        inner = to_solve["QPSolver solve"]
        @test haskey(inner, "residual evaluation")
        @test haskey(inner, "Jacobian evaluation")
        @test haskey(inner, "linear solve")
    end

    @testset "NonlinearSolver construction records timing sections" begin
        to = TimerOutput()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            to=to
        )
        @test haskey(to, "NonlinearSolver construction")
        inner = to["NonlinearSolver construction"]
        @test haskey(inner, "variable setup")
        @test haskey(inner, "approximate KKT setup")
        @test haskey(inner, "ParametricMCP build")
        @test haskey(inner, "linear solver init")
    end

    @testset "NonlinearSolver solve records timing sections" begin
        to_construct = TimerOutput()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            to=to_construct
        )

        to_solve = TimerOutput()
        result = solve(solver, param_values; to=to_solve)

        @test haskey(to_solve, "NonlinearSolver solve")
        inner = to_solve["NonlinearSolver solve"]
        @test haskey(inner, "compute K evals")
        @test haskey(inner, "residual evaluation")
        @test haskey(inner, "Jacobian evaluation")
        @test haskey(inner, "Newton step")
        @test haskey(inner, "line search")
    end

    # Disable timing for backward compatibility test
    disable_timing!()

    @testset "Default to (no timer passed) works without error" begin
        # QPSolver should work without to kwarg (backward compatible)
        solver_qp = QPSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )
        result_qp = solve(solver_qp, param_values)
        @test result_qp isa JointStrategy

        # NonlinearSolver should work without to kwarg
        solver_nl = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )
        result_nl = solve(solver_nl, param_values)
        @test result_nl isa JointStrategy
    end

    # Re-enable for accumulation test
    enable_timing!()

    @testset "Single TimerOutput accumulates across construction and solve" begin
        to = TimerOutput()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            to=to
        )
        result = solve(solver, param_values; to=to)

        # Both construction and solve sections should be in the same TimerOutput
        @test haskey(to, "NonlinearSolver construction")
        @test haskey(to, "NonlinearSolver solve")
    end

    # Restore default state
    disable_timing!()
end
