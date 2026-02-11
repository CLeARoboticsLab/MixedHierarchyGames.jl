using Test
using Graphs: SimpleDiGraph, add_edge!, nv
using LinearAlgebra: norm
using MixedHierarchyGames:
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    compute_K_evals,
    setup_problem_parameter_variables,
    default_backend,
    NonlinearSolver,
    solve_raw

using TrajectoryGamesBase: unflatten_trajectory

#=
    Test Helpers: Reuse the same problem factories as test_inplace_strategies.jl
=#

"""
Create a simple 2-player chain hierarchy game for testing.
P1 -> P2 (P1 is leader, P2 is follower)
"""
function make_two_player_chain_problem_ksolve(; T=3, state_dim=2, control_dim=2)
    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

    function J1(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        goal = [1.0, 1.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J2(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        goal = [2.0, 2.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
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

"""
Create a simple 3-player chain hierarchy game for testing.
P1 -> P2 -> P3 (P1 leads P2, P2 leads P3)
"""
function make_three_player_chain_problem_ksolve(; T=3, state_dim=2, control_dim=2)
    N = 3
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)

    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    backend = default_backend()
    θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

    function J1(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        goal = [1.0, 1.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J2(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        goal = [2.0, 2.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J3(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z3, state_dim, control_dim)
        goal = [3.0, 3.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    Js = Dict(1 => J1, 2 => J2, 3 => J3)

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

@testset "In-place K Solve via ldiv! (Strategy B)" begin

    @testset "compute_K_evals: inplace_ldiv matches backslash (2-player)" begin
        prob = make_two_player_chain_problem_ksolve()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        z_current = zeros(length(precomputed.all_variables))
        all_K_default, info_default = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info
        )
        all_K_ldiv, info_ldiv = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            inplace_ldiv=true
        )

        # Results must match to high precision
        @test all_K_default ≈ all_K_ldiv atol=1e-12
        @test info_default.status == info_ldiv.status

        # Check each player's K matrix individually
        for ii in 1:prob.N
            if info_default.K_evals[ii] !== nothing
                @test info_default.K_evals[ii] ≈ info_ldiv.K_evals[ii] atol=1e-12
            end
        end
    end

    @testset "compute_K_evals: inplace_ldiv matches backslash (3-player chain)" begin
        prob = make_three_player_chain_problem_ksolve()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        z_current = randn(length(precomputed.all_variables))
        all_K_default, info_default = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info
        )
        all_K_ldiv, info_ldiv = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            inplace_ldiv=true
        )

        @test all_K_default ≈ all_K_ldiv atol=1e-12
        @test info_default.status == info_ldiv.status

        for ii in 1:prob.N
            if info_default.K_evals[ii] !== nothing
                @test info_default.K_evals[ii] ≈ info_ldiv.K_evals[ii] atol=1e-12
            end
        end
    end

    @testset "compute_K_evals: inplace_ldiv repeated calls reuse K buffers" begin
        prob = make_two_player_chain_problem_ksolve()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        for trial in 1:5
            z_current = randn(length(precomputed.all_variables))
            all_K_default, _ = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info
            )
            all_K_ldiv, _ = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info;
                inplace_ldiv=true
            )
            @test all_K_default ≈ all_K_ldiv atol=1e-12
        end
    end

    @testset "compute_K_evals: inplace_ldiv + inplace_MN combined" begin
        # Both flags should work together
        prob = make_two_player_chain_problem_ksolve()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        z_current = randn(length(precomputed.all_variables))
        all_K_default, info_default = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info
        )
        all_K_both, info_both = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            inplace_MN=true, inplace_ldiv=true
        )

        @test all_K_default ≈ all_K_both atol=1e-12
        @test info_default.status == info_both.status
    end

    @testset "run_nonlinear_solver: inplace_ldiv produces identical solution (2-player)" begin
        prob = make_two_player_chain_problem_ksolve()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0])

        result_default = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=100, tol=1e-6
        )
        result_ldiv = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=100, tol=1e-6, inplace_ldiv=true
        )

        @test result_default.converged
        @test result_ldiv.converged
        @test result_default.sol ≈ result_ldiv.sol atol=1e-10
        @test result_default.iterations == result_ldiv.iterations
    end

    @testset "run_nonlinear_solver: inplace_ldiv produces identical solution (3-player chain)" begin
        prob = make_three_player_chain_problem_ksolve()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0], 3 => [0.0, 0.0])

        result_default = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=100, tol=1e-6
        )
        result_ldiv = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=100, tol=1e-6, inplace_ldiv=true
        )

        @test result_default.converged
        @test result_ldiv.converged
        @test result_default.sol ≈ result_ldiv.sol atol=1e-10
        @test result_default.iterations == result_ldiv.iterations
    end

    @testset "NonlinearSolver: inplace_ldiv option threads through" begin
        prob = make_two_player_chain_problem_ksolve()

        # Construct solver with inplace_ldiv=true
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            inplace_ldiv=true
        )
        @test solver.options.inplace_ldiv == true

        # Solve and verify convergence
        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0])
        result = solve_raw(solver, initial_states)
        @test result.converged

        # Compare with default (inplace_ldiv=false)
        solver_default = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )
        result_default = solve_raw(solver_default, initial_states)
        @test result.sol ≈ result_default.sol atol=1e-10
    end

    @testset "solve_raw: inplace_ldiv override at solve time" begin
        prob = make_two_player_chain_problem_ksolve()

        # Construct with default (inplace_ldiv=false)
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.0, 0.0])

        # Override at solve time
        result = solve_raw(solver, initial_states; inplace_ldiv=true)
        @test result.converged

        # Compare with default
        result_default = solve_raw(solver, initial_states)
        @test result.sol ≈ result_default.sol atol=1e-10
    end

end
