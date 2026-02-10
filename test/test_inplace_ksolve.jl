using Test
using LinearAlgebra: norm, lu, ldiv!
using Graphs: SimpleDiGraph, add_edge!
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
    Test Helpers: Shared problem constructors for inplace K-solve tests
=#

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

#=
    Tests for use_inplace_ksolve flag
=#

@testset "In-place K solve (ldiv!)" begin

    @testset "NonlinearSolver accepts use_inplace_ksolve keyword" begin
        prob = make_two_player_chain_problem_ksolve()

        # Default should be false
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim
        )
        @test solver.options.use_inplace_ksolve == false
    end

    @testset "NonlinearSolver stores use_inplace_ksolve=true" begin
        prob = make_two_player_chain_problem_ksolve()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            use_inplace_ksolve=true
        )
        @test solver.options.use_inplace_ksolve == true
    end

    @testset "compute_K_evals accepts use_inplace_ksolve flag" begin
        prob = make_two_player_chain_problem_ksolve()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        z_current = zeros(length(precomputed.all_variables))

        # Should not throw with use_inplace_ksolve=true
        K_vec_inplace, info_inplace = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_inplace_ksolve=true
        )
        @test info_inplace.status == :ok
    end

    @testset "ldiv! produces identical results to backslash — 2-player" begin
        prob = make_two_player_chain_problem_ksolve()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        z_current = randn(length(precomputed.all_variables))

        # Backslash path (default)
        K_vec_default, info_default = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_inplace_ksolve=false
        )

        # ldiv! path
        K_vec_inplace, info_inplace = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_inplace_ksolve=true
        )

        @test info_default.status == :ok
        @test info_inplace.status == :ok
        @test norm(K_vec_inplace - K_vec_default) / max(norm(K_vec_default), 1.0) < 1e-10
    end

    @testset "ldiv! produces identical results to backslash — 3-player" begin
        prob = make_three_player_chain_problem_ksolve()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        z_current = randn(length(precomputed.all_variables))

        K_vec_default, info_default = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_inplace_ksolve=false
        )

        K_vec_inplace, info_inplace = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_inplace_ksolve=true
        )

        @test info_default.status == :ok
        @test info_inplace.status == :ok
        @test norm(K_vec_inplace - K_vec_default) / max(norm(K_vec_default), 1.0) < 1e-10

        # Check individual K matrices
        for ii in 1:prob.N
            K_d = info_default.K_evals[ii]
            K_i = info_inplace.K_evals[ii]
            if isnothing(K_d)
                @test isnothing(K_i)
            else
                @test norm(K_i - K_d) / max(norm(K_d), 1.0) < 1e-10
            end
        end
    end

    @testset "Solver solutions identical with use_inplace_ksolve — 2-player" begin
        prob = make_two_player_chain_problem_ksolve()
        parameter_values = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        solver_default = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            use_inplace_ksolve=false
        )

        solver_inplace = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            use_inplace_ksolve=true
        )

        result_default = solve_raw(solver_default, parameter_values)
        result_inplace = solve_raw(solver_inplace, parameter_values)

        @test result_default.converged
        @test result_inplace.converged
        @test result_default.sol ≈ result_inplace.sol atol=1e-10
    end

    @testset "Solver solutions identical with use_inplace_ksolve — 3-player" begin
        prob = make_three_player_chain_problem_ksolve()
        parameter_values = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5], 3 => [1.0, 1.0])

        solver_default = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            use_inplace_ksolve=false
        )

        solver_inplace = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            use_inplace_ksolve=true
        )

        result_default = solve_raw(solver_default, parameter_values)
        result_inplace = solve_raw(solver_inplace, parameter_values)

        @test result_default.converged
        @test result_inplace.converged
        @test result_default.sol ≈ result_inplace.sol atol=1e-10
    end

    @testset "solve_raw passes use_inplace_ksolve through" begin
        prob = make_two_player_chain_problem_ksolve()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            use_inplace_ksolve=true
        )

        parameter_values = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])
        result = solve_raw(solver, parameter_values)
        @test result.converged
    end

    @testset "use_inplace_ksolve can be overridden at solve time" begin
        prob = make_two_player_chain_problem_ksolve()

        # Construct with default (false)
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            use_inplace_ksolve=false
        )

        parameter_values = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Override to true at solve time
        result = solve_raw(solver, parameter_values; use_inplace_ksolve=true)
        @test result.converged
    end

    @testset "Handles singular M matrix gracefully with ldiv!" begin
        # Follower with constant cost → singular M
        state_dim = 2
        control_dim = 2
        T = 3
        N = 2

        G = SimpleDiGraph(N)
        add_edge!(G, 1, 2)

        primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
        primal_dims = fill(primal_dim_per_player, N)

        backend = default_backend()
        θs = setup_problem_parameter_variables(fill(state_dim, N); backend)

        J1(z1, z2; θ=nothing) = begin
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((xs[end] .- [1.0, 1.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end

        J2_const(z1, z2; θ=nothing) = 0.0

        Js = Dict(1 => J1, 2 => J2_const)

        function make_dyn(player_idx)
            function dyn(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                constraints = []
                for t in 1:T
                    push!(constraints, xs[t+1] - xs[t] - us[t])
                end
                push!(constraints, xs[1] - θs[player_idx])
                return vcat(constraints...)
            end
            return dyn
        end

        gs = [make_dyn(i) for i in 1:N]

        precomputed = preoptimize_nonlinear_solver(
            G, Js, gs, primal_dims, θs;
            state_dim=state_dim, control_dim=control_dim
        )

        z_current = zeros(length(precomputed.all_variables))

        # Should NOT throw — should handle singular M gracefully
        K_vec, info = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_inplace_ksolve=true
        )

        @test info.status == :singular_matrix
        K2 = info.K_evals[2]
        @test !isnothing(K2)
        @test all(isnan, K2)
    end
end
