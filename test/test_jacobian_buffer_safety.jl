using Test
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: norm
using SparseArrays: nonzeros
using MixedHierarchyGames:
    QPSolver,
    NonlinearSolver,
    solve_raw,
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    setup_problem_parameter_variables,
    default_backend

using TrajectoryGamesBase: unflatten_trajectory

@testset "Jacobian Buffer Safety" begin

    # Shared problem constructors (same as test_allocation_optimization.jl)
    function make_qp_problem()
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

        function make_dynamics(player_idx)
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

        gs = [make_dynamics(i) for i in 1:2]
        return (; G, Js, gs, primal_dims, θs, state_dim, control_dim)
    end

    function make_nonlinear_problem()
        N = 2
        G = SimpleDiGraph(N)
        add_edge!(G, 1, 2)

        state_dim = 2
        control_dim = 2
        T = 3
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

        function make_dynamics(player_idx)
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

        gs = [make_dynamics(i) for i in 1:N]
        return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N)
    end

    @testset "QPSolver: corrupted J_buffer is fully overwritten" begin
        prob = make_qp_problem()
        solver = QPSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                         prob.state_dim, prob.control_dim)

        params = Dict(1 => [0.0, 0.0], 2 => [1.0, 0.5])

        # Get baseline result
        result_clean = solve_raw(solver, params)
        @test result_clean.status == :solved

        # Corrupt the Jacobian buffer with large garbage values
        J_buf = solver.precomputed.J_buffer
        nzvals = nonzeros(J_buf)
        fill!(nzvals, 1e10)

        # Also corrupt F_buffer and z0_buffer
        fill!(solver.precomputed.F_buffer, 1e10)
        fill!(solver.precomputed.z0_buffer, 1e10)

        # Solve again — must produce identical results despite corrupted buffers
        result_corrupted = solve_raw(solver, params)
        @test result_corrupted.status == :solved
        @test isapprox(result_clean.sol, result_corrupted.sol, atol=1e-12)
    end

    @testset "QPSolver: corrupted buffers with varying parameters" begin
        prob = make_qp_problem()
        solver = QPSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                         prob.state_dim, prob.control_dim)

        param_sets = [
            Dict(1 => [0.0, 0.0], 2 => [1.0, 0.5]),
            Dict(1 => [0.5, -0.3], 2 => [-1.0, 0.8]),
            Dict(1 => [2.0, 1.0], 2 => [0.0, 0.0]),
        ]

        # Collect baseline results (clean buffers reset by fill! internally)
        baseline_results = [solve_raw(solver, p) for p in param_sets]
        for r in baseline_results
            @test r.status == :solved
        end

        # Now corrupt buffers and re-solve in different order
        for (i, p) in enumerate(reverse(param_sets))
            J_buf = solver.precomputed.J_buffer
            fill!(nonzeros(J_buf), 9.99e15)
            fill!(solver.precomputed.F_buffer, -Inf)
            fill!(solver.precomputed.z0_buffer, NaN)

            result = solve_raw(solver, p)
            # Compare to corresponding baseline (reversed index)
            baseline = baseline_results[length(param_sets) - i + 1]
            @test result.status == :solved
            @test isapprox(baseline.sol, result.sol, atol=1e-12)
        end
    end

    @testset "NonlinearSolver: corrupted ∇F buffer is fully overwritten" begin
        prob = make_nonlinear_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim
        )

        initial_states = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        # Get baseline result
        result_clean = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=50, tol=1e-8, verbose=false
        )
        @test result_clean.converged

        # Corrupt the Jacobian result_buffer in the MCP object
        # This is the template from which ∇F is copied
        jac_buf = precomputed.mcp_obj.jacobian_z!.result_buffer
        fill!(nonzeros(jac_buf), 1e10)

        # Solve again — the ∇F buffer is allocated fresh via copy() each call,
        # but the result_buffer template is shared
        result_corrupted = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=50, tol=1e-8, verbose=false
        )
        @test result_corrupted.converged
        @test isapprox(result_clean.sol, result_corrupted.sol, atol=1e-10)
        @test result_clean.iterations == result_corrupted.iterations
    end

    @testset "jacobian_z! fully overwrites sparse buffer nonzeros" begin
        # Directly verify that jacobian_z! writes every nzval position
        prob = make_qp_problem()
        solver = QPSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                         prob.state_dim, prob.control_dim)

        mcp = solver.precomputed.parametric_mcp
        J = copy(mcp.jacobian_z!.result_buffer)
        n = size(J, 1)

        # Fill with sentinel value
        sentinel = 1.23456789e42
        fill!(nonzeros(J), sentinel)

        # Evaluate Jacobian at a known point
        z0 = zeros(n)
        θ_vals = zeros(mcp.parameter_dimension)
        mcp.jacobian_z!(J, z0, θ_vals)

        # Verify no sentinel values remain in nzval
        nzvals = nonzeros(J)
        @test all(v -> v != sentinel, nzvals)
        @test all(isfinite, nzvals)
    end
end
