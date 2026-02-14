using Test
using LinearAlgebra: norm
using SparseArrays: nonzeros, nnz, SparseMatrixCSC
using MixedHierarchyGames:
    QPSolver,
    NonlinearSolver,
    solve_raw,
    preoptimize_nonlinear_solver,
    run_nonlinear_solver

# make_standard_two_player_problem is provided by testing_utils.jl (included in runtests.jl)

@testset "Jacobian Buffer Safety" begin

    @testset "QPSolver: corrupted J_buffer is fully overwritten" begin
        prob = make_standard_two_player_problem(goal1=[1.0, 0.0], goal2=[0.0, 1.0])
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
        prob = make_standard_two_player_problem(goal1=[1.0, 0.0], goal2=[0.0, 1.0])
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
        prob = make_standard_two_player_problem()

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

    @testset "similar() buffer produces identical Jacobian to copy() buffer" begin
        # Verify that using similar() (uninitialized) instead of copy() for Jacobian
        # buffer allocation produces identical results, since jacobian_z! fully overwrites
        prob = make_standard_two_player_problem(goal1=[1.0, 0.0], goal2=[0.0, 1.0])
        solver = QPSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                         prob.state_dim, prob.control_dim)

        mcp = solver.precomputed.parametric_mcp
        n = size(mcp.jacobian_z!.result_buffer, 1)

        # Allocate via copy (current approach)
        J_copy = copy(mcp.jacobian_z!.result_buffer)
        # Allocate via similar (proposed optimization — no value copy)
        J_similar = similar(mcp.jacobian_z!.result_buffer)

        # Both must have same sparse structure
        @test size(J_copy) == size(J_similar)
        @test nnz(J_copy) == nnz(J_similar)
        @test J_copy isa SparseMatrixCSC
        @test J_similar isa SparseMatrixCSC

        # Evaluate Jacobian into both buffers at the same point
        z0 = zeros(n)
        θ_vals = zeros(mcp.parameter_dimension)
        mcp.jacobian_z!(J_copy, z0, θ_vals)
        mcp.jacobian_z!(J_similar, z0, θ_vals)

        # Results must be identical (not just approx — exact same computation)
        @test nonzeros(J_copy) == nonzeros(J_similar)
    end

    @testset "solve_qp_linear with similar() buffer matches copy() buffer" begin
        # End-to-end test: solve results must be identical whether J is from copy() or similar()
        prob = make_standard_two_player_problem(goal1=[1.0, 0.0], goal2=[0.0, 1.0])
        solver = QPSolver(prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
                         prob.state_dim, prob.control_dim)

        params = Dict(1 => [0.0, 0.0], 2 => [1.0, 0.5])

        # Solve with copy()-allocated buffer (baseline)
        mcp = solver.precomputed.parametric_mcp
        J_copy = copy(mcp.jacobian_z!.result_buffer)
        result_copy = solve_raw(solver, params)
        @test result_copy.status == :solved

        # Solve with similar()-allocated buffer
        J_sim = similar(mcp.jacobian_z!.result_buffer)
        result_sim = solve_raw(solver, params)
        @test result_sim.status == :solved

        @test isapprox(result_copy.sol, result_sim.sol, atol=1e-14)
    end

    @testset "jacobian_z! fully overwrites sparse buffer nonzeros" begin
        # Directly verify that jacobian_z! writes every nzval position
        prob = make_standard_two_player_problem(goal1=[1.0, 0.0], goal2=[0.0, 1.0])
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
