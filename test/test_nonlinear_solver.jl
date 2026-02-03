using Test
using Graphs: SimpleDiGraph, add_edge!
using LinearAlgebra: norm
using MixedHierarchyGames: preoptimize_nonlinear_solver, run_nonlinear_solver, compute_K_evals

@testset "Nonlinear Solver Preoptimization" begin
    @testset "Returns compiled evaluation functions" begin
        # preoptimize should return functions for:
        # - KKT evaluation
        # - Jacobian evaluation
        # - M, N matrix computation
        @test_skip "Implement preoptimize_nonlinear_solver first"
    end

    @testset "Precomputed functions are callable" begin
        # The returned functions should accept numerical inputs
        @test_skip "Implement preoptimize_nonlinear_solver first"
    end
end

@testset "K Matrix Evaluation" begin
    @testset "Returns K matrices per player" begin
        # compute_K_evals should return Dict{Int, Matrix}
        @test_skip "Implement compute_K_evals first"
    end

    @testset "K matrices computed in reverse topological order" begin
        # Followers computed before leaders (bottom-up)
        @test_skip "Implement compute_K_evals first"
    end

    @testset "K dimensions match player info structure" begin
        # K maps from information to decision: u = K * y
        @test_skip "Implement compute_K_evals first"
    end
end

@testset "Nonlinear Solver Convergence" begin
    @testset "Converges on simple LQ problem" begin
        # On a linear-quadratic problem, nonlinear solver should converge
        # to same solution as QP solver
        @test_skip "Implement run_nonlinear_solver first"
    end

    @testset "Convergence within tolerance" begin
        # Final KKT residual should be below specified tolerance
        @test_skip "Implement run_nonlinear_solver first"
    end

    @testset "Returns convergence info" begin
        # Should return: converged flag, iterations, final residual
        @test_skip "Implement run_nonlinear_solver first"
    end
end

@testset "Nonlinear vs QP Solver Comparison" begin
    @testset "Same solution on LQ problem" begin
        # On a pure LQ problem, both solvers should find same solution
        # (within numerical tolerance)
        @test_skip "Implement both solvers first"
    end
end
