#=
    Tests for two-phase linesearch (:twophase method).

    Scout phase uses cheap residual (stale K) to narrow α range,
    then verify phase uses full K recomputation at the promising candidate.

    Usage (standalone): julia --project=. test/test_twophase_linesearch.jl
    Also runs as part of: julia --project=. test/runtests.jl
=#

# Standalone bootstrap: load packages when run directly
if abspath(PROGRAM_FILE) == @__FILE__
    using Test
    using MixedHierarchyGames
    using MixedHierarchyGames: setup_problem_parameter_variables
    using Graphs: SimpleDiGraph, add_edge!
    using TrajectoryGamesBase: unflatten_trajectory
    using LinearAlgebra: norm
    include("testing_utils.jl")
end

@testset "Two-phase Linesearch" begin
    @testset "Solver convergence with :twophase method" begin
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:twophase, tol=1e-8
        )
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = solve_raw(solver, params)
        @test result.converged
        @test result.residual < 1e-8
    end

    @testset "Same solution as geometric method" begin
        prob = make_standard_two_player_problem()
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        solver_geo = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:geometric, tol=1e-10
        )
        solver_twophase = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:twophase, tol=1e-10
        )

        result_geo = solve_raw(solver_geo, params)
        result_twophase = solve_raw(solver_twophase, params)

        @test result_geo.converged
        @test result_twophase.converged
        @test norm(result_geo.sol - result_twophase.sol) < 1e-6
    end

    @testset "All step sizes bounded by 1.0" begin
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:twophase
        )
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        step_sizes = Float64[]
        callback = info -> push!(step_sizes, info.step_size)

        result = solve_raw(solver, params; callback=callback)
        @test result.converged

        for α in step_sizes
            @test α <= 1.0
        end
    end

    @testset "Works with recompute_policy_in_linesearch=false" begin
        # When K is not recomputed in linesearch, cheap and full evals are the same
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:twophase,
            recompute_policy_in_linesearch=false,
            tol=1e-8
        )
        params = Dict(1 => [0.0, 0.0], 2 => [0.5, 0.5])

        result = solve_raw(solver, params)
        @test result.converged
        @test result.residual < 1e-8
    end

    @testset "Works with different initial conditions" begin
        # Test with non-zero initial states that may require more iterations
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:twophase, tol=1e-8
        )

        for offset in [0.0, 0.5, 1.0, 2.0]
            params = Dict(1 => [offset, offset], 2 => [offset + 0.5, offset + 0.5])
            result = solve_raw(solver, params)
            @test result.converged
            @test result.residual < 1e-8
        end
    end
end
