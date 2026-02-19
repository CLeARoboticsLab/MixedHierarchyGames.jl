#=
    Tests for cubic interpolation linesearch (armijo_interpolation).

    Uses quadratic interpolation after 1 failure, then cubic interpolation
    after 2+ failures (Nocedal & Wright eq 3.57-3.58).

    Usage (standalone): julia --project=. test/test_cubic_linesearch.jl
    Also runs as part of: julia --project=. test/runtests.jl
=#

# Standalone bootstrap: load packages when run directly
if abspath(PROGRAM_FILE) == @__FILE__
    using Test
    using Logging
    using MixedHierarchyGames
    using MixedHierarchyGames: armijo_interpolation, setup_problem_parameter_variables
    using Graphs: SimpleDiGraph, add_edge!
    using TrajectoryGamesBase: unflatten_trajectory
    using LinearAlgebra: norm, dot
    include("testing_utils.jl")
end

@testset "Cubic Interpolation Linesearch" begin
    @testset "Full step accepted for well-scaled descent" begin
        r(x) = x
        x = [2.0]
        d = [-2.0]

        α = armijo_interpolation(r, x, d, 1.0)
        @test α == 1.0
    end

    @testset "Finds good step on known cubic/quartic ϕ" begin
        # r(x) = x.^3, so ϕ(α) = ||r(x+αd)||² involves high-degree terms.
        # Cubic interpolation should handle this well.
        r(x) = x .^ 3
        x = [1.0]
        d = [-10.0]

        α = armijo_interpolation(r, x, d, 1.0)
        @test 0.0 < α < 1.0
        @test norm(r(x .+ α .* d))^2 < norm(r(x))^2
    end

    @testset "Returns zero for ascent direction" begin
        r(x) = x
        x = [1.0]
        d = [1.0]

        α = armijo_interpolation(r, x, d, 1.0; max_iters=5)
        @test α == 0.0
    end

    @testset "Handles negative discriminant in cubic" begin
        # When b²-3aϕ'₀ < 0 in the cubic formula, should fall back gracefully.
        # Use a function that creates a tricky merit function.
        r(x) = x .^ 2
        x = [2.0, 3.0]
        d = [-1.0, -1.5]

        α = armijo_interpolation(r, x, d, 1.0)
        @test α > 0.0
        f_x = r(x)
        f_new = r(x .+ α .* d)
        @test dot(f_new, f_new) <= dot(f_x, f_x) + 1e-4 * α * (-2 * dot(f_x, f_x))
    end

    @testset "x_buffer gives same result as allocating" begin
        r(x) = x .^ 3
        x = [1.0, 2.0]
        d = [-5.0, -10.0]
        x_buffer = similar(x)

        α_alloc = armijo_interpolation(r, x, d, 1.0)
        α_buf = armijo_interpolation(r, x, d, 1.0; x_buffer)

        @test α_buf == α_alloc
        @test α_buf > 0.0
    end

    @testset "Respects alpha_init parameter" begin
        r(x) = x
        x = [2.0]
        d = [-2.0]
        alpha_init = 0.25

        α = armijo_interpolation(r, x, d, alpha_init)
        @test α <= alpha_init
        @test α > 0.0
    end

    @testset "Multidimensional problem" begin
        r(x) = x
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = -x

        α = armijo_interpolation(r, x, d, 1.0)
        @test α == 1.0
        @test norm(r(x .+ α .* d)) < 1e-14
    end

    @testset "Solver convergence with :armijo_interp method" begin
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:armijo_interp, tol=1e-8
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
        solver_interp = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:armijo_interp, tol=1e-10
        )

        result_geo = solve_raw(solver_geo, params)
        result_interp = solve_raw(solver_interp, params)

        @test result_geo.converged
        @test result_interp.converged
        @test norm(result_geo.sol - result_interp.sol) < 1e-6
    end

    @testset "Warning on failure" begin
        r(x) = x
        x = [1.0]
        d = [1.0]

        logs, _ = Test.collect_test_logs() do
            armijo_interpolation(r, x, d, 1.0; max_iters=3)
        end
        @test length(logs) >= 1
        @test logs[1].level == Logging.Warn
    end
end
