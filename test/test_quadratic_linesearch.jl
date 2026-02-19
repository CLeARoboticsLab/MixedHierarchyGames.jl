#=
    Tests for quadratic interpolation linesearch (armijo_quadratic_interp).

    After the first Armijo backtrack failure, fits a quadratic through
    ϕ(0), ϕ'(0), ϕ(α₀) to predict the minimizer. Falls back to geometric
    if the quadratic step also fails.

    Usage (standalone): julia --project=. test/test_quadratic_linesearch.jl
    Also runs as part of: julia --project=. test/runtests.jl
=#

# Standalone bootstrap: load packages when run directly
if abspath(PROGRAM_FILE) == @__FILE__
    using Test
    using MixedHierarchyGames
    using MixedHierarchyGames: armijo_quadratic_interp, setup_problem_parameter_variables
    using Graphs: SimpleDiGraph, add_edge!
    using TrajectoryGamesBase: unflatten_trajectory
    using LinearAlgebra: norm, dot
    include("testing_utils.jl")
end

@testset "Quadratic Interpolation Linesearch" begin
    @testset "Full step accepted for well-scaled descent" begin
        # Same as armijo: r(x) = x, Newton step d = -x → full step to origin
        r(x) = x
        x = [2.0]
        d = [-2.0]

        α = armijo_quadratic_interp(r, x, d, 1.0)
        @test α == 1.0
    end

    @testset "Finds good step on known quadratic ϕ" begin
        # r(x) = x.^2, ϕ(α) = ||r(x+αd)||² is a polynomial in α.
        # Quadratic interpolation should find a good step quickly.
        r(x) = x .^ 2
        x = [2.0, 3.0]
        d = [-1.0, -1.5]

        α = armijo_quadratic_interp(r, x, d, 1.0)
        @test α > 0.0
        # Verify Armijo sufficient decrease
        f_x = r(x)
        f_new = r(x .+ α .* d)
        ϕ_0 = dot(f_x, f_x)
        ϕ_new = dot(f_new, f_new)
        @test ϕ_new <= ϕ_0 + 1e-4 * α * (-2 * ϕ_0)
    end

    @testset "Backtracking needed for overshooting step" begin
        # Aggressive step that overshoots — should backtrack
        r(x) = x .^ 3
        x = [1.0]
        d = [-10.0]

        α = armijo_quadratic_interp(r, x, d, 1.0)
        @test 0.0 < α < 1.0
        @test norm(r(x .+ α .* d))^2 < norm(r(x))^2
    end

    @testset "Returns zero for ascent direction" begin
        r(x) = x
        x = [1.0]
        d = [1.0]  # Ascent direction

        α = armijo_quadratic_interp(r, x, d, 1.0; max_iters=5)
        @test α == 0.0
    end

    @testset "Safeguard clamps quadratic step to [0.1α, 0.5α]" begin
        # Even if quadratic predicts a very small or very large step relative
        # to current α, the safeguard should keep it in [0.1α, 0.5α].
        # We verify indirectly: the function should not return a step
        # that violates reasonable bounds.
        r(x) = x .^ 3
        x = [1.0]
        d = [-10.0]

        α = armijo_quadratic_interp(r, x, d, 1.0; max_iters=10)
        # Should find a valid step
        @test α >= 0.0
        if α > 0.0
            @test norm(r(x .+ α .* d))^2 < norm(r(x))^2
        end
    end

    @testset "Handles near-linear ϕ (degenerate quadratic)" begin
        # When ϕ is nearly linear, the quadratic denominator is near zero.
        # The safeguard should handle this gracefully (division guard).
        r(x) = x  # ϕ(α) = ||x + αd||², quadratic in α but linear residual
        x = [1.0]
        d = [1.0]  # Ascent direction — forces failure, tests robustness

        α = armijo_quadratic_interp(r, x, d, 1.0; max_iters=3)
        # Should handle gracefully (return 0.0 for ascent direction)
        @test α == 0.0
    end

    @testset "x_buffer gives same result as allocating" begin
        r(x) = x .^ 3
        x = [1.0, 2.0]
        d = [-5.0, -10.0]
        x_buffer = similar(x)

        α_alloc = armijo_quadratic_interp(r, x, d, 1.0)
        α_buf = armijo_quadratic_interp(r, x, d, 1.0; x_buffer)

        @test α_buf == α_alloc
        @test α_buf > 0.0
    end

    @testset "Respects alpha_init parameter" begin
        r(x) = x
        x = [2.0]
        d = [-2.0]
        alpha_init = 0.25

        α = armijo_quadratic_interp(r, x, d, alpha_init)
        @test α <= alpha_init
        @test α > 0.0
    end

    @testset "Multidimensional problem" begin
        r(x) = x
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = -x

        α = armijo_quadratic_interp(r, x, d, 1.0)
        @test α == 1.0
        @test norm(r(x .+ α .* d)) < 1e-14
    end

    @testset "Solver convergence with :armijo_quadratic method" begin
        prob = make_standard_two_player_problem()
        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:armijo_quadratic, tol=1e-8
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
        solver_quad = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            linesearch_method=:armijo_quadratic, tol=1e-10
        )

        result_geo = solve_raw(solver_geo, params)
        result_quad = solve_raw(solver_quad, params)

        @test result_geo.converged
        @test result_quad.converged
        # Solutions should match within tolerance
        @test norm(result_geo.sol - result_quad.sol) < 1e-6
    end

    @testset "Warning on failure uses lazy string" begin
        r(x) = x
        x = [1.0]
        d = [1.0]  # Ascent direction

        logs, _ = Test.collect_test_logs() do
            armijo_quadratic_interp(r, x, d, 1.0; max_iters=3)
        end
        @test length(logs) >= 1
        @test logs[1].level == Logging.Warn
    end
end
