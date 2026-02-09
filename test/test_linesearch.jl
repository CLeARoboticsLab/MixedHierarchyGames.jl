using Test
using LinearAlgebra: norm
using MixedHierarchyGames: armijo_backtracking

@testset "Armijo Backtracking Linesearch" begin
    @testset "Full step accepted for well-scaled descent" begin
        # Minimize f(x) = x^2 via Newton-like step from x = [2.0]
        # Residual r(x) = x (so ||r||^2 = x^2), Newton step d = -x
        r(x) = x
        x = [2.0]
        d = [-2.0]  # Full Newton step toward zero

        α = armijo_backtracking(r, x, d, 1.0)

        # Full step lands at origin — should be accepted at α = 1.0
        @test α == 1.0
    end

    @testset "Backtracking needed for overshooting step" begin
        # r(x) = x.^3, so ||r||^2 = sum(x.^6)
        # At x = [1.0], an aggressive step d = [-10.0] overshoots
        r(x) = x .^ 3
        x = [1.0]
        d = [-10.0]  # Way too aggressive

        α = armijo_backtracking(r, x, d, 1.0)

        # Should backtrack to a smaller step size
        @test 0.0 < α < 1.0
        # Verify the accepted step actually decreases the merit
        @test norm(r(x .+ α .* d))^2 < norm(r(x))^2
    end

    @testset "Returns zero for ascent direction" begin
        # r(x) = x, d = [1.0] is an ascent direction from x = [1.0]
        # No α > 0 satisfies Armijo condition since we're going uphill
        r(x) = x
        x = [1.0]
        d = [1.0]  # Ascent direction (moving away from zero)

        α = armijo_backtracking(r, x, d, 1.0; max_iters=5)

        # Should signal failure by returning 0.0
        @test α == 0.0
    end

    @testset "Armijo sufficient decrease condition holds" begin
        # Verify the returned α satisfies:
        #   ||r(x + α*d)||^2 ≤ ||r(x)||^2 + c1 * α * (-2 * ||r(x)||^2)
        r(x) = x .^ 2
        x = [2.0, 3.0]
        d = [-1.0, -1.5]
        c1 = 1e-4

        α = armijo_backtracking(r, x, d, 1.0; c1=c1)

        ϕ_0 = norm(r(x))^2
        ϕ_new = norm(r(x .+ α .* d))^2
        @test ϕ_new <= ϕ_0 + c1 * α * (-2 * ϕ_0)
    end

    @testset "Respects alpha_init parameter" begin
        # Starting from α_init < 1.0 should limit the maximum step
        r(x) = x
        x = [2.0]
        d = [-2.0]
        alpha_init = 0.25

        α = armijo_backtracking(r, x, d, alpha_init)

        # Cannot exceed the initial step size
        @test α <= alpha_init
        # Should still accept this step (descent direction)
        @test α > 0.0
    end

    @testset "Rho controls backtracking rate" begin
        # With rho=0.1 (aggressive shrinkage), we should backtrack faster
        r(x) = x .^ 3
        x = [1.0]
        d = [-10.0]

        α_slow = armijo_backtracking(r, x, d, 1.0; rho=0.9)
        α_fast = armijo_backtracking(r, x, d, 1.0; rho=0.1)

        # Faster shrinkage should yield smaller or equal step
        @test α_fast <= α_slow
    end

    @testset "Multidimensional problem" begin
        # r(x) = x for a 5D vector
        r(x) = x
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = -x  # Newton step toward origin

        α = armijo_backtracking(r, x, d, 1.0)

        @test α == 1.0
        @test norm(r(x .+ α .* d)) < 1e-14
    end
end
