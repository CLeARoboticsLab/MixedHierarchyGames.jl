using Test
using LinearAlgebra: norm
using MixedHierarchyGames: armijo_backtracking, geometric_reduction, constant_step

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

@testset "Geometric Reduction Linesearch" begin
    @testset "Full step accepted when it decreases merit" begin
        # r(x) = x, Newton step d = -x lands at origin
        r(x) = x
        x = [2.0]
        d = [-2.0]

        α = geometric_reduction(r, x, d, 1.0)

        # Full step gives ||r(0)||^2 = 0 < ||r(2)||^2 = 4
        @test α == 1.0
    end

    @testset "Reduction sequence follows α = alpha_init * rho^k" begin
        # Track which alphas are evaluated by using a function that
        # only accepts a specific step size
        calls = Float64[]
        r(x) = (push!(calls, x[1]); x)
        x = [1.0]
        d = [-1.0]

        # With rho=0.5, sequence is: 1.0, 0.5, 0.25, 0.125, ...
        # r(x) = x, so any step toward zero decreases merit.
        # First trial x + 1.0*d = 0.0 should be accepted immediately.
        α = geometric_reduction(r, x, d, 1.0; rho=0.5)
        @test α == 1.0

        # Now test that backtracking produces the geometric sequence
        # Use a function where only small steps decrease merit
        function r_hard(x)
            # Merit ||r||^2 = x^2, but add a bump: if |x| > 0.6 and x < 1, merit is huge
            val = x[1]
            if 0.3 < abs(val) < 0.8
                return [100.0]  # Artificial bump
            end
            return x
        end
        x2 = [1.0]
        d2 = [-1.0]
        # alpha_init=1.0, rho=0.5: trials at x=0.0, 0.5, 0.75, 0.875, ...
        # x=0.0 → r=[0] → ||r||^2=0 < 1 ✓ (accepted at α=1.0)
        # But wait, x + 1.0*(-1) = 0.0, which is outside the bump.
        # Need a case where first trial fails. Use a large overshoot.
        function r_overshoot(x)
            # ||r||^2 increases when x goes negative
            return [x[1]^2 - 1.0]  # zero at x=1, x=-1
        end
        x3 = [2.0]
        d3 = [-10.0]  # Full step lands at x=-8, ||r||^2 = 63^2 > ||r(2)||^2 = 3^2
        α3 = geometric_reduction(r_overshoot, x3, d3, 1.0; rho=0.5, max_iters=20)

        # Should backtrack. Check it's a power of 0.5
        if α3 > 0.0
            k = round(Int, log(α3) / log(0.5))
            @test α3 ≈ 0.5^k atol = 1e-14
        end
        @test 0.0 < α3 < 1.0
    end

    @testset "Returns zero when no step decreases merit" begin
        # r(x) = x with ascent direction: every step increases merit
        r(x) = x
        x = [1.0]
        d = [1.0]  # Moving away from zero

        α = geometric_reduction(r, x, d, 1.0; max_iters=5)

        @test α == 0.0
    end

    @testset "Simple decrease condition (no Armijo constant)" begin
        # Geometric reduction only requires ||f(x+αd)||^2 < ||f(x)||^2
        # Unlike Armijo, there's no c1 parameter for sufficient decrease
        r(x) = x .^ 2
        x = [2.0, 3.0]
        d = [-1.0, -1.5]

        α = geometric_reduction(r, x, d, 1.0)

        # Just verify strict decrease in merit
        ϕ_0 = norm(r(x))^2
        ϕ_new = norm(r(x .+ α .* d))^2
        @test ϕ_new < ϕ_0
        @test α > 0.0
    end

    @testset "Rho controls reduction rate" begin
        # Same problem, different rho values
        r(x) = x .^ 3
        x = [1.0]
        d = [-10.0]  # Overshooting step

        α_slow = geometric_reduction(r, x, d, 1.0; rho=0.9)
        α_fast = geometric_reduction(r, x, d, 1.0; rho=0.1)

        # More aggressive reduction should give smaller or equal step
        @test α_fast <= α_slow
    end

    @testset "Respects alpha_init parameter" begin
        r(x) = x
        x = [2.0]
        d = [-2.0]
        alpha_init = 0.25

        α = geometric_reduction(r, x, d, alpha_init)

        @test α <= alpha_init
        @test α > 0.0
    end

    @testset "Multidimensional convergence" begin
        # r(x) = x for 5D, Newton step toward origin
        r(x) = x
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = -x

        α = geometric_reduction(r, x, d, 1.0)

        @test α == 1.0
        @test norm(r(x .+ α .* d)) < 1e-14
    end
end

@testset "Constant Step Linesearch" begin
    @testset "Returns the fixed step size regardless of objective" begin
        # constant_step returns a closure that always returns the same alpha
        ls = constant_step(0.42)

        # Same interface as other methods: (f, x, d, alpha_init) -> α
        r(x) = x
        x = [2.0]
        d = [-2.0]

        α = ls(r, x, d, 1.0)
        @test α == 0.42
    end

    @testset "Ignores the objective function entirely" begin
        ls = constant_step(0.1)

        # Even with a pathological residual, constant_step returns the fixed value
        r_pathological(x) = [Inf]
        x = [1.0]
        d = [-1.0]

        α = ls(r_pathological, x, d, 1.0)
        @test α == 0.1
    end

    @testset "Ignores alpha_init argument" begin
        ls = constant_step(0.5)

        r(x) = x
        x = [1.0]
        d = [-1.0]

        # alpha_init is 0.01, but constant_step should ignore it
        α = ls(r, x, d, 0.01)
        @test α == 0.5
    end

    @testset "Works with different fixed step sizes" begin
        for fixed_alpha in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]
            ls = constant_step(fixed_alpha)
            r(x) = x
            α = ls(r, [1.0], [-1.0], 1.0)
            @test α == fixed_alpha
        end
    end

    @testset "Multidimensional problem" begin
        ls = constant_step(0.25)

        r(x) = x
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = -x

        α = ls(r, x, d, 1.0)
        @test α == 0.25
    end
end

@testset "Pre-allocated x_buffer" begin
    @testset "armijo_backtracking: buffer gives same result as allocating" begin
        r(x) = x .^ 3
        x = [1.0, 2.0]
        d = [-5.0, -10.0]
        x_buffer = similar(x)

        α_alloc = armijo_backtracking(r, x, d, 1.0)
        α_buf = armijo_backtracking(r, x, d, 1.0; x_buffer)

        @test α_buf == α_alloc
        @test α_buf > 0.0
    end

    @testset "armijo_backtracking: buffer is actually used (no extra allocation)" begin
        r(x) = x
        x = [2.0, 3.0, 4.0]
        d = -x
        x_buffer = similar(x)

        α = armijo_backtracking(r, x, d, 1.0; x_buffer)
        @test α == 1.0
        # Buffer should have been written to (last trial point)
        @test x_buffer ≈ x .+ α .* d atol = 1e-14
    end

    @testset "geometric_reduction: buffer gives same result as allocating" begin
        r(x) = x .^ 3
        x = [1.0, 2.0]
        d = [-5.0, -10.0]
        x_buffer = similar(x)

        α_alloc = geometric_reduction(r, x, d, 1.0)
        α_buf = geometric_reduction(r, x, d, 1.0; x_buffer)

        @test α_buf == α_alloc
        @test α_buf > 0.0
    end

    @testset "geometric_reduction: buffer is actually used (no extra allocation)" begin
        r(x) = x
        x = [2.0, 3.0, 4.0]
        d = -x
        x_buffer = similar(x)

        α = geometric_reduction(r, x, d, 1.0; x_buffer)
        @test α == 1.0
        # Buffer should have been written to (last trial point)
        @test x_buffer ≈ x .+ α .* d atol = 1e-14
    end

    @testset "armijo_backtracking: multidimensional with backtracking" begin
        r(x) = x .^ 2
        x = [2.0, 3.0, 4.0, 5.0]
        d = [-20.0, -30.0, -40.0, -50.0]
        x_buffer = similar(x)

        α_alloc = armijo_backtracking(r, x, d, 1.0; rho=0.5, max_iters=20)
        α_buf = armijo_backtracking(r, x, d, 1.0; rho=0.5, max_iters=20, x_buffer)

        @test α_buf == α_alloc
        @test 0.0 < α_buf < 1.0
    end

    @testset "perform_linesearch: z_trial_buffer gives same result" begin
        # Simple residual norm function
        residual_norm_fn(z) = sum(abs2, z)
        z_est = [1.0, 2.0, 3.0]
        δz = -z_est  # Newton step toward origin
        current_norm = residual_norm_fn(z_est)
        z_trial_buffer = similar(z_est)

        α_alloc = perform_linesearch(residual_norm_fn, z_est, δz, current_norm; use_armijo=true)
        α_buf = perform_linesearch(residual_norm_fn, z_est, δz, current_norm; use_armijo=true, z_trial_buffer)

        @test α_buf == α_alloc
    end
end
