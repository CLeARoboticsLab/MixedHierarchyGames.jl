using Test
using LinearAlgebra: norm
using MixedHierarchyGames: armijo_backtracking_linesearch

@testset "Armijo Backtracking Linesearch" begin
    @testset "Returns full step when sufficient decrease achieved" begin
        # If f(z + δz) already satisfies Armijo condition, return α = 1
        f_eval(z) = z.^2  # Simple quadratic
        z = [2.0, 2.0]
        δz = [-1.0, -1.0]  # Descent direction
        f_z = f_eval(z)

        α = armijo_backtracking_linesearch(f_eval, z, δz, f_z)

        # Should return α = 1 or close to it for good descent direction
        @test α > 0.5
    end

    @testset "Backtracks when needed" begin
        # If initial step too large, should reduce α
        f_eval(z) = z.^4  # Vector residual
        z = [1.0, 1.0]
        δz = [-10.0, -10.0]  # Too aggressive
        f_z = f_eval(z)

        α = armijo_backtracking_linesearch(f_eval, z, δz, f_z)

        # Should backtrack to smaller step
        @test α < 1.0
        @test α > 0.0
    end

    @testset "Respects minimum step size" begin
        # Should not return α below minimum threshold
        f_eval(z) = [z[1]^2 + 100*z[2]^2]  # Vector residual
        z = [0.1, 0.1]
        δz = [1.0, 1.0]  # Ascent direction (bad)
        f_z = f_eval(z)

        α = armijo_backtracking_linesearch(f_eval, z, δz, f_z; max_iters=5)

        # Should return something or indicate failure
        @test α >= 0.0
    end

    @testset "Sufficient decrease condition" begin
        # Verify Armijo condition: f(z + αδz) ≤ f(z) + c₁*α*∇f'*δz
        f_eval(z) = z.^2  # Vector residual
        z = [2.0, 3.0]
        δz = [-1.0, -1.5]  # Gradient descent direction
        f_z = f_eval(z)

        α = armijo_backtracking_linesearch(f_eval, z, δz, f_z; σ=1e-4)

        # Check sufficient decrease
        f_new = f_eval(z + α * δz)
        @test norm(f_new) < norm(f_z)  # Should decrease
    end
end
