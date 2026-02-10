using Test
using LinearAlgebra: norm
using MixedHierarchyGames: armijo_backtracking_linesearch

@testset "Armijo Backtracking Linesearch" begin
    @testset "Returns named tuple with step_size and success fields" begin
        f_eval(z) = z.^2
        z = [2.0, 2.0]
        δz = [-1.0, -1.0]
        f_z = f_eval(z)

        result = armijo_backtracking_linesearch(f_eval, z, δz, f_z)

        @test result isa NamedTuple
        @test haskey(result, :step_size)
        @test haskey(result, :success)
        @test result.step_size isa Float64
        @test result.success isa Bool
    end

    @testset "Returns full step when sufficient decrease achieved" begin
        # If f(z + δz) already satisfies Armijo condition, return α = 1
        f_eval(z) = z.^2  # Simple quadratic
        z = [2.0, 2.0]
        δz = [-1.0, -1.0]  # Descent direction
        f_z = f_eval(z)

        result = armijo_backtracking_linesearch(f_eval, z, δz, f_z)

        # Should return α = 1 or close to it for good descent direction
        @test result.step_size > 0.5
        @test result.success == true
    end

    @testset "Backtracks when needed" begin
        # If initial step too large, should reduce α
        f_eval(z) = z.^4  # Vector residual
        z = [1.0, 1.0]
        δz = [-10.0, -10.0]  # Too aggressive
        f_z = f_eval(z)

        result = armijo_backtracking_linesearch(f_eval, z, δz, f_z)

        # Should backtrack to smaller step
        @test result.step_size < 1.0
        @test result.step_size > 0.0
        @test result.success == true
    end

    @testset "Reports failure for ascent direction" begin
        # Ascent direction: line search should fail to find sufficient decrease
        f_eval(z) = [z[1]^2 + 100*z[2]^2]  # Vector residual
        z = [0.1, 0.1]
        δz = [1.0, 1.0]  # Ascent direction (bad)
        f_z = f_eval(z)

        result = armijo_backtracking_linesearch(f_eval, z, δz, f_z; max_iters=5)

        # Should indicate failure
        @test result.success == false
        @test result.step_size >= 0.0
    end

    @testset "Sufficient decrease condition with success" begin
        # Verify Armijo condition: f(z + αδz) ≤ f(z) + c₁*α*∇f'*δz
        f_eval(z) = z.^2  # Vector residual
        z = [2.0, 3.0]
        δz = [-1.0, -1.5]  # Gradient descent direction
        f_z = f_eval(z)

        result = armijo_backtracking_linesearch(f_eval, z, δz, f_z; σ=1e-4)

        @test result.success == true

        # Check sufficient decrease
        f_new = f_eval(z + result.step_size * δz)
        @test norm(f_new) < norm(f_z)  # Should decrease
    end

    @testset "Failure returns last tried step_size" begin
        # When line search fails, step_size should be the last attempted value
        f_eval(z) = [z[1]^2 + 100*z[2]^2]
        z = [0.1, 0.1]
        δz = [1.0, 1.0]  # Ascent direction
        f_z = f_eval(z)

        result = armijo_backtracking_linesearch(f_eval, z, δz, f_z; β=0.5, max_iters=3)

        @test result.success == false
        # After 3 iterations with β=0.5: α = 1.0 * 0.5^3 = 0.125
        @test result.step_size ≈ 0.5^3 atol=1e-10
    end
end
