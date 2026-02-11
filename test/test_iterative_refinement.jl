using Test
using LinearAlgebra: norm, I, Diagonal, qr, lu, cond
using Random: MersenneTwister

using MixedHierarchyGames
using MixedHierarchyGames: compute_K_evals, preoptimize_nonlinear_solver,
    setup_problem_parameter_variables

using Graphs: SimpleDiGraph, add_edge!
using TrajectoryGamesBase: unflatten_trajectory

@testset "Iterative Refinement for M\\N Solve" begin

    @testset "Default behavior unchanged (refinement_steps=0)" begin
        # A well-conditioned system should produce the same result with or without refinement
        n = 10
        M = randn(MersenneTwister(1), n, n) + 5.0 * I  # Well-conditioned: diagonally dominant
        N_mat = randn(MersenneTwister(2), n, 3)

        K_default = MixedHierarchyGames._solve_K(M, N_mat, 1)
        K_no_refinement = MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=0)

        @test K_default ≈ K_no_refinement atol=1e-15
    end

    @testset "Refinement correctness: K still satisfies M*K ≈ N" begin
        # Verify that refinement doesn't break the basic invariant
        n = 15
        M = randn(MersenneTwister(50), n, n) + 3.0 * I
        N_mat = randn(MersenneTwister(51), n, 4)

        for steps in [0, 1, 2, 3]
            K = MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=steps)
            residual = norm(N_mat - M * K) / norm(N_mat)
            @test residual < 1e-10
        end
    end

    @testset "Refinement accuracy analysis across condition numbers" begin
        # This test documents the actual behavior of iterative refinement
        # at various condition numbers. It serves as analysis, not assertion
        # of improvement — because standard double-precision refinement
        # provides inconsistent benefit.
        n = 20
        U_qr = qr(randn(MersenneTwister(42), n, n))
        V_qr = qr(randn(MersenneTwister(43), n, n))
        N_mat = randn(MersenneTwister(44), n, 5)

        for cond_exp in [6, 8, 10, 12]
            σ = 10.0 .^ range(0, -cond_exp, length=n)
            M = Matrix(U_qr.Q) * Diagonal(σ) * Matrix(V_qr.Q)'
            K_exact = Float64.(BigFloat.(M) \ BigFloat.(N_mat))

            K_base = MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=0)
            K_ref1 = MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=1)
            K_ref3 = MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=3)

            err_base = norm(K_base - K_exact) / norm(K_exact)
            err_ref1 = norm(K_ref1 - K_exact) / norm(K_exact)
            err_ref3 = norm(K_ref3 - K_exact) / norm(K_exact)

            @info "cond=1e$cond_exp: base=$err_base, ref1=$err_ref1, ref3=$err_ref3"

            # All solutions should be finite
            @test all(isfinite, K_base)
            @test all(isfinite, K_ref1)
            @test all(isfinite, K_ref3)

            # Error should be bounded by condition number * machine epsilon
            # (within an order of magnitude)
            @test err_base < 10.0^cond_exp * eps(Float64) * 100
        end
    end

    @testset "Well-conditioned system: refinement has negligible effect" begin
        n = 10
        M = randn(MersenneTwister(999), n, n) + 10.0 * I
        N_mat = randn(MersenneTwister(1000), n, 4)

        K_exact = Float64.(BigFloat.(M) \ BigFloat.(N_mat))

        K_no_refine = MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=0)
        K_refined = MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=2)

        error_no_refine = norm(K_no_refine - K_exact) / norm(K_exact)
        error_refined = norm(K_refined - K_exact) / norm(K_exact)

        # Both should be very accurate for well-conditioned M
        @test error_no_refine < 1e-10
        @test error_refined < 1e-10

        @info "Well-conditioned: no_refine_error=$error_no_refine, refined_error=$error_refined"
    end

    @testset "Refinement with singular matrix still returns NaN fallback" begin
        # Singular matrix should still produce NaN, not crash
        M = zeros(3, 3)
        M[1,1] = 1.0  # rank 1
        N_mat = ones(3, 2)

        K = MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=2)
        # Should get NaN fallback (singular matrix handling)
        @test any(isnan, K) || any(!isfinite, K)
    end

    @testset "Refinement with use_sparse=true" begin
        n = 15
        M = randn(MersenneTwister(200), n, n) + 5.0 * I
        N_mat = randn(MersenneTwister(201), n, 4)

        K_dense = MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=2)
        K_sparse = MixedHierarchyGames._solve_K(M, N_mat, 1; use_sparse=true, refinement_steps=2)

        @test norm(K_sparse - K_dense) / norm(K_dense) < 1e-10
    end

    @testset "compute_K_evals passes refinement_steps through" begin
        # Build a small game problem and verify refinement_steps is accepted
        N_players = 2
        G = SimpleDiGraph(N_players)
        add_edge!(G, 1, 2)

        state_dim = 2
        control_dim = 2
        T = 3
        primal_dim = state_dim * (T + 1) + control_dim * (T + 1)
        primal_dims_vec = fill(primal_dim, N_players)

        θs = setup_problem_parameter_variables(fill(state_dim, N_players))

        function J1(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
            sum((xs[end] .- [1.0, 1.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end

        function J2(z1, z2; θ=nothing)
            (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
            sum((xs[end] .- [2.0, 2.0]) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
        end

        Js = Dict(1 => J1, 2 => J2)

        function make_dynamics(player_idx)
            function dynamics(z)
                (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
                constraints = []
                for t in 1:T
                    push!(constraints, xs[t+1] - xs[t] - us[t])
                end
                push!(constraints, xs[1] - θs[player_idx])
                return vcat(constraints...)
            end
        end
        gs_vec = [make_dynamics(i) for i in 1:N_players]

        precomputed = preoptimize_nonlinear_solver(
            G, Js, gs_vec, primal_dims_vec, θs;
            state_dim=state_dim, control_dim=control_dim, verbose=false
        )

        z_test = randn(MersenneTwister(42), length(precomputed.all_variables))

        # Should work with default (no refinement)
        K_vec_default, info_default = compute_K_evals(
            z_test, precomputed.problem_vars, precomputed.setup_info
        )

        # Should accept refinement_steps keyword
        K_vec_refined, info_refined = compute_K_evals(
            z_test, precomputed.problem_vars, precomputed.setup_info;
            refinement_steps=2
        )

        # Results should be very close (game M matrices are usually well-conditioned)
        @test norm(K_vec_refined - K_vec_default) / max(norm(K_vec_default), 1.0) < 1e-10
    end

    @testset "Benchmark: refinement overhead on well-conditioned solve" begin
        n = 20
        M = randn(MersenneTwister(777), n, n) + 10.0 * I
        N_mat = randn(MersenneTwister(778), n, 5)

        # Warmup
        MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=0)
        MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=1)
        MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=3)

        n_iters = 5000
        t_baseline = @elapsed for _ in 1:n_iters
            MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=0)
        end

        t_1step = @elapsed for _ in 1:n_iters
            MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=1)
        end

        t_3steps = @elapsed for _ in 1:n_iters
            MixedHierarchyGames._solve_K(M, N_mat, 1; refinement_steps=3)
        end

        overhead_1 = t_1step / t_baseline
        overhead_3 = t_3steps / t_baseline

        @info "Refinement overhead (n=$n): baseline=$(round(t_baseline/n_iters*1e6, digits=1))μs, " *
              "1-step=$(round(t_1step/n_iters*1e6, digits=1))μs ($(round(overhead_1, digits=2))x), " *
              "3-step=$(round(t_3steps/n_iters*1e6, digits=1))μs ($(round(overhead_3, digits=2))x)"

        # Informational test — always passes
        @test true
    end
end
