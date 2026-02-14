using Test
using LinearAlgebra: norm, I, cond, Diagonal, qr

using MixedHierarchyGames:
    _solve_K!,
    compute_K_evals,
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    NonlinearSolver,
    solve_raw

# make_standard_two_player_problem is provided by testing_utils.jl (included in runtests.jl)

@testset "Numerical Regularization (Tikhonov)" begin

    #==========================================================================
        Unit tests for _solve_K! with regularization parameter
    ==========================================================================#

    @testset "_solve_K! regularization parameter" begin

        @testset "default regularization=0.0 matches original behavior" begin
            # Well-conditioned system
            M = [2.0 1.0; 1.0 3.0]
            N = [1.0 0.0; 0.0 1.0]

            K_default = _solve_K!(M, N, 1)
            K_explicit_zero = _solve_K!(M, N, 1; regularization=0.0)

            @test norm(K_default - K_explicit_zero) < 1e-14
        end

        @testset "regularization prevents singular exception" begin
            # Singular M matrix — without regularization, this returns NaN
            M = [1.0 2.0; 2.0 4.0]  # rank 1 (row 2 = 2 * row 1)
            N = [1.0; 2.0][:, :]     # Make it a matrix

            K_no_reg = _solve_K!(M, N, 1)
            @test any(isnan, K_no_reg)  # Should produce NaN fallback

            # With regularization, should get a finite result
            K_reg = _solve_K!(M, N, 1; regularization=1e-6)
            @test all(isfinite, K_reg)
        end

        @testset "regularization handles near-singular M" begin
            # Near-singular: condition number ~1e15
            n = 5
            U = qr(randn(n, n)).Q |> Matrix
            S = Diagonal([1.0, 1.0, 1.0, 1e-8, 1e-15])
            M = U * S * U'
            N = randn(n, 3)

            # Without regularization — result may have huge values
            K_no_reg = _solve_K!(M, N, 1)

            # With regularization — should be better conditioned
            K_reg = _solve_K!(M, N, 1; regularization=1e-6)
            @test all(isfinite, K_reg)

            # Regularized K should have bounded norm
            @test norm(K_reg) < 1e10
        end

        @testset "small regularization does not distort well-conditioned solutions" begin
            # Well-conditioned M (condition number ~3)
            M = [4.0 1.0; 1.0 3.0]
            N = [5.0 2.0; 3.0 7.0]

            K_exact = M \ N
            K_reg = _solve_K!(M, N, 1; regularization=1e-10)

            # With tiny regularization, error should be negligible
            relative_error = norm(K_reg - K_exact) / norm(K_exact)
            @test relative_error < 1e-8
        end

        @testset "regularization does not mutate input M matrix" begin
            M = [2.0 1.0; 1.0 3.0]
            N = [1.0 0.0; 0.0 1.0]
            M_copy = copy(M)

            _solve_K!(M, N, 1; regularization=1e-4)

            # M should be restored after the call (within floating-point roundtrip tolerance)
            @test M ≈ M_copy atol=1e-14
        end

        @testset "regularization does not mutate M even on singular matrix" begin
            M = [1.0 2.0; 2.0 4.0]  # singular
            N = [1.0; 2.0][:, :]
            M_copy = copy(M)

            _solve_K!(M, N, 1; regularization=1e-6)

            # M should be restored after the call (within floating-point roundtrip tolerance)
            @test M ≈ M_copy atol=1e-14
        end

        @testset "non-Singular exceptions are rethrown (not swallowed)" begin
            # DimensionMismatch is not SingularException or LAPACKException,
            # so it should be rethrown, not caught by the NaN fallback
            M = [1.0 0.0; 0.0 1.0]
            N_bad = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]  # 3×3, incompatible with 2×2 M

            @test_throws DimensionMismatch _solve_K!(M, N_bad, 1; regularization=1e-6)
        end

        @testset "M is restored even when exception is thrown" begin
            M = [1.0 0.0; 0.0 1.0]
            N_bad = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]  # triggers DimensionMismatch
            M_copy = copy(M)

            try
                _solve_K!(M, N_bad, 1; regularization=0.5)
            catch
                # expected
            end

            # M should be restored by the finally block
            @test M ≈ M_copy atol=1e-14
        end

        @testset "regularization works with use_sparse=true" begin
            M = [1.0 2.0; 2.0 4.0]  # singular
            N = [1.0; 2.0][:, :]

            K_dense = _solve_K!(M, N, 1; regularization=1e-6)
            K_sparse = _solve_K!(M, N, 1; regularization=1e-6, use_sparse=true)

            @test all(isfinite, K_dense)
            @test all(isfinite, K_sparse)
            @test norm(K_dense - K_sparse) / max(norm(K_dense), 1.0) < 1e-10
        end
    end

    #==========================================================================
        NaN-fill error path: allocation avoidance
    ==========================================================================#

    @testset "NaN-fill error paths avoid unnecessary allocation" begin

        @testset "non-finite path returns NaN-filled array of correct size" begin
            # M is invertible but M \ N overflows to Inf due to tiny diagonal + large N.
            # This triggers the non-finite check (line 739) rather than SingularException.
            M = [1.0 0.0; 0.0 1e-308]
            N = [1.0 0.0; 1e308 0.0]

            result = @test_warn r"non-finite values" _solve_K!(copy(M), copy(N), 1)
            @test size(result) == size(N)
            @test all(isnan, result)
        end

        @testset "singular exception path returns NaN-filled array of correct size" begin
            # Exactly singular M: duplicate rows
            M = [1.0 2.0; 2.0 4.0]
            N = [1.0 0.0; 0.0 1.0]

            result = @test_warn r"Singular M matrix" _solve_K!(copy(M), copy(N), 1)
            @test size(result) == size(N)
            @test all(isnan, result)
        end

        @testset "non-finite path returns correct dimensions for non-square N" begin
            # M is invertible but M \ N overflows to Inf.
            # Tests with non-square N to verify dimensions are preserved.
            M = [1.0 0.0; 0.0 1e-308]
            N = [1.0 2.0 3.0; 1e308 0.0 0.0]

            result = @test_warn r"non-finite values" _solve_K!(copy(M), copy(N), 1)
            @test size(result) == (2, 3)
            @test all(isnan, result)
        end

        @testset "singular exception path returns correct dimensions for non-square N" begin
            # Singular M with non-square N
            M = [0.0 0.0; 0.0 0.0]
            N = [1.0 2.0 3.0; 4.0 5.0 6.0]

            result = @test_warn r"Singular M matrix" _solve_K!(copy(M), copy(N), 1)
            @test size(result) == (2, 3)
            @test all(isnan, result)
        end

        @testset "N buffer is not mutated by error paths" begin
            # Ensure the input N matrix is not modified by either error path
            M_singular = [1.0 2.0; 2.0 4.0]
            N = [1.0 0.0; 0.0 1.0]
            N_copy = copy(N)

            _solve_K!(copy(M_singular), N, 1)
            @test N == N_copy  # N must not be mutated
        end
    end

    #==========================================================================
        Accuracy analysis: distortion vs regularization strength
    ==========================================================================#

    @testset "accuracy analysis: distortion vs regularization strength" begin

        @testset "well-conditioned system: error scales with λ" begin
            # Well-conditioned M with known solution
            M = Float64[4 1; 1 3]
            N = Float64[5 2; 3 7]
            K_exact = M \ N

            lambdas = [1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
            errors = Float64[]

            for λ in lambdas
                K_reg = _solve_K!(M, N, 1; regularization=λ)
                push!(errors, norm(K_reg - K_exact) / norm(K_exact))
            end

            # Errors should be monotonically increasing with λ
            for i in 1:(length(errors)-1)
                @test errors[i] ≤ errors[i+1] + 1e-15  # allow tiny numerical noise
            end

            # Very small λ should cause negligible distortion
            @test errors[1] < 1e-10  # λ=1e-12
            @test errors[2] < 1e-8   # λ=1e-10

            # Report results
            @debug "Distortion analysis (well-conditioned, cond(M)=$(round(cond(M), digits=1))):"
            for (λ, err) in zip(lambdas, errors)
                @debug "  λ=$λ → relative error=$err"
            end
        end

        @testset "moderately ill-conditioned: regularization tradeoff" begin
            # cond(M) ~ 1e6
            n = 4
            U = qr(randn(n, n)).Q |> Matrix
            S = Diagonal([1.0, 0.1, 0.01, 1e-6])
            M = U * S * U'
            M = (M + M') / 2  # ensure symmetric
            N = randn(n, 2)

            K_exact = M \ N  # may be slightly inaccurate due to conditioning

            # Compare different regularization values
            lambdas = [1e-10, 1e-8, 1e-6, 1e-4]
            for λ in lambdas
                K_reg = _solve_K!(M, N, 1; regularization=λ)
                @test all(isfinite, K_reg)
            end

            @debug "Moderately ill-conditioned (cond(M)=$(round(cond(M), sigdigits=3))):"
            for λ in lambdas
                K_reg = _solve_K!(M, N, 1; regularization=λ)
                err = norm(K_reg - K_exact) / norm(K_exact)
                @debug "  λ=$λ → relative error=$err"
            end
        end
    end

    #==========================================================================
        Integration: compute_K_evals with regularization
    ==========================================================================#

    @testset "compute_K_evals with regularization parameter" begin

        @testset "regularization=0.0 matches default behavior" begin
            prob = make_standard_two_player_problem()

            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim,
                control_dim=prob.control_dim,
                verbose=false
            )

            z_current = randn(length(precomputed.all_variables))

            K_vec_default, info_default = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info
            )

            K_vec_zero, info_zero = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info;
                regularization=0.0
            )

            @test norm(K_vec_default - K_vec_zero) < 1e-14
        end

        @testset "small regularization does not distort K significantly" begin
            prob = make_standard_two_player_problem()

            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim,
                control_dim=prob.control_dim,
                verbose=false
            )

            z_current = randn(length(precomputed.all_variables))

            K_vec_base, _ = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info
            )

            K_vec_reg, _ = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info;
                regularization=1e-10
            )

            # With well-conditioned M from a real problem, tiny regularization
            # should barely change K
            relative_error = norm(K_vec_reg - K_vec_base) / max(norm(K_vec_base), 1.0)
            @test relative_error < 1e-6
        end
    end

    #==========================================================================
        Integration: run_nonlinear_solver with regularization
    ==========================================================================#

    @testset "run_nonlinear_solver accepts regularization parameter" begin
        prob = make_standard_two_player_problem()

        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim,
            control_dim=prob.control_dim,
            verbose=false
        )

        initial_states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        # Default (no regularization) should work
        result_default = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=20, tol=1e-6
        )
        @test result_default.status in (:solved, :solved_initial_point, :max_iters_reached)

        # Explicit regularization=0.0 should give same result
        result_zero = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=20, tol=1e-6, regularization=0.0
        )
        @test norm(result_default.sol - result_zero.sol) < 1e-12

        # Small regularization should also converge with similar solution
        result_reg = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=20, tol=1e-6, regularization=1e-10
        )
        @test result_reg.status in (:solved, :solved_initial_point, :max_iters_reached)
    end

    #==========================================================================
        Integration: NonlinearSolver constructor and solve/solve_raw API
    ==========================================================================#

    @testset "NonlinearSolver constructor accepts regularization" begin
        prob = make_standard_two_player_problem()

        # Default (no regularization)
        solver_default = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            max_iters=20
        )
        @test solver_default.options.regularization == 0.0

        # Explicit regularization
        solver_reg = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            max_iters=20, regularization=1e-8
        )
        @test solver_reg.options.regularization == 1e-8
    end

    @testset "solve_raw passes regularization through" begin
        prob = make_standard_two_player_problem()

        solver = NonlinearSolver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs,
            prob.state_dim, prob.control_dim;
            max_iters=20
        )

        initial_states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        # Default solve
        result_default = solve_raw(solver, initial_states)
        @test result_default.status in (:solved, :solved_initial_point, :max_iters_reached)

        # Override regularization at solve time
        result_reg = solve_raw(solver, initial_states; regularization=1e-10)
        @test result_reg.status in (:solved, :solved_initial_point, :max_iters_reached)
    end
end
