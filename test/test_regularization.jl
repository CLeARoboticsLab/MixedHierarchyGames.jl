using Test
using LinearAlgebra: norm, I, cond, Diagonal, qr

using MixedHierarchyGames:
    _solve_K,
    compute_K_evals,
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    setup_problem_parameter_variables,
    NonlinearSolver,
    solve_raw

using Graphs: SimpleDiGraph, add_edge!
using TrajectoryGamesBase: unflatten_trajectory

#=
    Test helpers
=#

"""
    make_two_player_chain_for_regularization(; T=3, state_dim=2, control_dim=2)

Simple 2-player Stackelberg chain (P1 -> P2) for testing regularization.
"""
function make_two_player_chain_for_regularization(; T=3, state_dim=2, control_dim=2)
    N = 2
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)

    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)

    θs = setup_problem_parameter_variables(fill(state_dim, N))

    function J1(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        goal = [1.0, 1.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J2(z1, z2; θ=nothing)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        goal = [2.0, 2.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    Js = Dict(1 => J1, 2 => J2)

    function make_dynamics_constraint(player_idx)
        function dynamics_constraint(z)
            (; xs, us) = unflatten_trajectory(z, state_dim, control_dim)
            constraints = []
            for t in 1:T
                push!(constraints, xs[t+1] - xs[t] - us[t])
            end
            push!(constraints, xs[1] - θs[player_idx])
            return vcat(constraints...)
        end
        return dynamics_constraint
    end

    gs = [make_dynamics_constraint(i) for i in 1:N]
    return (; G, Js, gs, primal_dims, θs, state_dim, control_dim, T, N)
end

@testset "Numerical Regularization (Tikhonov)" begin

    #==========================================================================
        Unit tests for _solve_K with regularization parameter
    ==========================================================================#

    @testset "_solve_K regularization parameter" begin

        @testset "default regularization=0.0 matches original behavior" begin
            # Well-conditioned system
            M = [2.0 1.0; 1.0 3.0]
            N = [1.0 0.0; 0.0 1.0]

            K_default = _solve_K(M, N, 1)
            K_explicit_zero = _solve_K(M, N, 1; regularization=0.0)

            @test norm(K_default - K_explicit_zero) < 1e-14
        end

        @testset "regularization prevents singular exception" begin
            # Singular M matrix — without regularization, this returns NaN
            M = [1.0 2.0; 2.0 4.0]  # rank 1 (row 2 = 2 * row 1)
            N = [1.0; 2.0][:, :]     # Make it a matrix

            K_no_reg = _solve_K(M, N, 1)
            @test any(isnan, K_no_reg)  # Should produce NaN fallback

            # With regularization, should get a finite result
            K_reg = _solve_K(M, N, 1; regularization=1e-6)
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
            K_no_reg = _solve_K(M, N, 1)

            # With regularization — should be better conditioned
            K_reg = _solve_K(M, N, 1; regularization=1e-6)
            @test all(isfinite, K_reg)

            # Regularized K should have bounded norm
            @test norm(K_reg) < 1e10
        end

        @testset "small regularization does not distort well-conditioned solutions" begin
            # Well-conditioned M (condition number ~3)
            M = [4.0 1.0; 1.0 3.0]
            N = [5.0 2.0; 3.0 7.0]

            K_exact = M \ N
            K_reg = _solve_K(M, N, 1; regularization=1e-10)

            # With tiny regularization, error should be negligible
            relative_error = norm(K_reg - K_exact) / norm(K_exact)
            @test relative_error < 1e-8
        end

        @testset "regularization works with use_sparse=true" begin
            M = [1.0 2.0; 2.0 4.0]  # singular
            N = [1.0; 2.0][:, :]

            K_dense = _solve_K(M, N, 1; regularization=1e-6)
            K_sparse = _solve_K(M, N, 1; regularization=1e-6, use_sparse=true)

            @test all(isfinite, K_dense)
            @test all(isfinite, K_sparse)
            @test norm(K_dense - K_sparse) / max(norm(K_dense), 1.0) < 1e-10
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
                K_reg = _solve_K(M, N, 1; regularization=λ)
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
            @info "Distortion analysis (well-conditioned, cond(M)=$(round(cond(M), digits=1))):"
            for (λ, err) in zip(lambdas, errors)
                @info "  λ=$λ → relative error=$err"
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
                K_reg = _solve_K(M, N, 1; regularization=λ)
                @test all(isfinite, K_reg)
            end

            @info "Moderately ill-conditioned (cond(M)=$(round(cond(M), sigdigits=3))):"
            for λ in lambdas
                K_reg = _solve_K(M, N, 1; regularization=λ)
                err = norm(K_reg - K_exact) / norm(K_exact)
                @info "  λ=$λ → relative error=$err"
            end
        end
    end

    #==========================================================================
        Integration: compute_K_evals with regularization
    ==========================================================================#

    @testset "compute_K_evals with regularization parameter" begin

        @testset "regularization=0.0 matches default behavior" begin
            prob = make_two_player_chain_for_regularization()

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
            prob = make_two_player_chain_for_regularization()

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
        prob = make_two_player_chain_for_regularization()

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
        prob = make_two_player_chain_for_regularization()

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
        prob = make_two_player_chain_for_regularization()

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
