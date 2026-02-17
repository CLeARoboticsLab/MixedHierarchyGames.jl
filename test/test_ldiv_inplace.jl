using Test
using LinearAlgebra: norm, lu, lu!, I
using SparseArrays: sparse
using Graphs: nv

using MixedHierarchyGames:
    _solve_K!,
    compute_K_evals,
    preoptimize_nonlinear_solver,
    run_nonlinear_solver,
    NonlinearSolver,
    solve_raw,
    has_leader

# make_standard_two_player_problem is provided by testing_utils.jl (included in runtests.jl)

@testset "In-place ldiv! for K = M \\ N" begin

    #==========================================================================
        Unit tests: _solve_K! with K_buffer keyword argument
    ==========================================================================#

    @testset "_solve_K! with K_buffer" begin

        @testset "K_buffer produces identical results to default (dense)" begin
            M = [4.0 1.0; 1.0 3.0]
            N = [5.0 2.0; 3.0 7.0]

            K_default = _solve_K!(copy(M), copy(N), 1)
            K_buffer = Matrix{Float64}(undef, 2, 2)
            K_inplace = _solve_K!(copy(M), copy(N), 1; K_buffer)

            @test K_inplace === K_buffer  # must be the same object
            @test norm(K_inplace - K_default) < 1e-14
        end

        @testset "K_buffer produces identical results to default (sparse)" begin
            M = [4.0 1.0; 1.0 3.0]
            N = [5.0 2.0; 3.0 7.0]

            K_default = _solve_K!(copy(M), copy(N), 1; use_sparse=true)
            K_buffer = Matrix{Float64}(undef, 2, 2)
            K_inplace = _solve_K!(copy(M), copy(N), 1; K_buffer, use_sparse=true)

            @test K_inplace === K_buffer
            @test norm(K_inplace - K_default) < 1e-14
        end

        @testset "K_buffer with regularization produces correct results" begin
            M = [4.0 1.0; 1.0 3.0]
            N = [5.0 2.0; 3.0 7.0]

            K_default = _solve_K!(copy(M), copy(N), 1; regularization=1e-6)
            K_buffer = Matrix{Float64}(undef, 2, 2)
            K_inplace = _solve_K!(copy(M), copy(N), 1; K_buffer, regularization=1e-6)

            @test K_inplace === K_buffer
            @test norm(K_inplace - K_default) < 1e-14
        end

        @testset "K_buffer with regularization + sparse produces correct results" begin
            M = [4.0 1.0; 1.0 3.0]
            N = [5.0 2.0; 3.0 7.0]

            K_default = _solve_K!(copy(M), copy(N), 1; regularization=1e-6, use_sparse=true)
            K_buffer = Matrix{Float64}(undef, 2, 2)
            K_inplace = _solve_K!(copy(M), copy(N), 1; K_buffer, regularization=1e-6, use_sparse=true)

            @test K_inplace === K_buffer
            @test norm(K_inplace - K_default) < 1e-14
        end

        @testset "K_buffer=nothing falls back to allocating (backward compat)" begin
            M = [4.0 1.0; 1.0 3.0]
            N = [5.0 2.0; 3.0 7.0]

            K_result = _solve_K!(copy(M), copy(N), 1; K_buffer=nothing)
            K_expected = M \ N
            @test norm(K_result - K_expected) < 1e-14
        end

        @testset "K_buffer reduces allocation on dense path" begin
            M = randn(20, 20)
            M = M' * M + 5I  # positive definite
            N = randn(20, 10)
            K_buffer = Matrix{Float64}(undef, 20, 10)

            # Warmup
            _solve_K!(copy(M), copy(N), 1; K_buffer)

            # Measure allocation with buffer
            alloc_with = @allocated _solve_K!(copy(M), copy(N), 1; K_buffer)

            # Measure allocation without buffer
            alloc_without = @allocated _solve_K!(copy(M), copy(N), 1)

            # With buffer should allocate less (no K result matrix)
            # The K result is 20*10*8 = 1600 bytes, so savings should be at least that
            @test alloc_with < alloc_without
        end

        @testset "K_buffer on sparse path produces correct results (large matrix)" begin
            M = randn(20, 20)
            M = M' * M + 5I  # positive definite
            N = randn(20, 10)
            K_buffer = Matrix{Float64}(undef, 20, 10)

            K_default = _solve_K!(copy(M), copy(N), 1; use_sparse=true)
            K_inplace = _solve_K!(copy(M), copy(N), 1; K_buffer, use_sparse=true)

            @test K_inplace === K_buffer
            @test norm(K_inplace - K_default) < 1e-10
        end

        @testset "singular matrix NaN fallback works with K_buffer" begin
            M = [1.0 2.0; 2.0 4.0]  # singular
            N = [1.0 0.0; 0.0 1.0]
            K_buffer = Matrix{Float64}(undef, 2, 2)

            result = @test_warn r"Singular M matrix" _solve_K!(copy(M), copy(N), 1; K_buffer)
            @test all(isnan, result)
            @test size(result) == (2, 2)
        end

        @testset "non-finite fallback works with K_buffer" begin
            M = [1.0 0.0; 0.0 1e-308]
            N = [1.0 0.0; 1e308 0.0]
            K_buffer = Matrix{Float64}(undef, 2, 2)

            result = @test_warn r"non-finite values" _solve_K!(copy(M), copy(N), 1; K_buffer)
            @test all(isnan, result)
            @test result === K_buffer  # NaN should be filled into buffer
        end

        @testset "M is not permanently mutated with K_buffer (regularization)" begin
            M = [4.0 1.0; 1.0 3.0]
            N = [5.0 2.0; 3.0 7.0]
            M_copy = copy(M)
            K_buffer = Matrix{Float64}(undef, 2, 2)

            _solve_K!(M, N, 1; K_buffer, regularization=1e-4)

            @test M ≈ M_copy atol=1e-14
        end
    end

    #==========================================================================
        Integration: compute_K_evals with K_buffers
    ==========================================================================#

    @testset "compute_K_evals with K_buffers" begin

        @testset "K_buffers produces identical results to default" begin
            prob = make_standard_two_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
            )
            z_current = randn(length(precomputed.all_variables))

            K_vec_default, _ = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info
            )

            # Build K_buffers
            π_sizes = precomputed.setup_info.π_sizes
            ys = precomputed.problem_vars.ys
            graph = precomputed.setup_info.graph
            K_buffers = Dict{Int, Matrix{Float64}}()
            for ii in 1:nv(graph)
                if has_leader(graph, ii)
                    K_buffers[ii] = Matrix{Float64}(undef, π_sizes[ii], length(ys[ii]))
                end
            end

            K_vec_buffered, _ = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info;
                K_buffers
            )

            @test norm(K_vec_buffered - K_vec_default) < 1e-14
        end

        @testset "K_buffers with sparse and regularization" begin
            prob = make_standard_two_player_problem()
            precomputed = preoptimize_nonlinear_solver(
                prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
                state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
            )
            z_current = randn(length(precomputed.all_variables))

            K_vec_default, _ = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info;
                use_sparse=:always, regularization=1e-10
            )

            π_sizes = precomputed.setup_info.π_sizes
            ys = precomputed.problem_vars.ys
            graph = precomputed.setup_info.graph
            K_buffers = Dict{Int, Matrix{Float64}}()
            for ii in 1:nv(graph)
                if has_leader(graph, ii)
                    K_buffers[ii] = Matrix{Float64}(undef, π_sizes[ii], length(ys[ii]))
                end
            end

            K_vec_buffered, _ = compute_K_evals(
                z_current, precomputed.problem_vars, precomputed.setup_info;
                K_buffers, use_sparse=:always, regularization=1e-10
            )

            @test norm(K_vec_buffered - K_vec_default) < 1e-14
        end
    end

    #==========================================================================
        Integration: run_nonlinear_solver still converges with K_buffers
    ==========================================================================#

    @testset "run_nonlinear_solver convergence with K_buffers" begin
        prob = make_standard_two_player_problem()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )
        initial_states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        result = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            max_iters=20, tol=1e-6
        )
        @test result.status in (:solved, :solved_initial_point, :max_iters_reached)
    end
end
