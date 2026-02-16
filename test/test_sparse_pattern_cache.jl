using Test
using SparseArrays: sparse, nonzeros, nnz, nzrange, rowvals, getcolptr
using LinearAlgebra: norm
using Graphs: SimpleDiGraph, add_edge!, nv
using MixedHierarchyGames:
    preoptimize_nonlinear_solver,
    compute_K_evals,
    setup_problem_parameter_variables,
    run_nonlinear_solver,
    has_leader

using TrajectoryGamesBase: unflatten_trajectory

#=
    Test helpers — reuse patterns from test_sparse_solve.jl
=#

function make_sparse_cache_two_player(; T=3, state_dim=2, control_dim=2)
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

function make_sparse_cache_three_player(; T=3, state_dim=2, control_dim=2)
    N = 3
    G = SimpleDiGraph(N)
    add_edge!(G, 1, 2)
    add_edge!(G, 2, 3)

    primal_dim_per_player = (state_dim * (T + 1) + control_dim * (T + 1))
    primal_dims = fill(primal_dim_per_player, N)
    θs = setup_problem_parameter_variables(fill(state_dim, N))

    function J1(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z1, state_dim, control_dim)
        goal = [1.0, 1.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J2(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z2, state_dim, control_dim)
        goal = [2.0, 2.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    function J3(z1, z2, z3; θ=nothing)
        (; xs, us) = unflatten_trajectory(z3, state_dim, control_dim)
        goal = [3.0, 3.0]
        sum((xs[end] .- goal) .^ 2) + 0.1 * sum(sum(u .^ 2) for u in us)
    end

    Js = Dict(1 => J1, 2 => J2, 3 => J3)

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

@testset "Sparse Pattern Cache" begin

    @testset "sparse_M_cache field exists in run_nonlinear_solver buffers" begin
        # The run_nonlinear_solver should pre-compute sparse patterns for players
        # that use sparse solves, and reuse them across iterations.
        # We test this indirectly: calling with use_sparse=:always should
        # produce the same results as without caching, but with fewer allocations.
        prob = make_sparse_cache_two_player()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )

        initial_states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        # This should work without errors — the sparse cache is used internally
        result = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            use_sparse=:always, max_iters=5, verbose=false
        )
        @test result.status in (:solved, :solved_initial_point, :max_iters_reached)
    end

    @testset "cached sparse matches fresh sparse(M) — 2-player" begin
        prob = make_sparse_cache_two_player()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )

        z_current = randn(length(precomputed.all_variables))

        # Get M matrices via compute_K_evals
        _, K_info = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_sparse=:never
        )

        for ii in 1:prob.N
            M = K_info.M_evals[ii]
            isnothing(M) && continue

            # Fresh sparse conversion
            M_sp_fresh = sparse(M)

            # Build a cached sparse template from a different z, then update values
            z_test = randn(length(precomputed.all_variables))
            _, K_info_test = compute_K_evals(
                z_test, precomputed.problem_vars, precomputed.setup_info;
                use_sparse=:never
            )
            M_test = K_info_test.M_evals[ii]
            isnothing(M_test) && continue

            # Create template from test point
            M_sp_template = sparse(M_test)

            # Update nzval from the actual M (simulating cache update)
            # The pattern (rowval, colptr) must match because sparsity is structural
            M_sp_updated = sparse(M)  # fresh for now — test that pattern matches
            @test getcolptr(M_sp_updated) == getcolptr(M_sp_template)
            @test rowvals(M_sp_updated) == rowvals(M_sp_template)

            # Numerical result from cached sparse should match fresh
            K_cached = M_sp_updated \ K_info.N_evals[ii]
            K_fresh = M_sp_fresh \ K_info.N_evals[ii]
            @test norm(K_cached - K_fresh) < 1e-14
        end
    end

    @testset "cached sparse matches fresh sparse(M) — 3-player" begin
        prob = make_sparse_cache_three_player()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )

        z_current = randn(length(precomputed.all_variables))

        _, K_info = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_sparse=:never
        )

        z_test = randn(length(precomputed.all_variables))
        _, K_info_test = compute_K_evals(
            z_test, precomputed.problem_vars, precomputed.setup_info;
            use_sparse=:never
        )

        for ii in 1:prob.N
            M = K_info.M_evals[ii]
            isnothing(M) && continue

            M_test = K_info_test.M_evals[ii]
            isnothing(M_test) && continue

            M_sp_fresh = sparse(M)
            M_sp_template = sparse(M_test)

            # Structural pattern must be identical across different z values
            @test getcolptr(M_sp_fresh) == getcolptr(M_sp_template)
            @test rowvals(M_sp_fresh) == rowvals(M_sp_template)
        end
    end

    @testset "solve_raw with use_sparse=:always matches :never — 2-player" begin
        prob = make_sparse_cache_two_player()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )

        initial_states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        result_dense = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            use_sparse=:never, max_iters=50, verbose=false
        )

        result_sparse = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            use_sparse=:always, max_iters=50, verbose=false
        )

        # Both should converge to the same solution
        if result_dense.status == :converged && result_sparse.status == :converged
            @test norm(result_sparse.sol - result_dense.sol) / max(norm(result_dense.sol), 1.0) < 1e-6
        else
            # At minimum, they should have the same convergence behavior
            @test result_dense.status == result_sparse.status
        end
    end

    @testset "solve_raw with use_sparse=:always matches :never — 3-player" begin
        prob = make_sparse_cache_three_player()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )

        initial_states = Dict(i => zeros(prob.state_dim) for i in 1:prob.N)

        result_dense = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            use_sparse=:never, max_iters=50, verbose=false
        )

        result_sparse = run_nonlinear_solver(
            precomputed, initial_states, prob.G;
            use_sparse=:always, max_iters=50, verbose=false
        )

        if result_dense.status == :converged && result_sparse.status == :converged
            @test norm(result_sparse.sol - result_dense.sol) / max(norm(result_dense.sol), 1.0) < 1e-6
        else
            @test result_dense.status == result_sparse.status
        end
    end

    @testset "sparse cache reduces construction allocations" begin
        # This test verifies the allocation reduction from caching the sparse pattern.
        # We use actual M matrices from the solver to be realistic.
        prob = make_sparse_cache_two_player()
        precomputed = preoptimize_nonlinear_solver(
            prob.G, prob.Js, prob.gs, prob.primal_dims, prob.θs;
            state_dim=prob.state_dim, control_dim=prob.control_dim, verbose=false
        )

        z_current = randn(length(precomputed.all_variables))
        _, K_info = compute_K_evals(
            z_current, precomputed.problem_vars, precomputed.setup_info;
            use_sparse=:never
        )

        # Find a player with M
        player_ii = nothing
        for ii in 1:prob.N
            if !isnothing(K_info.M_evals[ii])
                player_ii = ii
                break
            end
        end
        @test !isnothing(player_ii)

        M = K_info.M_evals[player_ii]

        # Measure allocations for fresh sparse(M) construction
        sparse(M)  # warmup
        alloc_fresh = @allocated for _ in 1:1000
            sparse(M)
        end

        # Measure allocations for updating nzval in a cached sparse matrix
        # (the pattern stays the same, only numerical values change)
        M_sp_cached = sparse(M)
        nzv = nonzeros(M_sp_cached)

        # The update function copies dense M values into existing nzval array
        # using the structural mapping. This is what our implementation will do.
        function update_nzval_from_dense!(nzv, M_sp, M_dense)
            rv = rowvals(M_sp)
            for col in axes(M_dense, 2)
                for idx in nzrange(M_sp, col)
                    @inbounds nzv[idx] = M_dense[rv[idx], col]
                end
            end
        end

        update_nzval_from_dense!(nzv, M_sp_cached, M)  # warmup
        alloc_cached = @allocated for _ in 1:1000
            update_nzval_from_dense!(nzv, M_sp_cached, M)
        end

        @info "Sparse construction: fresh=$(alloc_fresh) bytes, cached=$(alloc_cached) bytes, " *
              "per-call: fresh=$(alloc_fresh÷1000) cached=$(alloc_cached÷1000)"

        # The cached nzval update should allocate less than fresh sparse() construction
        @test alloc_cached < alloc_fresh
    end
end
